#!/usr/bin/env python3
"""
Daily Juz Tafsir Generator
Fetches Ibn Kathir tafsir for today's juz from Quran.com API,
summarises it with Claude Haiku, and generates a static HTML page.
"""

import os
import json
import re
import html as html_mod
import requests
import anthropic
from datetime import datetime, timezone
from pathlib import Path

# --- Config ---
TAFSIR_RESOURCE_ID = 169  # Ibn Kathir (English, abridged)
QURAN_API_BASE = "https://api.quran.com/api/v4"
TEMPLATE_PATH = Path(__file__).parent / "templates" / "page.html"
OUTPUT_DIR = Path(__file__).parent / "site"

# Juz metadata: which surahs each juz covers (approximate descriptions)
JUZ_NAMES = {
    1: "Alif Lam Mim", 2: "Sayaqool", 3: "Tilkal Rusul",
    4: "Lan Tanaloo", 5: "Wal Muhsanat", 6: "La Yuhibbullah",
    7: "Wa Iza Sami'oo", 8: "Wa Lau Annana", 9: "Qalal Mala'u",
    10: "Wa A'lamu", 11: "Ya'tazeroon", 12: "Wa Ma Min Dabbah",
    13: "Wa Ma Ubarri'u", 14: "Rubama", 15: "Subhanallazi",
    16: "Qal Alam", 17: "Iqtaraba", 18: "Qad Aflaha",
    19: "Wa Qalallazina", 20: "A'man Khalaq", 21: "Utlu Ma Oohiya",
    22: "Wa Man Yaqnut", 23: "Wa Mali", 24: "Faman Azlamu",
    25: "Ilayhi Yuraddu", 26: "Ha Mim", 27: "Qala Fama Khatbukum",
    28: "Qad Sami Allahu", 29: "Tabarakallazi", 30: "Amma Yatasa'aloon",
}


def get_today_juz() -> int:
    """Determine which juz to cover today (cycles 1-30)."""
    # Use a fixed start date so the cycle is predictable
    start_date = datetime(2026, 2, 23, tzinfo=timezone.utc)
    today = datetime.now(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0)
    day_number = (today - start_date).days
    return (day_number % 30) + 1


def get_juz_verse_keys(juz_number: int) -> list[str]:
    """Get all verse keys (e.g. '4:148') for a given juz."""
    resp = requests.get(f"{QURAN_API_BASE}/juzs", timeout=30)
    resp.raise_for_status()
    juzs = resp.json().get("juzs", [])

    juz = next((j for j in juzs if j["juz_number"] == juz_number), None)
    if not juz:
        raise ValueError(f"Juz {juz_number} not found")

    verse_keys = []
    for surah, verse_range in juz["verse_mapping"].items():
        start, end = verse_range.split("-")
        for v in range(int(start), int(end) + 1):
            verse_keys.append(f"{surah}:{v}")
    return verse_keys


def fetch_tafsir(juz_number: int) -> list[dict]:
    """Fetch tafsir for every verse in a juz via the per-ayah endpoint."""
    verse_keys = get_juz_verse_keys(juz_number)
    tafsirs = []

    for vk in verse_keys:
        url = f"{QURAN_API_BASE}/tafsirs/{TAFSIR_RESOURCE_ID}/by_ayah/{vk}"
        resp = requests.get(url, timeout=30)
        resp.raise_for_status()
        data = resp.json().get("tafsir", {})
        text = data.get("text", "")
        if text.strip():
            tafsirs.append({"verse_key": vk, "text": text})

    return tafsirs


def strip_html_tags(text: str) -> str:
    """Remove HTML tags for the plain text version sent to Claude."""
    clean = re.sub(r"<[^>]+>", "", text)
    return html_mod.unescape(clean)


def markdown_to_html(text: str) -> str:
    """Convert basic markdown to HTML."""
    lines = text.split("\n")
    result = []
    in_list = False

    for line in lines:
        stripped = line.strip()

        # Headers
        if stripped.startswith("### "):
            if in_list:
                result.append("</ul>")
                in_list = False
            result.append(f"<h4>{stripped[4:]}</h4>")
        elif stripped.startswith("## "):
            if in_list:
                result.append("</ul>")
                in_list = False
            result.append(f"<h3>{stripped[3:]}</h3>")
        elif stripped.startswith("# "):
            if in_list:
                result.append("</ul>")
                in_list = False
            result.append(f"<h3>{stripped[2:]}</h3>")
        elif stripped.startswith("- "):
            if not in_list:
                result.append("<ul>")
                in_list = True
            result.append(f"<li>{stripped[2:]}</li>")
        elif stripped == "":
            if in_list:
                result.append("</ul>")
                in_list = False
            result.append("")
        else:
            if in_list:
                result.append("</ul>")
                in_list = False
            result.append(f"<p>{stripped}</p>")

    if in_list:
        result.append("</ul>")

    html = "\n".join(result)
    # Bold
    html = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", html)
    return html


def build_plain_text(tafsirs: list[dict]) -> str:
    """Combine all tafsir entries into a single plain text document."""
    parts = []
    for t in tafsirs:
        verse_key = t.get("verse_key", "")
        text = strip_html_tags(t.get("text", ""))
        if text.strip():
            parts.append(f"[{verse_key}]\n{text}")
    return "\n\n".join(parts)


def summarise(plain_text: str, juz_number: int) -> str:
    """Send the tafsir text to Claude Haiku for summarisation."""
    client = anthropic.Anthropic()  # uses ANTHROPIC_API_KEY env var

    prompt = f"""You are summarising Juz {juz_number} of the Quran's Tafsir Ibn Kathir for a Muslim audience.

Write a detailed overview summary that:
- Opens with which surahs/verses this juz covers
- Highlights the major themes, stories, and lessons
- Notes any key rulings or guidance mentioned
- Closes with the overarching message of the juz
- Is written in clear, accessible English
- Is around 400-600 words

Here is the full tafsir text:

{plain_text}"""

    message = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=1024,
        messages=[{"role": "user", "content": prompt}],
    )
    return message.content[0].text


def build_html(juz_number: int, summary: str, tafsirs: list[dict], word_count: int) -> str:
    """Render the final HTML page from the template."""
    template = TEMPLATE_PATH.read_text()

    # Build the full tafsir HTML (preserving the API's HTML formatting)
    tafsir_sections = []
    for t in tafsirs:
        verse_key = t.get("verse_key", "")
        text = t.get("text", "")
        if text.strip():
            tafsir_sections.append(
                f'<div class="verse-tafsir">'
                f'<h3 class="verse-key">{html_mod.escape(verse_key)}</h3>'
                f'<div class="verse-text">{text}</div>'
                f'</div>'
            )
    full_tafsir_html = "\n".join(tafsir_sections)

    today = datetime.now(timezone.utc).strftime("%A, %d %B %Y")
    juz_name = JUZ_NAMES.get(juz_number, "")

    # Convert markdown summary to HTML
    summary_html = markdown_to_html(summary)

    # Replace placeholders
    output = template.replace("{{JUZ_NUMBER}}", str(juz_number))
    output = output.replace("{{JUZ_NAME}}", html_mod.escape(juz_name))
    output = output.replace("{{DATE}}", today)
    output = output.replace("{{SUMMARY}}", summary_html)
    output = output.replace("{{FULL_TAFSIR}}", full_tafsir_html)
    output = output.replace("{{WORD_COUNT}}", f"{word_count:,}")

    return output


def main():
    juz_number_override = os.environ.get("JUZ_NUMBER")
    juz_number = int(juz_number_override) if juz_number_override else get_today_juz()

    print(f"Generating tafsir for Juz {juz_number} ({JUZ_NAMES.get(juz_number, '')})...")

    # Fetch
    print("Fetching tafsir from Quran.com API...")
    tafsirs = fetch_tafsir(juz_number)
    print(f"  Got {len(tafsirs)} tafsir entries")

    # Build plain text and count words
    plain_text = build_plain_text(tafsirs)
    word_count = len(plain_text.split())
    print(f"  Total words: {word_count:,}")

    # Summarise
    print("Summarising with Claude Haiku...")
    summary = summarise(plain_text, juz_number)
    print(f"  Summary: {len(summary.split())} words")

    # Generate HTML
    print("Generating HTML page...")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    page_html = build_html(juz_number, summary, tafsirs, word_count)

    # Write today's page and the index
    (OUTPUT_DIR / f"juz-{juz_number}.html").write_text(page_html)
    (OUTPUT_DIR / "index.html").write_text(page_html)

    # Write a manifest for the archive page
    manifest_path = OUTPUT_DIR / "manifest.json"
    manifest = {}
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
    manifest[str(juz_number)] = {
        "date": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
        "word_count": word_count,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2))

    print(f"Done! Output: site/juz-{juz_number}.html")


if __name__ == "__main__":
    main()
