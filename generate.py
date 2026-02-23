#!/usr/bin/env python3
"""
Daily Juz Tafsir Generator
Fetches Ibn Kathir tafsir for today's juz from Quran.com API,
summarises it with Claude Haiku, and generates a static HTML page.

Usage:
  python generate.py              # normal run (API + Claude)
  python generate.py --local      # local test with mock data (no API calls)
  JUZ_NUMBER=5 python generate.py # override juz number
"""

import os
import sys
import json
import re
import html as html_mod
from datetime import datetime, timezone
from pathlib import Path

# --- Config ---
TAFSIR_RESOURCE_ID = 169  # Ibn Kathir (English, abridged)
QURAN_API_BASE = "https://api.quran.com/api/v4"
TEMPLATE_PATH = Path(__file__).parent / "templates" / "page.html"
INDEX_TEMPLATE_PATH = Path(__file__).parent / "templates" / "index.html"
OUTPUT_DIR = Path(__file__).parent / "site"

# Juz metadata: transliteration and Arabic names
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

JUZ_NAMES_AR = {
    1: "الٓمٓ", 2: "سَيَقُولُ", 3: "تِلْكَ ٱلرُّسُلُ",
    4: "لَن تَنَالُوا۟", 5: "وَٱلْمُحْصَنَـٰتُ", 6: "لَا يُحِبُّ ٱللَّهُ",
    7: "وَإِذَا سَمِعُوا۟", 8: "وَلَوْ أَنَّنَا", 9: "قَالَ ٱلْمَلَأُ",
    10: "وَٱعْلَمُوٓا۟", 11: "يَعْتَذِرُونَ", 12: "وَمَا مِن دَآبَّةٍ",
    13: "وَمَآ أُبَرِّئُ", 14: "رُبَمَا", 15: "سُبْحَـٰنَ ٱلَّذِىٓ",
    16: "قَالَ أَلَمْ", 17: "ٱقْتَرَبَ", 18: "قَدْ أَفْلَحَ",
    19: "وَقَالَ ٱلَّذِينَ", 20: "أَمَّنْ خَلَقَ", 21: "ٱتْلُ مَآ أُوحِىَ",
    22: "وَمَن يَقْنُتْ", 23: "وَمَآ لِىَ", 24: "فَمَنْ أَظْلَمُ",
    25: "إِلَيْهِ يُرَدُّ", 26: "حمٓ", 27: "قَالَ فَمَا خَطْبُكُمْ",
    28: "قَدْ سَمِعَ ٱللَّهُ", 29: "تَبَـٰرَكَ ٱلَّذِى", 30: "عَمَّ يَتَسَآءَلُونَ",
}

# First verse key of each juz (for fetching Uthmani text)
JUZ_FIRST_VERSE = {
    1: "1:1", 2: "2:142", 3: "2:253", 4: "3:93", 5: "4:24",
    6: "4:148", 7: "5:82", 8: "6:111", 9: "7:88", 10: "8:41",
    11: "9:93", 12: "11:6", 13: "12:53", 14: "15:1", 15: "17:1",
    16: "18:75", 17: "21:1", 18: "23:1", 19: "25:21", 20: "27:56",
    21: "29:46", 22: "33:31", 23: "36:28", 24: "39:32", 25: "41:47",
    26: "46:1", 27: "51:31", 28: "58:1", 29: "67:1", 30: "78:1",
}


def get_today_juz() -> int:
    """Determine which juz to cover today (cycles 1-30)."""
    start_date = datetime(2026, 2, 23, tzinfo=timezone.utc)
    today = datetime.now(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0)
    day_number = (today - start_date).days
    return (day_number % 30) + 1


# ---------------------------------------------------------------------------
# Fetch tafsir from Quran.com API
# ---------------------------------------------------------------------------

def get_juz_verse_keys(juz_number: int) -> list[str]:
    """Get all verse keys (e.g. '4:148') for a given juz."""
    import requests
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


def fetch_uthmani_text(verse_keys: list[str]) -> dict[str, str]:
    """Fetch Uthmani script text for a list of verse keys."""
    import requests
    uthmani = {}
    # Batch by chunks of 50 to avoid huge URLs
    for i in range(0, len(verse_keys), 50):
        batch = verse_keys[i:i+50]
        keys_param = ",".join(batch)
        url = f"{QURAN_API_BASE}/quran/verses/uthmani"
        resp = requests.get(url, params={"verse_key": keys_param}, timeout=30)
        resp.raise_for_status()
        for v in resp.json().get("verses", []):
            uthmani[v["verse_key"]] = v.get("text_uthmani", "")
    return uthmani


def fetch_tafsir(juz_number: int) -> list[dict]:
    """Fetch tafsir and Uthmani text for every verse in a juz."""
    import requests
    verse_keys = get_juz_verse_keys(juz_number)

    # Fetch Uthmani text for all verses
    print("  Fetching Uthmani text...")
    uthmani = fetch_uthmani_text(verse_keys)

    tafsirs = []
    for vk in verse_keys:
        url = f"{QURAN_API_BASE}/tafsirs/{TAFSIR_RESOURCE_ID}/by_ayah/{vk}"
        resp = requests.get(url, timeout=30)
        resp.raise_for_status()
        data = resp.json().get("tafsir", {})
        text = data.get("text", "")
        if text.strip():
            tafsirs.append({
                "verse_key": vk,
                "text": text,
                "uthmani": uthmani.get(vk, ""),
            })

    return tafsirs


# ---------------------------------------------------------------------------
# Mock data for --local testing
# ---------------------------------------------------------------------------

def mock_tafsir(juz_number: int) -> list[dict]:
    """Return sample tafsir entries for local testing without API calls."""
    samples = [
        {
            "verse_key": f"{juz_number}:1",
            "uthmani": "بِسْمِ ٱللَّهِ ٱلرَّحْمَـٰنِ ٱلرَّحِيمِ",
            "text": (
                "<h2>Commentary on the Opening Verse</h2>"
                "<p>Ibn Kathir explains that this verse establishes the foundational "
                "theme of the surah. The scholars have noted its significance in "
                "understanding the broader context of divine guidance.</p>"
                "<p>Al-Tabari and others have reported that this verse was revealed "
                "in connection with the events following the migration to Madinah.</p>"
            ),
        },
        {
            "verse_key": f"{juz_number}:2",
            "uthmani": "ٱلْحَمْدُ لِلَّهِ رَبِّ ٱلْعَـٰلَمِينَ",
            "text": (
                "<h2>The Command to Reflect</h2>"
                "<p>This verse calls upon the believers to reflect deeply on the "
                "signs of Allah in creation. Ibn Kathir draws upon multiple hadith "
                "to illustrate how the Prophet (peace be upon him) exemplified this "
                "quality of contemplation.</p>"
            ),
        },
        {
            "verse_key": f"{juz_number}:3",
            "uthmani": "ٱلرَّحْمَـٰنِ ٱلرَّحِيمِ",
            "text": (
                "<h2>Rulings and Guidance</h2>"
                "<p>Here the tafsir elaborates on the specific rulings derived from "
                "this verse, including matters of worship, social conduct, and the "
                "importance of maintaining family ties. The scholars of fiqh have "
                "derived several important principles from this passage.</p>"
            ),
        },
    ]
    return samples


def mock_summary(juz_number: int) -> str:
    """Return a sample markdown summary for local testing."""
    name = JUZ_NAMES.get(juz_number, "")
    return f"""## Juz {juz_number} — {name}

This juz covers key themes of **divine guidance**, **reflection on creation**, and **practical rulings** for the Muslim community.

### Major Themes

The opening verse ({juz_number}:1) establishes the foundational message of the surah, connecting the believers to the broader narrative of prophetic history. Ibn Kathir draws upon classical scholarship to illuminate the depth of each verse.

In {juz_number}:2, we find a powerful call to reflect upon the signs of Allah in creation. The scholars have noted how this connects to the broader themes of gratitude and awareness.

### Key Rulings

- Matters of worship and their proper observance ({juz_number}:1)
- Social conduct and the rights of others ({juz_number}:2)
- The importance of maintaining family ties and community bonds ({juz_number}:3)

### Spiritual Lessons

The tafsir elaborates on the specific rulings in {juz_number}:3, emphasising that true understanding of the Quran requires both intellectual engagement and spiritual sincerity. The scholars remind us that each verse carries layers of meaning that reveal themselves to those who approach with humility.

### Overarching Message

Juz {juz_number} calls upon the believers to combine faith with action, knowledge with practice, and individual devotion with communal responsibility. This is the path to success in both worlds."""


# ---------------------------------------------------------------------------
# Text processing
# ---------------------------------------------------------------------------

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
    html = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", html)
    return html


def linkify_verses(html: str, verse_keys: set[str]) -> str:
    """Replace verse references (e.g. 4:148) with links to the full tafsir section."""
    def replace_match(m):
        ref = m.group(0)
        if ref in verse_keys:
            verse_id = f"v-{ref.replace(':', '-')}"
            return (
                f'<a href="#{verse_id}" class="verse-ref" '
                f'onclick="openVerse(\'{verse_id}\')">{ref}</a>'
            )
        return ref

    # Match patterns like 2:142, 4:148, 112:1 — but not inside existing tags/attributes
    return re.sub(r'(?<![#\w/-])(\d{1,3}:\d{1,3})(?!["\w])', replace_match, html)


def build_plain_text(tafsirs: list[dict]) -> str:
    """Combine all tafsir entries into a single plain text document."""
    parts = []
    for t in tafsirs:
        verse_key = t.get("verse_key", "")
        text = strip_html_tags(t.get("text", ""))
        if text.strip():
            parts.append(f"[{verse_key}]\n{text}")
    return "\n\n".join(parts)


# ---------------------------------------------------------------------------
# Summarisation (chunk-and-merge)
# ---------------------------------------------------------------------------

def chunk_text(plain_text: str, max_chars: int = 120_000) -> list[str]:
    """Split tafsir text into chunks that fit within model context.

    Splits on verse boundaries (double newlines) to keep entries intact.
    """
    if len(plain_text) <= max_chars:
        return [plain_text]

    chunks = []
    entries = plain_text.split("\n\n")
    current = []
    current_len = 0

    for entry in entries:
        entry_len = len(entry) + 2
        if current_len + entry_len > max_chars and current:
            chunks.append("\n\n".join(current))
            current = []
            current_len = 0
        current.append(entry)
        current_len += entry_len

    if current:
        chunks.append("\n\n".join(current))

    return chunks


def summarise(plain_text: str, juz_number: int) -> str:
    """Chunk the tafsir, summarise each chunk, then merge into a final summary."""
    import anthropic
    client = anthropic.Anthropic()
    chunks = chunk_text(plain_text)

    if len(chunks) == 1:
        return _summarise_single(client, plain_text, juz_number)

    # Phase 1: summarise each chunk
    print(f"  Text split into {len(chunks)} chunks for summarisation")
    chunk_summaries = []
    for i, chunk in enumerate(chunks, 1):
        print(f"  Summarising chunk {i}/{len(chunks)}...")
        message = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=1024,
            messages=[{"role": "user", "content": f"""Summarise this section of Tafsir Ibn Kathir from Juz {juz_number}.
Cover the key themes, stories, rulings, and lessons. Be thorough — this will be merged with other section summaries.
Write 200-300 words.

{chunk}"""}],
        )
        chunk_summaries.append(message.content[0].text)

    # Phase 2: merge into final summary
    print("  Merging chunk summaries into final summary...")
    merged_input = "\n\n---\n\n".join(
        f"Section {i}:\n{s}" for i, s in enumerate(chunk_summaries, 1)
    )

    message = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=1500,
        messages=[{"role": "user", "content": f"""You are writing the final summary for Juz {juz_number} of the Quran's Tafsir Ibn Kathir, for a Muslim audience.

Below are summaries of each section of the juz. Merge them into one cohesive summary that:
- Opens with which surahs/verses this juz covers
- Highlights the major themes, stories, and lessons
- Notes any key rulings or guidance mentioned
- References specific verses using the format surah:verse (e.g. 5:82, 6:1) when discussing key points, so readers can look them up
- Closes with the overarching message of the juz
- Is written in clear, accessible English
- Is around 500-700 words
- Flows naturally as one piece (not a list of sections)

Section summaries:

{merged_input}"""}],
    )
    return message.content[0].text


def _summarise_single(client, plain_text: str, juz_number: int) -> str:
    """Summarise when the full text fits in one call."""
    message = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=1024,
        messages=[{"role": "user", "content": f"""You are summarising Juz {juz_number} of the Quran's Tafsir Ibn Kathir for a Muslim audience.

Write a detailed overview summary that:
- Opens with which surahs/verses this juz covers
- Highlights the major themes, stories, and lessons
- Notes any key rulings or guidance mentioned
- References specific verses using the format surah:verse (e.g. 5:82, 6:1) when discussing key points, so readers can look them up
- Closes with the overarching message of the juz
- Is written in clear, accessible English
- Is around 400-600 words

Here is the full tafsir text:

{plain_text}"""}],
    )
    return message.content[0].text


# ---------------------------------------------------------------------------
# HTML generation
# ---------------------------------------------------------------------------

def build_html(juz_number: int, summary: str, tafsirs: list[dict], word_count: int) -> str:
    """Render the final HTML page from the template."""
    template = TEMPLATE_PATH.read_text()

    tafsir_sections = []
    jumpbar_links = []
    for t in tafsirs:
        verse_key = t.get("verse_key", "")
        text = t.get("text", "")
        uthmani = t.get("uthmani", "")
        if text.strip():
            verse_id = f"v-{verse_key.replace(':', '-')}"
            uthmani_html = (
                f'<div class="verse-arabic">{uthmani}</div>' if uthmani else ""
            )
            # Truncated Arabic preview for the collapsed header
            preview_ar = uthmani[:60] + "..." if len(uthmani) > 60 else uthmani
            tafsir_sections.append(
                f'<div class="verse-tafsir" id="{verse_id}">'
                f'<div class="verse-header">'
                f'<div class="verse-header-left">'
                f'<span class="verse-key">{html_mod.escape(verse_key)}</span>'
                f'<span class="verse-preview-ar">{preview_ar}</span>'
                f'</div>'
                f'<span class="verse-toggle-icon">&#9662;</span>'
                f'</div>'
                f'<div class="verse-body">'
                f'{uthmani_html}'
                f'<div class="verse-text">{text}</div>'
                f'</div>'
                f'</div>'
            )
            jumpbar_links.append(
                f'<a href="#{verse_id}">{html_mod.escape(verse_key)}</a>'
            )
    full_tafsir_html = "\n".join(tafsir_sections)
    jumpbar_html = "\n".join(jumpbar_links)

    today = datetime.now(timezone.utc).strftime("%A, %d %B %Y")
    juz_name = JUZ_NAMES.get(juz_number, "")
    juz_name_ar = JUZ_NAMES_AR.get(juz_number, "")

    summary_html = markdown_to_html(summary)
    # Link verse references in the summary to the full tafsir cards
    available_verses = {t.get("verse_key", "") for t in tafsirs}
    summary_html = linkify_verses(summary_html, available_verses)

    output = template.replace("{{JUZ_NUMBER}}", str(juz_number))
    output = output.replace("{{JUZ_NAME_AR}}", juz_name_ar)
    output = output.replace("{{JUZ_NAME}}", html_mod.escape(juz_name))
    output = output.replace("{{DATE}}", today)
    output = output.replace("{{SUMMARY}}", summary_html)
    output = output.replace("{{VERSE_JUMPBAR}}", jumpbar_html)
    output = output.replace("{{FULL_TAFSIR}}", full_tafsir_html)
    output = output.replace("{{WORD_COUNT}}", f"{word_count:,}")

    return output


def build_index(manifest: dict) -> str:
    """Render the archive index page from the template."""
    template = INDEX_TEMPLATE_PATH.read_text()

    # Find the latest juz
    # Highest juz number is always the latest (sequential daily generation)
    latest_juz = max((int(k) for k in manifest), default=None)
    latest_date = manifest[str(latest_juz)]["date"] if latest_juz else ""

    # Latest card
    if latest_juz:
        name = JUZ_NAMES.get(latest_juz, "")
        name_ar = JUZ_NAMES_AR.get(latest_juz, "")
        date_obj = datetime.strptime(latest_date, "%Y-%m-%d")
        date_str = date_obj.strftime("%A, %d %B %Y")
        latest_card = (
            f'<a class="latest-card" href="juz-{latest_juz}.html">'
            f'<div class="latest-badge">Latest</div>'
            f'<div class="latest-arabic">{name_ar}</div>'
            f'<h2>Juz {latest_juz} — {html_mod.escape(name)}</h2>'
            f'<div class="meta">{date_str} &middot; '
            f'{manifest[str(latest_juz)]["word_count"]:,} words</div>'
            f'</a>'
        )
    else:
        latest_card = ""

    # Build cards for all 30 juz
    cards = []
    for n in range(1, 31):
        name = JUZ_NAMES.get(n, "")
        name_ar = JUZ_NAMES_AR.get(n, "")
        key = str(n)
        if key in manifest:
            date_obj = datetime.strptime(manifest[key]["date"], "%Y-%m-%d")
            date_str = date_obj.strftime("%d %b %Y")
            cards.append(
                f'<a class="juz-card" href="juz-{n}.html">'
                f'<div class="juz-num">Juz {n}</div>'
                f'<div class="juz-arabic">{name_ar}</div>'
                f'<div class="juz-title">{html_mod.escape(name)}</div>'
                f'<div class="juz-date">{date_str}</div>'
                f'</a>'
            )
        else:
            cards.append(
                f'<div class="juz-card upcoming">'
                f'<div class="juz-num">Juz {n}</div>'
                f'<div class="juz-arabic">{name_ar}</div>'
                f'<div class="juz-title">{html_mod.escape(name)}</div>'
                f'<div class="juz-date">Coming soon</div>'
                f'</div>'
            )

    output = template.replace("{{LATEST_CARD}}", latest_card)
    output = output.replace("{{JUZ_CARDS}}", "\n            ".join(cards))
    return output


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    local_mode = "--local" in sys.argv

    juz_number_override = os.environ.get("JUZ_NUMBER")
    juz_number = int(juz_number_override) if juz_number_override else get_today_juz()

    print(f"Generating tafsir for Juz {juz_number} ({JUZ_NAMES.get(juz_number, '')})...")
    if local_mode:
        print("  [LOCAL MODE] Using mock data — no API calls")

    # Fetch or mock
    if local_mode:
        tafsirs = mock_tafsir(juz_number)
    else:
        print("Fetching tafsir from Quran.com API...")
        tafsirs = fetch_tafsir(juz_number)
    print(f"  Got {len(tafsirs)} tafsir entries")

    # Build plain text and count words
    plain_text = build_plain_text(tafsirs)
    word_count = len(plain_text.split())
    print(f"  Total words: {word_count:,}")

    # Summarise or mock
    if local_mode:
        summary = mock_summary(juz_number)
    else:
        print("Summarising with Claude Haiku...")
        summary = summarise(plain_text, juz_number)
    print(f"  Summary: {len(summary.split())} words")

    # Generate HTML
    print("Generating HTML pages...")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    page_html = build_html(juz_number, summary, tafsirs, word_count)

    # Write juz page
    (OUTPUT_DIR / f"juz-{juz_number}.html").write_text(page_html)

    # Update manifest
    manifest_path = OUTPUT_DIR / "manifest.json"
    manifest = {}
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
    manifest[str(juz_number)] = {
        "date": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
        "word_count": word_count,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2))

    # Generate index/archive page
    index_html = build_index(manifest)
    (OUTPUT_DIR / "index.html").write_text(index_html)

    print(f"Done! Output: site/juz-{juz_number}.html + site/index.html")


if __name__ == "__main__":
    main()
