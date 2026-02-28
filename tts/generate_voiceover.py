"""Generate AI voiceovers for juz summaries using Chatterbox TTS.

Workflow:
  1. Place full script at tts/scripts/juz10.txt
  2. Run: python tts/generate_voiceover.py juz10
  3. Script auto-chunks by theme + word count (max 65 words per chunk)
  4. Generates audio for each chunk with quality validation
  5. Auto-retries bad generations (truncated, hallucinated, wrong pace)
  6. Trims silence, joins into final wav

Quality checks per chunk:
  - Token ceiling: rejects any chunk hitting exactly 40.0s (truncated)
  - Speech rate: expects 2.0-3.5 words/sec (flags hallucination/drift)
  - Minimum duration: rejects suspiciously short audio
  - Retries up to MAX_RETRIES times with increasing temperature
"""
import torch
import torchaudio as ta
import os
import re
import glob
import sys
import subprocess

# --- Mac MPS/CPU setup ---
device = "mps" if torch.backends.mps.is_available() else "cpu"
print(f"Using device: {device}")

_original_load = torch.load
def _patched_load(*args, **kwargs):
    if "map_location" not in kwargs:
        kwargs["map_location"] = torch.device(device)
    return _original_load(*args, **kwargs)
torch.load = _patched_load

from chatterbox.tts import ChatterboxTTS

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
VOICE_REF = os.path.join(BASE_DIR, "reference", "calm-narrator.wav")
SCRIPTS_DIR = os.path.join(BASE_DIR, "scripts")
AUDIO_DIR = os.path.join(BASE_DIR, "audio")

TTS_PARAMS = {
    "exaggeration": 0.2,
    "cfg_weight": 1.5,
    "temperature": 0.5,
}

MAX_WORDS = 65          # safe limit — keeps well under 1000-token ceiling
MIN_WORDS = 30          # minimum chunk size — merge small orphans back
SILENCE_SECS = 0.7      # pause between chunks in final output
SILENCE_THRESH_DB = -35  # threshold for trimming trailing silence
MAX_INTERNAL_SILENCE = 0.5  # compress any internal silence longer than this

# Quality thresholds
TOKEN_CEILING_SECS = 39.5  # anything >= this likely hit the 40s token limit
MIN_WORDS_PER_SEC = 1.8    # below this = too slow / padding / silence
MAX_WORDS_PER_SEC = 4.0    # above this = too fast / skipping words
MIN_DURATION_SECS = 3.0    # suspiciously short
MAX_RETRIES = 3            # retry bad generations this many times


# ---------------------------------------------------------------------------
# Auto-chunking
# ---------------------------------------------------------------------------

def is_section_header(line):
    """Detect section headers: short lines without ending punctuation."""
    line = line.strip()
    if not line:
        return False
    if len(line.split()) > 8:
        return False
    if line[-1] in ".!?,;:":
        return False
    return True


def split_into_sentences(text):
    """Split text into sentences, keeping the delimiter attached."""
    parts = re.split(r'(?<=[.!?])\s+', text.strip())
    return [p.strip() for p in parts if p.strip()]


def chunk_script(text):
    """Split a full juz script into TTS-ready chunks (max MAX_WORDS each).

    Strategy:
      1. Split on blank lines into paragraphs/sections
      2. Group: if a line is a section header, merge it with the next paragraph
      3. If a group exceeds MAX_WORDS, split on sentence boundaries
    """
    lines = text.strip().split("\n")

    # Step 1: group into paragraphs (split on blank lines)
    paragraphs = []
    current = []
    for line in lines:
        if line.strip() == "":
            if current:
                paragraphs.append("\n".join(current))
                current = []
        else:
            current.append(line.strip())
    if current:
        paragraphs.append("\n".join(current))

    # Step 2: merge section headers with following paragraph
    merged = []
    i = 0
    while i < len(paragraphs):
        para_lines = paragraphs[i].split("\n")
        # If this paragraph is just a header line, merge with next
        if len(para_lines) == 1 and is_section_header(para_lines[0]) and i + 1 < len(paragraphs):
            merged.append(para_lines[0] + ". " + paragraphs[i + 1].replace("\n", " "))
            i += 2
        else:
            merged.append(paragraphs[i].replace("\n", " "))
            i += 1

    # Step 3: split any group exceeding MAX_WORDS on sentence boundaries
    chunks = []
    for group in merged:
        words = group.split()
        if len(words) <= MAX_WORDS:
            chunks.append(group)
            continue

        sentences = split_into_sentences(group)
        current_chunk = []
        current_words = 0

        for sentence in sentences:
            s_words = len(sentence.split())
            if current_words + s_words > MAX_WORDS and current_chunk:
                chunks.append(" ".join(current_chunk))
                current_chunk = [sentence]
                current_words = s_words
            else:
                current_chunk.append(sentence)
                current_words += s_words

        if current_chunk:
            chunks.append(" ".join(current_chunk))

    # Merge small orphan chunks into neighbors
    # Allow slight overflow (up to MAX_WORDS + 15) to avoid tiny orphans
    merge_limit = MAX_WORDS + 15
    final = []
    i = 0
    while i < len(chunks):
        wc = len(chunks[i].split())

        if wc < MIN_WORDS:
            # Try merge with previous
            if final:
                prev_wc = len(final[-1].split())
                if prev_wc + wc <= merge_limit:
                    final[-1] = final[-1] + " " + chunks[i]
                    i += 1
                    continue
            # Try merge with next
            if i + 1 < len(chunks):
                next_wc = len(chunks[i + 1].split())
                if wc + next_wc <= merge_limit:
                    chunks[i + 1] = chunks[i] + " " + chunks[i + 1]
                    i += 1
                    continue

        final.append(chunks[i])
        i += 1

    return final


# ---------------------------------------------------------------------------
# Audio generation + quality validation
# ---------------------------------------------------------------------------

def validate_chunk(wav, sr, word_count):
    """Check if generated audio passes quality gates.

    Returns (ok, reason) tuple.
    """
    dur = wav.shape[1] / sr
    wps = word_count / dur if dur > 0 else 0

    if dur >= TOKEN_CEILING_SECS:
        return False, f"hit token ceiling ({dur:.1f}s >= {TOKEN_CEILING_SECS}s)"

    if dur < MIN_DURATION_SECS and word_count > 10:
        return False, f"too short ({dur:.1f}s for {word_count} words)"

    if wps < MIN_WORDS_PER_SEC:
        return False, f"too slow ({wps:.1f} w/s < {MIN_WORDS_PER_SEC})"

    if wps > MAX_WORDS_PER_SEC:
        return False, f"too fast ({wps:.1f} w/s > {MAX_WORDS_PER_SEC})"

    return True, f"ok ({dur:.1f}s, {wps:.1f} w/s)"


def generate_chunk(model, text, out_path):
    """Generate a single chunk with quality validation and retries."""
    word_count = len(text.split())
    best_wav = None
    best_dur = 0
    best_reason = ""

    for attempt in range(1, MAX_RETRIES + 1):
        # Slightly vary temperature on retries to get different output
        params = TTS_PARAMS.copy()
        if attempt > 1:
            params["temperature"] = min(0.8, params["temperature"] + 0.1 * (attempt - 1))
            print(f"    Retry {attempt}/{MAX_RETRIES} (temp={params['temperature']:.1f})")

        wav = model.generate(text, audio_prompt_path=VOICE_REF, **params)
        dur = wav.shape[1] / model.sr
        ok, reason = validate_chunk(wav, model.sr, word_count)

        if ok:
            ta.save(out_path, wav, model.sr)
            return dur, reason

        print(f"    FAILED: {reason}")

        # Keep the best attempt so far (closest to expected duration)
        expected_dur = word_count / 2.7  # ~2.7 w/s is typical
        if best_wav is None or abs(dur - expected_dur) < abs(best_dur - expected_dur):
            best_wav = wav
            best_dur = dur
            best_reason = reason

    # All retries failed — use best attempt with warning
    print(f"    WARNING: using best of {MAX_RETRIES} attempts ({best_reason})")
    ta.save(out_path, best_wav, model.sr)
    return best_dur, f"WARN: {best_reason}"


def trim_silence(in_path, out_path):
    """Trim trailing silence and compress internal silence gaps."""
    af_filters = (
        f"silenceremove=stop_periods=-1"
        f":stop_duration={MAX_INTERNAL_SILENCE}"
        f":stop_threshold={SILENCE_THRESH_DB}dB,"
        f"areverse,"
        f"silenceremove=start_periods=1"
        f":start_silence=0.1"
        f":start_threshold={SILENCE_THRESH_DB}dB,"
        f"areverse"
    )
    cmd = ["ffmpeg", "-y", "-i", in_path, "-af", af_filters, out_path]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        subprocess.run(["cp", in_path, out_path])
        return 0

    dur_cmd = subprocess.run(
        ["ffprobe", "-v", "quiet", "-show_entries", "format=duration",
         "-of", "default=noprint_wrappers=1:nokey=1", out_path],
        capture_output=True, text=True,
    )
    return float(dur_cmd.stdout.strip()) if dur_cmd.stdout.strip() else 0


def concat_chunks(chunk_paths, out_path):
    """Join chunk WAVs with silence gaps using ffmpeg."""
    inputs = []
    filter_parts = []
    for i, path in enumerate(chunk_paths):
        inputs.extend(["-i", path])
        filter_parts.append(f"[{i}:a]aresample=24000[a{i}]")

    n = len(chunk_paths)
    concat_inputs = ""
    for i in range(n):
        if i > 0:
            filter_parts.append(f"aevalsrc=0:s=24000:d={SILENCE_SECS}[s{i}]")
            concat_inputs += f"[s{i}]"
        concat_inputs += f"[a{i}]"

    total_streams = n + (n - 1)
    filter_parts.append(f"{concat_inputs}concat=n={total_streams}:v=0:a=1[out]")
    filter_str = ";".join(filter_parts)

    cmd = ["ffmpeg", "-y"] + inputs + [
        "-filter_complex", filter_str,
        "-map", "[out]",
        "-c:a", "pcm_s16le",
        out_path,
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"  ffmpeg error: {result.stderr[-500:]}")
        sys.exit(1)

    dur_cmd = subprocess.run(
        ["ffprobe", "-v", "quiet", "-show_entries", "format=duration",
         "-of", "default=noprint_wrappers=1:nokey=1", out_path],
        capture_output=True, text=True,
    )
    total_dur = float(dur_cmd.stdout.strip())
    size_mb = os.path.getsize(out_path) / (1024 * 1024)
    print(f"\nFinal: {out_path}")
    print(f"  Duration: {int(total_dur//60)}m {total_dur%60:.0f}s | Size: {size_mb:.1f} MB")


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def process_juz(model, juz_name, text, regen_chunks=None):
    """Full pipeline: chunk → generate → trim → join.

    Args:
        regen_chunks: optional set of chunk numbers to regenerate (e.g. {1, 7}).
                      If None, generates all missing chunks.
    """
    chunks = chunk_script(text)

    print(f"\n{'='*60}")
    print(f"  {juz_name}: {len(text.split())} words → {len(chunks)} chunks")
    print(f"{'='*60}")
    for i, c in enumerate(chunks):
        print(f"  [{i+1:02d}] {len(c.split()):3d} words | {c[:70]}...")
    print()

    chunk_dir = os.path.join(AUDIO_DIR, juz_name, "chunks")
    trimmed_dir = os.path.join(AUDIO_DIR, juz_name, "trimmed")
    os.makedirs(chunk_dir, exist_ok=True)
    os.makedirs(trimmed_dir, exist_ok=True)

    scripts_out = os.path.join(SCRIPTS_DIR, juz_name)
    os.makedirs(scripts_out, exist_ok=True)

    results = []
    trimmed_paths = []
    for i, chunk_text in enumerate(chunks):
        chunk_num = i + 1
        chunk_name = f"chunk{chunk_num:02d}"
        raw_path = os.path.join(chunk_dir, f"{chunk_name}.wav")
        trim_path = os.path.join(trimmed_dir, f"{chunk_name}.wav")
        script_path = os.path.join(scripts_out, f"{chunk_name}.txt")
        trimmed_paths.append(trim_path)

        with open(script_path, "w") as f:
            f.write(chunk_text)

        # Skip if exists and not in regen set
        if os.path.exists(trim_path) and (regen_chunks is None or chunk_num not in regen_chunks):
            print(f"  Skipping {chunk_name} (already exists)")
            results.append((chunk_name, "skipped", ""))
            continue

        # Remove old files if regenerating
        for p in [raw_path, trim_path]:
            if os.path.exists(p):
                os.remove(p)

        print(f"\n[{chunk_num}/{len(chunks)}] {chunk_name} ({len(chunk_text.split())} words)")
        dur, reason = generate_chunk(model, chunk_text, raw_path)
        print(f"  Raw: {dur:.1f}s [{reason}]", end="")

        trim_dur = trim_silence(raw_path, trim_path)
        print(f" → Trimmed: {trim_dur:.1f}s (cut {dur - trim_dur:.1f}s)")
        results.append((chunk_name, reason, f"{trim_dur:.1f}s"))

    # Print quality report
    print(f"\n{'='*60}")
    print(f"  Quality Report")
    print(f"{'='*60}")
    warnings = 0
    for name, reason, dur in results:
        status = "SKIP" if reason == "skipped" else ("WARN" if "WARN" in reason else "OK")
        if "WARN" in reason:
            warnings += 1
        print(f"  {name}: [{status}] {reason} {dur}")

    if warnings:
        print(f"\n  ⚠ {warnings} chunk(s) had quality warnings — listen and re-run with --regen if needed")

    # Join all trimmed chunks
    final_path = os.path.join(AUDIO_DIR, f"{juz_name}.wav")
    print(f"\nJoining {len(trimmed_paths)} trimmed chunks...")
    concat_chunks(trimmed_paths, final_path)


def main():
    os.makedirs(AUDIO_DIR, exist_ok=True)

    # Parse arguments
    target = None
    regen_chunks = None

    args = sys.argv[1:]
    for arg in args:
        if arg.startswith("--regen="):
            # e.g. --regen=1,7 to regenerate chunks 1 and 7
            nums = arg.split("=")[1]
            regen_chunks = set(int(n) for n in nums.split(","))
        elif not arg.startswith("--"):
            target = arg

    if target:
        script_file = os.path.join(SCRIPTS_DIR, f"{target}.txt")
        if not os.path.isfile(script_file):
            print(f"Not found: {script_file}")
            sys.exit(1)
        with open(script_file) as f:
            text = f.read().strip()
        targets = [(target, text)]
    else:
        targets = []
        for item in sorted(os.listdir(SCRIPTS_DIR)):
            if item.endswith(".txt") and item.startswith("juz"):
                path = os.path.join(SCRIPTS_DIR, item)
                name = item.replace(".txt", "")
                with open(path) as f:
                    targets.append((name, f.read().strip()))

    if not targets:
        print("No scripts found. Add juz##.txt files to tts/scripts/")
        sys.exit(1)

    print(f"Voice ref: {VOICE_REF}")
    print(f"Params: {TTS_PARAMS}")
    print(f"Max words/chunk: {MAX_WORDS}")
    print(f"Silence between chunks: {SILENCE_SECS}s")
    if regen_chunks:
        print(f"Regenerating chunks: {sorted(regen_chunks)}")

    print("\nLoading model...")
    model = ChatterboxTTS.from_pretrained(device=device)
    print("Model loaded.")

    for juz_name, text in targets:
        process_juz(model, juz_name, text, regen_chunks)

    print("\nAll done!")


if __name__ == "__main__":
    main()
