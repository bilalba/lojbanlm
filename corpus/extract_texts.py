#!/usr/bin/env python3
"""
Extract clean Lojban (and English where applicable) text from downloaded HTML files.

Each HTML source has a different structure:
  1. Esther       - MediaWiki page, content in div.mw-parser-output <p> tags
  2. In A Grove   - XHTML5/ePub-style, text in <div> and <section> elements
  3. Metamorphosis - MediaWiki page with tab-separated Lojban\tEnglish lines
  4. Little Prince - Wayback Machine HTML with <span title="gloss"> word-by-word
  5. Wizard of Oz  - Plain HTML with text in <p>, <span>, <td> inside <table>

Run from: /Users/billy/repo/lojban_experiment/corpus/
"""

import os
import re
import sys
import html as html_module
import textwrap

# ---------------------------------------------------------------------------
# Try BeautifulSoup first; fall back to stdlib html.parser
# ---------------------------------------------------------------------------
try:
    from bs4 import BeautifulSoup, NavigableString, Comment, Tag
    HAS_BS4 = True
except ImportError:
    HAS_BS4 = False
    from html.parser import HTMLParser

CORPUS_DIR = os.path.dirname(os.path.abspath(__file__))

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def read_file(rel_path, encoding="utf-8", fallback_encoding="latin-1"):
    """Read an HTML file, trying utf-8 first then a fallback encoding."""
    path = os.path.join(CORPUS_DIR, rel_path)
    try:
        with open(path, "r", encoding=encoding) as f:
            return f.read()
    except UnicodeDecodeError:
        with open(path, "r", encoding=fallback_encoding) as f:
            return f.read()


def write_output(rel_path, text):
    """Write extracted text to a file."""
    path = os.path.join(CORPUS_DIR, rel_path)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)
    return path


def normalize_whitespace(text):
    """Collapse runs of whitespace (except newlines) and strip each line."""
    # Replace non-breaking spaces and other whitespace chars with regular space
    text = text.replace("\u00a0", " ")
    text = text.replace("\xa0", " ")
    # Collapse multiple spaces/tabs into one
    text = re.sub(r"[^\S\n]+", " ", text)
    # Strip each line
    lines = [line.strip() for line in text.split("\n")]
    # Collapse multiple blank lines into at most two
    cleaned = []
    blank_count = 0
    for line in lines:
        if not line:
            blank_count += 1
            if blank_count <= 2:
                cleaned.append("")
        else:
            blank_count = 0
            cleaned.append(line)
    return "\n".join(cleaned).strip() + "\n"


def file_stats(path):
    """Return (lines, words, chars) for a file."""
    with open(path, "r", encoding="utf-8") as f:
        text = f.read()
    lines = text.count("\n")
    words = len(text.split())
    chars = len(text)
    return lines, words, chars


# ===================================================================
# 1. Book of Esther
# ===================================================================

def extract_esther():
    """Extract Lojban text from the Esther wiki page."""
    print("--- Extracting: Book of Esther ---")
    raw = read_file("esther/esther_lojban.html")

    if HAS_BS4:
        soup = BeautifulSoup(raw, "html.parser")
        content_div = soup.find("div", class_="mw-parser-output")
        if not content_div:
            print("  ERROR: Could not find mw-parser-output div")
            return []

        paragraphs = []
        for elem in content_div.children:
            if isinstance(elem, Tag):
                if elem.name == "p":
                    text = elem.get_text()
                    text = text.strip()
                    if text:
                        paragraphs.append(text)
                elif elem.name == "pre":
                    # The pre block contains a formatted list (names)
                    text = elem.get_text()
                    text = text.strip()
                    if text:
                        paragraphs.append(text)
                elif elem.name == "center":
                    text = elem.get_text().strip()
                    if text:
                        paragraphs.append(text)
    else:
        # Fallback: regex extraction
        # Find content between mw-parser-output and printfooter
        match = re.search(
            r'class="mw-parser-output">(.*?)<div class="printfooter"',
            raw, re.DOTALL
        )
        if not match:
            print("  ERROR: Could not find content")
            return []
        body = match.group(1)
        # Extract <p> contents
        paragraphs = []
        for m in re.finditer(r"<p>(.*?)</p>", body, re.DOTALL):
            text = re.sub(r"<[^>]+>", "", m.group(1))
            text = html_module.unescape(text).strip()
            if text:
                paragraphs.append(text)
        # Extract <pre> contents
        for m in re.finditer(r"<pre[^>]*>(.*?)</pre>", body, re.DOTALL):
            text = re.sub(r"<[^>]+>", "", m.group(1))
            text = html_module.unescape(text).strip()
            if text:
                paragraphs.append(text)
        # Extract <center> contents
        for m in re.finditer(r"<center>(.*?)</center>", body, re.DOTALL):
            text = re.sub(r"<[^>]+>", "", m.group(1))
            text = html_module.unescape(text).strip()
            if text:
                paragraphs.append(text)

    full_text = "\n\n".join(paragraphs)
    full_text = normalize_whitespace(full_text)
    out = write_output("esther/esther_lojban.txt", full_text)
    return [out]


# ===================================================================
# 2. In A Grove
# ===================================================================

def extract_in_a_grove():
    """Extract Lojban text from In A Grove HTML (ePub-style XHTML)."""
    print("--- Extracting: In A Grove ---")
    raw = read_file("in_a_grove/in_a_grove_lojban.html")

    if HAS_BS4:
        soup = BeautifulSoup(raw, "html.parser")

        # Remove all HTML comments (which contain Japanese source text)
        for comment in soup.find_all(string=lambda t: isinstance(t, Comment)):
            comment.extract()

        # Remove the nav/toc
        nav = soup.find("nav")
        if nav:
            nav.decompose()

        # Extract text from all sections and divs within body
        body = soup.find("body")
        if not body:
            print("  ERROR: Could not find body")
            return []

        lines = []
        for elem in body.descendants:
            if isinstance(elem, Tag):
                if elem.name in ("h1", "h2", "h3"):
                    text = elem.get_text().strip()
                    if text:
                        lines.append("")
                        lines.append(text)
                        lines.append("")
                elif elem.name == "div":
                    # Get direct text content of this div only
                    # (to avoid double-counting nested elements)
                    text = elem.get_text().strip()
                    # Only add if this div has no child divs/sections
                    child_blocks = elem.find_all(["div", "section", "h1", "h2", "h3", "nav"])
                    if not child_blocks and text:
                        lines.append(text)
                elif elem.name == "img":
                    # Include image alt text for reference
                    alt = elem.get("alt", "")
                    if alt and alt != "cover":
                        lines.append(f"[{alt}]")
    else:
        # Fallback: regex-based extraction
        # Remove HTML comments
        cleaned = re.sub(r"<!--.*?-->", "", raw, flags=re.DOTALL)
        # Remove nav
        cleaned = re.sub(r"<nav[^>]*>.*?</nav>", "", cleaned, flags=re.DOTALL)

        lines = []
        # Extract headings
        for m in re.finditer(r"<h[123][^>]*>(.*?)</h[123]>", cleaned, re.DOTALL):
            text = re.sub(r"<[^>]+>", "", m.group(1))
            text = html_module.unescape(text).strip()
            if text:
                lines.append("")
                lines.append(text)
                lines.append("")

        # Extract div text
        for m in re.finditer(r"<div[^>]*>(.*?)</div>", cleaned, re.DOTALL):
            inner = m.group(1)
            # Skip divs that contain other divs
            if "<div" in inner:
                continue
            text = re.sub(r"<[^>]+>", "", inner)
            text = html_module.unescape(text).strip()
            if text:
                lines.append(text)

    full_text = "\n".join(lines)
    full_text = normalize_whitespace(full_text)
    out = write_output("in_a_grove/in_a_grove_lojban.txt", full_text)
    return [out]


# ===================================================================
# 3. The Metamorphosis (Lojban + English interleaved)
# ===================================================================

def extract_metamorphosis():
    """
    Extract Lojban and English text from the Metamorphosis wiki page.
    The content is tab-separated: Lojban<TAB>English on each line within
    a single <p> inside div.mw-parser-output.
    """
    print("--- Extracting: The Metamorphosis ---")
    raw = read_file("metamorphosis/metamorphosis_lojban.html")

    if HAS_BS4:
        soup = BeautifulSoup(raw, "html.parser")
        content_div = soup.find("div", class_="mw-parser-output")
        if not content_div:
            print("  ERROR: Could not find mw-parser-output div")
            return []
        # Get all paragraph text
        body_text = content_div.get_text()
    else:
        match = re.search(
            r'class="mw-parser-output">(.*?)<div class="printfooter"',
            raw, re.DOTALL
        )
        if not match:
            print("  ERROR: Could not find content")
            return []
        body_text = re.sub(r"<[^>]+>", "", match.group(1))
        body_text = html_module.unescape(body_text)

    # Split into lines and separate Lojban/English by tab character
    lojban_parts = []
    english_parts = []

    for line in body_text.split("\n"):
        line = line.strip()
        if not line:
            continue
        # Skip the wiki-table markup artifacts
        if line.startswith("<tab") or line.startswith("</tab") or line.startswith("&lt;tab") or line.startswith("&lt;/tab"):
            continue

        if "\t" in line:
            parts = line.split("\t", 1)
            loj = parts[0].strip()
            eng = parts[1].strip() if len(parts) > 1 else ""

            # Clean up non-breaking space markers
            loj = loj.replace("\u00a0", " ").strip()
            eng = eng.replace("\u00a0", " ").strip()

            # Skip header/meta lines that are just formatting
            if loj:
                lojban_parts.append(loj)
            if eng:
                english_parts.append(eng)
        else:
            # Lines without tabs are likely Lojban continuation
            if line:
                lojban_parts.append(line)

    lojban_text = "\n\n".join(lojban_parts)
    english_text = "\n\n".join(english_parts)

    lojban_text = normalize_whitespace(lojban_text)
    english_text = normalize_whitespace(english_text)

    outputs = []
    out1 = write_output("metamorphosis/metamorphosis_lojban.txt", lojban_text)
    outputs.append(out1)
    out2 = write_output("metamorphosis/metamorphosis_english.txt", english_text)
    outputs.append(out2)
    return outputs


# ===================================================================
# 4. The Little Prince
# ===================================================================

def extract_little_prince():
    """
    Extract Lojban text from the Little Prince archive HTML.
    Each word is in a <span title="gloss">word</span>.
    We want just the visible text, not the title attributes.
    """
    print("--- Extracting: The Little Prince ---")
    raw = read_file("little_prince/lpp_archive.html")

    if HAS_BS4:
        soup = BeautifulSoup(raw, "html.parser")

        # Remove script and style tags
        for tag in soup.find_all(["script", "style", "link", "meta"]):
            tag.decompose()

        body = soup.find("body")
        if not body:
            print("  ERROR: Could not find body")
            return []

        lines = []

        # Process structural elements
        for elem in body.descendants:
            if isinstance(elem, Tag):
                if elem.name in ("h1", "h2", "h3", "h4", "h5", "h6"):
                    text = elem.get_text().strip()
                    if text:
                        lines.append("")
                        lines.append(text)
                        lines.append("")
                elif elem.name == "p":
                    text = elem.get_text().strip()
                    if text:
                        lines.append(text)

        # Join and clean up
        full_text = "\n".join(lines)
    else:
        # Fallback: strip all tags, keep text
        # Remove script/style blocks
        cleaned = re.sub(r"<script[^>]*>.*?</script>", "", raw, flags=re.DOTALL)
        cleaned = re.sub(r"<style[^>]*>.*?</style>", "", cleaned, flags=re.DOTALL)

        lines = []
        # Extract headings
        for m in re.finditer(r"<h[1-6][^>]*>(.*?)</h[1-6]>", cleaned, re.DOTALL):
            text = re.sub(r"<[^>]+>", "", m.group(1))
            text = html_module.unescape(text).strip()
            if text:
                lines.append("")
                lines.append(text)
                lines.append("")

        # Extract paragraphs
        for m in re.finditer(r"<p[^>]*>(.*?)</p>", cleaned, re.DOTALL):
            text = re.sub(r"<[^>]+>", "", m.group(1))
            text = html_module.unescape(text).strip()
            if text:
                lines.append(text)

        full_text = "\n".join(lines)

    full_text = normalize_whitespace(full_text)
    out = write_output("little_prince/little_prince_lojban.txt", full_text)
    return [out]


# ===================================================================
# 5. Wizard of Oz
# ===================================================================

def extract_wizard_of_oz():
    """
    Extract Lojban text from the Wizard of Oz plain HTML.
    The layout uses nested tables with <td> cells. Some rows have 2 columns
    (text + image, or two text columns), some have a single full-width cell.
    We collect text from each <td> cell in document order, deduplicating
    paragraphs that appear in both a <p> child and its parent <td>.
    File uses some non-UTF-8 bytes, so we fall back to latin-1.
    """
    print("--- Extracting: Wizard of Oz ---")
    raw = read_file("wizard_of_oz/wizard_of_oz_plain.html",
                     encoding="utf-8", fallback_encoding="latin-1")

    # Decode HTML entities like &agrave; etc.
    raw = html_module.unescape(raw)

    if HAS_BS4:
        soup = BeautifulSoup(raw, "html.parser")

        # Remove script, style
        for tag in soup.find_all(["script", "style"]):
            tag.decompose()

        body = soup.find("body")
        if not body:
            print("  ERROR: Could not find body")
            return []

        # The HTML uses a multi-table layout. Each <tr> contains either
        # a single full-width <td> or two side-by-side <td>s (text + image).
        # The content in each row is unique (no overlap between rows).
        # Within cells, the HTML is messy (nested <p>, <span>, <font>,
        # <big> etc.), so we extract the full text per <tr> row to avoid
        # duplicates from nested elements.
        paragraphs = []

        for tr in body.find_all("tr"):
            text = tr.get_text().strip()
            if text:
                paragraphs.append(text)

        # Also check for any content outside tables (unlikely but safe)
        seen = set(paragraphs)
        for div in body.find_all("div", recursive=False):
            if not div.find("table"):
                text = div.get_text().strip()
                if text and text not in seen:
                    paragraphs.append(text)
                    seen.add(text)

    else:
        # Fallback regex
        cleaned = re.sub(r"<script[^>]*>.*?</script>", "", raw, flags=re.DOTALL)
        cleaned = re.sub(r"<style[^>]*>.*?</style>", "", cleaned, flags=re.DOTALL)

        paragraphs = []
        # Extract from <td> cells
        for m in re.finditer(r"<td[^>]*>(.*?)</td>", cleaned, re.DOTALL):
            inner = m.group(1)
            # Try <p> inside
            p_matches = re.findall(r"<p[^>]*>(.*?)</p>", inner, re.DOTALL)
            if p_matches:
                for pm in p_matches:
                    text = re.sub(r"<[^>]+>", "", pm)
                    text = html_module.unescape(text).strip()
                    if text:
                        paragraphs.append(text)
            else:
                text = re.sub(r"<[^>]+>", "", inner)
                text = html_module.unescape(text).strip()
                if text:
                    paragraphs.append(text)

    full_text = "\n\n".join(paragraphs)
    full_text = normalize_whitespace(full_text)
    out = write_output("wizard_of_oz/wizard_of_oz_lojban.txt", full_text)
    return [out]


# ===================================================================
# Main
# ===================================================================

def main():
    print(f"Corpus directory: {CORPUS_DIR}")
    print(f"Using BeautifulSoup: {HAS_BS4}")
    print()

    all_outputs = []

    extractors = [
        extract_esther,
        extract_in_a_grove,
        extract_metamorphosis,
        extract_little_prince,
        extract_wizard_of_oz,
    ]

    for extractor in extractors:
        try:
            outputs = extractor()
            all_outputs.extend(outputs)
            for out in outputs:
                lines, words, chars = file_stats(out)
                print(f"  -> {os.path.relpath(out, CORPUS_DIR)}")
                print(f"     {lines:,} lines | {words:,} words | {chars:,} chars")
            print()
        except Exception as e:
            print(f"  ERROR: {e}")
            import traceback
            traceback.print_exc()
            print()

    # Summary
    print("=" * 60)
    print("SUMMARY OF OUTPUT FILES")
    print("=" * 60)
    for out in all_outputs:
        lines, words, chars = file_stats(out)
        rel = os.path.relpath(out, CORPUS_DIR)
        print(f"  {rel:<50s}  {lines:>6,} lines  {words:>8,} words")
    print("=" * 60)
    print(f"Total files produced: {len(all_outputs)}")


if __name__ == "__main__":
    main()
