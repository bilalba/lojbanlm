"""Lojban dictionary: parse gismu-data.js + jbovlaste XML, build English→Lojban index."""

import json
import re
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path

from .config import (
    GISMU_DATA_PATH,
    JBOVLASTE_DIR,
    JBOVLASTE_URL,
    JBOVLASTE_XML_PATH,
    MAX_HINTS_PER_CHUNK,
)

# Common English stopwords to skip during lookup
STOPWORDS = frozenset(
    "a an the is are was were be been being have has had do does did will would "
    "shall should may might can could of in to for on with at by from as into "
    "through during before after above below between under again further then "
    "once here there when where why how all both each few more most other some "
    "such no nor not only own same so than too very and but if or because until "
    "while about that this these those it its he she they them their his her "
    "we you your our my me him us what which who whom also just still even "
    "however although though yet already always never often sometimes usually "
    "many much one two three four five six seven eight nine ten first second "
    "third new old good great little long big small well back over".split()
)


def parse_gismu_data(path: Path = GISMU_DATA_PATH) -> dict[str, str]:
    """Parse ilmentufa/glosser/gismu-data.js into {lojban_word: english_gloss}.

    Filters out entries where the value is a grammar code (all-caps like "GOI").
    """
    text = path.read_text(encoding="utf-8")

    # Extract key-value pairs directly with regex (avoids JSON parse issues
    # from JS comments, stray block-comment closers, blank lines, etc.)
    result = {}
    for m in re.finditer(r'"([^"]+)"\s*:\s*"([^"]*)"', text):
        word = m.group(1).strip()
        defn = m.group(2).strip()
        # Skip grammar codes (all uppercase, like "GOI", "KOhA3", "JOI")
        if re.match(r"^[A-Z][A-Za-z0-9]*$", defn):
            continue
        # Skip empty or placeholder definitions
        if not defn or defn.startswith("("):
            continue
        result[word] = defn

    return result


def download_jbovlaste(output_dir: Path = JBOVLASTE_DIR) -> Path:
    """Download jbovlaste en.xml from GitHub if not already cached."""
    output_dir.mkdir(parents=True, exist_ok=True)
    xml_path = output_dir / "en.xml"

    if xml_path.exists():
        return xml_path

    print(f"Downloading jbovlaste dictionary to {xml_path}...")
    urllib.request.urlretrieve(JBOVLASTE_URL, xml_path)
    print(f"  Downloaded ({xml_path.stat().st_size / 1024:.0f} KB)")
    return xml_path


def parse_jbovlaste_xml(path: Path = JBOVLASTE_XML_PATH) -> dict[str, dict]:
    """Parse jbovlaste XML into {lojban_word: {type, definition, glosses}}.

    Returns dict like:
        {"karce": {"type": "gismu", "definition": "x1 is a car...",
                   "glosses": ["car", "automobile"]}}
    """
    if not path.exists():
        path = download_jbovlaste(path.parent)

    tree = ET.parse(path)
    root = tree.getroot()

    result = {}
    for valsi in root.iter("valsi"):
        word = valsi.get("word", "").strip()
        word_type = valsi.get("type", "").strip()
        if not word:
            continue

        definition = ""
        glosses = []

        for child in valsi:
            if child.tag == "definition":
                definition = (child.text or "").strip()
            elif child.tag == "glossword":
                gw = child.get("word", "").strip()
                if gw:
                    glosses.append(gw.lower())

        # Also extract keywords from definition text
        # e.g. "x1 is a car/automobile/truck" → ["car", "automobile", "truck"]
        if definition:
            # Find words after "is a" or "is an" patterns
            for m in re.finditer(r"(?:is (?:a|an) |are )([a-z/]+)", definition.lower()):
                for w in m.group(1).split("/"):
                    w = w.strip()
                    if w and w not in glosses and len(w) > 2:
                        glosses.append(w)

        if definition or glosses:
            result[word] = {
                "type": word_type,
                "definition": definition,
                "glosses": glosses,
            }

    return result


def build_reverse_index(
    jbovlaste: dict[str, dict] | None = None,
    gismu_data: dict[str, str] | None = None,
) -> dict[str, list[tuple[str, str]]]:
    """Build English word → [(lojban_word, short_definition)] reverse index.

    Merges both dictionary sources. Prefers gismu over lujvo when both match.
    """
    if gismu_data is None:
        gismu_data = parse_gismu_data()
    if jbovlaste is None:
        try:
            jbovlaste = parse_jbovlaste_xml()
        except Exception:
            jbovlaste = {}

    index: dict[str, list[tuple[str, str]]] = {}

    def _add(eng_word: str, loj_word: str, short_def: str, priority: int):
        eng_word = eng_word.lower().strip()
        if not eng_word or eng_word in STOPWORDS or len(eng_word) < 3:
            return
        if eng_word not in index:
            index[eng_word] = []
        # Avoid duplicates
        for existing_loj, _ in index[eng_word]:
            if existing_loj == loj_word:
                return
        index[eng_word].append((loj_word, short_def))

    # Add gismu-data.js entries (high priority — curated glosses)
    for word, defn in gismu_data.items():
        # Split multi-word definitions: "to teach" → index on "teach"
        for part in re.split(r"[;,/]", defn):
            part = part.strip()
            # Remove "to " prefix for verbs
            if part.startswith("to "):
                part = part[3:]
            _add(part, word, defn, priority=0)
            # Also index individual words for multi-word glosses
            for w in part.split():
                if w not in STOPWORDS and len(w) >= 3:
                    _add(w, word, defn, priority=1)

    # Add jbovlaste entries (fills gaps with lujvo etc.)
    for word, info in jbovlaste.items():
        short_def = info["glosses"][0] if info["glosses"] else info["definition"][:40]
        for gloss in info["glosses"]:
            _add(gloss, word, short_def, priority=2)

    return index


def lookup_hints(
    english_text: str,
    reverse_index: dict[str, list[tuple[str, str]]],
    max_hints: int = MAX_HINTS_PER_CHUNK,
) -> str:
    """Extract content words from English text, look up Lojban equivalents.

    Returns formatted string like "car=karce, teach=ctuca, book=cukta"
    """
    # Tokenize: extract lowercase words
    words = re.findall(r"[a-zA-Z]+", english_text.lower())

    # Deduplicate while preserving order
    seen = set()
    unique_words = []
    for w in words:
        if w not in seen and w not in STOPWORDS and len(w) >= 3:
            seen.add(w)
            unique_words.append(w)

    hints = []
    for w in unique_words:
        if w in reverse_index:
            matches = reverse_index[w]
            # Take the first (highest priority) match
            loj, _ = matches[0]
            hints.append(f"{w}={loj}")
            if len(hints) >= max_hints:
                break

    return ", ".join(hints)


class LojbanDictionary:
    """High-level dictionary interface. Loads and caches all sources."""

    def __init__(self):
        self._gismu_data: dict[str, str] | None = None
        self._jbovlaste: dict[str, dict] | None = None
        self._reverse_index: dict[str, list[tuple[str, str]]] | None = None

    def load(self):
        """Load all dictionary sources and build the reverse index."""
        print("Loading Lojban dictionaries...")
        self._gismu_data = parse_gismu_data()
        print(f"  gismu-data.js: {len(self._gismu_data)} entries")

        try:
            xml_path = download_jbovlaste()
            self._jbovlaste = parse_jbovlaste_xml(xml_path)
            print(f"  jbovlaste XML: {len(self._jbovlaste)} entries")
        except Exception as e:
            print(f"  jbovlaste XML: failed ({e}), using gismu-data.js only")
            self._jbovlaste = {}

        self._reverse_index = build_reverse_index(self._jbovlaste, self._gismu_data)
        print(f"  Reverse index: {len(self._reverse_index)} English words mapped")

    def lookup(self, english_text: str, max_hints: int = MAX_HINTS_PER_CHUNK) -> str:
        """Look up vocabulary hints for an English text chunk."""
        if self._reverse_index is None:
            self.load()
        return lookup_hints(english_text, self._reverse_index, max_hints)
