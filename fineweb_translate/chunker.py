"""Split FineWeb documents into translation-sized chunks."""

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

from .config import FINEWEB_PATH, MAX_CHUNK_CHARS, MIN_CHUNK_CHARS


@dataclass
class Chunk:
    doc_index: int
    chunk_index: int
    chunk_id: str  # "doc_00000_chunk_000"
    text: str
    char_offset: int  # offset within train.txt for verification


def iter_documents(path: Path = FINEWEB_PATH) -> Iterator[tuple[int, str]]:
    """Yield (doc_index, doc_text) lazily from file.

    Documents are separated by blank lines (download_fineweb.py writes
    text + "\\n\\n", producing one blank line between documents).
    """
    doc_index = 0
    current_lines: list[str] = []

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line == "\n":
                # Blank line = document boundary
                if current_lines:
                    doc_text = "".join(current_lines).strip()
                    if doc_text:
                        yield doc_index, doc_text
                        doc_index += 1
                    current_lines = []
            else:
                current_lines.append(line)

    # Flush last document
    if current_lines:
        doc_text = "".join(current_lines).strip()
        if doc_text:
            yield doc_index, doc_text


def chunk_document(
    doc_index: int,
    doc_text: str,
    max_chars: int = MAX_CHUNK_CHARS,
    min_chars: int = MIN_CHUNK_CHARS,
) -> list[Chunk]:
    """Split a document into chunks at paragraph boundaries.

    Greedy: accumulate paragraphs until hitting max_chars, then start a new chunk.
    Chunks smaller than min_chars get merged with the previous chunk.
    """
    paragraphs = doc_text.split("\n")
    chunks: list[Chunk] = []
    current_paras: list[str] = []
    current_len = 0
    char_offset = 0

    for para in paragraphs:
        para_len = len(para) + 1  # +1 for the newline

        if current_len + para_len > max_chars and current_paras:
            # Flush current chunk
            text = "\n".join(current_paras)
            chunk_id = f"doc_{doc_index:05d}_chunk_{len(chunks):03d}"
            chunks.append(Chunk(doc_index, len(chunks), chunk_id, text, char_offset))
            char_offset += len(text) + 1
            current_paras = []
            current_len = 0

        current_paras.append(para)
        current_len += para_len

    # Flush remaining
    if current_paras:
        text = "\n".join(current_paras)
        if chunks and current_len < min_chars:
            # Merge with previous chunk
            prev = chunks[-1]
            merged_text = prev.text + "\n" + text
            chunks[-1] = Chunk(
                prev.doc_index, prev.chunk_index, prev.chunk_id,
                merged_text, prev.char_offset
            )
        else:
            chunk_id = f"doc_{doc_index:05d}_chunk_{len(chunks):03d}"
            chunks.append(Chunk(doc_index, len(chunks), chunk_id, text, char_offset))

    return chunks


def iter_chunks(
    path: Path = FINEWEB_PATH,
    start_doc: int = 0,
    max_chars: int = 0,
) -> Iterator[Chunk]:
    """Iterate all chunks from the FineWeb file.

    Supports resumption via start_doc. Stops after max_chars total if set.
    """
    total_chars = 0

    for doc_index, doc_text in iter_documents(path):
        if doc_index < start_doc:
            continue

        for chunk in chunk_document(doc_index, doc_text):
            yield chunk
            total_chars += len(chunk.text)

            if max_chars > 0 and total_chars >= max_chars:
                return
