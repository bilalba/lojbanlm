// Minimal BPE tokenizer in JavaScript
// Port of HuggingFace ByteLevel BPE tokenizer
// Loads vocab + merges from the tokenizer JSON (results/v5/tokenizer_*.json)

const NEWLINE_PLACEHOLDER = '\u0126'; // Ħ

class BPETokenizer {
  constructor(name) {
    this.name = name;
    this.vocab = null;       // byte-level token string -> id
    this.vocabInv = null;    // id -> byte-level token string
    this.mergeRank = null;   // "a b" -> rank (priority)
    this.ready = false;
  }

  async load(url) {
    const resp = await fetch(url);
    const data = await resp.json();
    this.vocab = data.model.vocab;
    this.vocabInv = {};
    for (const [token, id] of Object.entries(this.vocab)) {
      this.vocabInv[id] = token;
    }
    this.mergeRank = {};
    for (let i = 0; i < data.model.merges.length; i++) {
      const key = data.model.merges[i][0] + ' ' + data.model.merges[i][1];
      this.mergeRank[key] = i;
    }
    this.ready = true;
  }

  // ── GPT-2 bytes_to_unicode mapping ─────────────────────────────────
  static bytesToUnicode() {
    if (BPETokenizer._b2u) return BPETokenizer._b2u;
    const bs = [];
    for (let i = 33; i <= 126; i++) bs.push(i);
    for (let i = 161; i <= 172; i++) bs.push(i);
    for (let i = 174; i <= 255; i++) bs.push(i);
    const cs = [...bs];
    let n = 0;
    for (let b = 0; b < 256; b++) {
      if (!bs.includes(b)) {
        bs.push(b);
        cs.push(256 + n);
        n++;
      }
    }
    const b2u = {};
    const u2b = {};
    for (let i = 0; i < bs.length; i++) {
      b2u[bs[i]] = String.fromCharCode(cs[i]);
      u2b[String.fromCharCode(cs[i])] = bs[i];
    }
    BPETokenizer._b2u = b2u;
    BPETokenizer._u2b = u2b;
    return b2u;
  }

  static unicodeToBytes() {
    if (BPETokenizer._u2b) return BPETokenizer._u2b;
    BPETokenizer.bytesToUnicode();
    return BPETokenizer._u2b;
  }

  // ── Convert text to byte-level unicode string ──────────────────────
  // " went" -> "Ġwent"  (space byte 0x20 -> Ġ U+0120)
  // "Ħ" -> "Ä¦"         (UTF-8 0xC4 0xA6 -> Ä ¦)
  textToByteLevelString(text) {
    const b2u = BPETokenizer.bytesToUnicode();
    const encoder = new TextEncoder();
    const bytes = encoder.encode(text);
    let result = '';
    for (const b of bytes) {
      result += b2u[b];
    }
    return result;
  }

  // ── Pre-tokenize: split ORIGINAL text with HF regex ────────────────
  // HuggingFace ByteLevel pre-tokenizer:
  //   1. Split original text using GPT-2 regex (with Unicode \p{L}, \p{N})
  //   2. Convert each piece to byte-level unicode
  // Key: Ħ (U+0126) is \p{L} so "ĦBob" stays together as one word
  preTokenize(text) {
    const pattern = /'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+/gu;
    return text.match(pattern) || [text];
  }

  // ── Apply BPE merges ───────────────────────────────────────────────
  applyBPE(tokens) {
    if (tokens.length < 2) return tokens;
    let word = [...tokens];

    while (word.length >= 2) {
      let bestPair = null;
      let bestRank = Infinity;
      for (let i = 0; i < word.length - 1; i++) {
        const key = word[i] + ' ' + word[i + 1];
        const rank = this.mergeRank[key];
        if (rank !== undefined && rank < bestRank) {
          bestRank = rank;
          bestPair = [word[i], word[i + 1]];
        }
      }
      if (bestPair === null) break;

      const merged = bestPair[0] + bestPair[1];
      const newWord = [];
      let i = 0;
      while (i < word.length) {
        if (i < word.length - 1 && word[i] === bestPair[0] && word[i + 1] === bestPair[1]) {
          newWord.push(merged);
          i += 2;
        } else {
          newWord.push(word[i]);
          i++;
        }
      }
      word = newWord;
    }
    return word;
  }

  // ── Encode text to token IDs ───────────────────────────────────────
  encode(text) {
    if (!this.ready) return [];

    // Replace newlines with placeholder (same as Python BPETokenizerWrapper)
    text = text.replace(/\n/g, NEWLINE_PLACEHOLDER);

    // Split original text into words using HF regex
    const words = this.preTokenize(text);

    // For each word: convert to byte-level, split into chars, apply BPE
    const ids = [];
    for (const word of words) {
      const byteLevelStr = this.textToByteLevelString(word);
      const chars = [...byteLevelStr];
      const merged = this.applyBPE(chars);
      for (const token of merged) {
        const id = this.vocab[token];
        if (id !== undefined) {
          ids.push(id);
        }
      }
    }
    return ids;
  }

  // ── Decode token IDs to text ───────────────────────────────────────
  decode(ids) {
    if (!this.ready) return '';
    const u2b = BPETokenizer.unicodeToBytes();
    const text = ids.map(id => this.vocabInv[id] || '').join('');
    const bytes = [];
    for (const ch of text) {
      const b = u2b[ch];
      if (b !== undefined) bytes.push(b);
    }
    let result = new TextDecoder().decode(new Uint8Array(bytes));
    return result.replace(new RegExp(NEWLINE_PLACEHOLDER, 'g'), '\n');
  }

  // ── Tokenize for colorized display ─────────────────────────────────
  tokenize(text) {
    if (!this.ready) return [];
    text = text.replace(/\n/g, NEWLINE_PLACEHOLDER);
    const words = this.preTokenize(text);
    const result = [];
    const u2b = BPETokenizer.unicodeToBytes();

    for (const word of words) {
      const byteLevelStr = this.textToByteLevelString(word);
      const chars = [...byteLevelStr];
      const merged = this.applyBPE(chars);
      for (const token of merged) {
        const bytes = [];
        for (const ch of token) {
          const b = u2b[ch];
          if (b !== undefined) bytes.push(b);
        }
        let decoded;
        try {
          decoded = new TextDecoder().decode(new Uint8Array(bytes));
          decoded = decoded.replace(new RegExp(NEWLINE_PLACEHOLDER, 'g'), '\\n');
        } catch {
          decoded = token;
        }
        result.push({ token, decoded, id: this.vocab[token] });
      }
    }
    return result;
  }
}

// Color palette for token visualization
const TOKEN_COLORS = [
  'rgba(74, 158, 255, 0.25)',
  'rgba(255, 179, 71, 0.25)',
  'rgba(124, 92, 252, 0.25)',
  'rgba(74, 222, 128, 0.25)',
  'rgba(239, 68, 68, 0.25)',
  'rgba(251, 191, 36, 0.25)',
  'rgba(168, 85, 247, 0.25)',
  'rgba(236, 72, 153, 0.25)',
  'rgba(34, 211, 238, 0.25)',
  'rgba(163, 230, 53, 0.25)',
];

function renderTokens(tokens, containerEl, countEl) {
  containerEl.innerHTML = '';
  if (!tokens.length) {
    countEl.textContent = '';
    return;
  }
  tokens.forEach((t, i) => {
    const span = document.createElement('span');
    span.className = 'token-span';
    span.style.backgroundColor = TOKEN_COLORS[i % TOKEN_COLORS.length];
    span.textContent = t.decoded;
    span.title = `ID: ${t.id}`;
    containerEl.appendChild(span);
  });
  countEl.textContent = `${tokens.length} tokens`;
}

// Global tokenizer instances (loaded from results/v5/tokenizer_*.json)
const tokenizerEN = new BPETokenizer('English');
const tokenizerLJ = new BPETokenizer('Lojban');
