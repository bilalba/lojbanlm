// Verify JS BPE tokenizer produces correct IDs matching Python HuggingFace tokenizer
import { readFileSync } from 'fs';

const NEWLINE_PLACEHOLDER = '\u0126';

// ── Byte-level mapping (same as bpe.js) ──────────────────────────────
function bytesToUnicode() {
  const bs = [];
  for (let i = 33; i <= 126; i++) bs.push(i);
  for (let i = 161; i <= 172; i++) bs.push(i);
  for (let i = 174; i <= 255; i++) bs.push(i);
  const cs = [...bs];
  let n = 0;
  for (let b = 0; b < 256; b++) {
    if (!bs.includes(b)) { bs.push(b); cs.push(256 + n); n++; }
  }
  const b2u = {}, u2b = {};
  for (let i = 0; i < bs.length; i++) {
    b2u[bs[i]] = String.fromCharCode(cs[i]);
    u2b[String.fromCharCode(cs[i])] = bs[i];
  }
  return { b2u, u2b };
}

const { b2u, u2b } = bytesToUnicode();

function textToByteLevelString(text) {
  const encoder = new TextEncoder();
  const bytes = encoder.encode(text);
  let result = '';
  for (const b of bytes) result += b2u[b];
  return result;
}

function preTokenize(text) {
  const pattern = /'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+/gu;
  return text.match(pattern) || [text];
}

function applyBPE(tokens, mergeRank) {
  if (tokens.length < 2) return tokens;
  let word = [...tokens];
  while (word.length >= 2) {
    let bestPair = null, bestRank = Infinity;
    for (let i = 0; i < word.length - 1; i++) {
      const key = word[i] + ' ' + word[i + 1];
      const rank = mergeRank[key];
      if (rank !== undefined && rank < bestRank) { bestRank = rank; bestPair = [word[i], word[i + 1]]; }
    }
    if (bestPair === null) break;
    const merged = bestPair[0] + bestPair[1];
    const newWord = [];
    let i = 0;
    while (i < word.length) {
      if (i < word.length - 1 && word[i] === bestPair[0] && word[i + 1] === bestPair[1]) { newWord.push(merged); i += 2; }
      else { newWord.push(word[i]); i++; }
    }
    word = newWord;
  }
  return word;
}

// Load tokenizer
const data = JSON.parse(readFileSync('docs/assets/tokenizer_english.json', 'utf8'));
const vocab = data.model.vocab;
const mergeRank = {};
for (let i = 0; i < data.model.merges.length; i++) {
  const key = data.model.merges[i][0] + ' ' + data.model.merges[i][1];
  mergeRank[key] = i;
}

function encode(text) {
  text = text.replace(/\n/g, NEWLINE_PLACEHOLDER);
  const words = preTokenize(text);
  const ids = [];
  for (const word of words) {
    const bl = textToByteLevelString(word);
    const chars = [...bl];
    const merged = applyBPE(chars, mergeRank);
    for (const token of merged) {
      const id = vocab[token];
      if (id !== undefined) ids.push(id);
    }
  }
  return ids;
}

// Test cases - expected IDs from Python HuggingFace tokenizer
const tests = [
  {
    text: 'Alice went to the garden.\nBob went to the kitchen.\nWhere is Alice?',
    expected: [612, 174, 161, 152, 252, 13, 506, 174, 161, 152, 282, 13, 227, 160, 461, 30]
  },
  { text: 'hello', expected: null },
  { text: 'the cat sat on the mat', expected: null },
];

let passed = 0;
for (const t of tests) {
  const ids = encode(t.text);
  if (t.expected) {
    const match = JSON.stringify(ids) === JSON.stringify(t.expected);
    console.log(`${match ? 'PASS' : 'FAIL'}: "${t.text.substring(0, 40)}..."`);
    if (!match) {
      console.log(`  Expected: ${JSON.stringify(t.expected)}`);
      console.log(`  Got:      ${JSON.stringify(ids)}`);
    } else {
      passed++;
    }
  } else {
    console.log(`INFO: "${t.text}" -> ${JSON.stringify(ids)}`);
  }
}
console.log(`\n${passed}/${tests.filter(t=>t.expected).length} tests passed`);
