// Test ONNX inference using the exact same logic as playground.js
// This isolates whether the bug is in tensor creation, logits indexing, or sampling

import { readFileSync } from 'fs';
import ort from 'onnxruntime-node';

const ONNX_MODEL_URL = 'docs/assets/phase2_model.onnx';
const MODEL_CTX_LEN = 256;
const MODEL_VOCAB_SIZE = 1024;

// ── Load tokenizer (same logic as bpe.js) ────────────────────────────
const NEWLINE_PLACEHOLDER = '\u0126'; // Ħ

function bytesToUnicode() {
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
    let bestPair = null;
    let bestRank = Infinity;
    for (let i = 0; i < word.length - 1; i++) {
      const key = word[i] + ' ' + word[i + 1];
      const rank = mergeRank[key];
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

function encode(text, vocab, mergeRank) {
  text = text.replace(/\n/g, NEWLINE_PLACEHOLDER);
  const words = preTokenize(text);
  const ids = [];
  for (const word of words) {
    const byteLevelStr = textToByteLevelString(word);
    const chars = [...byteLevelStr];
    const merged = applyBPE(chars, mergeRank);
    for (const token of merged) {
      const id = vocab[token];
      if (id !== undefined) ids.push(id);
    }
  }
  return ids;
}

function decode(ids, vocabInv) {
  const text = ids.map(id => vocabInv[id] || '').join('');
  const bytes = [];
  for (const ch of text) {
    const b = u2b[ch];
    if (b !== undefined) bytes.push(b);
  }
  let result = new TextDecoder().decode(new Uint8Array(bytes));
  return result.replace(new RegExp(NEWLINE_PLACEHOLDER, 'g'), '\n');
}

// ── Main test ────────────────────────────────────────────────────────
async function main() {
  // Load tokenizer
  const tokData = JSON.parse(readFileSync('docs/assets/tokenizer_english.json', 'utf8'));
  const vocab = tokData.model.vocab;
  const vocabInv = {};
  for (const [token, id] of Object.entries(vocab)) vocabInv[id] = token;
  const mergeRank = {};
  for (let i = 0; i < tokData.model.merges.length; i++) {
    const key = tokData.model.merges[i][0] + ' ' + tokData.model.merges[i][1];
    mergeRank[key] = i;
  }

  // Encode test prompt
  const promptText = 'Alice went to the garden.\nBob went to the kitchen.\nWhere is Alice?';
  let tokenIds = encode(promptText, vocab, mergeRank);
  console.log('Prompt:', JSON.stringify(promptText));
  console.log('Token IDs:', tokenIds);
  console.log('Num tokens:', tokenIds.length);

  // Load ONNX model
  console.log('\nLoading ONNX model...');
  const session = await ort.InferenceSession.create(ONNX_MODEL_URL);
  console.log('Model loaded. Input names:', session.inputNames, 'Output names:', session.outputNames);

  // ── Test 1: Check tensor creation (same as playground.js) ──────────
  console.log('\n=== Test 1: Tensor creation ===');
  const inputArray = BigInt64Array.from(tokenIds.map(id => BigInt(id)));
  console.log('BigInt64Array:', [...inputArray].map(Number));
  const inputTensor = new ort.Tensor('int64', inputArray, [1, tokenIds.length]);
  console.log('Tensor shape:', inputTensor.dims, 'type:', inputTensor.type);

  // ── Test 2: Run inference ──────────────────────────────────────────
  console.log('\n=== Test 2: Inference ===');
  const results = await session.run({ input_ids: inputTensor });
  const logits = results.logits.data;
  console.log('Logits type:', results.logits.type, 'dims:', results.logits.dims);
  console.log('Logits total elements:', logits.length);
  console.log('Expected elements:', tokenIds.length * MODEL_VOCAB_SIZE);

  // ── Test 3: Extract last position logits (same as playground.js) ───
  console.log('\n=== Test 3: Last position logits ===');
  const vocabSize = MODEL_VOCAB_SIZE;
  const lastPos = tokenIds.length - 1;
  const offset = lastPos * vocabSize;
  console.log('lastPos:', lastPos, 'offset:', offset);

  const lastLogits = new Float32Array(vocabSize);
  for (let v = 0; v < vocabSize; v++) {
    lastLogits[v] = logits[offset + v];
  }

  // Check if logits look reasonable
  let minLogit = Infinity, maxLogit_val = -Infinity;
  for (let v = 0; v < vocabSize; v++) {
    if (lastLogits[v] < minLogit) minLogit = lastLogits[v];
    if (lastLogits[v] > maxLogit_val) maxLogit_val = lastLogits[v];
  }
  console.log('Logit range:', minLogit.toFixed(4), 'to', maxLogit_val.toFixed(4));

  // Top 10 by raw logits (greedy)
  const indices = Array.from({ length: vocabSize }, (_, i) => i);
  indices.sort((a, b) => lastLogits[b] - lastLogits[a]);
  console.log('\nTop 10 tokens (greedy):');
  for (let i = 0; i < 10; i++) {
    const id = indices[i];
    const token = vocabInv[id] || '???';
    // Decode token to readable text
    const bytes = [];
    for (const ch of token) {
      const b = u2b[ch];
      if (b !== undefined) bytes.push(b);
    }
    let decoded;
    try {
      decoded = new TextDecoder().decode(new Uint8Array(bytes));
    } catch { decoded = token; }
    console.log(`  ${i}: id=${id} token="${token}" decoded="${decoded}" logit=${lastLogits[id].toFixed(4)}`);
  }

  // ── Test 4: Generate with temperature=0.8 (like playground default) ─
  console.log('\n=== Test 4: Autoregressive generation (20 tokens) ===');
  const temperature = 0.8;
  const maxTokens = 20;
  const promptLen = tokenIds.length;

  for (let i = 0; i < maxTokens; i++) {
    const inputArr = BigInt64Array.from(tokenIds.map(id => BigInt(id)));
    const inTensor = new ort.Tensor('int64', inputArr, [1, tokenIds.length]);
    const res = await session.run({ input_ids: inTensor });
    const lgt = res.logits.data;

    const lp = tokenIds.length - 1;
    const off = lp * vocabSize;
    const ll = new Float32Array(vocabSize);
    for (let v = 0; v < vocabSize; v++) {
      ll[v] = lgt[off + v];
    }

    // Temperature
    for (let v = 0; v < vocabSize; v++) ll[v] /= temperature;

    // Softmax
    let maxL = -Infinity;
    for (let v = 0; v < vocabSize; v++) if (ll[v] > maxL) maxL = ll[v];
    let sumExp = 0;
    for (let v = 0; v < vocabSize; v++) {
      ll[v] = Math.exp(ll[v] - maxL);
      sumExp += ll[v];
    }
    for (let v = 0; v < vocabSize; v++) ll[v] /= sumExp;

    // Top-k sampling (k=40)
    const k = 40;
    const idx = Array.from({ length: vocabSize }, (_, i) => i);
    idx.sort((a, b) => ll[b] - ll[a]);
    const topK = idx.slice(0, k);

    let topSum = 0;
    for (const id of topK) topSum += ll[id];
    const probs = topK.map(id => ll[id] / topSum);

    // Sample
    const r = Math.random();
    let cumsum = 0;
    let nextToken = topK[0];
    for (let j = 0; j < probs.length; j++) {
      cumsum += probs[j];
      if (r < cumsum) { nextToken = topK[j]; break; }
    }

    tokenIds.push(nextToken);
    if (tokenIds.length > MODEL_CTX_LEN) {
      tokenIds = tokenIds.slice(tokenIds.length - MODEL_CTX_LEN);
    }
  }

  const generated = decode(tokenIds.slice(promptLen), vocabInv);
  console.log('Generated text:', JSON.stringify(generated));

  // ── Test 5: Also check what Math.max(...) does vs manual loop ──────
  console.log('\n=== Test 5: Math.max(...) safety check ===');
  const testArr = new Float32Array(1024);
  for (let i = 0; i < 1024; i++) testArr[i] = Math.random() * 10 - 5;
  testArr[500] = 99.0;
  const spreadMax = Math.max(...testArr);
  let loopMax = -Infinity;
  for (let i = 0; i < testArr.length; i++) if (testArr[i] > loopMax) loopMax = testArr[i];
  console.log('Math.max(...) =', spreadMax, 'Loop max =', loopMax, 'Match:', spreadMax === loopMax);
}

main().catch(err => {
  console.error('Test failed:', err);
  process.exit(1);
});
