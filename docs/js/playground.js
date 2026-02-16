// Interactive playground with live tokenization + ONNX inference
// Requires: bpe.js, ort (ONNX Runtime Web)

const MODEL_CTX_LEN = 256;
const MODEL_VOCAB_SIZE = 1024;

const models = {
  en: { url: 'assets/phase2_model.onnx', session: null, tokenizer: tokenizerEN },
  lj: { url: 'assets/phase2_model_lojban.onnx', session: null, tokenizer: tokenizerLJ }
};

function initPlayground() {
  initTokenizerDemo();
  initPlaygroundControls();
  loadONNXModel('en');
  loadONNXModel('lj');
}

// ── Tokenizer Demo (in "What is Lojban?" section) ────────────────────
function initTokenizerDemo() {
  const inputEN = document.getElementById('tokenizer-input-en');
  const inputLJ = document.getElementById('tokenizer-input-lj');
  const displayEN = document.getElementById('token-display-en');
  const displayLJ = document.getElementById('token-display-lj');
  const countEN = document.getElementById('token-count-en');
  const countLJ = document.getElementById('token-count-lj');

  if (!inputEN || !inputLJ) return;

  function updateEN() {
    if (!tokenizerEN.ready) return;
    const tokens = tokenizerEN.tokenize(inputEN.value);
    renderTokens(tokens, displayEN, countEN);
  }

  function updateLJ() {
    if (!tokenizerLJ.ready) return;
    const tokens = tokenizerLJ.tokenize(inputLJ.value);
    renderTokens(tokens, displayLJ, countLJ);
  }

  inputEN.addEventListener('input', updateEN);
  inputLJ.addEventListener('input', updateLJ);

  // Load tokenizers and render initial text
  Promise.all([
    tokenizerEN.load('assets/tokenizer_english.json'),
    tokenizerLJ.load('assets/tokenizer_lojban.json')
  ]).then(() => {
    updateEN();
    updateLJ();
  }).catch(err => {
    console.warn('Tokenizer load failed:', err);
    if (displayEN) displayEN.textContent = 'Tokenizer not available (serve from HTTP)';
    if (displayLJ) displayLJ.textContent = 'Tokenizer not available (serve from HTTP)';
  });
}

// ── ONNX Model Loading ───────────────────────────────────────────────
async function loadONNXModel(lang) {
  const statusEl = document.getElementById(`model-status-${lang}`);
  if (!statusEl) return;

  statusEl.textContent = 'Loading model...';
  statusEl.className = 'model-badge model-loading';

  try {
    models[lang].session = await ort.InferenceSession.create(models[lang].url, {
      executionProviders: ['wasm'],
    });

    statusEl.textContent = 'Model ready';
    statusEl.className = 'model-badge model-ready';

    const btn = document.getElementById(`btn-generate-${lang}`);
    if (btn) btn.classList.remove('disabled');
  } catch (err) {
    console.error(`ONNX model load failed (${lang}):`, err);
    statusEl.textContent = 'Model unavailable';
    statusEl.className = 'model-badge model-error';
  }
}

// ── Playground Controls ──────────────────────────────────────────────
function initPlaygroundControls() {
  // Temperature sliders
  setupSlider('temp-en', 'temp-en-val');
  setupSlider('temp-lj', 'temp-lj-val');
  setupSlider('len-en', 'len-en-val');
  setupSlider('len-lj', 'len-lj-val');

  // Generate buttons
  const btnEN = document.getElementById('btn-generate-en');
  if (btnEN) {
    btnEN.addEventListener('click', () => generateText('en'));
  }

  const btnLJ = document.getElementById('btn-generate-lj');
  if (btnLJ) {
    btnLJ.addEventListener('click', () => generateText('lj'));
  }
}

function setupSlider(sliderId, valId) {
  const slider = document.getElementById(sliderId);
  const val = document.getElementById(valId);
  if (!slider || !val) return;
  slider.addEventListener('input', () => { val.textContent = slider.value; });
}

// ── Text Generation with ONNX ────────────────────────────────────────
async function generateText(lang) {
  const model = models[lang];
  if (!model.session || !model.tokenizer.ready) return;

  const btn = document.getElementById(`btn-generate-${lang}`);
  const output = document.getElementById(`playground-output-${lang}`);
  const input = document.getElementById(`playground-input-${lang}`);
  const temperature = parseFloat(document.getElementById(`temp-${lang}`).value);
  const maxTokens = parseInt(document.getElementById(`len-${lang}`).value);

  if (!btn || !output || !input) return;

  btn.textContent = 'Generating...';
  btn.classList.add('generating');
  output.innerHTML = '';

  const promptText = input.value;
  let tokenIds = model.tokenizer.encode(promptText);

  // Truncate to fit context
  if (tokenIds.length > MODEL_CTX_LEN - 1) {
    tokenIds = tokenIds.slice(tokenIds.length - MODEL_CTX_LEN + 1);
  }

  const promptLen = tokenIds.length;

  // Show prompt text
  const promptSpan = document.createElement('span');
  promptSpan.className = 'output-prompt';
  promptSpan.textContent = promptText;
  output.appendChild(promptSpan);

  const genSpan = document.createElement('span');
  genSpan.className = 'output-generated';
  output.appendChild(genSpan);

  try {
    for (let i = 0; i < maxTokens; i++) {
      // Create input tensor
      const inputArray = BigInt64Array.from(tokenIds.map(id => BigInt(id)));
      const inputTensor = new ort.Tensor('int64', inputArray, [1, tokenIds.length]);

      // Run inference
      const results = await model.session.run({ input_ids: inputTensor });
      const logits = results.logits.data;

      // Get logits for last position
      const vocabSize = MODEL_VOCAB_SIZE;
      const lastPos = tokenIds.length - 1;
      const offset = lastPos * vocabSize;
      const lastLogits = new Float32Array(vocabSize);
      for (let v = 0; v < vocabSize; v++) {
        lastLogits[v] = logits[offset + v];
      }

      // Apply temperature
      for (let v = 0; v < vocabSize; v++) {
        lastLogits[v] /= temperature;
      }

      // Softmax
      let maxLogit = -Infinity;
      for (let v = 0; v < vocabSize; v++) {
        if (lastLogits[v] > maxLogit) maxLogit = lastLogits[v];
      }
      let sumExp = 0;
      for (let v = 0; v < vocabSize; v++) {
        lastLogits[v] = Math.exp(lastLogits[v] - maxLogit);
        sumExp += lastLogits[v];
      }
      for (let v = 0; v < vocabSize; v++) {
        lastLogits[v] /= sumExp;
      }

      // Top-k sampling (k=40)
      const k = 40;
      const indices = Array.from({ length: vocabSize }, (_, i) => i);
      indices.sort((a, b) => lastLogits[b] - lastLogits[a]);
      const topK = indices.slice(0, k);

      // Renormalize top-k
      let topSum = 0;
      for (const idx of topK) topSum += lastLogits[idx];
      const probs = topK.map(idx => lastLogits[idx] / topSum);

      // Sample
      const r = Math.random();
      let cumsum = 0;
      let nextToken = topK[0];
      for (let j = 0; j < probs.length; j++) {
        cumsum += probs[j];
        if (r < cumsum) {
          nextToken = topK[j];
          break;
        }
      }

      tokenIds.push(nextToken);

      // Truncate to context length
      if (tokenIds.length > MODEL_CTX_LEN) {
        tokenIds = tokenIds.slice(tokenIds.length - MODEL_CTX_LEN);
      }

      // Decode and display incrementally
      const generated = model.tokenizer.decode(tokenIds.slice(promptLen));
      genSpan.textContent = generated;
      output.scrollTop = output.scrollHeight;

      // Yield to UI
      if (i % 4 === 0) {
        await new Promise(r => setTimeout(r, 0));
      }
    }
  } catch (err) {
    console.error('Generation error:', err);
    genSpan.textContent = '\n[Generation error: ' + err.message + ']';
  }

  btn.textContent = 'Generate';
  btn.classList.remove('generating');
}
