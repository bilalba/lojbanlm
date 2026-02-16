// Embedded experiment data for the demo website
// Sources: CLAUDE.md summary tables + V4 result.json files

const DATA = {
  // ── V2 Results (small ~835K params, medium ~3.2M params) ──────────────
  v2: {
    small: {
      english: { val_bpc: 1.801, test_bpc: 2.770, grammar: 0.735, memorized: 0, params: 835000 },
      lojban:  { val_bpc: 1.477, test_bpc: 1.860, grammar: 1.000, memorized: 4.5, params: 835000 }
    },
    medium: {
      english: { val_bpc: 1.757, test_bpc: 2.899, grammar: 0.796, memorized: 0, params: 3200000 },
      lojban:  { val_bpc: 1.442, test_bpc: 1.874, grammar: 1.000, memorized: 4.5, params: 3200000 }
    }
  },

  // ── V3 Results (base ~837K params, early stopping) ────────────────────
  v3: {
    base: {
      english: { val_bpc: 0.960, test_bpc: 2.827, grammar: 0.968, babi_seen: 0.464, babi_unseen: 0.380, avg_best_step: 4367, params: 837000 },
      lojban:  { val_bpc: 0.977, test_bpc: 2.857, grammar: 1.000, babi_seen: 0.205, babi_unseen: 0.180, avg_best_step: 1100, params: 837000 }
    }
  },

  // ── V3.1 Results (fixed 10K steps, no early stopping) ─────────────────
  v3_1: {
    nano: {
      english: { val_bpc: 1.244, test_bpc: 3.453, grammar: 0.982, babi_seen: 0.190, babi_unseen: 0.162, avg_best_step: 6800, params: 67000 },
      lojban:  { val_bpc: 0.946, test_bpc: 2.813, grammar: 1.000, babi_seen: 0.208, babi_unseen: 0.183, avg_best_step: 7533, params: 67000 }
    },
    micro: {
      english: { val_bpc: 1.117, test_bpc: 3.064, grammar: 0.987, babi_seen: 0.244, babi_unseen: 0.210, avg_best_step: 7733, params: 164000 },
      lojban:  { val_bpc: 0.915, test_bpc: 2.604, grammar: 1.000, babi_seen: 0.209, babi_unseen: 0.191, avg_best_step: 5133, params: 164000 }
    },
    mini: {
      english: { val_bpc: 1.200, test_bpc: 3.083, grammar: 0.983, babi_seen: 0.286, babi_unseen: 0.236, avg_best_step: 7450, params: 261000 },
      lojban:  { val_bpc: 0.962, test_bpc: 2.899, grammar: 1.000, babi_seen: 0.221, babi_unseen: 0.190, avg_best_step: 2100, params: 261000 }
    }
  },

  // ── V4 Results (BPE tokenization, medium ~570K, large ~758K) ──────────
  v4: {
    medium: {
      english: { val_bpc: 3.448, test_bpc: 2.657, grammar: 0.992, babi_seen: 0.208, babi_unseen: 0.144, avg_best_step: 967, params: 570000 },
      lojban:  { val_bpc: 3.298, test_bpc: 2.251, grammar: 1.000, babi_seen: 0.195, babi_unseen: 0.101, avg_best_step: 767, params: 570000 }
    },
    large: {
      english: { val_bpc: 3.337, test_bpc: 2.543, grammar: 0.989, babi_seen: 0.208, babi_unseen: 0.174, avg_best_step: 1433, params: 758000 },
      lojban:  { val_bpc: 3.192, test_bpc: 2.321, grammar: 1.000, babi_seen: 0.145, babi_unseen: 0.107, avg_best_step: 533, params: 758000 }
    }
  },

  // ── Training dynamics: V4 medium seed 42 (val_bpc over steps) ─────────
  trainingLog: {
    english: [
      { step: 1, val_bpc: 10.061 },
      { step: 100, val_bpc: 6.957 },
      { step: 200, val_bpc: 4.385 },
      { step: 300, val_bpc: 3.793 },
      { step: 400, val_bpc: 3.711 },
      { step: 500, val_bpc: 3.701 },
      { step: 600, val_bpc: 3.737 },
      { step: 700, val_bpc: 3.516 },
      { step: 800, val_bpc: 3.534 },
      { step: 900, val_bpc: 3.458 },
      { step: 1000, val_bpc: 3.448 },
      { step: 1500, val_bpc: 3.675 },
      { step: 2000, val_bpc: 3.803 },
      { step: 2500, val_bpc: 3.768 },
      { step: 3000, val_bpc: 3.900 },
      { step: 3500, val_bpc: 4.095 },
      { step: 4000, val_bpc: 4.125 },
      { step: 4500, val_bpc: 4.097 },
      { step: 5000, val_bpc: 4.145 },
      { step: 5500, val_bpc: 4.167 },
      { step: 6000, val_bpc: 4.196 },
      { step: 6500, val_bpc: 4.196 },
      { step: 7000, val_bpc: 4.204 },
      { step: 7500, val_bpc: 4.227 },
      { step: 8000, val_bpc: 4.273 },
      { step: 8500, val_bpc: 4.263 },
      { step: 9000, val_bpc: 4.287 },
      { step: 9500, val_bpc: 4.282 },
      { step: 10000, val_bpc: 4.282 }
    ],
    lojban: [
      { step: 1, val_bpc: 10.020 },
      { step: 100, val_bpc: 7.036 },
      { step: 200, val_bpc: 4.145 },
      { step: 300, val_bpc: 3.593 },
      { step: 400, val_bpc: 3.500 },
      { step: 500, val_bpc: 3.455 },
      { step: 600, val_bpc: 3.450 },
      { step: 700, val_bpc: 3.352 },
      { step: 800, val_bpc: 3.305 },
      { step: 900, val_bpc: 3.303 },
      { step: 1000, val_bpc: 3.298 },
      { step: 1500, val_bpc: 3.418 },
      { step: 2000, val_bpc: 3.568 },
      { step: 2500, val_bpc: 3.712 },
      { step: 3000, val_bpc: 3.582 },
      { step: 3500, val_bpc: 3.630 },
      { step: 4000, val_bpc: 3.749 },
      { step: 4500, val_bpc: 3.631 },
      { step: 5000, val_bpc: 3.682 },
      { step: 5500, val_bpc: 3.691 },
      { step: 6000, val_bpc: 3.639 },
      { step: 6500, val_bpc: 3.761 },
      { step: 7000, val_bpc: 3.672 },
      { step: 7500, val_bpc: 3.705 },
      { step: 8000, val_bpc: 3.688 },
      { step: 8500, val_bpc: 3.666 },
      { step: 9000, val_bpc: 3.680 },
      { step: 9500, val_bpc: 3.660 },
      { step: 10000, val_bpc: 3.668 }
    ]
  },

  // ── Grammar across scales (all versions) ──────────────────────────────
  grammarByScale: [
    { label: "V2 Small\n835K",     english: 73.5,  lojban: 100.0 },
    { label: "V2 Medium\n3.2M",    english: 79.6,  lojban: 100.0 },
    { label: "V3.1 Nano\n67K",     english: 98.2,  lojban: 100.0 },
    { label: "V3.1 Micro\n164K",   english: 98.7,  lojban: 100.0 },
    { label: "V3.1 Mini\n261K",    english: 98.3,  lojban: 100.0 },
    { label: "V4 Medium\n570K",    english: 99.2,  lojban: 100.0 },
    { label: "V4 Large\n758K",     english: 98.9,  lojban: 100.0 }
  ],

  // ── BPC across versions ───────────────────────────────────────────────
  bpcComparison: [
    { label: "V2 Small",   english: 2.770, lojban: 1.860 },
    { label: "V2 Medium",  english: 2.899, lojban: 1.874 },
    { label: "V3.1 Nano",  english: 3.453, lojban: 2.813 },
    { label: "V3.1 Micro", english: 3.064, lojban: 2.604 },
    { label: "V3.1 Mini",  english: 3.083, lojban: 2.899 },
    { label: "V4 Medium",  english: 2.657, lojban: 2.251 },
    { label: "V4 Large",   english: 2.543, lojban: 2.321 }
  ],

  // ── bAbI confound story ───────────────────────────────────────────────
  babiConfound: [
    { label: "V3 (early stop)",    english: 46.4, lojban: 20.5, gap: 25.9 },
    { label: "V3.1 (fixed 10K)",   english: 28.6, lojban: 22.1, gap: 6.5 },
    { label: "V4 (BPE + 10K)",     english: 20.8, lojban: 19.5, gap: 1.3 }
  ],

  // ── Curated samples from V4 medium seed 42 ───────────────────────────
  samples: {
    babi: {
      english: [
        {
          label: "Location tracking (Task 1/2)",
          prompt: "Bill went to the kitchen.\nBill got the bag.\nLucy went to the restaurant.\nLucy got the milk.\nWhy did Bill go to the kitchen? bag",
          generated: "Eve went to the bathroom.\nBill went to the park.\nSam went to the garden.\nWhere is Daniel? kitchen\n\nLucy went to the restaurant.\nFred went to the bathroom.\nWhere is Daniel? playground\n\nSandra picked up the ball.\nSam went to the market.\nLucy went to the bedroom.\nSandra went to the bathroom.\nWhere is Frank? park",
          note: "Generates coherent bAbI structure but answers are wrong (Daniel, Frank not tracked)"
        },
        {
          label: "Yes/No questions (Task 6)",
          prompt: "Mary got the book.\nBob went to the hallway.\nBob got the bag.\nWhy did Mary go to the kitchen? book\n\nFred went to the bathroom.",
          generated: "Is David in the restaurant? yes\n\nKate went to the restaurant.\nEve went to the garden.\nIs John in the hallway? no\n\nSandra went to the market.\nKate went to the school.\nSam went to the market.\nIs Fred in the playground? no",
          note: "Well-formed questions/answers but no factual grounding"
        },
        {
          label: "Spatial reasoning (Task 17/18)",
          prompt: "The bedroom is west of the bathroom.\nHow do you go from the office to the bathroom? west",
          generated: "The market is south of the hallway.\nThe playground is above the market.\nThe playground is west of the garden.\nWhat is west of the restaurant? playground\n\nThe school is north of the restaurant.\nThe hallway is west of the garden.",
          note: "Generates spatial task structure but answers are random guesses"
        }
      ],
      lojban: [
        {
          label: "Location tracking (Task 1/2)",
          prompt: "la .sandr. pu klama lo kelci stuzi\nla .sandr. pu cpacu lo kabri\nla .sandr. mu'i ma pu klama lo kelci stuzi? lo kabri",
          generated: "la .bab. ca na zvati lo ckule\nla .bil. pu lo panka\nla .sandr. pu klama lo ckule\nla .alis. pu klama lo briju\nla .bab. pu zvati ma? lo purdi\n\nla .an. ca lo larcu kumfa\nla .sam. pu klama lo kelci stuzi\nla .frank. pu klama lo vrogai",
          note: "Grammatically valid Lojban throughout, same vocabulary domain"
        },
        {
          label: "Object tracking (Task 5)",
          prompt: "la .deiv. pu klama lo larcu kumfa\nla .deiv. pu cpacu lo penbi\nla .deiv. mu'i ma pu klama lo larcu kumfa? lo penbi",
          generated: "la .alis. pu klama lo ckule\nla .pet. pu klama lo zarci\nla .sam. pu zvati lo kelci stuzi\nla .sandr. pu lo sipna kumfa\nla .ninas. pu lo gusta\nla .ninas. cu zvati ma? lo briju",
          note: "Uses proper Lojban grammar: articles (lo), tense markers (pu), question particles (ma)"
        },
        {
          label: "Yes/No questions",
          prompt: "la .an. pu klama lo kelci stuzi\nla .an. pu cpacu lo tanxe",
          generated: "xu la .pet. cu zvati lo tanxe? go'i\n\nca lo vanci la .suz. pu klama lo prulamdei\nla .frank. pu klama lo ckule\nxu la .dorotis. cu se vasru lo kelci stuzi? na go'i",
          note: "Correct yes/no structure: xu...? go'i (yes) / na go'i (no)"
        }
      ]
    },
    narrative: {
      english: [
        {
          label: "Out-of-domain: Metamorphosis prompt",
          prompt: "oked Gregor a little, and only when she had shoved him from his place without any resistance did she become",
          generated: "'I am you, so he will be his haled, \"Will the poor the king.\n\n'I am to stangained a met.\n\n'RHhy,\" said the gans, but Dorothy.\n\n\"You'vely. And he shouldde the Woodman.",
          note: "Garbled word salad mixing Oz/Alice vocabulary. No coherent sentences despite 570K params."
        },
        {
          label: "Out-of-domain: Metamorphosis prompt",
          prompt: "approach. She cried out \"Finally!\" to her parents, as she turned the key in the lock",
          generated: "ing of her\n\n'I'te,\" said the Scarecrow.\n17: 'I's he went to the coer, and the Tin Woodman.\n'RHAs of Dorothy, I't sindions as he did not you,' said the Tin Woodman, and the Gryphon.",
          note: "Character fragments, mixed sources. Grammar checker says 99.2% but sentences are meaningless."
        }
      ],
      lojban: [
        {
          label: "Out-of-domain: Metamorphosis prompt",
          prompt: "xa'o zvati lo vorme cu gasnu lo nu carna fa lo stedu .i na'e mulno go'i ki'u lo nu gy ganse",
          generated: "'u lo nu lo nu la .i .oz. cu ba bo li'o \u00ablu je la palci mudypre cu cusku\u2014\nni'u\u00bb ni'u\u00bb ni'u .i .i ku'u\u00bb ni'o \u00ablu ku'e lo nu viska lo nu da se pi'o \u00ablu'i",
          note: "Uses real Lojban grammar particles (ni'o, ku'e, li'o) and proper quotation structure. More structured than English despite errors."
        },
        {
          label: "Out-of-domain: Metamorphosis prompt",
          prompt: "my xruti lo ka kansa la .greten.\n\nni'o la .gregor. cu rapli cusku fi gy fe",
          generated: "lo nu ku'i dy cu ka ska lo nu lo nu za'o \u2014sei la'e da .abu ba'a ca lo ri'a lo nixli cu cusku\u2014 li'u\u00bb ni'u\u00bb ni'e lo nu bo se ki'u li'e mi ka'o",
          note: "Maintains Lojban discourse markers (ni'o, ku'i, ri'a) and quotation nesting. 100% parseable grammar."
        }
      ]
    }
  },

  // ── Timeline data for The Journey section ─────────────────────────────
  timeline: [
    {
      version: "V1",
      title: "Single Book",
      params: "10.8M params",
      change: "Train on Alice in Wonderland in both languages",
      finding: "100% memorization. Both models just recited the book back.",
      lesson: "10.8M params on 130K chars = pure memorization. Need more data, smaller models.",
      status: "invalidated"
    },
    {
      version: "V2",
      title: "Multi-Corpus",
      params: "0.8M\u20133.2M",
      change: "4 books, 3 seeds, held-out test book, BPC metric",
      finding: "Lojban BPC 33% lower, grammar 100% at all sizes. English grammar 73\u219280% with 4\u00d7 params.",
      lesson: "Lojban generalizes better, but we can\u2019t measure reasoning with narrative generation alone.",
      status: "completed"
    },
    {
      version: "V3",
      title: "bAbI Reasoning",
      params: "67K\u2013837K",
      change: "Added 20 bAbI reasoning tasks. 5 model sizes, combined training.",
      finding: "English 46% vs Lojban 21% on bAbI. But Lojban early-stopped 4\u00d7 sooner!",
      lesson: "Early stopping on character loss systematically undertrains lower-entropy languages.",
      status: "confounded"
    },
    {
      version: "V3.1",
      title: "Fixed Steps",
      params: "67K\u2013261K",
      change: "Fixed 10K steps, no early stopping. Same architecture as V3.",
      finding: "bAbI gap narrowed from 26pp to 6.5pp. Checkpoint selection still biased.",
      lesson: "Checkpoint selection by narrative loss is orthogonal to reasoning performance.",
      status: "improved"
    },
    {
      version: "V4",
      title: "BPE Tokenization",
      params: "570K\u2013758K",
      change: "BPE tokenizer (vocab=1024) to equalize token-level information.",
      finding: "bAbI gap: 1.3pp. Neither language shows reasoning above chance.",
      lesson: "Both languages at <1M params are too small for bAbI reasoning regardless of tokenization.",
      status: "completed"
    }
  ],

  // ── Model architecture details for methodology section ────────────────
  architectures: {
    v2: [
      { size: "Small",  d_model: 128, n_layer: 4, n_head: 2, params: "~835K",  dropout: 0.15 },
      { size: "Medium", d_model: 256, n_layer: 4, n_head: 4, params: "~3.2M",  dropout: 0.20 }
    ],
    v3: [
      { size: "Nano",  d_model: 48,  n_layer: 2, n_head: 2, params: "~67K",  ctx: 128, dropout: 0.05 },
      { size: "Micro", d_model: 64,  n_layer: 3, n_head: 2, params: "~164K", ctx: 128, dropout: 0.08 },
      { size: "Mini",  d_model: 80,  n_layer: 3, n_head: 2, params: "~261K", ctx: 256, dropout: 0.10 },
      { size: "Small", d_model: 96,  n_layer: 4, n_head: 2, params: "~480K", ctx: 256, dropout: 0.12 },
      { size: "Base",  d_model: 128, n_layer: 4, n_head: 2, params: "~837K", ctx: 256, dropout: 0.15 }
    ],
    v4: [
      { size: "Medium", d_model: 96,  n_layer: 4, n_head: 4, params: "~570K", ctx: 256, dropout: 0.15, vocab: 1024 },
      { size: "Large",  d_model: 128, n_layer: 4, n_head: 4, params: "~758K", ctx: 256, dropout: 0.15, vocab: 1024 }
    ]
  }
};
