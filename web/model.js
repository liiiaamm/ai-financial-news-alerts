// In-browser inference for the joint BiLSTM NER + sentiment model.
// Weights are exported by scripts/export_web_model.py as float16 and expanded to float32 here.
(function (root) {
  "use strict";

  // Same tokenizer as pipeline.tokenize(): numbers, dotted abbreviations (U.S.), words with
  // apostrophes, a separate 's, and every other symbol on its own, matching the training data.
  const TOKEN = /\d+(?:[.,]\d+)*|[A-Za-z]+(?:\.[A-Za-z]+)+\.?|[\p{L}\p{N}_]+(?:['’][\p{L}\p{N}_]+)*|'s|[^\p{L}\p{N}_\s]/gu;
  const YEAR = /^(19|20)\d{2}$/;

  function tokenize(text) {
    const out = [];
    for (const m of text.matchAll(TOKEN)) out.push({ text: m[0], start: m.index, end: m.index + m[0].length });
    return out;
  }

  function halfToFloat(h) {
    const s = (h & 0x8000) >> 15, e = (h & 0x7c00) >> 10, f = h & 0x03ff;
    if (e === 0) return (s ? -1 : 1) * Math.pow(2, -14) * (f / 1024);
    if (e === 0x1f) return f ? NaN : (s ? -1 : 1) * Infinity;
    return (s ? -1 : 1) * Math.pow(2, e - 15) * (1 + f / 1024);
  }

  class NewsModel {
    static async load(base, fetchImpl = root.fetch.bind(root)) {
      const [manifest, vocab, buf] = await Promise.all([
        fetchImpl(base + "manifest.json").then((r) => r.json()),
        fetchImpl(base + "vocab.json").then((r) => r.json()),
        fetchImpl(base + "weights.bin").then((r) => r.arrayBuffer()),
      ]);
      return new NewsModel(manifest, vocab, buf);
    }

    constructor(manifest, vocab, buffer) {
      this.manifest = manifest;
      this.tags = manifest.tags;
      this.sentiments = manifest.sentiments;
      this.maxLen = manifest.max_len;
      this.unk = manifest.unk;
      this.word2idx = new Map(vocab.map((w, i) => [w, i]));
      const half = new Uint16Array(buffer);
      const table = new Float32Array(65536);
      for (let i = 0; i < 65536; i++) table[i] = halfToFloat(i);
      this.W = {};
      for (const t of manifest.tensors) {
        const n = t.shape.reduce((a, b) => a * b, 1);
        const arr = new Float32Array(n);
        for (let i = 0; i < n; i++) arr[i] = table[half[t.offset + i]];
        this.W[t.name] = { data: arr, shape: t.shape };
      }
      this.H = this.W["lstm.weight_hh_l0"].shape[1];
      this.E = this.W["embedding.weight"].shape[1];
      // fold the two LSTM biases together once
      for (const l of [0, 1]) for (const s of ["", "_reverse"]) {
        const a = this.W[`lstm.bias_ih_l${l}${s}`].data, b = this.W[`lstm.bias_hh_l${l}${s}`].data;
        const sum = new Float32Array(a.length);
        for (let i = 0; i < a.length; i++) sum[i] = a[i] + b[i];
        this.W[`bias_l${l}${s}`] = sum;
      }
    }

    // One direction of one LSTM layer. X is T x D (row-major), returns T x H.
    _lstm(X, T, D, layer, reverse) {
      const s = reverse ? "_reverse" : "";
      const Wi = this.W[`lstm.weight_ih_l${layer}${s}`].data, Wh = this.W[`lstm.weight_hh_l${layer}${s}`].data;
      const b = this.W[`bias_l${layer}${s}`];
      const H = this.H, G = 4 * H;
      const out = new Float32Array(T * H);
      const h = new Float32Array(H), c = new Float32Array(H), g = new Float32Array(G);
      const sig = (x) => 1 / (1 + Math.exp(-x));
      for (let step = 0; step < T; step++) {
        const t = reverse ? T - 1 - step : step;
        const xo = t * D;
        for (let r = 0; r < G; r++) {
          let acc = b[r];
          const wi = r * D, wh = r * H;
          for (let k = 0; k < D; k++) acc += Wi[wi + k] * X[xo + k];
          for (let k = 0; k < H; k++) acc += Wh[wh + k] * h[k];
          g[r] = acc;
        }
        for (let k = 0; k < H; k++) {
          const i = sig(g[k]), f = sig(g[H + k]), cc = Math.tanh(g[2 * H + k]), o = sig(g[3 * H + k]);
          c[k] = f * c[k] + i * cc;
          h[k] = o * Math.tanh(c[k]);
          out[t * H + k] = h[k];
        }
      }
      return out;
    }

    analyse(text) {
      const all = tokenize(text);
      const tokens = all.slice(0, this.maxLen);
      const T = tokens.length;
      if (!T) return { tokens: [], allTokens: all, entities: [], sentiment: "neutral", probs: { positive: 0, negative: 0, neutral: 1 }, attention: [] };
      const E = this.E, H = this.H, emb = this.W["embedding.weight"].data;
      let X = new Float32Array(T * E);
      const ids = tokens.map((t) => (this.word2idx.has(t.text.toLowerCase()) ? this.word2idx.get(t.text.toLowerCase()) : this.unk));
      ids.forEach((id, t) => X.set(emb.subarray(id * E, id * E + E), t * E));
      let D = E;
      for (const layer of [0, 1]) {
        const f = this._lstm(X, T, D, layer, false), r = this._lstm(X, T, D, layer, true);
        const next = new Float32Array(T * 2 * H);
        for (let t = 0; t < T; t++) {
          next.set(f.subarray(t * H, t * H + H), t * 2 * H);
          next.set(r.subarray(t * H, t * H + H), t * 2 * H + H);
        }
        X = next;
        D = 2 * H;
      }
      // NER head
      const nw = this.W["ner_head.weight"].data, nb = this.W["ner_head.bias"].data, K = nb.length;
      const tagIdx = [];
      for (let t = 0; t < T; t++) {
        let best = -Infinity, arg = 0;
        for (let k = 0; k < K; k++) {
          let acc = nb[k];
          for (let j = 0; j < D; j++) acc += nw[k * D + j] * X[t * D + j];
          if (acc > best) { best = acc; arg = k; }
        }
        tagIdx.push(arg);
      }
      // attention + sentiment head
      const aw = this.W["W_attention.weight"].data;
      const scores = new Float32Array(T);
      let mx = -Infinity;
      for (let t = 0; t < T; t++) {
        let acc = 0;
        for (let j = 0; j < D; j++) acc += aw[j] * X[t * D + j];
        scores[t] = Math.tanh(acc);
        mx = Math.max(mx, scores[t]);
      }
      let z = 0;
      const attention = Array.from(scores, (s) => { const e = Math.exp(s - mx); z += e; return e; }).map((e) => e / z);
      const ctx = new Float32Array(D);
      for (let t = 0; t < T; t++) for (let j = 0; j < D; j++) ctx[j] += attention[t] * X[t * D + j];
      const sw = this.W["sa_head.weight"].data, sb = this.W["sa_head.bias"].data;
      const logits = Array.from(sb, (bias, k) => { let acc = bias; for (let j = 0; j < D; j++) acc += sw[k * D + j] * ctx[j]; return acc; });
      const lm = Math.max(...logits);
      const ex = logits.map((l) => Math.exp(l - lm));
      const zs = ex.reduce((a, b) => a + b, 0);
      const probs = {};
      this.sentiments.forEach((s, k) => { probs[s] = ex[k] / zs; });
      const sentiment = this.sentiments[logits.indexOf(lm)];

      const tags = tagIdx.map((k) => this.tags[k]);
      return { tokens, allTokens: all, tags, entities: extractEntities(text, tokens, tags), sentiment, probs, attention };
    }
  }

  // Years are tagged by rule (the training data has no YEAR label); then BIO spans are merged.
  function extractEntities(text, tokens, tags) {
    const fixed = tags.map((tag, i) => (YEAR.test(tokens[i].text) ? (i > 0 && YEAR.test(tokens[i - 1].text) ? "YEAR_I" : "YEAR_B") : tag));
    const out = [];
    let cur = null;
    const flush = () => { if (cur) { cur.text = text.slice(cur.start, cur.end); out.push(cur); cur = null; } };
    fixed.forEach((tag, i) => {
      if (tag === "O") return flush();
      const [type, bio] = tag.split("_");
      if (bio === "B" || !cur || cur.type !== type) { flush(); cur = { type, start: tokens[i].start, end: tokens[i].end, first: i, last: i }; }
      else { cur.end = tokens[i].end; cur.last = i; }
    });
    flush();
    return out;
  }

  // Port of pipeline._rule_based().
  function ruleAlert(entities, sentiment) {
    const of = (t) => entities.filter((e) => e.type === t);
    const orgs = of("ORG"), pers = of("PER"), locs = of("LOC");
    const list = (l) => l.map((e) => `${e.text} [${e.type}]`).join(", ");
    if (orgs.length && sentiment === "negative") return ["Reputation risk", `${orgs[0].text} mentioned negatively.`];
    if (pers.length && sentiment === "negative") return ["Reputation risk", `${pers[0].text} mentioned negatively.`];
    if (orgs.length && sentiment === "positive") return ["Positive update", `${orgs[0].text} reported positively.`];
    if (pers.length && sentiment === "positive") return ["Positive update", `${pers[0].text} mentioned in a positive context.`];
    if (locs.length && sentiment === "negative") return ["Geopolitical risk", `Negative news involving ${locs[0].text}.`];
    if (locs.length && sentiment === "positive") return ["Positive development", `Good news from ${locs[0].text}.`];
    if (entities.length > 1 && sentiment === "negative") return ["Risk alert", `Negative sentiment. Entities: ${list(entities)}.`];
    if (entities.length > 1 && sentiment === "positive") return ["Positive alert", `Positive sentiment. Entities: ${list(entities)}.`];
    if (!entities.length && sentiment === "negative") return ["Sentiment alert", "Negative sentiment, no named entities."];
    if (!entities.length && sentiment === "positive") return ["Sentiment alert", "Positive sentiment, no named entities."];
    if (entities.length && sentiment === "neutral") return ["Info", `Neutral content. Entities: ${list(entities)}.`];
    return ["No alert", "Neutral content with no named entities."];
  }

  const api = { NewsModel, tokenize, ruleAlert };
  if (typeof module !== "undefined") module.exports = api;
  else root.NewsWire = api;
})(typeof window !== "undefined" ? window : globalThis);
