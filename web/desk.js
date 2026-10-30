// The night desk: loads the model and the archive, marks up a story and prints the alert.
(() => {
  "use strict";
  const $ = (id) => document.getElementById(id);
  const reduceMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
  const esc = (s) => s.replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));
  const TYPE_CLASS = { ORG: "org", PER: "per", LOC: "loc", YEAR: "year" };
  const TYPE_NAME = { ORG: "Organisations", PER: "People", LOC: "Places", YEAR: "Years" };
  const STAMP = { positive: "POSITIVE", negative: "NEGATIVE", neutral: "NEUTRAL" };

  let model, samples, current = null, timers = [];
  const later = (fn, ms) => timers.push(setTimeout(fn, reduceMotion ? 0 : ms));
  const clearTimers = () => { timers.forEach(clearTimeout); timers = []; };

  // ------------------------------------------------------------------ story rendering

  function renderStory(text, result) {
    const toks = result.allTokens;
    const read = result.tokens.length;
    const att = result.attention;
    // Only the words with the most attention get a pencil mark: the top ~12 %, scaled
    // between that cut-off and the maximum, so the marks show where it really looked.
    const sorted = [...att].sort((x, y) => y - x);
    const cutA = sorted[Math.max(0, Math.ceil(sorted.length * 0.12) - 1)] || 0;
    const maxA = sorted[0] || 1;
    const tokClass = new Array(toks.length).fill("");
    const gapClass = new Array(toks.length).fill("");
    result.entities.forEach((e) => {
      const c = `hl ${TYPE_CLASS[e.type]}`;
      for (let i = e.first; i <= e.last; i++) { tokClass[i] = c; if (i > e.first) gapClass[i] = c; }
    });
    let html = "", pos = 0;
    toks.forEach((t, i) => {
      const gap = text.slice(pos, t.start);
      if (gap) html += gapClass[i] ? `<span class="gap ${gapClass[i]}" data-g="${i}">${esc(gap)}</span>` : esc(gap);
      const raw = i < read ? att[i] : 0;
      const a = raw >= cutA && maxA > cutA ? 0.35 + 0.65 * ((raw - cutA) / (maxA - cutA)) : raw >= cutA ? 1 : 0;
      const style = a > 0 ? ` style="--a:${a.toFixed(3)}"` : "";
      html += `<span class="tok ${tokClass[i]}${i >= read ? " late" : ""}" data-i="${i}"${style}>${esc(t.text)}</span>`;
      pos = t.end;
    });
    html += esc(text.slice(pos));
    const story = $("story");
    story.innerHTML = html.replace(/\n/g, "<br>");
    const cut = $("cutoff");
    if (toks.length > read) {
      cut.hidden = false;
      cut.textContent = `The model reads the first ${read} tokens; the remaining ${toks.length - read} are greyed out.`;
    } else cut.hidden = true;
  }

  // ------------------------------------------------------------------ the markup sequence

  function animate(result) {
    clearTimers();
    const story = $("story");
    const stamp = $("stamp"), receipt = $("receipt");
    stamp.className = "stamp";
    receipt.className = "receipt";
    receipt.innerHTML = "";

    // 1. pencil follows the reading, left to right
    const toks = [...story.querySelectorAll(".tok:not(.late)")];
    const pencilTime = Math.min(900, 7 * toks.length);
    const batches = 24;
    for (let b = 0; b < batches; b++) {
      const slice = toks.slice(Math.floor((b * toks.length) / batches), Math.floor(((b + 1) * toks.length) / batches));
      later(() => slice.forEach((el) => el.classList.add("pen")), (b * pencilTime) / batches);
    }

    // 2. highlighters, one entity after another
    let t = pencilTime + 120;
    // all highlighting fits in about 1.4 s however many entities there are
    const perEntity = result.entities.length ? Math.min(220, 1400 / result.entities.length) : 0;
    result.entities.forEach((e) => {
      const els = [];
      for (let i = e.first; i <= e.last; i++) {
        const g = story.querySelector(`.gap[data-g="${i}"]`);
        if (g) els.push(g);
        els.push(story.querySelector(`.tok[data-i="${i}"]`));
      }
      const dur = Math.min(perEntity / 1000, 0.08 + 0.02 * (e.end - e.start));
      els.forEach((el, k) => {
        el.style.setProperty("--dur", `${(dur / els.length).toFixed(3)}s`);
        later(() => el.classList.add("on"), t + (k * dur * 1000) / els.length);
      });
      t += perEntity;
    });

    // 3. the stamp
    t += 200;
    later(() => {
      const conf = result.probs[result.sentiment];
      $("stampWord").textContent = STAMP[result.sentiment];
      $("stampConf").textContent = `${Math.round(conf * 100)}% sure`;
      stamp.className = `stamp ${result.sentiment}`;
      void stamp.offsetWidth;
      stamp.classList.add("down");
    }, t);

    // 4. the receipt
    later(() => printReceipt(result), t + 380);
  }

  function printReceipt(result) {
    const [type, msg] = NewsWire.ruleAlert(result.entities, result.sentiment);
    const now = new Date();
    const stampTime = now.toISOString().slice(0, 16).replace("T", " ");
    const groups = {};
    result.entities.forEach((e) => {
      const g = (groups[e.type] = groups[e.type] || []);
      if (!g.includes(e.text)) g.push(e.text);
    });
    const ents = Object.keys(TYPE_NAME).filter((k) => groups[k])
      .map((k) => `<p><b>${TYPE_NAME[k]}</b><br>${groups[k].map(esc).join(", ")}</p>`).join("") || "<p>None found</p>";
    const bars = model.sentiments.map((s) => {
      const p = result.probs[s];
      return `<div class="row"><span>${s}</span><span><span class="bar" style="width:${Math.round(p * 60)}px"></span> ${(p * 100).toFixed(0)}%</span></div>`;
    }).join("");
    const receipt = $("receipt");
    receipt.innerHTML = `
      <h3>Night desk alert</h3>
      <p>${stampTime}</p>
      <p>${current ? esc(current.title) : "Your own story"}</p>
      <hr>
      <p><b>${esc(type)}</b></p>
      <p>${esc(msg)}</p>
      <hr>
      ${ents}
      <hr>
      ${bars}
      <hr>
      <p>Read ${result.tokens.length} of ${result.allTokens.length} tokens in ${result.ms} ms, on this device.</p>`;
    receipt.classList.add("printed");
  }

  // ------------------------------------------------------------------ actions

  function run(text) {
    const btn = $("runBtn");
    btn.disabled = true;
    btn.textContent = "Reading…";
    setTimeout(() => {
      const t0 = performance.now();
      const result = model.analyse(text);
      result.ms = Math.round(performance.now() - t0);
      $("compose").hidden = true;
      $("story").hidden = false;
      renderStory(text, result);
      animate(result);
      btn.disabled = false;
      btn.textContent = "Mark it up again";
    }, 30);
  }

  function loadSample(s, { autorun = true } = {}) {
    current = s;
    $("storyTitle").textContent = s.title;
    document.querySelectorAll(".cutting").forEach((c) => c.setAttribute("aria-current", String(c.dataset.id === s.id)));
    const clip = $("clip");
    if (s.chart) {
      clip.hidden = false;
      $("chartImg").src = s.chart;
      $("chartImg").alt = `Chart that came with the story: ${s.title}`;
      $("caption").textContent = s.caption || "";
      $("captionMore").hidden = !s.caption || s.caption.length < 260;
      clip.querySelector(".sticky").classList.remove("open");
    } else clip.hidden = true;
    $("compose").hidden = true;
    $("story").hidden = false;
    $("editBtn").textContent = "Write your own";
    if (autorun) run(s.text);
  }

  function openEditor() {
    clearTimers();
    current = null;
    $("storyTitle").textContent = "Your own story";
    $("clip").hidden = true;
    $("story").hidden = true;
    $("cutoff").hidden = true;
    $("compose").hidden = false;
    $("runBtn").textContent = "Mark it up";
    document.querySelectorAll(".cutting").forEach((c) => c.setAttribute("aria-current", "false"));
    const ta = $("composeText");
    ta.focus();
  }

  function buildArchive() {
    const box = $("cuttings");
    samples.forEach((s, i) => {
      const b = document.createElement("button");
      b.type = "button";
      b.className = "cutting";
      b.dataset.id = s.id;
      b.style.setProperty("--r", `${[-1.6, 1.1, -0.6, 1.8, -1.2, 0.7, -1.9, 1.4, -0.4, 0.9][i % 10]}deg`);
      const words = s.text.split(/\s+/).length;
      b.innerHTML = `${s.chart ? `<img src="${s.chart}" alt="" loading="lazy">` : ""}<strong>${esc(s.title)}</strong><span>${words} words</span>`;
      b.addEventListener("click", () => {
        loadSample(s);
        $("sheet").scrollIntoView({ behavior: reduceMotion ? "auto" : "smooth", block: "start" });
      });
      box.appendChild(b);
    });
  }

  $("runBtn").addEventListener("click", () => {
    if (!$("compose").hidden) {
      const text = $("composeText").value.trim();
      if (!text) { $("composeText").focus(); return; }
      run(text);
      $("editBtn").textContent = "Edit the text";
    } else if (current) run(current.text);
    else run($("composeText").value.trim());
  });
  $("editBtn").addEventListener("click", () => {
    if (!current && $("compose").hidden) { $("compose").hidden = false; $("story").hidden = true; $("cutoff").hidden = true; $("composeText").focus(); return; }
    openEditor();
  });
  $("captionMore").addEventListener("click", (e) => {
    const st = $("clip").querySelector(".sticky");
    st.classList.toggle("open");
    e.target.textContent = st.classList.contains("open") ? "Show less" : "Read all";
  });

  Promise.all([
    NewsWire.NewsModel.load("web/model/"),
    fetch("web/samples.json").then((r) => r.json()),
  ]).then(([m, s]) => {
    model = m;
    samples = s;
    buildArchive();
    $("runBtn").disabled = false;
    loadSample(samples.find((x) => x.id === "p3"));
  }).catch((err) => {
    $("storyTitle").textContent = "The model could not be loaded";
    $("story").textContent = "If you opened this file directly, serve the folder instead: python -m http.server";
    console.error(err);
  });
})();
