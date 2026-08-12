// Interactive visualizations for franken-ocr.com.
//
// Every number drawn here is measured or is arithmetic over measured numbers,
// and every visualization is an ENHANCEMENT of markup that already carries the
// same facts: the pipeline stepper is built from the <ol> of stages in the
// page, the bar charts are built from real <table>s. Turn JavaScript off and
// you lose the motion, not the content.
//
// Vanilla ES modules, SVG + DOM. No framework, no bundler, no network.

const reduced = window.matchMedia("(prefers-reduced-motion: reduce)").matches;

const NS = "http://www.w3.org/2000/svg";

/** Create an SVG element with attributes, appended to `parent`. */
function svg(tag, attrs = {}, parent = null) {
  const el = document.createElementNS(NS, tag);
  for (const [k, v] of Object.entries(attrs)) el.setAttribute(k, String(v));
  if (parent) parent.appendChild(el);
  return el;
}

/** Create an HTML element with props/attrs. */
function h(tag, props = {}, ...children) {
  const el = document.createElement(tag);
  for (const [k, v] of Object.entries(props)) {
    if (k === "class") el.className = v;
    else if (k === "text") el.textContent = v;
    else if (k === "html") el.innerHTML = v;
    else if (k.startsWith("on")) el.addEventListener(k.slice(2), v);
    else if (v !== null && v !== undefined) el.setAttribute(k, String(v));
  }
  for (const c of children) if (c) el.append(c);
  return el;
}

function text(parent, x, y, str, opts = {}) {
  const t = svg(
    "text",
    {
      x,
      y,
      fill: opts.fill ?? "#94a3b8",
      "font-size": opts.size ?? 12.5,
      "font-weight": opts.weight ?? 600,
      "text-anchor": opts.anchor ?? "middle",
      "font-family": opts.mono ? "JetBrains Mono, monospace" : "Inter, sans-serif",
      ...(opts.extra ?? {}),
    },
    parent,
  );
  t.textContent = str;
  return t;
}

/**
 * Build on approach — but never trust intersection alone. An instant scroll
 * jump (End key, scrollbar drag, deep link) can skip an element without a
 * single callback, so a timeout guarantees every visualization exists.
 */
function lazyBuild(el, build, { rootMargin = "220px", timeout = 1800 } = {}) {
  let done = false;
  const once = () => {
    if (done) return;
    done = true;
    try {
      build();
    } catch (err) {
      // A broken visualization must never take the page (or the playground)
      // down with it; the static markup underneath stays readable.
      console.error("viz build failed", el?.id, err);
    }
  };
  const obs = new IntersectionObserver(
    (entries, o) => {
      if (entries.some((e) => e.isIntersecting)) {
        o.disconnect();
        once();
      }
    },
    { rootMargin },
  );
  obs.observe(el);
  setTimeout(once, timeout);
}

/** Run once when the element is actually on screen (or after a timeout). */
function onceVisible(el, run, { threshold = 0.25, timeout = 1500 } = {}) {
  let done = false;
  const fire = () => {
    if (done) return;
    done = true;
    run();
  };
  const obs = new IntersectionObserver(
    (entries, o) => {
      if (entries.some((e) => e.isIntersecting)) {
        o.disconnect();
        fire();
      }
    },
    { threshold },
  );
  obs.observe(el);
  setTimeout(fire, timeout);
}

/**
 * Run a RAF loop only while `el` is near the viewport. Returns nothing; the
 * observer owns the loop's lifetime, so scrolling away costs zero CPU.
 */
function rafWhileVisible(el, tick) {
  if (reduced) return;
  let raf = null;
  const loop = (now) => {
    tick(now);
    raf = requestAnimationFrame(loop);
  };
  new IntersectionObserver(
    (entries) => {
      for (const e of entries) {
        if (e.isIntersecting && raf === null) raf = requestAnimationFrame(loop);
        else if (!e.isIntersecting && raf !== null) {
          cancelAnimationFrame(raf);
          raf = null;
        }
      }
    },
    { rootMargin: "150px" },
  ).observe(el);
}

const C = {
  accent: "#34d399",
  amber: "#fbbf24",
  dim: "#94a3b8",
  mid: "#cbd5e1",
  faint: "#748496",
  line: "rgba(255,255,255,0.09)",
  fillIdle: "rgba(255,255,255,0.03)",
};

/* ==========================================================================
   1. Pipeline anatomy — image → SAM → compressor → CLIP → projector → MoE → text
   ========================================================================== */

const pipelineRoot = document.getElementById("pipeline-viz");

function buildPipeline() {
  const list = document.getElementById("pipeline-stages");
  if (!list) return;

  // The stages ARE the markup: read them so the diagram and the prose can
  // never drift apart.
  const stages = [...list.querySelectorAll("li")].map((li) => ({
    name: li.dataset.name,
    shape: li.dataset.shape,
    cost: li.dataset.cost ?? "",
    kind: li.dataset.kind ?? "vision",
    title: li.querySelector("h3")?.textContent ?? li.dataset.name,
    body: li.querySelector("p")?.innerHTML ?? "",
  }));
  if (!stages.length) return;

  const W = 980;
  const H = 232;
  const root = svg("svg", {
    viewBox: `0 0 ${W} ${H}`,
    role: "img",
    "aria-label":
      "The Unlimited-OCR pipeline: a page image through SAM-ViT-B, a sixteen-times " +
      "convolutional token compressor, CLIP-L/14, a linear projector, and a twelve-layer " +
      "mixture-of-experts decoder, to Markdown text.",
  });
  root.style.width = "100%";
  root.style.height = "auto";
  pipelineRoot.appendChild(root);

  const n = stages.length;
  const padX = 14;
  const gap = 12;
  const boxW = (W - padX * 2 - gap * (n - 1)) / n;
  const boxH = 92;
  const boxY = 58;

  const nodes = stages.map((stage, i) => {
    const x = padX + i * (boxW + gap);
    const g = svg("g", { class: "stage-node" }, root);
    const isDecoder = stage.kind === "decoder";
    const rect = svg(
      "rect",
      {
        x,
        y: boxY,
        width: boxW,
        height: boxH,
        rx: 12,
        fill: isDecoder ? "rgba(16,185,129,0.07)" : C.fillIdle,
        stroke: isDecoder ? "rgba(52,211,153,0.32)" : C.line,
        "stroke-width": 1.2,
      },
      g,
    );
    // Stage name, wrapped by hand onto at most two lines (SVG has no wrapping).
    const words = stage.name.split(" ");
    const lines = words.length > 2 ? [words.slice(0, -1).join(" "), words.at(-1)] : [stage.name];
    const cx = x + boxW / 2;
    lines.forEach((line, li) =>
      text(g, cx, boxY + (lines.length === 1 ? 34 : 27 + li * 15), line, {
        size: 12.5,
        weight: 800,
        fill: isDecoder ? "#a7f3d0" : C.mid,
      }),
    );
    text(g, cx, boxY + 60, stage.shape, { size: 10.5, fill: C.dim, mono: true, weight: 700 });
    if (stage.cost) text(g, cx, boxY + 78, stage.cost, { size: 10, fill: C.faint, mono: true });
    // Index badge above the box.
    text(g, cx, boxY - 12, String(i + 1).padStart(2, "0"), {
      size: 10,
      weight: 800,
      fill: C.faint,
      mono: true,
    });
    // Connector arrow into the next box.
    if (i < n - 1) {
      const from = x + boxW;
      const to = from + gap;
      svg(
        "path",
        {
          d: `M ${from + 1} ${boxY + boxH / 2} L ${to - 4} ${boxY + boxH / 2}`,
          stroke: "rgba(148,163,184,0.4)",
          "stroke-width": 1.4,
        },
        root,
      );
      svg(
        "path",
        {
          d: `M ${to - 6} ${boxY + boxH / 2 - 3.5} L ${to - 1} ${boxY + boxH / 2} L ${to - 6} ${boxY + boxH / 2 + 3.5}`,
          fill: "none",
          stroke: "rgba(148,163,184,0.6)",
          "stroke-width": 1.4,
        },
        root,
      );
    }
    return { g, rect, x, cx, stage };
  });

  // Grouping brackets: which stages are the vision tower, which the decoder.
  const visionEnd = nodes.findLastIndex((nd) => nd.stage.kind === "vision");
  const bracket = (i0, i1, label, color) => {
    const x0 = nodes[i0].x - 5;
    const x1 = nodes[i1].x + boxW + 5;
    const y = boxY + boxH + 16;
    svg(
      "path",
      { d: `M ${x0} ${y} L ${x0} ${y + 6} L ${x1} ${y + 6} L ${x1} ${y}`, fill: "none", stroke: color, "stroke-width": 1, opacity: 0.5 },
      root,
    );
    text(root, (x0 + x1) / 2, y + 22, label, { size: 10, weight: 800, fill: color, extra: { "letter-spacing": "0.18em" } });
  };
  if (visionEnd > 0) {
    bracket(1, visionEnd, "VISION TOWER · KEPT BF16", "rgba(167,139,250,0.85)");
    if (visionEnd + 1 <= n - 2) {
      bracket(visionEnd + 1, n - 2, "DECODER · INT4 + INT8", "rgba(52,211,153,0.85)");
    }
  }

  // A travelling pulse that traces the whole chain: the one thing static
  // markup cannot say is "this is one forward pass, left to right".
  const pulse = svg("circle", { r: 5, cy: boxY + boxH / 2, cx: padX, fill: C.accent, opacity: 0 }, root);
  if (!reduced) {
    const x0 = padX + boxW / 2;
    const x1 = W - padX - boxW / 2;
    const PERIOD = 6200;
    rafWhileVisible(pipelineRoot, (now) => {
      const t = (now % PERIOD) / PERIOD;
      // Ease so the pulse dwells inside each box rather than sliding evenly.
      const eased = t < 0.86 ? t / 0.86 : 1;
      pulse.setAttribute("cx", (x0 + eased * (x1 - x0)).toFixed(1));
      pulse.setAttribute("opacity", t < 0.86 ? 0.95 : Math.max(0, 1 - (t - 0.86) / 0.14));
    });
  }

  /* --- the stage detail panel: the <ol> becomes a keyboard-driven stepper --- */
  const items = [...list.querySelectorAll("li")];
  const detail = h("div", { class: "step-body", "aria-live": "polite" });
  const head = h("div", { class: "step-head" });
  const counter = h("span", { class: "step-count" });
  const title = h("span", { class: "step-title" });
  const stat = h("span", { class: "step-stat" });
  head.append(counter, title, stat);

  const dots = h("div", { class: "step-dots", "aria-label": "Pipeline stages" });
  const dotEls = stages.map((s, i) =>
    h("button", {
      type: "button",
      class: "step-dot",
      "aria-label": `Stage ${i + 1}: ${s.name}`,
      onclick: () => show(i, true),
    }),
  );
  dots.append(...dotEls);
  const prev = h("button", { type: "button", class: "btn", text: "← Prev" });
  const next = h("button", { type: "button", class: "btn primary", text: "Next →" });
  const nav = h("div", { class: "step-nav" }, prev, dots, next);

  let index = 0;
  let auto = !reduced;

  function show(i, byUser = false) {
    index = (i + stages.length) % stages.length;
    if (byUser) auto = false;
    const s = stages[index];
    counter.textContent = `${index + 1} / ${stages.length}`;
    title.textContent = s.title;
    stat.textContent = s.shape;
    detail.innerHTML = s.body;
    dotEls.forEach((d, j) => d.setAttribute("aria-current", j === index ? "true" : "false"));
    nodes.forEach((nd, j) => {
      const on = j === index;
      nd.rect.setAttribute(
        "fill",
        on ? "rgba(52,211,153,0.2)" : nd.stage.kind === "decoder" ? "rgba(16,185,129,0.07)" : C.fillIdle,
      );
      nd.rect.setAttribute("stroke", on ? C.accent : nd.stage.kind === "decoder" ? "rgba(52,211,153,0.32)" : C.line);
      nd.rect.setAttribute("stroke-width", on ? 2 : 1.2);
    });
  }

  prev.addEventListener("click", () => show(index - 1, true));
  next.addEventListener("click", () => show(index + 1, true));
  // Hovering or focusing a box in the diagram selects it — desktop affordance
  // layered on top of the buttons, which are the accessible path.
  nodes.forEach((nd, i) => {
    nd.g.style.cursor = "pointer";
    nd.g.addEventListener("mouseenter", () => show(i, true));
    nd.g.addEventListener("click", () => show(i, true));
  });
  const stepper = h("div", { class: "stepper" }, head, detail, nav);
  stepper.tabIndex = 0;
  stepper.setAttribute("aria-label", "Pipeline stage details; use the left and right arrow keys");
  stepper.addEventListener("keydown", (e) => {
    if (e.key === "ArrowLeft") { e.preventDefault(); show(index - 1, true); }
    if (e.key === "ArrowRight") { e.preventDefault(); show(index + 1, true); }
  });

  // Replace the static list with the stepper (the list's content lives on
  // inside `stages`, so nothing is lost — it is re-rendered on demand).
  list.replaceWith(stepper);
  for (const li of items) li.remove();
  show(0);

  // A guided tour that runs ONCE and then hands over: an animation that loops
  // forever competes with the reader for the rest of the page. A timer (rather
  // than a RAF loop) because 4.2 s is not a frame-rate problem, and it only
  // ticks while the stepper is actually on screen.
  if (auto) {
    let timer = null;
    const stop = () => {
      if (timer !== null) clearInterval(timer);
      timer = null;
    };
    new IntersectionObserver((entries) => {
      for (const e of entries) {
        if (e.isIntersecting && auto && timer === null) {
          timer = setInterval(() => {
            if (!auto) return stop();
            if (index === stages.length - 1) {
              auto = false;
              return stop();
            }
            show(index + 1);
          }, 4200);
        } else if (!e.isIntersecting) {
          stop();
        }
      }
    }, { rootMargin: "0px" }).observe(stepper);
    stepper.addEventListener("pointerdown", () => { auto = false; stop(); });
  }
}

if (pipelineRoot) lazyBuild(pipelineRoot, buildPipeline);

/* ==========================================================================
   2. MoE routing — top-6 of 64, plus 2 shared, per token, per layer
   ========================================================================== */

const moeRoot = document.getElementById("moe-viz");

function buildMoe() {
  const W = 900;
  const H = 300;
  const root = svg("svg", {
    viewBox: `0 0 ${W} ${H}`,
    role: "img",
    "aria-label":
      "Mixture-of-experts routing: for every token the f32 router picks the top six of " +
      "sixty-four experts, and two shared experts always run.",
  });
  root.style.width = "100%";
  root.style.height = "auto";
  moeRoot.appendChild(root);

  // Router column
  text(root, 88, 34, "ONE TOKEN", { size: 10.5, weight: 800, fill: C.faint, extra: { "letter-spacing": "0.18em" } });
  const tokenBox = svg("rect", { x: 24, y: 48, width: 128, height: 40, rx: 9, fill: "rgba(167,139,250,0.1)", stroke: "rgba(167,139,250,0.4)" }, root);
  text(root, 88, 73, "hidden 1280", { size: 12, weight: 700, fill: "#c4b5fd", mono: true });

  svg("rect", { x: 24, y: 116, width: 128, height: 52, rx: 9, fill: "rgba(251,191,36,0.07)", stroke: "rgba(251,191,36,0.42)" }, root);
  text(root, 88, 136, "ROUTER", { size: 11, weight: 800, fill: C.amber, extra: { "letter-spacing": "0.16em" } });
  text(root, 88, 154, "f32 · never quantized", { size: 10.5, fill: C.dim, mono: true });
  text(root, 88, 192, "softmax → top-6 greedy", { size: 11, fill: C.dim });
  text(root, 88, 208, "no renormalization", { size: 11, fill: C.faint });

  // Expert grid: 8 x 8 = 64 routed experts
  const gx = 268;
  const gy = 46;
  const cell = 26;
  const pad = 5;
  const experts = [];
  for (let i = 0; i < 64; i++) {
    const col = i % 8;
    const row = (i / 8) | 0;
    experts.push(
      svg(
        "rect",
        {
          x: gx + col * (cell + pad),
          y: gy + row * (cell + pad),
          width: cell,
          height: cell,
          rx: 6,
          fill: C.fillIdle,
          stroke: C.line,
          "stroke-width": 1,
          style: reduced ? "" : "transition: fill .3s ease, stroke .3s ease",
        },
        root,
      ),
    );
  }
  const gridW = 8 * cell + 7 * pad;
  text(root, gx, gy - 14, "64 ROUTED EXPERTS", {
    size: 10,
    weight: 800,
    fill: C.faint,
    anchor: "start",
    extra: { "letter-spacing": "0.14em" },
  });

  // Shared experts: always on
  const sx = gx + gridW + 44;
  text(root, sx + 54, gy - 14, "ALWAYS ON", { size: 10, weight: 800, fill: C.amber, extra: { "letter-spacing": "0.14em" } });
  for (let i = 0; i < 2; i++) {
    svg(
      "rect",
      { x: sx, y: gy + i * 44, width: 108, height: 34, rx: 8, fill: "rgba(251,191,36,0.12)", stroke: "rgba(251,191,36,0.5)" },
      root,
    );
    text(root, sx + 54, gy + 22 + i * 44, `shared ${i + 1}`, { size: 11, weight: 700, fill: "#fde68a", mono: true });
  }
  text(root, sx + 54, gy + 108, "intermediate", { size: 10.5, fill: C.dim });
  text(root, sx + 54, gy + 123, "1792, fused", { size: 10.5, fill: C.dim, mono: true });

  // Read-out line under the grid
  const readout = text(root, gx + gridW / 2, gy + 8 * (cell + pad) + 22, "", { size: 12.5, weight: 700, fill: C.mid, mono: true });
  const readout2 = text(root, gx + gridW / 2, gy + 8 * (cell + pad) + 40, "", { size: 11.5, fill: C.dim });

  // Measured: routed-expert tensors total 1.716 GB over 11 MoE layers x 64
  // experts; shared experts total 0.054 GB over 11 layers. Per token a layer
  // reads 6 of its 64 routed experts plus both shared ones.
  const ROUTED_GB = 1.716;
  const SHARED_GB = 0.054;
  const perTokenGB = (ROUTED_GB * 6) / 64 + SHARED_GB;
  readout2.textContent =
    "each expert a 1280↔896 SiLU-gated MLP · top-6 of 64 = 9.4% of the bank, per token, per layer";

  // Deterministic pseudo-random choice so every visitor sees the same run and
  // the picture is reproducible (an "aha" you can point at, not noise).
  let seed = 20260811;
  const rnd = () => ((seed = (seed * 1103515245 + 12345) & 0x7fffffff) / 0x7fffffff);
  const pickSix = () => {
    const chosen = new Set();
    while (chosen.size < 6) chosen.add((rnd() * 64) | 0);
    return [...chosen];
  };

  let tokenNo = 0;
  const render = (chosen) => {
    experts.forEach((e, i) => {
      const on = chosen.includes(i);
      e.setAttribute("fill", on ? "rgba(52,211,153,0.6)" : C.fillIdle);
      e.setAttribute("stroke", on ? C.accent : C.line);
    });
    readout.textContent =
      `token ${tokenNo} · experts ${chosen.slice().sort((a, b) => a - b).map((i) => `#${i}`).join(" ")}`;
  };

  tokenNo = 1;
  render(pickSix());
  if (!reduced) {
    // One token every 1.25 s is a timer, not an animation: the cells cross-fade
    // in CSS. A RAF loop here would wake 60 times a second to do nothing.
    let timer = null;
    new IntersectionObserver(
      (entries) => {
        for (const e of entries) {
          if (e.isIntersecting && timer === null) {
            timer = setInterval(() => {
              tokenNo += 1;
              render(pickSix());
            }, 1250);
          } else if (!e.isIntersecting && timer !== null) {
            clearInterval(timer);
            timer = null;
          }
        }
      },
      { rootMargin: "120px" },
    ).observe(moeRoot);
  }

  const foot = h("p", {
    class: "viz-caption",
    html:
      `Stored: <b class="mono hl">${ROUTED_GB.toFixed(2)} GB</b> of routed experts across 11 mixture-of-experts layers ` +
      `(layer 0 is dense). Read per output token: about <b class="mono hl">${perTokenGB.toFixed(2)} GB</b> ` +
      `of expert weights — arithmetic over the measured tensor table, not a benchmark. ` +
      `Sparsity is why a 3.3-billion-parameter model fits a browser tab's patience at all.`,
  });
  moeRoot.appendChild(foot);
}

if (moeRoot) lazyBuild(moeRoot, buildMoe);

/* ==========================================================================
   3. Bits — bf16 versus int4 nibbles with group scales
   ========================================================================== */

const bitsRoot = document.getElementById("bits-viz");

function buildBits() {
  const W = 900;
  const H = 214;
  const root = svg("svg", {
    viewBox: `0 0 ${W} ${H}`,
    role: "img",
    "aria-label":
      "Sixteen weights stored as bf16 versus as int4 nibbles sharing one f32 group scale.",
  });
  root.style.width = "100%";
  root.style.height = "auto";
  bitsRoot.appendChild(root);

  const N = 16; // weights drawn
  const left = 128;
  const right = W - 24;
  const laneW = right - left;

  // --- bf16 lane: 16 bits per weight
  text(root, left - 12, 42, "bf16", { size: 13, weight: 800, fill: C.faint, anchor: "end", mono: true });
  text(root, left - 12, 58, "16 bits each", { size: 10.5, fill: C.faint, anchor: "end" });
  const bfCellW = laneW / N;
  for (let i = 0; i < N; i++) {
    const x = left + i * bfCellW;
    svg("rect", { x: x + 1, y: 26, width: bfCellW - 2, height: 30, rx: 4, fill: "rgba(100,116,139,0.22)", stroke: "rgba(148,163,184,0.35)" }, root);
    // 16 tick marks = 16 bits, so the width difference is literal, not a metaphor.
    for (let b = 0; b < 16; b++) {
      svg("rect", { x: x + 2 + b * ((bfCellW - 4) / 16), y: 32, width: Math.max(0.8, (bfCellW - 4) / 16 - 0.7), height: 18, fill: "rgba(203,213,225,0.5)" }, root);
    }
  }
  text(root, (left + right) / 2, 76, `${N} weights = ${N * 16} bits = ${N * 2} bytes`, { size: 11.5, fill: C.dim, mono: true });

  // --- int4 lane: 4 bits per weight + one f32 scale for the group
  text(root, left - 12, 128, "int4", { size: 13, weight: 800, fill: C.accent, anchor: "end", mono: true });
  text(root, left - 12, 144, "4 bits each", { size: 10.5, fill: C.faint, anchor: "end" });
  // Draw the int4 lane at the SAME bits-per-pixel scale as the bf16 lane above:
  // 4/16 of the width. Anything else would lie about the saving.
  const q4W = laneW / 4;
  const nibW = q4W / N;
  for (let i = 0; i < N; i++) {
    const x = left + i * nibW;
    svg("rect", { x: x + 0.6, y: 112, width: nibW - 1.2, height: 30, rx: 2.5, fill: "rgba(16,185,129,0.35)", stroke: "rgba(52,211,153,0.55)", "stroke-width": 0.8 }, root);
  }
  // The group scale, drawn to the same scale: 32 bits = two bf16 cells wide.
  const scaleX = left + q4W + 8;
  const scaleW = bfCellW * 2;
  svg("rect", { x: scaleX, y: 112, width: scaleW, height: 30, rx: 4, fill: "rgba(251,191,36,0.18)", stroke: "rgba(251,191,36,0.55)" }, root);
  text(root, scaleX + scaleW / 2, 132, "f32 scale", { size: 10.5, weight: 700, fill: "#fde68a", mono: true });
  svg("path", { d: `M ${left} 150 L ${left} 156 L ${scaleX + scaleW} 156 L ${scaleX + scaleW} 150`, fill: "none", stroke: "rgba(52,211,153,0.5)" }, root);
  const totalLabel = text(root, (left + scaleX + scaleW) / 2, 172, "", { size: 11.5, fill: C.mid, mono: true, weight: 700 });
  const perLabel = text(root, (left + scaleX + scaleW) / 2, 190, "", { size: 11, fill: C.dim });


  /* --- controls: group size drives everything, including a measured check --- */
  const GROUPS = [16, 32, 64, 128];
  // Measured from the shipped artifact's own tensor table.
  const EXPERT_PARAMS = 4.844e9 / 2; // bf16 bytes / 2 = parameters in the routed expert bank
  const OTHER_GB = 3.004 - 1.716; // everything that is not a routed expert, as shipped

  const slider = h("input", {
    type: "range",
    min: "0",
    max: String(GROUPS.length - 1),
    value: "0",
    step: "1",
    class: "slider",
    "aria-label": "Quantization group size",
  });
  const readout = h("div", { class: "readout" });
  const shipped = h("button", { type: "button", class: "btn", text: "Shipped recipe" });
  const controls = h("div", { class: "viz-controls" }, slider, readout, shipped);
  const verdict = h("p", { class: "viz-caption", "aria-live": "polite" });

  const bitsPerWeight = (g) => 4 + 32 / g;

  const paint = (bpw, groupLabel, note) => {
    const gb = (EXPERT_PARAMS * bpw) / 8 / 1e9;
    totalLabel.textContent = `${N} weights + 1 scale = ${N * 4 + 32} bits = ${(N * 4 + 32) / 8} bytes`;
    perLabel.textContent = `${bpw.toFixed(2)} bits per weight at group size ${groupLabel}`;
    readout.innerHTML =
      `<span class="readout-big">${bpw.toFixed(2)}</span>` +
      `<span class="readout-dim">bits / weight</span>`;
    verdict.innerHTML =
      `Routed experts at this setting: <b class="mono hl">${gb.toFixed(2)} GB</b> ` +
      `(from <b class="mono">4.84 GB</b> in bf16), whole artifact ` +
      `<b class="mono hl">${(gb + OTHER_GB).toFixed(2)} GB</b>. ${note}`;
  };

  const applySlider = () => {
    const g = GROUPS[Number(slider.value)];
    paint(
      bitsPerWeight(g),
      String(g),
      g === 16
        ? "Small groups track local outliers best and cost the most metadata."
        : g >= 64
          ? "Cheap, but one scale now covers a wide span of very differently-sized weights."
          : "A middle setting.",
    );
  };
  slider.addEventListener("input", applySlider);
  shipped.addEventListener("click", () => {
    // The real recipe is a MIX: gate_proj and down_proj at g16, up_proj at g32
    // — two thirds of the expert bank at 6 bits, one third at 5.
    paint(
      (2 * bitsPerWeight(16) + bitsPerWeight(32)) / 3,
      "16 / 16 / 32",
      "The shipped mix: <b>gate_proj</b> and <b>down_proj</b> at group 16, <b>up_proj</b> at group 32; " +
        "which lands within 1 MB of the artifact's measured 1.716 GB expert bank.",
    );
  });

  bitsRoot.append(controls, verdict);
  applySlider();
}

if (bitsRoot) lazyBuild(bitsRoot, buildBits);

/* ==========================================================================
   4. Table → bars. Any <table data-bars> becomes an animated bar chart, and
      degrades to the table itself when this file never runs.
   ========================================================================== */

function tableToBars(table) {
  const max = Number(table.dataset.barsMax);
  const valueCol = Number(table.dataset.barsValue ?? 1);
  const refCol = table.dataset.barsRef ? Number(table.dataset.barsRef) : null;
  const unit = table.dataset.barsUnit ?? "";
  const rows = [...table.querySelectorAll("tbody tr")];
  const wrap = h("div", { class: "bars" });

  const bar = (name, value, display, cls) => {
    const track = h("div", { class: "bar-track" });
    const fill = h("div", { class: `bar-fill ${cls}`.trim() });
    // An exact zero draws NO bar and says so in words. Giving it a hairline
    // would make "we measured 0" and "we measured 0.0011" look identical, which
    // is the one thing this chart exists to distinguish.
    const pct = value <= 0 ? 0 : Math.max(1.2, Math.min(100, (value / max) * 100));
    fill.dataset.width = `${pct}%`;
    if (value <= 0) track.classList.add("is-zero");
    track.append(fill);
    return h(
      "div",
      { class: "bar-row" },
      h("span", { class: "bar-name", text: name }),
      track,
      h("span", { class: "bar-val", html: display }),
    );
  };

  for (const tr of rows) {
    const cells = [...tr.children];
    const name = cells[0].textContent.trim();
    const v = Number(cells[valueCol].dataset.v ?? cells[valueCol].textContent);
    const cls = cells[valueCol].dataset.cls ?? "";
    if (refCol !== null) {
      const rv = Number(cells[refCol].dataset.v ?? cells[refCol].textContent);
      const pair = h("div", { class: "bar-pair" });
      pair.append(bar(name, rv, `${cells[refCol].textContent.trim()}${unit} <small>ref</small>`, "ref"));
      pair.append(
        bar(
          "",
          v,
          `${cells[valueCol].textContent.trim()}${unit}` +
            (v <= 0 ? ' <small>exact</small>' : ""),
          cls,
        ),
      );
      wrap.append(pair);
    } else {
      wrap.append(bar(name, v, `${cells[valueCol].textContent.trim()}${unit}`, cls));
    }
  }

  // The <table> stays in the accessibility tree as the authoritative version of
  // these numbers; the bars are the sighted rendering of the same rows, so they
  // are hidden from assistive tech to avoid reading everything twice.
  wrap.setAttribute("aria-hidden", "true");
  const host = table.closest(".table-scroll") ?? table;
  host.insertAdjacentElement("beforebegin", wrap);
  host.classList.add("sr-only");
  onceVisible(wrap, () => {
    for (const fill of wrap.querySelectorAll("[data-width]")) {
      fill.style.transition = reduced ? "none" : "width 1.1s cubic-bezier(.2,.8,.2,1)";
      requestAnimationFrame(() => {
        fill.style.width = fill.dataset.width;
      });
    }
  });
}

for (const table of document.querySelectorAll("table[data-bars-max]")) {
  lazyBuild(table, () => tableToBars(table));
}

/* ==========================================================================
   5. The five wasm walls — a stepper over markup that already lists them
   ========================================================================== */

const wallsRoot = document.getElementById("walls-viz");

function buildWalls() {
  const list = document.getElementById("walls-list");
  if (!list) return;
  const walls = [...list.querySelectorAll("li")].map((li) => ({
    title: li.querySelector("h3")?.textContent ?? "",
    stat: li.dataset.stat ?? "",
    rail: li.dataset.rail ?? "",
    wall: li.querySelector("[data-kind='wall']")?.innerHTML ?? "",
    why: li.querySelector("[data-kind='why']")?.innerHTML ?? "",
    fix: li.querySelector("[data-kind='fix']")?.innerHTML ?? "",
  }));
  if (!walls.length) return;

  // A thin progress rail: five walls, in the order they were actually hit.
  const rail = h("div", { class: "viz-scroll walls-rail" });
  const railSvg = svg("svg", { viewBox: "0 0 900 66", role: "presentation", focusable: "false" });
  railSvg.style.width = "100%";
  railSvg.style.height = "auto";
  rail.appendChild(railSvg);
  svg("line", { x1: 60, y1: 32, x2: 840, y2: 32, stroke: "rgba(255,255,255,0.12)", "stroke-width": 2 }, railSvg);
  // Progress is drawn as a full-length line revealed by stroke-dashoffset:
  // transitioning an SVG geometry attribute (x2) is not reliable everywhere,
  // but dashoffset has been animatable since forever.
  const RAIL_LEN = 780;
  const railDone = svg(
    "line",
    {
      x1: 60, y1: 32, x2: 840, y2: 32,
      stroke: C.accent,
      "stroke-width": 2,
      "stroke-dasharray": RAIL_LEN,
      "stroke-dashoffset": RAIL_LEN,
      style: reduced ? "" : "transition: stroke-dashoffset .5s cubic-bezier(.2,.8,.2,1)",
    },
    railSvg,
  );
  const marks = walls.map((w, i) => {
    const x = 60 + (i * 780) / (walls.length - 1);
    const g = svg("g", {}, railSvg);
    const c = svg("circle", { cx: x, cy: 32, r: 11, fill: "#0b1512", stroke: "rgba(255,255,255,0.2)", "stroke-width": 1.5 }, g);
    const t = text(g, x, 36, String(i + 1), { size: 11, weight: 800, fill: C.faint, mono: true });
    text(g, x, 60, w.rail, { size: 9.5, fill: C.faint, mono: true });
    return { c, t, x };
  });

  const head = h("div", { class: "step-head" });
  const counter = h("span", { class: "step-count" });
  const title = h("span", { class: "step-title" });
  const stat = h("span", { class: "step-stat" });
  head.append(counter, title, stat);
  const body = h("div", { class: "step-body", "aria-live": "polite" });
  const dots = h("div", { class: "step-dots", "aria-label": "The five walls" });
  const dotEls = walls.map((w, i) =>
    h("button", { type: "button", class: "step-dot", "aria-label": `Wall ${i + 1}: ${w.title}`, onclick: () => show(i) }),
  );
  dots.append(...dotEls);
  const prev = h("button", { type: "button", class: "btn", text: "← Prev" });
  const next = h("button", { type: "button", class: "btn primary", text: "Next →" });
  const nav = h("div", { class: "step-nav" }, prev, dots, next);

  let index = 0;
  const row = (kind, label, html) =>
    `<div class="step-row"><span class="step-kind ${kind}">${label}</span><span class="note">${html}</span></div>`;

  function show(i) {
    index = Math.max(0, Math.min(walls.length - 1, i));
    const w = walls[index];
    counter.textContent = `${index + 1} / ${walls.length}`;
    title.textContent = w.title;
    stat.textContent = w.stat;
    body.innerHTML = row("wall", "Wall", w.wall) + row("why", "Why", w.why) + row("fix", "Fix", w.fix);
    prev.disabled = index === 0;
    next.disabled = index === walls.length - 1;
    dotEls.forEach((d, j) => d.setAttribute("aria-current", j === index ? "true" : "false"));
    marks.forEach((m, j) => {
      const done = j <= index;
      m.c.setAttribute("fill", j === index ? C.accent : done ? "rgba(16,185,129,0.35)" : "#0b1512");
      m.c.setAttribute("stroke", done ? C.accent : "rgba(255,255,255,0.2)");
      m.t.setAttribute("fill", j === index ? "#04140d" : done ? "#a7f3d0" : C.faint);
    });
    railDone.setAttribute(
      "stroke-dashoffset",
      String(RAIL_LEN - (RAIL_LEN * index) / (walls.length - 1)),
    );
  }

  prev.addEventListener("click", () => show(index - 1));
  next.addEventListener("click", () => show(index + 1));
  const stepper = h("div", {}, rail, head, body, nav);
  stepper.tabIndex = 0;
  stepper.setAttribute("aria-label", "The five wasm walls; use the left and right arrow keys");
  stepper.addEventListener("keydown", (e) => {
    if (e.key === "ArrowLeft") { e.preventDefault(); show(index - 1); }
    if (e.key === "ArrowRight") { e.preventDefault(); show(index + 1); }
  });
  wallsRoot.appendChild(stepper);
  list.remove();
  show(0);
}

if (wallsRoot) lazyBuild(wallsRoot, buildWalls);

/* ==========================================================================
   6. Count-ups on the receipts stats
   ========================================================================== */

for (const el of document.querySelectorAll("[data-countup]")) {
  const target = Number.parseFloat(el.dataset.countup);
  const decimals = Number.parseInt(el.dataset.decimals ?? "0", 10);
  const suffix = el.dataset.suffix ?? "";
  if (reduced || !Number.isFinite(target)) {
    el.textContent = target.toFixed(decimals) + suffix;
    continue;
  }
  el.textContent = (0).toFixed(decimals) + suffix;
  onceVisible(el, () => {
    const t0 = performance.now();
    const DUR = 1000;
    const step = (now) => {
      const t = Math.min(1, (now - t0) / DUR);
      const eased = 1 - (1 - t) ** 3;
      el.textContent = (target * eased).toFixed(decimals) + suffix;
      if (t < 1) requestAnimationFrame(step);
    };
    requestAnimationFrame(step);
  });
}
