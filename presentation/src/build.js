// LIA temperature inversion talk — Great Aletsch Glacier.
const pptxgen = require("pptxgenjs");
const React = require("react");
const ReactDOMServer = require("react-dom/server");
const sharp = require("sharp");
const fa = require("react-icons/fa");
const path = require("path");
const { applyTheme } = require(process.env.PPTX_SKILL + "/scripts/apply_theme.js");

const OUT = process.argv[2] || "LIA_temperature_inversion.pptx";
const IMG = (f) => path.join(__dirname, "img", f);

const THEME = {
  name: "Aletsch Glacier",
  headFontFace: "Cambria",
  bodyFontFace: "Calibri",
  colors: {
    dk1: "102A3A", lt1: "FFFFFF", dk2: "2E5266", lt2: "EAF2F6",
    accent1: "2B7BB9", accent2: "D0532A", accent3: "8DC3DE",
    accent4: "8A9AA5", accent5: "E0A43A", accent6: "5B8C6A",
    hlink: "2B7BB9", folHlink: "2E5266",
  },
};
const HEX = THEME.colors;

const pres = new pptxgen();
pres.layout = "LAYOUT_WIDE"; // 13.333 x 7.5
pres.theme = { headFontFace: THEME.headFontFace, bodyFontFace: THEME.bodyFontFace };
pres.title = "Inverting glacier history for past temperature";
pres.author = "Kejdi Lleshi";
pres.subject = "Little Ice Age temperature inversion, Great Aletsch Glacier";
const C = pres.SchemeColor;

// ------------------------------------------------------------------ layouts
const W = 13.333;
const FOOT = "K. Lleshi  ·  Differentiable LIA temperature inversion  ·  Great Aletsch";

pres.defineSlideMaster({
  title: "TITLE",
  background: { color: HEX.dk1 },
  objects: [
    { placeholder: { options: { name: "title", type: "title", x: 0.7, y: 1.7, w: 7.6, h: 2.3, align: "left",
        fontSize: 42, bold: true, color: C.background1, valign: "bottom", margin: 0 }, text: "" } },
    { placeholder: { options: { name: "subtitle", type: "body", x: 0.7, y: 4.25, w: 7.4, h: 1.1,
        fontSize: 20, color: C.accent3, valign: "top", margin: 0 }, text: "" } },
    { placeholder: { options: { name: "author", type: "body", x: 0.7, y: 6.0, w: 7.4, h: 0.7,
        fontSize: 16, color: C.background1, valign: "top", margin: 0 }, text: "" } },
  ],
});

pres.defineSlideMaster({
  title: "SECTION",
  background: { color: HEX.dk1 },
  objects: [
    { placeholder: { options: { name: "kicker", type: "body", x: 0.7, y: 2.35, w: 7.6, h: 0.6,
        fontSize: 18, bold: true, color: C.accent3, margin: 0, charSpacing: 2 }, text: "" } },
    { placeholder: { options: { name: "title", type: "title", x: 0.7, y: 3.0, w: 7.6, h: 1.9, align: "left",
        fontSize: 40, bold: true, color: C.background1, valign: "top", margin: 0 }, text: "" } },
  ],
});

pres.defineSlideMaster({
  title: "CONTENT",
  background: { color: HEX.lt1 },
  margin: [0.5, 0.6, 0.6, 0.6],
  objects: [
    { placeholder: { options: { name: "title", type: "title", x: 0.6, y: 0.32, w: 12.1, h: 0.9, align: "left",
        fontSize: 36, bold: true, color: C.text1, valign: "middle", margin: 0 }, text: "" } },
    { text: { text: FOOT, options: { x: 0.6, y: 7.0, w: 9, h: 0.3, fontSize: 10,
        color: C.accent4, margin: 0, isTextBox: true } } },
  ],
  slideNumber: { x: 12.23, y: 7.0, w: 0.5, h: 0.3, fontSize: 10, color: HEX.accent4, align: "right" },
});

pres.defineSlideMaster({
  title: "END",
  background: { color: HEX.dk1 },
  objects: [
    { placeholder: { options: { name: "title", type: "title", x: 0.7, y: 2.4, w: 7.6, h: 1.5, align: "left",
        fontSize: 44, bold: true, color: C.background1, valign: "bottom", margin: 0 }, text: "" } },
    { placeholder: { options: { name: "body", type: "body", x: 0.7, y: 4.1, w: 7.4, h: 1.6,
        fontSize: 18, color: C.accent3, valign: "top", margin: 0 }, text: "" } },
  ],
});

// ------------------------------------------------------------------ helpers
async function icon(name, hex, px = 256) {
  const svg = ReactDOMServer.renderToStaticMarkup(
    React.createElement(fa[name], { color: "#" + hex, size: px }));
  const buf = await sharp(Buffer.from(svg)).png().toBuffer();
  return "image/png;base64," + buf.toString("base64");
}

// Icon in a filled circle: the deck's recurring motif.
async function iconCircle(slide, name, x, y, d, { fill = C.background2, fg = HEX.accent1, tag = "" } = {}) {
  slide.addShape(pres.shapes.OVAL, { x, y, w: d, h: d, fill: { color: fill },
    line: { type: "none" }, objectName: `icon-bg-${tag || name}` });
  const pad = d * 0.25;
  slide.addImage({ data: await icon(name, fg), x: x + pad, y: y + pad, w: d - 2 * pad, h: d - 2 * pad,
    objectName: `icon-${tag || name}` });
}

function card(slide, x, y, w, h, { fill = C.background2, tag = "card" } = {}) {
  slide.addShape(pres.shapes.ROUNDED_RECTANGLE, { x, y, w, h, rectRadius: 0.12,
    fill: { color: fill }, line: { type: "none" }, objectName: tag });
}

function text(slide, t, opts) {
  slide.addText(t, Object.assign({ isTextBox: true, margin: 0, fontSize: 16, color: C.text1,
    valign: "top", fit: "none" }, opts));
}

function stat(slide, big, label, x, y, w, { color = C.accent1, bigSize = 40, labelSize = 14, tag = "stat" } = {}) {
  text(slide, big, { x, y, w, h: 0.75, fontSize: bigSize, bold: true, color,
    fontFace: THEME.headFontFace, valign: "bottom", objectName: `${tag}-value` });
  text(slide, label, { x, y: y + 0.8, w, h: 0.75, fontSize: labelSize, color: C.text2,
    objectName: `${tag}-label` });
}

const chartText = { catAxisLabelFontFace: "+mn-lt", valAxisLabelFontFace: "+mn-lt",
  titleFontFace: "+mn-lt", legendFontFace: "+mn-lt", dataLabelFontFace: "+mn-lt",
  valAxisTitleFontFace: "+mn-lt", catAxisTitleFontFace: "+mn-lt" };
const axisStyle = () => Object.assign({
  catAxisLabelColor: HEX.dk2, valAxisLabelColor: HEX.dk2, catAxisLabelFontSize: 12,
  valAxisLabelFontSize: 12, valAxisTitleColor: HEX.dk2, catAxisTitleColor: HEX.dk2,
  valAxisTitleFontSize: 13, catAxisTitleFontSize: 13,
  valGridLine: { color: "D5DEE3", size: 0.75 }, catGridLine: { style: "none" },
  catAxisLineColor: "B8C4CB", valAxisLineShow: false,
  legendFontSize: 12, legendColor: HEX.dk2,
}, chartText);

function content(title, section) {
  const s = pres.addSlide({ masterName: "CONTENT", sectionTitle: section });
  s.addText(title, { placeholder: "title" });
  return s;
}

// ======================================================================= deck
(async () => {
  const S1 = "Motivation", S2 = "Method", S3 = "Great Aletsch", S4 = "Outlook";

  // 1 ---------------------------------------------------------------- title
  pres.addSection({ title: S1 });
  let s = pres.addSlide({ masterName: "TITLE", sectionTitle: S1 });
  s.addText("Inverting glacier history for past temperature", { placeholder: "title" });
  s.addText("A differentiable glacier model applied to the Little Ice Age at the Great Aletsch Glacier",
    { placeholder: "subtitle" });
  s.addText("Kejdi Lleshi  ·  University of Lausanne", { placeholder: "author" });
  s.addImage({ path: IMG("title_art.png"), x: W - 4.4, y: 0, w: 4.4, h: 7.5, objectName: "aletsch-hillshade" });
  s.addNotes(
`~0:45. Welcome. The question: what was the temperature in the Alps during the Little Ice Age, and how much can a single glacier tell us about it?
The image is the Great Aletsch Glacier today, from the model's own 100 m grid — it is the glacier we will invert at the end.`);

  // 2 --------------------------------------------------------- the problem
  s = content("Past climate is badly underconstrained", S1);
  text(s, "Temperature proxies are sparse, and most of them speak for a hemisphere or a broad region — not for the valley where a glacier sits.",
    { x: 0.6, y: 1.4, w: 12.1, h: 0.8, fontSize: 18, color: C.text2 });
  const proxies = [
    ["FaIcicles", "Ice cores", "Greenland, Antarctica", "Hemispheric, high-latitude", "Millennia"],
    ["FaTree", "Tree rings", "and lake sediments", "Regional, mostly summer", "Centuries to millennia"],
    ["FaThermometerHalf", "Instruments", "Station network", "Local", "Only since the mid-1800s"],
    ["FaMountain", "Glaciers", "Mapped extents, moraines", "One valley", "Centuries (moraines: millennia)"],
  ];
  const pw = 2.84, pg = 0.25, py = 2.45, ph = 3.55;
  for (let i = 0; i < proxies.length; i++) {
    const [ic, name, where, foot, span] = proxies[i];
    const x = 0.6 + i * (pw + pg);
    const hi = i === 3;
    card(s, x, py, pw, ph, { fill: hi ? C.text1 : C.background2, tag: `proxy-card-${i}` });
    await iconCircle(s, ic, x + 0.3, py + 0.3, 0.8,
      { fill: hi ? C.accent1 : C.background1, fg: hi ? HEX.lt1 : HEX.accent1, tag: `proxy-${i}` });
    text(s, name, { x: x + 0.3, y: py + 1.25, w: pw - 0.5, h: 0.45, fontSize: 20, bold: true,
      color: hi ? C.background1 : C.text1 });
    text(s, where, { x: x + 0.3, y: py + 1.7, w: pw - 0.5, h: 0.35, fontSize: 14,
      color: hi ? C.accent3 : C.text2 });
    text(s, [
      { text: "Footprint", options: { fontSize: 12, color: hi ? C.accent3 : C.accent4, breakLine: true } },
      { text: foot, options: { fontSize: 15, bold: true, color: hi ? C.background1 : C.text1, breakLine: true } },
      { text: " ", options: { fontSize: 6, breakLine: true } },
      { text: "Time span", options: { fontSize: 12, color: hi ? C.accent3 : C.accent4, breakLine: true } },
      { text: span, options: { fontSize: 15, bold: true, color: hi ? C.background1 : C.text1 } },
    ], { x: x + 0.3, y: py + 2.15, w: pw - 0.5, h: 1.3 });
  }
  text(s, [
    { text: "Glaciers are local and long-lived, and well documented — ", options: {} },
    { text: "but they record climate through ice dynamics, so the record has to be inverted.", options: { bold: true } },
  ], { x: 0.6, y: 6.25, w: 12.1, h: 0.6, fontSize: 17, color: C.text1 });
  s.addNotes(
`~2:00. The problem. Paleo-temperature is severely underconstrained.
- Ice cores (e.g. Greenland) give a high-latitude, essentially hemispheric signal.
- Tree rings and lake sediments are regional and often summer-only.
- Instrumental series in the Alps only start in the mid-19th century — right at the end of the Little Ice Age.
So for a specific Alpine valley we usually borrow a temperature curve built for a much larger area.
Glaciers are the opposite: one valley, centuries of mapped extents, moraines going back millennia. The catch: a glacier is not a thermometer you can read directly.
[Add your preferred references here — e.g. PAGES 2k for regional reconstructions, Greenland ice-core temperature reconstructions.]`);

  // 3 ------------------------------------------------- slow thermometer
  s = content("A glacier is a slow, nonlinear thermometer", S1);
  const chain = [["Climate", "T(t), P(t)"], ["Mass balance", "accumulation, melt"],
    ["Ice flow", "deformation, sliding"], ["Geometry", "extent, thickness, moraines"]];
  const cw = 2.55, cg = 0.62, cy = 1.95, ch = 1.25;
  for (let i = 0; i < chain.length; i++) {
    const x = 0.6 + i * (cw + cg);
    card(s, x, cy, cw, ch, { fill: i === 0 ? C.accent2 : (i === 3 ? C.text1 : C.background2), tag: `chain-${i}` });
    const light = i === 0 || i === 3;
    text(s, chain[i][0], { x: x + 0.2, y: cy + 0.2, w: cw - 0.4, h: 0.45, fontSize: 20, bold: true,
      align: "center", color: light ? C.background1 : C.text1 });
    text(s, chain[i][1], { x: x + 0.2, y: cy + 0.68, w: cw - 0.4, h: 0.4, fontSize: 14,
      align: "center", color: light ? C.background1 : C.text2 });
    if (i < 3) s.addShape(pres.shapes.RIGHT_ARROW, { x: x + cw + 0.12, y: cy + 0.45, w: 0.38, h: 0.35,
      fill: { color: C.accent4 }, line: { type: "none" }, objectName: `chain-arrow-${i}` });
  }
  // forward / inverse arrows
  s.addShape(pres.shapes.LINE, { x: 0.9, y: 1.62, w: 11.5, h: 0, line: { color: C.accent4, width: 1.5, endArrowType: "triangle" }, objectName: "fwd-line" });
  text(s, "forward model: well understood", { x: 0.9, y: 1.28, w: 6, h: 0.3, fontSize: 13, color: C.accent4, italic: true });
  s.addShape(pres.shapes.LINE, { x: 0.9, y: 3.55, w: 11.5, h: 0, line: { color: C.accent2, width: 2.25, beginArrowType: "triangle" }, objectName: "inv-line" });
  text(s, "inverse problem: what we want", { x: 6.4, y: 3.65, w: 6, h: 0.3, fontSize: 13, color: C.accent2, italic: true, bold: true, align: "right" });
  const hard = [
    ["FaHistory", "Lag", "The tongue responds to climate over 100–200 years, so today's extent reflects past centuries."],
    ["FaChartLine", "Nonlinear", "Melt grows with positive degree-days; equal errors in either direction don't cost the same."],
    ["FaBalanceScale", "Non-unique", "Warmer melts like drier: temperature and precipitation trade off against each other."],
  ];
  for (let i = 0; i < 3; i++) {
    const x = 0.6 + i * 4.1;
    card(s, x, 4.45, 3.85, 2.3, { fill: C.background2, tag: `hard-card-${i}` });
    await iconCircle(s, hard[i][0], x + 0.25, 4.7, 0.75, { fill: C.background1, tag: `hard-${i}` });
    text(s, hard[i][1], { x: x + 1.2, y: 4.85, w: 2.5, h: 0.45, fontSize: 20, bold: true });
    text(s, hard[i][2], { x: x + 0.25, y: 5.6, w: 3.4, h: 1.1, fontSize: 15, color: C.text2 });
  }
  s.addNotes(
`~1:30. Climate sets the mass balance, the mass balance feeds ice flow, and ice flow sets the geometry we observe. Running that forward is routine. We want the reverse arrow.
Three things make it hard: lag (Aletsch responds over a century or two), nonlinearity of melt, and non-uniqueness — a warmer climate can look like a drier one. Keep the last one in mind; it comes back in the sensitivity results.`);

  // 4 ------------------------------------------------------ ensembles
  s = content("Ensembles: cost grows with every parameter", S1);
  text(s, "The usual inverse: sample parameters, run the forward model many times, keep the runs that fit.",
    { x: 0.6, y: 1.45, w: 6.4, h: 0.9, fontSize: 18, color: C.text2 });
  const cons = [
    ["One-at-a-time sensitivity", "≈ 2 forward runs per parameter, per perturbation size"],
    ["Grids & Monte Carlo", "cost multiplies with every added dimension"],
    ["So climate gets compressed", "into a few numbers: an offset, a step, a trend"],
    ["And data get thinned", "one observation type at a time, hand-tuned"],
  ];
  text(s, cons.flatMap(([a, b], i) => [
    { text: a, options: { bold: true, fontSize: 19, color: C.text1, breakLine: true } },
    { text: b, options: { fontSize: 16, color: C.text2, breakLine: i < cons.length - 1, paraSpaceAfter: 18 } },
  ]), { x: 0.6, y: 2.55, w: 6.3, h: 4.2 });
  // ensemble dot grid
  const gx = 7.85, gy = 1.5, n = 10, step = 0.385;
  const accepted = new Set([23, 24, 34, 35, 45, 46, 56]);
  for (let r = 0; r < n; r++) for (let c = 0; c < n; c++) {
    const k = r * n + c;
    s.addShape(pres.shapes.OVAL, { x: gx + c * step, y: gy + r * step, w: 0.26, h: 0.26,
      fill: { color: accepted.has(k) ? C.accent2 : C.accent3 }, line: { type: "none" }, objectName: `run-${k}` });
  }
  text(s, [
    { text: "100 forward runs", options: { bold: true, color: C.text1, breakLine: true } },
    { text: "7 accepted — and that is only 2 parameters", options: { color: C.text2 } },
  ], { x: gx, y: gy + n * step + 0.05, w: 4.6, h: 0.6, fontSize: 14 });
  card(s, gx - 0.05, 6.05, 4.4, 0.8, { fill: C.background2, tag: "grid-callout" });
  text(s, [
    { text: "11 temperature nodes × 5 values = ", options: { color: C.text2 } },
    { text: "5¹¹ ≈ 49 million runs", options: { bold: true, color: C.accent2 } },
  ], { x: gx + 0.15, y: 6.05, w: 4.1, h: 0.8, fontSize: 15, valign: "middle" });
  s.addNotes(
`~1:30. How is this usually done? Ensembles: sample the unknowns, run the model many times, keep the runs that match the moraines or the mapped extents.
That is robust, and it gives a distribution — but every extra parameter multiplies the cost. In practice the climate history is collapsed to a few numbers (a constant offset, a step change), and observations are used one type at a time.
To make it concrete: a time-varying temperature with 11 nodes, sampled at only 5 values each, is 49 million forward runs. That is not a thing anyone does.`);

  // 5 --------------------------------------------------- our approach
  pres.addSection({ title: S2 });
  s = content("Our approach: a differentiable glacier model", S2);
  const boxes = [
    ["Temperature", "T(t), 11 nodes", C.accent2, true],
    ["Mass balance", "PDD scheme", C.background2, false],
    ["Ice flow", "IGM emulator (CNN)", C.background2, false],
    ["Transport", "flux-limited upwind", C.background2, false],
    ["Misfit J", "vs. dated extents", C.text1, true],
  ];
  const bw = 2.05, bgp = 0.46, by = 1.75, bh = 1.15;
  for (let i = 0; i < boxes.length; i++) {
    const x = 0.6 + i * (bw + bgp);
    card(s, x, by, bw, bh, { fill: boxes[i][2], tag: `pipe-${i}` });
    text(s, boxes[i][0], { x: x + 0.1, y: by + 0.18, w: bw - 0.2, h: 0.4, fontSize: 18, bold: true, align: "center",
      color: boxes[i][3] ? C.background1 : C.text1 });
    text(s, boxes[i][1], { x: x + 0.1, y: by + 0.62, w: bw - 0.2, h: 0.35, fontSize: 13, align: "center",
      color: boxes[i][3] ? C.background1 : C.text2 });
    if (i < boxes.length - 1) s.addShape(pres.shapes.RIGHT_ARROW, { x: x + bw + 0.08, y: by + 0.42, w: 0.3, h: 0.3,
      fill: { color: C.accent4 }, line: { type: "none" }, objectName: `pipe-arrow-${i}` });
  }
  // time loop bracket around the three middle boxes
  const lx = 0.6 + (bw + bgp) - 0.15, lw = 3 * bw + 2 * bgp + 0.3;
  s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: lx, y: by - 0.3, w: lw, h: bh + 0.55, rectRadius: 0.15,
    fill: { type: "none" }, line: { color: C.accent1, width: 1.5, dashType: "dash" }, objectName: "time-loop" });
  text(s, "repeated every time step  ·  ~5,000 steps from 1500 to 2017", { x: lx + 0.2, y: by - 0.6, w: lw - 0.4, h: 0.3,
    fontSize: 13, color: C.accent1, align: "center", italic: true });
  // backward arrow
  s.addShape(pres.shapes.LINE, { x: 1.0, y: 3.55, w: 11.3, h: 0, line: { color: C.accent2, width: 2.5, beginArrowType: "triangle" }, objectName: "adjoint-line" });
  text(s, [
    { text: "Backward pass: ", options: { bold: true, color: C.accent2 } },
    { text: "automatic differentiation carries ∂J/∂T(t) back through every time step", options: { color: C.text1 } },
  ], { x: 1.0, y: 3.7, w: 11.3, h: 0.4, fontSize: 16, align: "center" });
  const facts = [
    ["FaCogs", "Real ice physics", "PyTorch port of IGM's pretrained ice-flow emulator; matches the TensorFlow original to 2.4 × 10⁻⁶."],
    ["FaLayerGroup", "Fits in memory", "Gradient checkpointing recomputes each step on the way back, so 517 years run on one GPU."],
    ["FaCrosshairs", "Gradients verified", "With a fixed time step, the adjoint matches finite differences exactly."],
  ];
  for (let i = 0; i < 3; i++) {
    const x = 0.6 + i * 4.1;
    card(s, x, 4.45, 3.85, 2.3, { fill: C.background2, tag: `fact-card-${i}` });
    await iconCircle(s, facts[i][0], x + 0.25, 4.7, 0.75, { fill: C.background1, tag: `fact-${i}` });
    text(s, facts[i][1], { x: x + 1.2, y: 4.85, w: 2.5, h: 0.45, fontSize: 19, bold: true });
    text(s, facts[i][2], { x: x + 0.25, y: 5.6, w: 3.4, h: 1.1, fontSize: 14, color: C.text2 });
  }
  s.addNotes(
`~2:15. Our approach: instead of sampling, we differentiate.
The forward model: a temperature history on 11 nodes drives a positive-degree-day mass balance; ice velocities come from IGM's pretrained CNN emulator; mass is transported with a flux-limited upwind scheme. That loop runs about 5,000 times between 1500 and 2017. At the end we compare simulated and observed extents.
Because every operation is in PyTorch, automatic differentiation gives the gradient of the misfit with respect to the whole temperature history in one backward pass, through all 5,000 steps.
Three ingredients make this work: the emulator port (agrees with IGM's TensorFlow model to 2.4e-6), checkpointing so the backward pass fits in memory, and gradients verified against finite differences.`);

  // 6 --------------------------------------------------- cost scaling
  s = content("One gradient gives every sensitivity", S2);
  const Ns = ["1", "3", "11", "100", "1,000", "10,000"];
  const nv = [1, 3, 11, 100, 1000, 10000];
  s.addChart(pres.charts.LINE, [
    { name: "Finite differences / one-at-a-time (N + 1 runs)", labels: Ns, values: nv.map((v) => v + 1) },
    { name: "Adjoint gradient (≈ 3.5 runs, any N)", labels: Ns, values: nv.map(() => 3.5) },
  ], Object.assign(axisStyle(), {
    x: 0.5, y: 1.4, w: 7.4, h: 5.35, chartColors: [HEX.accent4, HEX.accent2], lineSize: 3,
    lineDataSymbol: "circle", lineDataSymbolSize: 8,
    valAxisLogScaleBase: 10, valAxisMinVal: 1, valAxisMaxVal: 100000,
    showValAxisTitle: true, valAxisTitle: "Cost, in forward-run equivalents",
    showCatAxisTitle: true, catAxisTitle: "Number of parameters N",
    showLegend: true, legendPos: "t", valAxisLabelFormatCode: "#,##0",
    objectName: "cost-scaling-chart",
  }));
  stat(s, "≈ 3.5", "forward runs for one gradient: 173 s per forward + backward vs. ~50 s per forward (1500–2017, RTX 4090)",
    8.4, 1.45, 4.3, { color: C.accent2, tag: "stat-grad" });
  stat(s, "~3 h", "60 iterations invert the full 11-node temperature history", 8.4, 3.2, 4.3, { tag: "stat-inv" });
  stat(s, "≈ 9,000", "ice-covered cells: a spatial field (e.g. sliding τ_ref) costs one gradient too, vs. ~5 GPU-days of runs",
    8.4, 4.85, 4.3, { tag: "stat-field" });
  s.addNotes(
`~2:00. The key property: the cost of the gradient does not depend on the number of parameters.
A forward run from 1500 to 2017 takes about 50 s on an RTX 4090; forward + backward takes about 173 s, so one gradient costs ~3.5 forward runs — whether we ask about 1 parameter, the 11 temperature nodes, or a full 2D field.
Finite differences need N+1 runs. The lines cross at N≈3; beyond that the adjoint wins, and for a spatial field (~9,000 ice cells) it is ~3 minutes vs. ~5 GPU-days.
Honest caveat: a gradient is a local sensitivity. For full uncertainty you still want an ensemble — but a gradient-informed one can be much smaller.`);

  // 7 --------------------------------------------- observations in the loss
  s = content("Any observation becomes a term in the loss", S2);
  card(s, 0.6, 1.45, 12.1, 0.95, { fill: C.background2, tag: "loss-eq" });
  text(s, [
    { text: "J(T) = ", options: { bold: true } },
    { text: "Σ", options: { bold: true, color: C.accent1 } },
    { text: "k", options: { subscript: true, color: C.accent1 } },
    { text: " w", options: {} }, { text: "k", options: { subscript: true } },
    { text: " · misfit", options: {} }, { text: "k", options: { subscript: true } },
    { text: "(model, observation", options: {} }, { text: "k", options: { subscript: true } },
    { text: ")", options: {} },
    { text: "  +  λ · smoothness(T)", options: { color: C.text2 } },
  ], { x: 0.9, y: 1.45, w: 11.5, h: 0.95, fontSize: 24, valign: "middle", fontFace: THEME.headFontFace });
  const obs = [
    ["FaMap", "Dated extents", "Historical maps and DEMs, compared through a soft (sigmoid) ice mask so the margin stays differentiable.", "Used here: 5 epochs", C.accent2],
    ["FaRulerVertical", "Surface elevation & thickness", "Same DEMs, much richer signal: Aletsch lost 34 % of its volume but only 6 % of its area.", "Available: next step", C.accent1],
    ["FaMountain", "Moraine crests", "The ice surface must reach the crest height, scored just inboard of the ridge.", "Built for the Younger Dryas", C.accent1],
    ["FaFlagCheckered", "Termini, velocities, …", "Anything the model can predict differentiably adds a term — no re-design of the inversion.", "Possible", C.accent4],
  ];
  for (let i = 0; i < 4; i++) {
    const y = 2.7 + i * 1.05;
    await iconCircle(s, obs[i][0], 0.6, y, 0.75, { tag: `obs-${i}` });
    text(s, obs[i][1], { x: 1.6, y: y + 0.02, w: 3.6, h: 0.4, fontSize: 18, bold: true });
    text(s, obs[i][2], { x: 1.6, y: y + 0.42, w: 8.2, h: 0.55, fontSize: 14, color: C.text2 });
    s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: 10.1, y: y + 0.17, w: 2.6, h: 0.42, rectRadius: 0.21,
      fill: { color: obs[i][4] }, line: { type: "none" }, objectName: `obs-chip-${i}` });
    text(s, obs[i][3], { x: 10.1, y: y + 0.17, w: 2.6, h: 0.42, fontSize: 12, bold: true,
      color: C.background1, align: "center", valign: "middle" });
  }
  s.addNotes(
`~1:30. Second advantage: observations are just terms in a loss function. Each data type needs an observation operator — "what would the model show here?" — that is differentiable.
Today I use only dated extents, through a soft sigmoid mask. Surface elevation from the same DEMs is the obvious next term. For the Younger Dryas we already built a moraine operator: the ice surface must reach the crest height. More data types means more constraint, without changing the inversion machinery.`);

  // 8 ---------------------------------------------------------- section
  pres.addSection({ title: S3 });
  s = pres.addSlide({ masterName: "SECTION", sectionTitle: S3 });
  s.addText("APPLICATION", { placeholder: "kicker" });
  s.addText("Little Ice Age temperature at the Great Aletsch Glacier, 1500–2017", { placeholder: "title" });
  s.addImage({ path: IMG("title_art.png"), x: W - 4.4, y: 0, w: 4.4, h: 7.5, objectName: "aletsch-hillshade-section" });
  s.addNotes(`~0:15. Now the real case: the Great Aletsch Glacier, the largest glacier in the Alps.`);

  // 9 ------------------------------------------------------------- site
  s = content("Aletsch: 140 years of dated surfaces", S3);
  s.addImage({ path: IMG("site_map.png"), x: 0.6, y: 1.35, w: 3.25, h: 3.25 * 1353 / 793, objectName: "site-map" });
  const sx = 4.6;
  stat(s, "7", "dated surface DEMs, 1880–2017 (5 used: 1880, 1926, 1957, 1980, 2017)", sx, 1.45, 3.7, { tag: "stat-dems" });
  stat(s, "100 m", "model grid, 225 × 132 cells; started from the 2017 geometry", sx + 4.2, 1.45, 3.7, { tag: "stat-grid" });
  stat(s, "−34 %", "ice volume, 1880 → 2017 (20.7 → 13.7 km³)", sx, 3.35, 3.7, { color: C.accent2, tag: "stat-vol" });
  stat(s, "−6 %", "ice area over the same period (96 → 91 km²)", sx + 4.2, 3.35, 3.7, { color: C.accent2, tag: "stat-area" });
  card(s, sx, 5.35, 8.1, 1.3, { fill: C.background2, tag: "site-takeaway" });
  text(s, [
    { text: "Aletsch thins far more than it shrinks. ", options: { bold: true } },
    { text: "Extent is real information, but a coarse, terminus-weighted signal — the inversion has to work from the tongue.", options: { color: C.text2 } },
  ], { x: sx + 0.3, y: 5.35, w: 7.5, h: 1.3, fontSize: 16, valign: "middle" });
  s.addNotes(
`~1:30. The data: seven dated surface DEMs from historical maps and modern surveys; I use five of them as extents. 100 m grid.
Important numbers: between 1880 and 2017 the glacier lost a third of its volume but only 6% of its area. It thins in a deep trough rather than narrowing. So an extent-only inversion relies mostly on the tongue — keep that in mind when reading the result.`);

  // 10 ---------------------------------------------------------- moraine
  s = content("Independent check: LIA moraine = 1880 margin", S3);
  s.addImage({ path: IMG("moraine_1880.png"), x: 0.6, y: 1.4, w: 6.6, h: 6.6 * 1126 / 1364, objectName: "moraine-map" });
  stat(s, "100 m", "median distance from the mapped LIA moraine to the 1880 ice margin — one grid cell", 7.75, 1.45, 4.95, { tag: "stat-med" });
  stat(s, "86 %", "of moraine cells within 150 m of the 1880 margin", 7.75, 3.0, 4.95, { tag: "stat-86" });
  stat(s, "505 m", "median distance to the 2017 margin — the offset grows with every later epoch", 7.75, 4.55, 4.95,
    { color: C.accent4, tag: "stat-505" });
  text(s, "Two independent datasets agree: this validates the 1880 DEM and calibrates the moraine operator we will need for older, undated stands.",
    { x: 7.75, y: 6.05, w: 4.95, h: 0.85, fontSize: 14, color: C.text1, bold: true });
  s.addNotes(
`~1:30. Before inverting, a data check. The field-mapped LIA moraine and the 1880 DEM margin are independent products. If both record the same stand, the moraine should sit on the 1880 margin.
It does: median offset 100 m — one cell — and 86% of moraine cells within 150 m. The distance grows monotonically for later epochs, the signature of a moraine dating one specific stand.
So the 1880 extent is trustworthy, and this is the calibration for the moraine operator for the Younger Dryas later.`);

  // 11 ------------------------------------------------------- experiment
  s = content("Experiment: start from today, run 517 years", S3);
  const t0 = 1500, t1 = 2017, ax0 = 0.9, ax1 = 12.4, ay = 2.25;
  const tx = (t) => ax0 + (t - t0) / (t1 - t0) * (ax1 - ax0);
  s.addShape(pres.shapes.RECTANGLE, { x: tx(1500), y: ay - 0.55, w: tx(1880) - tx(1500), h: 1.1,
    fill: { color: C.background2 }, line: { type: "none" }, objectName: "spinup-band" });
  text(s, "Spin-up from the 2017 geometry: 380 years to regrow the full 1880 volume",
    { x: tx(1500) + 0.15, y: ay - 0.5, w: tx(1880) - tx(1500) - 0.3, h: 0.35, fontSize: 13, color: C.text2, italic: true });
  s.addShape(pres.shapes.LINE, { x: ax0, y: ay, w: ax1 - ax0, h: 0, line: { color: C.text1, width: 2 }, objectName: "time-axis" });
  for (let t = 1500; t <= 2000; t += 50) {
    s.addShape(pres.shapes.OVAL, { x: tx(t) - 0.09, y: ay - 0.09, w: 0.18, h: 0.18, fill: { color: C.accent1 },
      line: { color: C.background1, width: 1 }, objectName: `node-${t}` });
  }
  for (const t of [1500, 1600, 1700, 1800, 1900, 2000]) {
    text(s, String(t), { x: tx(t) - 0.4, y: ay + 0.62, w: 0.8, h: 0.3, fontSize: 13, color: C.text2, align: "center" });
  }
  for (const t of [1880, 1926, 1957, 1980, 2017]) {
    s.addShape(pres.shapes.DIAMOND, { x: tx(t) - 0.14, y: ay + 0.18, w: 0.28, h: 0.28, fill: { color: C.accent2 },
      line: { type: "none" }, objectName: `obs-${t}` });
  }
  text(s, [
    { text: "●", options: { color: C.accent1 } }, { text: "  11 temperature nodes, every 50 years      ", options: {} },
    { text: "◆", options: { color: C.accent2 } }, { text: "  5 observed extents: 1880, 1926, 1957, 1980, 2017", options: {} },
  ], { x: ax0, y: ay + 1.05, w: ax1 - ax0, h: 0.35, fontSize: 14, color: C.text1 });
  const setup = [
    ["Control", "T_lowest(t): mean annual temperature at the lowest grid point (799 m), lapse rate 7 °C/km"],
    ["Objective", "Soft-extent misfit, averaged over the 5 epochs, plus a smoothness penalty on T(t)"],
    ["Held fixed", "Melt factor 0.225, precipitation 1.8 m/yr, τ_ref 0.215 MPa, A = 78 MPa⁻³ yr⁻¹"],
    ["Optimiser", "Adam, learning rate 0.02, 60 iterations, one RTX 4090"],
  ];
  for (let i = 0; i < 4; i++) {
    const x = 0.6 + i * 3.1;
    card(s, x, 4.05, 2.85, 2.45, { fill: C.background2, tag: `setup-${i}` });
    text(s, setup[i][0], { x: x + 0.25, y: 4.3, w: 2.4, h: 0.4, fontSize: 19, bold: true });
    text(s, setup[i][1], { x: x + 0.25, y: 4.8, w: 2.4, h: 1.85, fontSize: 15, color: C.text2 });
  }
  s.addNotes(
`~1:45. Setup. We start from today's geometry and run forward from 1500 to 2017. Why 1500? I tested the spin-up: starting in 1700 undershoots the 1880 volume by 9% and leaves the terminus 1 km too short; 1500 recovers the full volume.
The control is the temperature at the lowest grid point, on 11 nodes every 50 years. The nodes before ~1700 have no data within a response time — they act as a warm-up.
Precipitation, melt factor and the flow parameters are fixed at values from an earlier calibration; the sensitivity slide shows what that costs.`);

  // 12 ------------------------------------------------- constant climate
  s = content("Constant climate cannot explain the retreat", S3);
  const ep = ["1880", "1926", "1957", "1980", "2017"];
  s.addChart(pres.charts.LINE, [
    { name: "Observed", labels: ep, values: [0, 1.1, 1.6, 2.1, 2.6] },
    { name: "Constant 6.5 °C", labels: ep, values: [0.8, 0.8, 0.8, 0.8, 0.8] },
    { name: "Constant 7.0 °C", labels: ep, values: [1.9, 1.9, 1.9, 1.9, 1.8] },
    { name: "Constant 7.5 °C", labels: ep, values: [2.6, 2.6, 2.6, 2.6, 2.6] },
    { name: "Constant 8.0 °C", labels: ep, values: [3.6, 3.6, 3.6, 3.6, 3.6] },
  ], Object.assign(axisStyle(), {
    x: 0.5, y: 1.4, w: 7.6, h: 5.35, chartColors: [HEX.accent2, "C9DCE6", "8DC3DE", "4F95C4", "2B7BB9"],
    lineSize: 2.5, lineDataSymbol: "circle", lineDataSymbolSize: 7,
    valAxisMinVal: 0, valAxisMaxVal: 4, valAxisMajorUnit: 1, valAxisLabelFormatCode: "0.0",
    showValAxisTitle: true, valAxisTitle: "Terminus retreat from the 1880 position (km)",
    showLegend: true, legendPos: "t", objectName: "terminus-chart",
  }));
  text(s, "Under any constant temperature the modelled tongue settles at its equilibrium length and stays there. The real one retreated 2.6 km.",
    { x: 8.6, y: 1.5, w: 4.1, h: 1.7, fontSize: 17, color: C.text1 });
  stat(s, "+1.36 °C", "warming 1880 → 2017 implied by matching each epoch's terminus to an equilibrium temperature",
    8.6, 3.35, 4.1, { color: C.accent2, tag: "stat-implied" });
  text(s, "A lower bound: the glacier lags its forcing, so the true temperatures must be warmer than equilibrium suggests.",
    { x: 8.6, y: 5.35, w: 4.1, h: 1.3, fontSize: 14, color: C.text2, italic: true });
  s.addNotes(
`~1:45. First result, from a cheap scan with constant climate: whatever constant temperature you choose, the modelled terminus sits still at its equilibrium length. The observed one retreats 2.6 km over the record. So the data demand a time-varying climate.
If we read off, epoch by epoch, which equilibrium temperature matches the observed terminus, we get +1.36 °C between 1880 and 2017. That ignores the lag, so it is a lower bound — and it gives us an independent target to judge the inversion against.`);

  // 13 ------------------------------------------------- inverted history
  s = content("The inversion finds +0.9 °C of warming", S3);
  const nodesLbl = ["1500", "1550", "1600", "1650", "1700", "1750", "1800", "1850", "1900", "1950", "2000"];
  s.addChart(pres.charts.LINE, [
    { name: "Iteration 1", labels: nodesLbl, values: [7.02, 7.02, 7.02, 7.02, 7.02, 7.02, 7.02, 7.02, 7.02, 7.02, 7.02] },
    { name: "Iteration 15", labels: nodesLbl, values: [7.286, 7.286, 7.286, 7.286, 7.286, 7.288, 7.293, 7.308, 7.316, 7.311, 7.306] },
    { name: "Iteration 30", labels: nodesLbl, values: [7.353, 7.353, 7.355, 7.365, 7.396, 7.513, 7.537, 7.586, 7.617, 7.611, 7.604] },
    { name: "Iteration 60 (best)", labels: nodesLbl, values: [7.269, 7.269, 7.27, 7.281, 7.316, 7.248, 7.414, 7.703, 8.012, 8.15, 8.148] },
  ], Object.assign(axisStyle(), {
    x: 0.5, y: 1.4, w: 7.6, h: 5.0, chartColors: ["C9D3D9", "8DC3DE", "2B7BB9", HEX.accent2],
    lineSize: 2.5, lineDataSymbol: "circle", lineDataSymbolSize: 7,
    valAxisMinVal: 6.9, valAxisMaxVal: 8.3, valAxisMajorUnit: 0.2, valAxisLabelFormatCode: "0.0",
    showValAxisTitle: true, valAxisTitle: "T_lowest (°C)", showLegend: true, legendPos: "t",
    objectName: "temperature-history-chart",
  }));
  text(s, "1500–1650: no observation within a response time — shaped by the smoothness prior only.",
    { x: 0.6, y: 6.42, w: 7.5, h: 0.45, fontSize: 12, color: C.accent4, italic: true });
  stat(s, "+0.90 °C", "from 1750 to 1950, recovered from extent alone", 8.6, 1.45, 4.1, { color: C.accent2, tag: "stat-trend" });
  stat(s, "66 %", "of the ≥ 1.36 °C the termini imply; the trend was still growing at the last iteration", 8.6, 3.0, 4.1, { tag: "stat-66" });
  stat(s, "0.90 → 0.94", "mean extent overlap (IoU) across the 5 epochs; loss down 47 %", 8.6, 4.55, 4.1, { tag: "stat-iou" });
  text(s, "T_lowest is a model parameter, not a station temperature: the trend is the interpretable quantity.",
    { x: 8.6, y: 6.05, w: 4.1, h: 0.8, fontSize: 13, color: C.text2, italic: true });
  s.addNotes(
`~2:30. The main result. Starting from a flat 7 °C, the optimiser first raises the whole curve, and from about iteration 20 a trend develops. After 60 iterations (about 3 hours on one GPU) the history is flat near 7.3 °C until 1750, then warms to 8.15 °C by 1950–2000: +0.9 °C.
That is two thirds of the +1.36 °C the termini imply, and it was still growing — so read 0.9 °C as a lower bound from this run. Fit improves from IoU 0.90 to 0.94.
Two caveats: the early nodes are unconstrained; and absolute values are not station temperatures — T_lowest is a degree-day parameter at 799 m. The trend is what to interpret.`);

  // 14 ---------------------------------------------------------- extents
  s = content("Simulated extents match all five epochs", S3);
  const ew = 12.1, eh = ew * 1075 / 2700;
  s.addImage({ path: IMG("extent_fit.png"), x: 0.6, y: 1.3, w: ew, h: eh, objectName: "extent-fit-maps" });
  const leg = [["Agree", "BFD9E8"], ["Observed only", HEX.accent1], ["Simulated only", HEX.accent2]];
  let lx2 = 0.6;
  for (const [lab, col] of leg) {
    s.addShape(pres.shapes.RECTANGLE, { x: lx2, y: 6.32, w: 0.3, h: 0.3, fill: { color: col }, line: { type: "none" }, objectName: `legend-${lab}` });
    text(s, lab, { x: lx2 + 0.4, y: 6.3, w: 1.9, h: 0.35, fontSize: 14, color: C.text1, valign: "middle" });
    lx2 += 2.3;
  }
  text(s, "Residuals: a one-cell band at the terminus, and in 1880 thin ice on steep headwalls that the real glacier sheds by avalanching.",
    { x: 7.4, y: 6.18, w: 5.3, h: 0.7, fontSize: 13, color: C.text2 });
  s.addNotes(
`~1:30. Spatially: grey-blue is agreement, blue is observed ice the model misses, orange is simulated ice that isn't observed. IoU 0.91–0.95.
The residuals have structure: a one-cell band at the terminus, which is the resolution limit, and in 1880 patches high in the basin — ice the model grows on steep headwalls where the real glacier avalanches it away. No temperature can remove that; it is model error, and it matters (next-but-one slide).`);

  // 15 ------------------------------------------------------- sensitivity
  s = content("Mass balance, not ice flow, sets the answer", S3);
  s.addChart(pres.charts.BAR, [
    { name: "−25 %", labels: ["Melt factor", "Precipitation", "τ_ref (sliding)"], values: [-1.37, 0.62, -0.17] },
    { name: "−10 %", labels: ["Melt factor", "Precipitation", "τ_ref (sliding)"], values: [-0.43, 0.23, -0.04] },
    { name: "+10 %", labels: ["Melt factor", "Precipitation", "τ_ref (sliding)"], values: [0.23, -0.38, 0.01] },
    { name: "+25 %", labels: ["Melt factor", "Precipitation", "τ_ref (sliding)"], values: [0.51, -0.88, 0.07] },
  ], Object.assign(axisStyle(), {
    x: 0.5, y: 1.4, w: 7.6, h: 5.35, barDir: "col", barGrouping: "clustered", barGapWidthPct: 60,
    chartColors: ["1F5F8F", "8DC3DE", "EDB49E", HEX.accent2],
    valAxisMinVal: -1.5, valAxisMaxVal: 1.0, valAxisMajorUnit: 0.5, valAxisLabelFormatCode: "0.0",
    showValAxisTitle: true, valAxisTitle: "Bias in the inverted temperature (°C)",
    showLegend: true, legendPos: "t", catAxisLabelPos: "low",
    objectName: "sensitivity-chart",
  }));
  text(s, "If a fixed parameter is off by this much, the inversion returns a temperature this far off to compensate.",
    { x: 8.6, y: 1.5, w: 4.1, h: 1.1, fontSize: 15, color: C.text2 });
  stat(s, "0.2–0.4 °C", "bias from a 10 % error in melt factor or precipitation", 8.6, 2.6, 4.1, { color: C.accent2, tag: "stat-mb" });
  stat(s, "≤ 0.05 °C", "from the same error in sliding (τ_ref): 5–15× smaller", 8.6, 4.1, 4.1, { color: C.accent4, tag: "stat-tau" });
  text(s, "Error bars on T(t) are set by the mass-balance priors. Melt and precipitation trade off, so extent constrains roughly their ratio.",
    { x: 8.6, y: 5.6, w: 4.1, h: 1.25, fontSize: 14, color: C.text1, bold: true });
  s.addNotes(
`~2:00. How much does it matter that we fixed the other parameters? I perturbed each by ±10% and ±25% and converted the shift in terminus into an equivalent temperature bias.
Melt factor and precipitation dominate: 10% error ≈ 0.2–0.4 °C. Sliding is 5–15× smaller. Physically sensible: mass balance decides where the terminus ends up; dynamics mostly decide how fast it gets there. Note the melt-factor asymmetry (nonlinear PDD) and the melt–precipitation trade-off from slide 3.
Method note: this particular study was done the classical way — 16 perturbed forward runs. With the adjoint, one backward pass gives the same derivatives for all parameters at once, including spatial fields; that comparison is the next thing to run.`);

  // 16 ---------------------------------------------------------- lessons
  s = content("What the inversion taught us about itself", S3);
  const lessons = [
    ["FaSlidersH", "~20 iterations", "Level before shape",
      "Adam first moves all 11 nodes together; a trend appears only once the mean is right. Optimising level and shape separately makes the trend appear at once — learning rate still being tuned."],
    ["FaMountain", "36 %", "Not all misfit is climate",
      "Of the extent misfit (60 % at 1880) sits on ~34° headwalls above every possible ELA. Masking ice above 3000 m keeps every reducible cell and drops 96 % of the rest."],
    ["FaExclamationTriangle", "10⁸ ×", "Flat ice breaks gradients",
      "Gradient growth in 40 steps over a 2 cm-flat patch: the emulator's flow direction differentiates like 1/|∇s|. Flooring the slope at 0.1 % fixes it; the forward run is unchanged."],
  ];
  for (let i = 0; i < 3; i++) {
    const x = 0.6 + i * 4.1, y = 1.45, w = 3.85, h = 4.55;
    card(s, x, y, w, h, { fill: C.background2, tag: `lesson-${i}` });
    await iconCircle(s, lessons[i][0], x + 0.3, y + 0.3, 0.8, { fill: C.background1, tag: `lesson-${i}` });
    text(s, lessons[i][1], { x: x + 0.3, y: y + 1.25, w: w - 0.6, h: 0.75, fontSize: 36, bold: true,
      color: C.accent2, fontFace: THEME.headFontFace, valign: "bottom" });
    text(s, lessons[i][2], { x: x + 0.3, y: y + 2.1, w: w - 0.6, h: 0.45, fontSize: 19, bold: true });
    text(s, lessons[i][3], { x: x + 0.3, y: y + 2.6, w: w - 0.6, h: 1.85, fontSize: 15, color: C.text2 });
  }
  text(s, [
    { text: "Stopping at iteration 20 would have reported “no trend”: ", options: { bold: true } },
    { text: "a conditioning failure that looks like a scientific result.", options: { color: C.text2 } },
  ], { x: 0.6, y: 6.2, w: 12.1, h: 0.6, fontSize: 17, valign: "middle" });
  s.addNotes(
`~1:30 (optional — skip if short on time). Three lessons that are about the method, not the glacier.
1. Conditioning: with node values as parameters, Adam fixes the mean level first, and the trend has to emerge as a residue — hence 20 flat iterations. Stopping there would have given the wrong scientific answer ("no trend"). Splitting level and shape fixes that mechanism; the learning rate still needs tuning.
2. A third of the misfit is model-structure error on steep headwalls, out of reach for any temperature; at 1880 it is 60%, and 1880 anchors the cold end. A 3000 m mask removes it.
3. A numerical one: over a perfectly flat patch the emulator's gradient blows up; a tiny slope floor fixes it without changing the forward model.`);

  // 17 ------------------------------------------------------- conclusions
  pres.addSection({ title: S4 });
  s = content("Conclusions and next steps", S4);
  card(s, 0.6, 1.45, 6.3, 5.35, { fill: C.text1, tag: "concl-card" });
  text(s, "What we showed", { x: 0.95, y: 1.7, w: 5.6, h: 0.45, fontSize: 20, bold: true, color: C.accent3 });
  const concl = [
    "A differentiable glacier model inverts a 517-year temperature history from 5 dated extents in ~3 GPU-hours.",
    "Aletsch needs warming: +0.9 °C between 1750 and 1950 recovered; ≥ +1.36 °C implied by the termini.",
    "One gradient costs ~3.5 forward runs for any number of parameters: sensitivity without ensembles.",
    "Mass balance matters 5–15× more than ice flow for the answer.",
  ];
  text(s, concl.map((t, i) => ({ text: t, options: { bullet: { type: "number" }, breakLine: i < concl.length - 1,
    paraSpaceAfter: 20, color: C.background1 } })), { x: 0.95, y: 2.35, w: 5.65, h: 4.3, fontSize: 18 });
  text(s, "Next", { x: 7.4, y: 1.7, w: 5.3, h: 0.45, fontSize: 20, bold: true, color: C.accent1 });
  const next = [
    ["FaRulerVertical", "Add surface elevation", "volume −34 % vs. area −6 %: far more signal than extent"],
    ["FaWater", "Invert temperature and precipitation jointly", "test how much of the trend is accumulation in disguise"],
    ["FaChartLine", "Error bars", "gradient-informed ensembles over the melt and precipitation priors"],
    ["FaSnowflake", "Younger Dryas", "421 dated moraine cells (11.8–14.2 ka) already on the model grid"],
  ];
  for (let i = 0; i < 4; i++) {
    const y = 2.3 + i * 1.12;
    await iconCircle(s, next[i][0], 7.4, y, 0.65, { tag: `next-${i}` });
    text(s, next[i][1], { x: 8.25, y: y - 0.02, w: 4.45, h: 0.4, fontSize: 16, bold: true });
    text(s, next[i][2], { x: 8.25, y: y + 0.38, w: 4.45, h: 0.6, fontSize: 14, color: C.text2 });
  }
  s.addNotes(
`~1:30. To conclude: the differentiable model inverts a 500-year temperature history from extents in a few GPU-hours; the Aletsch record requires warming of at least ~0.9 °C from the mid-18th to the mid-20th century in this run; gradients deliver sensitivities at a fixed cost, independent of the number of parameters; and mass balance, not dynamics, dominates.
Next: add surface elevation (much more information than extent), invert precipitation jointly, put error bars on T(t), and move to the Younger Dryas moraines, which are already prepared on the same grid.`);

  // 18 --------------------------------------------------------------- end
  s = pres.addSlide({ masterName: "END", sectionTitle: S4 });
  s.addText("Thank you — questions?", { placeholder: "title" });
  s.addText([
    { text: "Kejdi Lleshi", options: { bold: true, color: C.background1, breakLine: true } },
    { text: "University of Lausanne  ·  kejdi.lleshi@unil.ch", options: {} },
  ], { placeholder: "body" });
  s.addImage({ path: IMG("title_art.png"), x: W - 4.4, y: 0, w: 4.4, h: 7.5, objectName: "aletsch-hillshade-end" });
  s.addNotes(`Questions. Backup material if asked: spin-up test (1500 vs 1600 vs 1700 start), the V2/V3 parameterisation experiments, and the flat-surface gradient fix.`);

  await pres.writeFile({ fileName: OUT });
  await applyTheme(OUT, THEME);
  console.log("wrote", OUT);
})().catch((e) => { console.error(e); process.exit(1); });
