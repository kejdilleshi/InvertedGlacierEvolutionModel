// Context slide (2) and joint-inversion slides (12-14) for the Argentière YD talk, built on the same theme, layouts
// and helpers as build.js (header copied from it), then merged into the existing deck by merge_slides.py:
//   ONLY=context,controls PPTX_SKILL=<pptx skill dir> node joint_slides.js new_slides.pptx
const pptxgen = require("pptxgenjs");
const React = require("react");
const ReactDOMServer = require("react-dom/server");
const sharp = require("sharp");
const fa = require("react-icons/fa");
const path = require("path");
const { applyTheme } = require(process.env.PPTX_SKILL + "/scripts/apply_theme.js");

const OUT = process.argv[2] || "Argentiere_YD_inversion.pptx";
const IMG = (f) => path.join(__dirname, "img", f);

// Same theme as the LIA talk (presentation/src/build.js), so the two decks match.
const THEME = {
  name: "Argentiere Glacier",
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
pres.title = "Younger Dryas temperature from the Argentière moraines";
pres.author = "Kejdi Lleshi";
pres.subject = "Lateglacial temperature inversion, Argentière, Mer de Glace and Talèfre";
const C = pres.SchemeColor;

// ------------------------------------------------------------------ layouts
const W = 13.333;
const FOOT = "K. Lleshi  ·  Younger Dryas temperature inversion  ·  Argentière";

pres.defineSlideMaster({
  title: "TITLE",
  background: { color: HEX.dk1 },
  objects: [
    { placeholder: { options: { name: "title", type: "title", x: 0.7, y: 1.5, w: 7.4, h: 2.5, align: "left",
        fontSize: 40, bold: true, color: C.background1, valign: "bottom", margin: 0 }, text: "" } },
    { placeholder: { options: { name: "subtitle", type: "body", x: 0.7, y: 4.25, w: 7.2, h: 1.2,
        fontSize: 20, color: C.accent3, valign: "top", margin: 0 }, text: "" } },
    { placeholder: { options: { name: "author", type: "body", x: 0.7, y: 6.0, w: 7.2, h: 0.7,
        fontSize: 16, color: C.background1, valign: "top", margin: 0 }, text: "" } },
  ],
});

pres.defineSlideMaster({
  title: "CONTENT",
  background: { color: HEX.lt1 },
  margin: [0.5, 0.6, 0.6, 0.6],
  objects: [
    { placeholder: { options: { name: "title", type: "title", x: 0.6, y: 0.32, w: 12.1, h: 0.9, align: "left",
        fontSize: 34, bold: true, color: C.text1, valign: "middle", margin: 0 }, text: "" } },
    { text: { text: FOOT, options: { x: 0.6, y: 7.0, w: 9, h: 0.3, fontSize: 10,
        color: C.accent4, margin: 0, isTextBox: true } } },
  ],
  slideNumber: { x: 12.23, y: 7.0, w: 0.5, h: 0.3, fontSize: 10, color: HEX.accent4, align: "right" },
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

// "Borrowed from the literature" badge: ties the inputs (slides 3-4) to the sensitivity (slide 10).
async function litBadge(slide, x, y, d = 0.34, tag = "lit") {
  await iconCircle(slide, "FaBook", x, y, d, { fill: C.accent5, fg: HEX.lt1, tag });
}

function card(slide, x, y, w, h, { fill = C.background2, tag = "card" } = {}) {
  slide.addShape(pres.shapes.ROUNDED_RECTANGLE, { x, y, w, h, rectRadius: 0.12,
    fill: { color: fill }, line: { type: "none" }, objectName: tag });
}

function text(slide, t, opts) {
  slide.addText(t, Object.assign({ isTextBox: true, margin: 0, fontSize: 16, color: C.text1,
    valign: "top", fit: "none" }, opts));
}

function stat(slide, big, label, x, y, w, { color = C.accent1, bigSize = 36, labelSize = 14, tag = "stat" } = {}) {
  text(slide, big, { x, y, w, h: 0.7, fontSize: bigSize, bold: true, color,
    fontFace: THEME.headFontFace, valign: "bottom", objectName: `${tag}-value` });
  text(slide, label, { x, y: y + 0.75, w, h: 0.6, fontSize: labelSize, color: C.text2,
    objectName: `${tag}-label` });
}

// Picture fitted inside a box (keeps its aspect), centred horizontally, top-aligned.
async function picture(slide, file, x, y, maxW, maxH, tag) {
  const m = await sharp(IMG(file)).metadata();
  const r = m.width / m.height;
  let w = maxW, h = maxW / r;
  if (h > maxH) { h = maxH; w = maxH * r; }
  slide.addImage({ path: IMG(file), x: x + (maxW - w) / 2, y, w, h, objectName: tag || file });
  return { x: x + (maxW - w) / 2, y, w, h };
}

const th = (t) => ({ text: t, options: { bold: true, color: C.background1, fill: { color: C.text1 } } });
const tableOpts = (o) => Object.assign({ fontSize: 14, color: C.text1, border: { type: "solid", pt: 0.5, color: "D5DEE3" },
  valign: "middle", margin: [3, 6, 3, 6], autoPage: false }, o);

function content(title, section) {
  const s = pres.addSlide({ masterName: "CONTENT", sectionTitle: section });
  s.addText(title, { placeholder: "title" });
  return s;
}

// ================================================================= slides
// ONLY=context,why,controls,result builds a subset (default: all), in that order.
const ONLY = (process.env.ONLY || "context,why,controls,result").split(",");
const S1 = "Setting", S5 = "Joint inversion";

const SLIDES = {
  // --------------------------------------------------------------- context
  context: async () => {
    const s = content("What made the Younger Dryas glacier?", S1);
    const what = [
      ["FaThermometerHalf", "Climate", "T(t) at 2000 m, precipitation P, seasonality T_s, melt factor DDF", HEX.accent2],
      ["FaMountain", "Glacier", "ice softness A (flow law) and basal sliding τ_ref", HEX.accent1],
    ];
    for (let i = 0; i < 2; i++) {
      const x = 0.6 + i * 6.15;
      card(s, x, 1.35, 5.95, 1.15, { tag: `goal-${i}` });
      await iconCircle(s, what[i][0], x + 0.2, 1.55, 0.75, { fill: C.background1, fg: what[i][3], tag: `goal-ic-${i}` });
      text(s, what[i][1], { x: x + 1.15, y: 1.5, w: 4.6, h: 0.4, fontSize: 18, bold: true });
      text(s, what[i][2], { x: x + 1.15, y: 1.88, w: 4.6, h: 0.6, fontSize: 14, color: C.text2 });
    }
    text(s, "Goal: the values with which the glacier model reproduces the dated Younger Dryas landforms", {
      x: 0.6, y: 2.68, w: 12.1, h: 0.35, fontSize: 16, bold: true, color: C.accent2 });
    const steps = [
      ["FaMapMarkedAlt", "Observations", "dated moraines and exposure ages, 13.6–9 ka", "27", "landforms"],
      ["FaRunning", "Cheap first run · 200 m", "T(t) only, literature parameters: the prior", "35.3 → 25.9", "loss"],
      ["FaThermometerHalf", "T(t) inversion · 100 m", "starts from the 200 m T(t) (slides 9–11)", "→ 22.7", "loss"],
      ["FaSlidersH", "Joint inversion · 100 m", "T(t) + P, T_s, DDF, A, τ_ref (slides 12–14)", "→ 13.4", "loss"],
    ];
    const sw = 2.8, sg = 0.3, sy = 3.25, sh = 2.8;
    for (let i = 0; i < 4; i++) {
      const x = 0.6 + i * (sw + sg), last = i === 3;
      card(s, x, sy, sw, sh, { fill: last ? C.text1 : C.background2, tag: `step-${i}` });
      await iconCircle(s, steps[i][0], x + 0.2, sy + 0.2, 0.6,
        { fill: last ? C.accent2 : C.background1, fg: last ? HEX.lt1 : HEX.accent1, tag: `step-ic-${i}` });
      text(s, steps[i][1], { x: x + 0.2, y: sy + 0.9, w: sw - 0.4, h: 0.6, fontSize: 15, bold: true,
        color: last ? C.background1 : C.text1 });
      text(s, steps[i][2], { x: x + 0.2, y: sy + 1.45, w: sw - 0.4, h: 0.6, fontSize: 13,
        color: last ? C.accent3 : C.text2 });
      text(s, steps[i][3], { x: x + 0.2, y: sy + 2.0, w: sw - 0.4, h: 0.42, fontSize: 22, bold: true,
        fontFace: THEME.headFontFace, color: last ? C.accent5 : C.text1 });
      text(s, steps[i][4], { x: x + 0.2, y: sy + 2.42, w: sw - 0.4, h: 0.28, fontSize: 12,
        color: last ? C.accent3 : C.accent4 });
      if (i < 3) s.addShape(pres.shapes.RIGHT_ARROW, { x: x + sw + 0.05, y: sy + 1.25, w: 0.2, h: 0.3,
        fill: { color: C.accent4 }, line: { type: "none" }, objectName: `step-arrow-${i}` });
    }
    text(s, "All results from here on are at 100 m; the 200 m run only provides the starting T(t).",
      { x: 0.6, y: 6.3, w: 12.1, h: 0.35, fontSize: 14, italic: true, color: C.text2 });
    s.addNotes("What we look for: the climate (temperature history at 2000 m, precipitation, seasonality, melt factor) and the glacier parameters (ice softness, sliding) with which the IGM-emulator glacier model reproduces the dated Younger Dryas landforms of Argentiere, Mer de Glace and Talefre. Path: a cheap first inversion of T(t) alone at 200 m with literature parameters (loss 35.3 -> 25.9) gives the starting T(t); the 100 m inversion refines it (22.7, slides 9-11); the joint 100 m inversion then frees the five climate and glacier parameters together with T(t) (13.4, slides 12-14). Only 100 m results are shown.");
  },

  // ------------------------------------------------------------------ why
  why: async () => {
    const s = content("Why free the parameters too", S5);
    const why = [
      ["FaMountain", "Glacier too thick for its length",
        "Right length (J2 met), but ice 55–80 m above the La Joux and Lognan crests. T(t) moves length and thickness together."],
      ["FaSlidersH", "Ice flow thins it, T barely moves",
        "τ_ref 0.15 MPa: loss 22.4 → 13.7, YD anomaly −0.07 °C. Softer ice the same way."],
      ["FaBalanceScale", "Invert T(t) + five parameters",
        "P, T_s, DDF, A, τ_ref (slide 5), with no control favoured."],
    ];
    for (let i = 0; i < 3; i++) {
      const y = 1.45 + i * 1.75;
      await iconCircle(s, why[i][0], 0.6, y, 0.75, { tag: `why-${i}`, fg: i === 2 ? HEX.accent2 : HEX.accent1 });
      text(s, why[i][1], { x: 1.6, y: y - 0.02, w: 5.0, h: 0.45, fontSize: 18, bold: true });
      text(s, why[i][2], { x: 1.6, y: y + 0.48, w: 5.0, h: 1.0, fontSize: 14, color: C.text2 });
    }
    s.addTable([
      [th("control"), th("start"), th("allowed range")],
      ["T(t), 24 nodes", "100 m inversion", "literature envelope"],
      ["P", "1.8 m/yr", "1.0 – 1.8 m/yr"],
      ["T_s", "9.4 °C", "7.9 – 10.9 °C"],
      ["DDF", "10 mm/K/d", "8 – 12 mm/K/d"],
      ["A (flow law)", "78 MPa⁻³/yr", "52 – 117"],
      ["τ_ref (sliding)", "0.215 MPa", "0.10 – 0.25 MPa"],
    ], tableOpts({ x: 7.0, y: 1.45, w: 5.7, colW: [1.9, 1.75, 2.05], fontSize: 13, rowH: 0.4 }));
    card(s, 7.0, 4.6, 5.7, 2.25, { fill: C.text1, tag: "equal-card" });
    text(s, "Equal weight", { x: 7.3, y: 4.78, w: 5.1, h: 0.4, fontSize: 17, bold: true, color: C.accent3 });
    text(s, [
      { text: "Every control is a fraction of its range, 0 … 1", options: { bullet: true, breakLine: true } },
      { text: "One Adam step size for all, no prior on the parameters", options: { bullet: true, breakLine: true } },
      { text: "T_present re-fitted to the geodetic balance at each step", options: { bullet: true } },
    ], { x: 7.3, y: 5.25, w: 5.2, h: 1.5, fontSize: 14, color: C.background1, paraSpaceAfter: 4 });
    s.addNotes("The fixed-parameter inversion fits the length (J2) but not the thickness: lateral moraines and bedrock samples sit 50-150 m under modelled ice. A temperature change moves length and thickness together, so T(t) alone cannot fix that. The forward checks showed that more sliding (tau_ref 0.15) or softer ice (A 117) thins the glacier at the same length and barely moves the YD temperature. So the five bold parameters of slide 5 are inverted together with T(t). Equal weight: each control is optimized as a fraction of its allowed range with one Adam learning rate (Adam normalizes each control by its own gradient history), and there is no prior on the parameters; the only prior left is the smoothness of T(t). P range 1.0-1.8 m/yr. Gradients: adjoint for T(t), P, T_s, DDF; finite differences for A and tau_ref (their emulator gradient was unreliable). Run: results/argentiere_lg_100m_joint, 30 iterations of ~53 min.");
  },

  // ------------------------------------------------------------- controls
  controls: async () => {
    const s = content("Ice flow does the work, not the temperature", S5);
    await picture(s, "joint_controls.png", 0.6, 1.3, 8.9, 3.15, "joint-controls-fig");
    const jst = [
      ["−41 %", "loss 22.7 → 13.4 in 30 iterations", C.accent6],
      ["0.215 → 0.146", "τ_ref (MPa): more sliding", C.accent1],
      ["78 → 105", "A (MPa⁻³/yr): softer ice", C.accent2],
    ];
    for (let i = 0; i < 3; i++) stat(s, jst[i][0], jst[i][1], 9.85, 1.35 + i * 1.45, 2.9,
      { color: jst[i][2], bigSize: i === 0 ? 36 : 26, tag: `jst-${i}` });
    // every changed control, largest move first (move = fraction of its allowed range, the inversion's units)
    const big = (t, col) => ({ text: t, options: { bold: true, color: col } });
    const dim = (t) => ({ text: t, options: { italic: true, color: C.accent4 } });
    s.addTable([
      [th("control"), th("start"), th("joint best"), th("moved (fraction of range)")],
      [big("τ_ref (sliding)", C.accent1), "0.215 MPa", big("0.146 MPa", C.accent1), big("0.46", C.accent1)],
      [big("A (flow law)", C.accent2), "78 MPa⁻³/yr", big("105 MPa⁻³/yr", C.accent2), big("0.41", C.accent2)],
      ["T(t), YD mean anomaly", "−4.65 °C", "−4.81 °C", "0.15 (all nodes, mean)"],
      ["P", "1.80 m/yr", "1.75 m/yr", "0.06"],
      ["T_s", "9.40 °C", "9.52 °C", "0.04"],
      ["DDF", "10.0 mm/K/d", "9.87 mm/K/d", "0.03"],
      [dim("T_present (re-fitted)"), dim("−0.89 °C"), dim("−1.00 °C"), dim("follows P, T_s, DDF")],
    ], tableOpts({ x: 0.6, y: 4.6, w: 8.9, colW: [2.5, 1.9, 2.0, 2.5], fontSize: 12, rowH: 0.27 }));
    s.addNotes("Left: loss per iteration, from the fixed-parameter optimum (22.74) to 13.36. Right: change of every control as a fraction of its allowed range, the units in which they are all treated alike. Table: every changed quantity, largest move first. Sliding and flow-law factor move furthest and steadily; the temperature nodes move 0.15 on average, P, T_s and DDF hardly. T_present is not a control: it is re-fitted to the 2012-2021 geodetic balance whenever P, T_s or DDF change. tau_ref and A were still drifting at iteration 30 (loss falling by ~0.05 per iteration), so the run is close to but not at convergence.");
  },

  // --------------------------------------------------------------- result
  result: async () => {
    const s = content("Same Younger Dryas, thinner glacier", S5);
    await picture(s, "joint_result.png", 0.6, 1.3, 12.1, 4.1, "joint-result-fig");
    const res = [
      ["−4.81 °C", "YD vs 2012–2021 (fixed parameters −4.65)", C.accent2],
      ["4 → 3", "bedrock samples under > 10 m of ice", C.text1],
    ];
    for (let i = 0; i < 2; i++) stat(s, res[i][0], res[i][1], 0.6 + i * 3.3, 5.5, 3.1,
      { color: res[i][2], bigSize: 32, labelSize: 13, tag: `jres-${i}` });
    card(s, 7.3, 5.6, 5.4, 1.25, { tag: "next-joint" });
    text(s, [
      { text: "Not yet converged: ", options: { bold: true } },
      { text: "τ_ref, A still drifting. 9.2–9.0 ka on the warm bound; La Joux still 40–75 m too high", options: {} },
    ], { x: 7.5, y: 5.7, w: 5.0, h: 1.05, fontSize: 13, color: C.text2, valign: "middle" });
    s.addNotes("Left: the YD anomaly is unchanged within its data uncertainty (-4.81 vs -4.65 C, sigma 0.22); the Allerod and 9.6-9.0 ka come out about 1 C warmer, and the last two nodes sit on the warm limit of the envelope. Right: modelled ice surface above each landform's target, fixed parameters (grey) to joint (red). Lognan L3 (62 -> 7 m), Aiguille du Moine C (96 -> 27 m), Trelaporte F (149 -> 87 m) and the La Joux erratic (83 -> 48 m) improve most; La Joux laterals still 40-75 m too high; Talefre T1/T2 now 35-44 m below their crests (within sigma). J2 stays met (0 m). Next: continue from this best point until tau_ref and A settle; widen the 9-10 ka warm limit; tau_ref is close to the lower end of the emulator's training range.");
  },

  // -------------------------------------------- compare: slide 8 figure, T(t) only vs joint
  compare: async () => {
    const s = content("How the fit improved: same length, thinner ice", S5);
    await picture(s, "scoring_compare.png", 0.6, 1.3, 9.0, 5.5, "scoring-compare-fig");
    const pts = [
      ["FaRulerVertical", "39 m thinner", "Same J2 terminus, lower surface along the whole valley."],
      ["FaWater", "Faster flow", "More sliding (τ_ref ↓) and softer ice (A ↑) carry the ice further with less thickness."],
      ["FaCheck", "Bedrock −7.3", "Most of the gain: samples no longer under ice. Laterals −1.7."],
      ["FaWaveSquare", "Cost +1.5", "T(t) a bit rougher (smoothness term)."],
    ];
    for (let i = 0; i < pts.length; i++) {
      const y = 1.35 + i * 1.38;
      await iconCircle(s, pts[i][0], 9.85, y, 0.55, { tag: `cmp-${i}` });
      text(s, pts[i][1], { x: 10.55, y: y + 0.08, w: 2.2, h: 0.4, fontSize: 17, bold: true,
        color: i === 3 ? C.text2 : C.accent2 });
      text(s, pts[i][2], { x: 9.85, y: y + 0.62, w: 2.9, h: 0.7, fontSize: 12, color: C.text2 });
    }
    s.addNotes("Slide-8 figure with both runs. Grey: T(t) only (fixed parameters, loss 22.74). Red: joint inversion (13.36). Top: Argentiere flowline when each run meets J2 (11.9 ka and 11.4 ka); the joint surface is 39 m lower on average between 6 and 13 km, mean thickness 273 -> 233 m. Bottom: J2 is met by both; Lognan L1 drops from 77 to 48 m above its crest (inside sigma); the bedrock sample, already ice-free, gets a wider margin. Right: loss by landform type. Bedrock and erratics carry most of the gain (8.5 -> 1.2: Trelaporte F 149 -> 87 m, Lehmann C 96 -> 27 m, La Joux erratic 83 -> 48 m), laterals 10.7 -> 9.0, preservation 2.3 -> 0.7, frontal 1.1 -> 0.7. The price: the T(t) smoothness term rises 0.2 -> 1.8. Why: a T change moves length and thickness together; sliding and the flow law change the thickness for a given length.");
  },
};

(async () => {
  for (const name of ONLY) {
    const sec = name === "context" ? S1 : S5;
    if (!pres.sections.some((x) => x.title === sec)) pres.addSection({ title: sec });
    await SLIDES[name]();
  }
  await pres.writeFile({ fileName: OUT });
  await applyTheme(OUT, THEME);
  console.log("wrote", OUT, ONLY);
})().catch((e) => { console.error(e); process.exit(1); });
