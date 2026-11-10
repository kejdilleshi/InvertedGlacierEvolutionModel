// Argentière Younger Dryas temperature inversion talk. Figures: figs.py (same folder).
//   PPTX_SKILL=<pptx skill dir> node build.js ../Argentiere_YD_inversion.pptx
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

// ======================================================================= deck
(async () => {
  const S1 = "Setting", S2 = "Forward run and loss", S3 = "Result", S4 = "Sensitivity";

  // 1 ---------------------------------------------------------------- title
  pres.addSection({ title: S1 });
  let s = pres.addSlide({ masterName: "TITLE", sectionTitle: S1 });
  s.addText("Younger Dryas temperature from the Argentière moraines", { placeholder: "title" });
  s.addText("Inverting T(t), 13.6–9.0 ka, from dated landforms with a differentiable glacier model",
    { placeholder: "subtitle" });
  s.addText("Kejdi Lleshi  ·  University of Lausanne", { placeholder: "author" });
  s.addImage({ path: IMG("site.png"), x: 7.678, y: 0.768, w: 5.656, h: 5.728, objectName: "title-map" });
  s.addNotes("Lateglacial window 13.6-9.0 ka; the target is the Younger Dryas temperature and how far we can trust it.");

  // 2 ------------------------------------------------------ site and data
  s = content("Dated landforms record the Lateglacial glacier", S1);
  await picture(s, "site.png", 0.6, 1.35, 5.4, 5.5, "site-map");
  s.addTable([
    [th("landform"), th("n"), th("age (ka)"), th("constrains")],
    ["Frontal moraine J2 (La Joux)", "1", "10.8", "terminus position"],
    ["Lateral, dated: Lognan L1–L3, Talèfre T1–T2", "5", "11.9–10.3", "ice-surface height"],
    ["Lateral, undated: La Joux, Lognan", "4", "13.1–10.8 *", "ice-surface height"],
    ["Bedrock / erratic exposure ages", "17", "19.1–9.8", "ice-free after exposure"],
  ], tableOpts({ x: 6.4, y: 1.5, w: 6.3, colW: [2.75, 0.5, 1.25, 1.8], fontSize: 13 }));
  text(s, "* age window from morphostratigraphy", { x: 6.4, y: 4.45, w: 6.3, h: 0.3, fontSize: 12, color: C.accent4 });
  card(s, 6.4, 5.0, 6.3, 1.75, { tag: "data-card" });
  text(s, [
    { text: "¹⁰Be ages: ", options: { bold: true } },
    { text: "AlpIce-1 (Kamleitner et al. 2026); Protin et al. 2019, 2021; Lehmann et al. 2019, 2020", options: { breakLine: true } },
    { text: "Start glacier: ", options: { bold: true } },
    { text: "LIA extent and surface (Reinthaler & Paul 2025)", options: { breakLine: true } },
    { text: "Grid: ", options: { bold: true } },
    { text: "100 m, 250 × 250 cells", options: {} },
  ], { x: 6.65, y: 5.2, w: 5.8, h: 1.4, fontSize: 14, color: C.text2, paraSpaceAfter: 4 });
  s.addNotes("Three valleys: Argentière, Mer de Glace, Talèfre. One frontal moraine (J2), nine lateral moraines, 17 exposure-dated bedrock/erratic samples. The five samples exposed before 15.6 ka cannot constrain this window.");

  // 3 ----------------------------------------------------- forward run chain
  pres.addSection({ title: S2 });
  s = content("A forward run: from T(t) to a glacier", S2);
  const chain = [
    ["T(t) at 2000 m", "24 nodes, the unknown", C.accent2, true, false],
    ["T(x, y, t)", "lapse rate", C.background2, false, true],
    ["Mass balance", "P, T_s, DDF, T_present", C.background2, false, true],
    ["Ice flow", "IGM emulator, fixed", C.background2, false, true],
    ["H(x, y, t)", "saved every 100 yr", C.background2, false, false],
    ["Loss", "vs dated landforms", C.text1, true, false],
  ];
  const bw = 1.78, bg = 0.284, by0 = 1.5, bh = 1.15;
  for (let i = 0; i < chain.length; i++) {
    const [h, sub, fill, light, lit] = chain[i];
    const x = 0.6 + i * (bw + bg);
    card(s, x, by0, bw, bh, { fill, tag: `chain-${i}` });
    text(s, h, { x: x + 0.1, y: by0 + 0.18, w: bw - 0.2, h: 0.4, fontSize: 17, bold: true, align: "center",
      color: light ? C.background1 : C.text1 });
    text(s, sub, { x: x + 0.1, y: by0 + 0.62, w: bw - 0.2, h: 0.4, fontSize: 13, align: "center",
      color: light ? C.background1 : C.text2 });
    if (lit) await litBadge(s, x + bw - 0.26, by0 - 0.1, 0.34, `lit-chain-${i}`);
    if (i < chain.length - 1) s.addShape(pres.shapes.RIGHT_ARROW, { x: x + bw + 0.04, y: by0 + 0.45, w: 0.2, h: 0.25,
      fill: { color: C.accent4 }, line: { type: "none" }, objectName: `chain-arrow-${i}` });
  }
  await litBadge(s, 0.6, 2.85, 0.3, "lit-legend");
  text(s, "= parameters borrowed from the literature (next slide), tested for sensitivity at the end",
    { x: 1.0, y: 2.87, w: 11, h: 0.3, fontSize: 13, color: C.text2 });
  await picture(s, "control.png", 0.6, 3.35, 12.1, 3.5, "control-fig");
  s.addNotes("The only unknown is T at a fixed elevation (2000 m), 24 values every 200 years; linear in between; the mass balance is recomputed every 10 years. Everything else is fixed: these are the parameters on the next slide.");

  // 4 ------------------------------------------------------ fixed inputs
  s = content("Fixed inputs, borrowed from the literature", S2);
  const P = [
    ["P", "1.8 m/yr", "precipitation; snow where T < 0 °C", "Chamonix 1.28 (1042 m), Col du Dôme ~2.5 (4250 m)", "1.5 – 2.1"],
    ["P scaling", "+7 %/°C", "Lateglacial P follows T(t)", "Clausius–Clapeyron", "×0.75 – ×1.25"],
    ["T_s", "9.4 °C", "seasonal half-range of the year", "Chamonix 1981–2010 (Jan −2.2, Jul 16.5 °C)", "7.9 – 10.9"],
    ["lapse rate", "7 °C/km", "T(z) from T at 2000 m", "standard value", "6 – 8"],
    ["DDF", "10 mm/K/d", "melt per positive degree-day", "upper literature bound", "8 – 12"],
    ["T_present", "−0.89 °C", "2012–2021 at 2000 m: anomaly reference", "fitted to geodetic −0.66 m w.e./yr (Davaze 2020)", "re-fitted"],
    ["A (flow law)", "78 MPa⁻³/yr", "ice softness", "IGM", "52 – 117"],
    ["τ_ref (sliding)", "0.215 MPa", "basal sliding", "IGM emulator", "0.15"],
    ["speed cap", "600 m/yr", "stability in the Lognan icefall", "model choice", "1000"],
    ["T bounds", "envelope", "keeps T(t) in the literature range", "Protin 2019, Heiri 2015, Samartin 2012", "–"],
  ];
  s.addTable([
    [th("parameter"), th("value"), th("what it does"), th("source"), th("tested range")],
    ...P.map((r) => [{ text: r[0], options: { bold: true } }, r[1], r[2],
      { text: r[3], options: { color: C.text2 } }, { text: r[4], options: { bold: true, color: C.accent2 } }]),
  ], tableOpts({ x: 0.6, y: 1.35, w: 12.1, colW: [1.65, 1.4, 3.35, 4.3, 1.4], fontSize: 13, rowH: 0.4 }));
  text(s, "Check: P = 1.8 m/yr → 1.40 m w.e./yr accumulation vs 1.47 measured (Davaze et al. 2020); the first calibration (P = 4.7) gave 3.2",
    { x: 0.6, y: 6.5, w: 12.1, h: 0.35, fontSize: 13, color: C.text2 });
  s.addNotes("Every row here is a choice we did not invert. The last column is what the sensitivity analysis perturbs (slide 10). T_present is re-fitted to the geodetic balance whenever a mass-balance parameter changes, so the result stays an anomaly to today.");

  // 5 ---------------------------------------------------------------- SMB
  s = content("How T(t), T_s and P make the mass balance", S2);
  await picture(s, "smb.png", 0.6, 1.3, 12.1, 4.3, "smb-fig");
  const eq = [
    ["T_m(z, t) = T(t) − 7 °C/km · (z − 2000 m)", "T(t) shifts the whole field up or down"],
    ["T(τ) = T_m − T_s · cos 2πτ", "T_s sets how far summer rises above the mean"],
    ["SMB = P(t) · f(T < 0) − DDF · Σ max(T, 0)", "P(t) = 1.8 m/yr · 1.07^(T(t) − T_present)"],
  ];
  for (let i = 0; i < 3; i++) {
    const x = 0.6 + i * 4.1;
    card(s, x, 5.75, 3.9, 1.1, { tag: `eq-${i}` });
    text(s, eq[i][0], { x: x + 0.2, y: 5.85, w: 3.5, h: 0.45, fontSize: 14, bold: true, fontFace: "Cambria" });
    text(s, eq[i][1], { x: x + 0.2, y: 6.32, w: 3.5, h: 0.45, fontSize: 13, color: C.text2 });
  }
  s.addNotes("Left: the year at 2400 m. Today summer rises above 0 °C for ~4 months; in the Younger Dryas almost never. Right: the result is an ELA ~600 m lower (2865 → 2258 m), even though P is 27 % smaller (7 %/°C).");

  // 6 --------------------------------------------------------------- loss
  s = content("The loss: landforms read along their valley", S2);
  await picture(s, "flowlines.png", 0.6, 1.3, 5.55, 5.55, "flowlines-fig");
  s.addTable([
    [th("landform"), th("asks"), th("σ")],
    ["frontal moraine", "terminus L = moraine, at some time in its age window", "300 m"],
    ["lateral moraine", "glacier reaches it, and axis surface = crest + 10 m, at some time in its window", "300 m, 60 m"],
    ["bedrock, erratic", "axis surface ≤ sample + 10 m at every time after exposure", "60 m"],
    ["moraine preserved", "no later advance overrides it", "weight 0.5"],
  ], tableOpts({ x: 6.45, y: 1.35, w: 6.25, colW: [1.55, 3.45, 1.25], fontSize: 13 }));
  card(s, 6.45, 4.55, 6.25, 2.3, { tag: "loss-card" });
  text(s, "J = Σ landforms  w · Huber(misfit)  +  Σ (ΔΔT)²", { x: 6.7, y: 4.72, w: 5.8, h: 0.45, fontSize: 17,
    bold: true, fontFace: "Cambria" });
  text(s, [
    { text: "Terminus L: soft maximum of ice along the flowline — defined and differentiable even when the glacier stops short", options: { bullet: true, breakLine: true } },
    { text: "Each landform counts once; beyond 3σ it pulls linearly (Huber)", options: { bullet: true, breakLine: true } },
    { text: "Second term: smoothness of the 24 nodes", options: { bullet: true } },
  ], { x: 6.7, y: 5.25, w: 5.8, h: 1.5, fontSize: 13, color: C.text2, paraSpaceAfter: 4 });
  s.addNotes("Map: one flowline per valley (thick part = where ice counts for the terminus). White lines: each landform is moved to its cross-section on the valley axis. Two quantities are measured there: the terminus position L along the flowline and the ice surface on the axis. The first version scored the ice at the landform's own cells: blind when the glacier stops short, and fooled by ice on valley sides and nunataks.");

  // 7 ------------------------------------------------------------ scoring
  s = content("Scoring one forward run", S2);
  await picture(s, "scoring.png", 0.6, 1.3, 9.55, 5.6, "scoring-fig");
  const steps = [
    ["FaRoute", "Space", "Each landform is moved to its cross-section of the valley axis."],
    ["FaClock", "Time", "Moraines: the best time in their age window. Bedrock: every time after exposure."],
    ["FaCalculator", "Score", "(distance / σ)², summed over landforms."],
  ];
  for (let i = 0; i < 3; i++) {
    const y = 1.45 + i * 1.8;
    await iconCircle(s, steps[i][0], 10.45, y, 0.6, { tag: `score-${i}` });
    text(s, steps[i][1], { x: 11.2, y: y + 0.1, w: 1.5, h: 0.4, fontSize: 17, bold: true });
    text(s, steps[i][2], { x: 10.45, y: y + 0.72, w: 2.3, h: 1.0, fontSize: 13, color: C.text2 });
  }
  s.addNotes("Top: the best run at 11.9 ka along the Argentière flowline. Red: lateral crests (+10 m) with their 60 m σ; dark triangles: bedrock samples, ice must stay below them once they are exposed; star: J2 with its 300 m band; open triangle: the modelled terminus. Bottom: the same three kinds through time. J2 is met at 11.9 ka, the old end of its window. Lognan L1 sits ~76 m too high for the whole window, outside its σ. The bedrock sample stays ice-free.");

  // 8 ------------------------------------------------------------- result
  pres.addSection({ title: S3 });
  s = content("Younger Dryas: 4.65 °C colder than 2012–2021", S3);
  await picture(s, "result.png", 0.6, 1.3, 8.9, 4.4, "result-fig");
  const st = [
    ["−4.65 °C", "YD mean vs 2012–2021"],
    ["−2.65 °C", "YD mean vs pre-industrial"],
    ["−5.65 °C", "coldest node, 12.2 ka (at 2000 m)"],
  ];
  for (let i = 0; i < 3; i++) stat(s, st[i][0], st[i][1], 9.85, 1.35 + i * 1.45, 2.9,
    { color: i === 0 ? C.accent2 : C.text1, tag: `res-${i}` });
  s.addTable([
    [th("check"), th("value")],
    ["Protin et al. 2019 (summer, YD / early Holocene)", "3.0–5.5 °C below present"],
    ["Loss over 30 iterations (from the 200 m result)", "23.66 → 22.74"],
    ["Nodes on the envelope bounds", "0 of 24"],
  ], tableOpts({ x: 0.6, y: 5.3, w: 8.9, colW: [5.4, 3.5], fontSize: 13 }));
  s.addNotes("Warming from 10.2 ka to −2.41 °C at 2000 m by 9.0 ka. The YD value sits inside Protin et al. 2019 and no node is held by the envelope, so the data, not the bounds, set it. But: the 100 m run started from the 200 m optimum, so the 30 iterations only refine it (35.3 → 25.9 happened at 200 m).");

  // 9 ------------------------------------------------------------- stands
  s = content("The glacier at the dated stands", S3);
  await picture(s, "stands.png", 0.6, 1.3, 12.1, 3.0, "stands-fig");
  text(s, "▽ modelled terminus   ★ J2 frontal moraine   red: moraine crests",
    { x: 0.6, y: 3.95, w: 12.1, h: 0.3, fontSize: 12, color: C.accent4 });
  const fitCols = [
    ["FaCheck", "Met", C.accent6, [
      "J2 terminus, at 11.9 ka (old end of its window)",
      "Lognan L2; Talèfre T1, T2",
      "13 of 17 bedrock samples ice-free"]],
    ["FaTimes", "Not met", C.accent2, [
      "La Joux, Lognan L1/L3: ice 55–80 m above the crests",
      "4 bedrock samples under 46–149 m of ice",
      "→ glacier too thick for its length"]],
  ];
  for (let i = 0; i < 2; i++) {
    const x = 0.6 + i * 6.2;
    card(s, x, 4.75, 5.9, 2.1, { tag: `fit-${i}` });
    await iconCircle(s, fitCols[i][0], x + 0.25, 4.95, 0.5, { fill: fitCols[i][2], fg: HEX.lt1, tag: `fit-ic-${i}` });
    text(s, fitCols[i][1], { x: x + 0.9, y: 5.02, w: 4, h: 0.4, fontSize: 18, bold: true });
    text(s, fitCols[i][3].map((t, k) => ({ text: t, options: { bullet: true, breakLine: k < 2,
      bold: i === 1 && k === 2 } })), { x: x + 0.3, y: 5.55, w: 5.4, h: 1.25, fontSize: 14, color: C.text2, paraSpaceAfter: 3 });
  }
  s.addNotes("Video of the full run: results/argentiere_lg_100m_phys/video/argentiere_lateglacial_100m.mp4. The remaining misfit is a glacier that is long enough but too thick: T(t) cannot fix that, since it moves length and thickness together. It points at accumulation, melt or ice flow.");

  // 10 ------------------------------------------------------- sensitivity
  pres.addSection({ title: S4 });
  s = content("Small temperature shifts, large changes in fit", S4);
  // Forward checks (results/argentiere_lg_100m_phys/sensitivity/*.json, tools/sensitivity_argentiere.py):
  // best uniform shift of T(t) for one parameter at an end of its range, as a change of the YD anomaly
  // (dT - dT_present) relative to the base parameters' own best shift (-0.12 C), and the loss there.
  const BASE_LOSS = 22.40, SIG_DATA = 0.22;  // base parabola vertex; data-only sigma of the YD mean (block_yd)
  const rows = [
    ["P", "1.5", "2.1 m/yr", [-0.54, 19.28], [0.47, 25.52]],
    ["DDF", "8", "12 mm/K/d", [0.34, 20.32], [-0.33, 24.95]],
    ["T_s", "7.9", "10.9 °C", [-0.16, 23.47], [0.08, 22.44]],
    ["A (flow law)", "52", "117", [0.01, 27.97], [-0.06, 18.32]],
    ["τ_ref (sliding)", "0.15 MPa", "", [-0.07, 13.73]],
    ["P scaling", "LG ×0.75", "×1.25"],
    ["lapse rate", "6", "8 °C/km"],
  ];
  const tx = 0.6, ty = 1.75, rh = 0.56, x0 = 4.9, scale = 1.6; // inches per °C
  const X = (v) => x0 + v * scale;
  await litBadge(s, tx, 1.33, 0.3, "lit-sens");
  text(s, "parameter (slide 4)", { x: tx + 0.4, y: 1.35, w: 2.2, h: 0.3, fontSize: 13, bold: true, color: C.text2 });
  text(s, "low / high", { x: 2.2, y: 1.35, w: 1.2, h: 0.3, fontSize: 13, bold: true, color: C.text2 });
  text(s, "change of the YD anomaly (°C)", { x: x0 - 1.5, y: 1.35, w: 3.0, h: 0.3, fontSize: 13, bold: true,
    color: C.text2, align: "center" });
  text(s, "loss", { x: 6.65, y: 1.35, w: 0.8, h: 0.3, fontSize: 13, bold: true, color: C.text2, align: "right" });
  for (let i = 0; i < rows.length; i++) if (i % 2 === 0) s.addShape(pres.shapes.RECTANGLE, { x: tx, y: ty + i * rh - 0.04,
    w: 6.9, h: rh, fill: { color: C.background2 }, line: { type: "none" }, objectName: `sens-band-${i}` });
  s.addShape(pres.shapes.RECTANGLE, { x: X(-SIG_DATA), y: ty - 0.1, w: 2 * SIG_DATA * scale, h: 4.06,
    fill: { color: C.accent4, transparency: 65 }, line: { type: "none" }, objectName: "sens-data-sigma" });
  const fmt = (v) => (v > 0 ? "+" : v < 0 ? "−" : "±") + Math.abs(v).toFixed(2);
  for (let i = 0; i < rows.length; i++) {
    const y = ty + i * rh, [nm, lab0, lab1, lo, hi] = rows[i];
    text(s, nm, { x: tx + 0.1, y: y + 0.1, w: 1.5, h: 0.35, fontSize: 14, bold: true });
    text(s, [{ text: lab0, options: { breakLine: true } }, { text: lab1 }],
      { x: 2.2, y: y + 0.02, w: 1.2, h: 0.48, fontSize: 12, color: C.text2, lineSpacing: 15 });
    if (lo === undefined) {
      text(s, "forward run pending", { x: x0 - 1.2, y: y + 0.1, w: 2.4, h: 0.32, fontSize: 13, italic: true,
        color: C.accent4, align: "center", valign: "middle" });
      continue;
    }
    [lo, hi].forEach((r, k) => {
      if (!r) return;
      const [d, loss] = r, yb = y + 0.05 + k * 0.23, better = loss < BASE_LOSS;
      const col = better ? C.accent6 : C.accent2;
      s.addShape(pres.shapes.RECTANGLE, { x: Math.min(X(0), X(d)), y: yb, w: Math.max(Math.abs(d) * scale, 0.03), h: 0.19,
        fill: { color: col }, line: { type: "none" }, objectName: `sens-bar-${i}-${k}` });
      text(s, fmt(d), d < 0 ? { x: X(d) - 0.62, y: yb - 0.02, w: 0.56, h: 0.23, fontSize: 12, align: "right", valign: "middle" }
        : { x: X(d) + 0.06, y: yb - 0.02, w: 0.56, h: 0.23, fontSize: 12, valign: "middle" });
      text(s, loss.toFixed(1), { x: 6.65, y: yb - 0.02, w: 0.8, h: 0.23, fontSize: 12, bold: true, color: col,
        align: "right", valign: "middle" });
    });
  }
  s.addShape(pres.shapes.LINE, { x: x0, y: ty - 0.1, w: 0, h: 4.06, line: { color: C.text1, width: 1 }, objectName: "sens-zero" });
  text(s, [
    { text: "upper: low end · lower: high end   " },
    { text: "■ ", options: { color: C.accent6 } }, { text: `better than base (${BASE_LOSS.toFixed(1)})   ` },
    { text: "■ ", options: { color: C.accent2 } }, { text: "worse   " },
    { text: "■ ", options: { color: C.accent4 } }, { text: `data ±${SIG_DATA.toFixed(2)} °C` },
  ], { x: tx, y: 6.08, w: 6.9, h: 0.3, fontSize: 12, color: C.text2 });
  text(s, "Forward runs, best uniform T shift relative to the base parameters; mass-balance cases re-fit T_present.",
    { x: tx, y: 6.45, w: 6.9, h: 0.3, fontSize: 11, color: C.accent4 });

  card(s, 7.85, 1.35, 4.85, 3.05, { fill: C.text1, tag: "p-card" });
  text(s, "Loss at the best T shift", { x: 8.15, y: 1.55, w: 4.3, h: 0.35, fontSize: 16, bold: true, color: C.accent3 });
  stat(s, "22.4", "literature parameters", 8.15, 1.85, 2.1, { color: C.background1, labelSize: 13, tag: "loss-base" });
  stat(s, "13.7", "τ_ref 0.15 MPa (more sliding)", 10.4, 1.85, 2.2, { color: C.accent5, labelSize: 13, tag: "loss-tau" });
  text(s, "YD moves ≤ 0.54 °C; the fit moves far more", { x: 8.15, y: 3.55, w: 4.3, h: 0.7, fontSize: 14,
    bold: true, color: C.background1 });

  card(s, 7.85, 4.6, 4.85, 2.25, { tag: "next-card" });
  text(s, "Still to come", { x: 8.15, y: 4.75, w: 4.3, h: 0.35, fontSize: 16, bold: true });
  text(s, [
    { text: "Forward checks: Lateglacial P, lapse rate", options: { bullet: true, breakLine: true } },
    { text: "First-order gradients were unreliable here (τ_ref: −0.80 predicted)", options: { bullet: true, breakLine: true } },
    { text: "Joint inversion T(t) + P, T_s, DDF, A, τ_ref: running", options: { bullet: true, bold: true } },
  ], { x: 8.15, y: 5.2, w: 4.35, h: 1.55, fontSize: 14, color: C.text2, paraSpaceAfter: 4 });
  s.addNotes("Forward checks: for each parameter at an end of its literature range, the uniform shift of T(t) that minimises the loss (parabola through 3-6 forward runs), as a change of the YD anomaly to today relative to the base parameters' own best shift (-0.12 C). Mass-balance parameters re-fit T_present to the geodetic balance first. Upper bar: low end, lower bar: high end; green = better fit than the base (22.4), red = worse; number on the right = loss at that shift. Grey band: data-only uncertainty of the YD mean (+-0.22 C). All tested parameters move the YD anomaly by at most 0.54 C (P 1.5), but the fit changes a lot: more sliding gives 13.7, softer ice 18.3, less precipitation 19.3, a lower DDF 20.3. The first-order gradient estimates (previous version) were poor for the flow parameters. Pending: Lateglacial precipitation factor and lapse rate. Running: joint inversion of T(t) with P, T_s, DDF, A and tau_ref.");

  // 11 ----------------------------------------------------------- best fit
  s = content("Best fit: more sliding, same temperature", S4);
  await picture(s, "bestfit.png", 0.6, 1.3, 8.9, 4.2, "bestfit-fig");
  s.addTable([
    [th("same T(t)"), th("τ_ref 0.215 (literature)"), th("τ_ref 0.15")],
    ["loss", "22.7", { text: "14.5", options: { bold: true, color: C.accent6 } }],
    ["landforms with misfit > 1", "10", { text: "6", options: { bold: true, color: C.accent6 } }],
    ["J2 terminus vs moraine", "−129 m", "−122 m"],
  ], tableOpts({ x: 0.6, y: 5.45, w: 8.9, colW: [3.9, 2.6, 2.4], fontSize: 13 }));
  const bf = [
    ["−36 %", "loss, without changing T(t)", C.accent6],
    ["20–50 m", "thinner ice at the landforms, same length", C.text1],
    ["−0.07 °C", "YD anomaly at the new best shift", C.text1],
  ];
  for (let i = 0; i < 3; i++) stat(s, bf[i][0], bf[i][1], 9.85, 1.35 + i * 1.45, 2.9, { color: bf[i][2], tag: `bf-${i}` });
  card(s, 9.85, 5.75, 2.85, 1.1, { tag: "bf-card" });
  text(s, "Softer ice (A 117) points the same way: loss 18.3", { x: 10.0, y: 5.85, w: 2.6, h: 0.9, fontSize: 13,
    color: C.text2 });
  s.addNotes("Same T(t) as the inversion, only the sliding parameter changed (tau_ref 0.215 -> 0.15 MPa). Faster flow drains the glacier: the ice surface drops 20-50 m at every landform, the terminus stays put (J2 -129 -> -122 m). That is exactly the 'too thick for its length' misfit of the inversion. Lateral moraines at La Joux and Lognan L1/L3 come into or close to their 60 m sigma; the La Joux erratic and Aiguille du Moine C become ice-free or nearly; Talefre T1/T2 and Lognan (undated 2) now sit 30-40 m below their crests. Caveat: one parameter value, one uniform T shift, not a re-inversion; the joint inversion now running frees tau_ref, A and the mass-balance parameters together with T(t).");

  await pres.writeFile({ fileName: OUT });
  await applyTheme(OUT, THEME);
  console.log("wrote", OUT);
})().catch((e) => { console.error(e); process.exit(1); });
