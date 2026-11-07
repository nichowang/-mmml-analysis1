// One editable slide: Idea 2 pipeline + design (coords in px of a 1920x1080 canvas, 1 px = 1/144 in = 0.5 pt)
const pptxgen = require("pptxgenjs");
const pres = new pptxgen();
pres.layout = "LAYOUT_WIDE";
pres.theme = { headFontFace: "Fira Sans", bodyFontFace: "Fira Sans" };
const s = pres.addSlide();
s.background = { color: "FFFFFF" };

const P = (v) => v / 144;
const pt = (px) => px * 0.5;
const Y = (y) => (y <= 150 || y >= 1000 ? y : y <= 548 ? 150 + (y - 150) * 1.12 : 596 + (y - 548) * 1.17);
const H = (y, h) => Y(y + h) - Y(y);
const RED = "A6192E", MOT = "3E6FA8", APP = "6E9440", REF = "D9A3A3", INK = "2B2B2B", GRAY = "6B6B6B", RULE = "C8C8C8";
const FONT = "Fira Sans";

// all gray text is drawn in black
const GRAYS = new Set([GRAY, "555555", "777777", "888888"]);
const ungray = (o) => (o && GRAYS.has(o.color) ? Object.assign({}, o, { color: INK }) : o);
function text(t, x, y, w, h, o = {}) {
  o = ungray(o);
  if (Array.isArray(t)) t = t.map((run) => ({ text: run.text, options: ungray(run.options) }));
  s.addText(t, Object.assign({ x: P(x), y: P(Y(y)), w: P(w), h: P(H(y, h)), fontFace: FONT, fontSize: pt(24),
    color: INK, margin: 0, valign: "middle", isTextBox: true }, o));
}
function rect(x, y, w, h, o = {}) {
  s.addShape(pres.shapes.RECTANGLE, Object.assign({ x: P(x), y: P(Y(y)), w: P(w), h: P(H(y, h)) }, o));
}
function line(x1, y1, x2, y2, o = {}) {
  y1 = Y(y1); y2 = Y(y2);
  s.addShape(pres.shapes.LINE, { x: P(Math.min(x1, x2)), y: P(Math.min(y1, y2)), w: P(Math.abs(x2 - x1)), h: P(Math.abs(y2 - y1)),
    flipH: x2 < x1, flipV: y2 < y1,
    line: { color: o.color || GRAY, width: o.width || 1.75, dashType: o.dash ? "dash" : "solid", endArrowType: o.arrow ? "triangle" : undefined } });
}
const r = (t, o = {}) => ({ text: t, options: o });
const sb = (t, o = {}) => ({ text: t, options: Object.assign({ subscript: true }, o) });
const solid = (c) => ({ fill: { color: c }, line: { color: c, width: 0 } });
const outline = (c, w = 1.5) => ({ fill: { color: "FFFFFF" }, line: { color: c, width: w } });

// ---------- header ----------
rect(0, 0, 1920, 4, solid(RED));
text("03  /  IDEA 2", 116, 36, 600, 32, { fontSize: pt(26), bold: true, color: RED, charSpacing: 1 });
text("Idea 2 — Which reference clips should define a person's style?", 116, 70, 1580, 76, { fontSize: pt(52), bold: true, color: RED });
rect(1743, 36, 85, 85, solid(RED));
text("Carnegie\nMellon\nUniversity", 1750, 49, 78, 60, { fontFace: "Georgia", fontSize: 7, color: "FFFFFF", valign: "top" });

// ---------- pipeline ----------
text("PIPELINE", 116, 164, 200, 28, { fontSize: pt(22), bold: true, color: GRAY, charSpacing: 1.5 });

// A. reference set
rect(116, 215, 270, 250, outline(RULE));
[["Rec 1", 232, 2], ["Rec 2", 284, 1], ["Rec 3", 336, 1]].forEach(([lab, y, n]) => {
  text(lab, 132, y, 80, 36, { fontSize: pt(22), color: GRAY });
  for (let i = 0; i < n; i++) rect(214 + 52 * i, y + 3, 44, 30, { fill: { color: REF }, line: { color: "AA7777", width: 0.75 } });
});
text("Noisy", 132, 388, 80, 36, { fontSize: pt(22), color: GRAY });
rect(214, 391, 44, 30, { fill: { color: REF }, line: { color: "888888", width: 1, dashType: "dash" } });
text("Other", 132, 424, 80, 36, { fontSize: pt(22), color: GRAY });
rect(214, 427, 44, 30, { fill: { color: "E3E3E3" }, line: { color: "888888", width: 1, dashType: "dash" } });
text("5 s\nblocks", 290, 404, 80, 56, { fontSize: pt(20), color: GRAY, align: "center" });

line(386, 260, 438, 260, { arrow: true });
line(386, 420, 438, 420, { arrow: true });

// B. encoders
rect(440, 225, 240, 70, solid(MOT));
text([r("Motion encoder", { bold: true, breakLine: true }), r("pose (SMPL-H), trained", { fontSize: pt(20), color: "E8EEF7" })],
  440, 225, 240, 70, { align: "center", color: "FFFFFF", fontSize: pt(26) });
rect(440, 385, 240, 70, solid(APP));
text([r("Appearance enc.", { bold: true, breakLine: true }), r("video, frozen", { fontSize: pt(20), color: "EEF3E6" })],
  440, 385, 240, 70, { align: "center", color: "FFFFFF", fontSize: pt(26) });

line(680, 260, 738, 260, { arrow: true, color: MOT });
text([r("h", { italic: true }), sb("b")], 694, 228, 36, 28, { fontSize: pt(24), color: MOT });
line(680, 420, 738, 420, { arrow: true, color: APP });
text([r("u", { italic: true }), sb("b")], 694, 388, 36, 28, { fontSize: pt(24), color: APP });

// C. trust signals
rect(740, 225, 250, 70, outline(MOT));
text([r("Motion agreement c"), sb("b", { breakLine: true }), r("agrees with other blocks?", { fontSize: pt(19), color: GRAY })],
  740, 225, 250, 70, { align: "center", color: MOT, fontSize: pt(24), bold: false });
rect(740, 385, 250, 70, outline(APP));
text([r("Appearance match κ"), sb("b", { breakLine: true }), r("consistent with the set?", { fontSize: pt(19), color: GRAY })],
  740, 385, 250, 70, { align: "center", color: APP, fontSize: pt(24) });

line(990, 260, 1020, 260); line(1020, 260, 1020, 310); line(1020, 310, 1048, 310, { arrow: true });
line(990, 420, 1020, 420); line(1020, 420, 1020, 370); line(1020, 370, 1048, 370, { arrow: true });

// D. gate
rect(1050, 265, 200, 150, solid(RED));
text([r("Weight α"), sb("b")], 1050, 272, 200, 34, { align: "center", color: "FFFFFF", fontSize: pt(26), bold: true });
text("softmax over blocks", 1050, 304, 200, 26, { align: "center", color: "F6DEDE", fontSize: pt(19) });
[[1078, 44], [1110, 40], [1142, 42], [1174, 38]].forEach(([x, h]) => rect(x, 398 - h, 24, h, solid("F2CACA")));
rect(1206, 393, 24, 5, solid("BBBBBB"));

// motion features bypass the gate
line(560, 225, 560, 196, { color: MOT, dash: true });
line(560, 196, 1340, 196, { color: MOT, dash: true });
line(1340, 196, 1340, 293, { color: MOT, dash: true, arrow: true });
text([r("motion features h"), sb("b"), r(" are the only thing summed")], 700, 162, 630, 28, { fontSize: pt(21), color: MOT, align: "right" });

// E. weighted sum
line(1250, 340, 1293, 340, { arrow: true });
s.addShape(pres.shapes.OVAL, { x: P(1295), y: P(Y(340) - 45), w: P(90), h: P(90), fill: { color: "FFFFFF" }, line: { color: MOT, width: 2 } });
text("Σ", 1295, 297, 86, 86, { align: "center", fontSize: pt(44), color: MOT });
text([r("z = Σ α"), sb("b"), r(" h"), sb("b")], 1250, 392, 180, 30, { align: "center", fontSize: pt(22), italic: true });

// F. generator
text([r("new speech (audio) "), r("a", { italic: true })], 1440, 214, 220, 30, { align: "center", fontSize: pt(22), color: GRAY });
line(1550, 246, 1550, 288, { arrow: true });
line(1385, 340, 1438, 340, { arrow: true });
rect(1440, 290, 220, 100, solid("333333"));
text([r("Generator", { bold: true, breakLine: true }), r("G(a, z)", { fontSize: pt(22), color: "DDDDDD" })],
  1440, 290, 220, 100, { align: "center", color: "FFFFFF", fontSize: pt(28) });
line(1660, 340, 1700, 340, { arrow: true });
text("gestures in\np's style", 1704, 300, 120, 80, { fontSize: pt(22), bold: true });

// stage captions
[["several recordings,\nfixed total D = 20 s", 116, 270],
 ["one embedding\nper block", 440, 240],
 ["two trust signals\nper block", 740, 250],
 ["bad blocks → α ≈ 0", 1040, 220],
 ["style code,\nmotion only", 1250, 180],
 ["speech → motion", 1440, 220]].forEach(([t, x, w]) =>
  text(t, x, 474, w, 52, { fontSize: pt(20), color: GRAY, align: x === 116 ? "left" : "center", valign: "top" }));

line(116, 548, 1810, 548, { color: RULE, width: 1 });

// ---------- three design questions ----------
const CW = 520, CX = [116, 703, 1290];
line(673, 572, 673, 880, { color: RULE, width: 1 });
line(1260, 572, 1260, 880, { color: RULE, width: 1 });
const head = (i, n, q) => {
  text(n, CX[i], 568, 34, 44, { fontSize: pt(34), bold: true, color: RED });
  text(q, CX[i] + 40, 568, CW - 40, 44, { fontSize: pt(28), bold: true });
};
const body = (i, paras, y = 795, h = 190) => {
  const runs = [];
  paras.forEach((p, k) => (Array.isArray(p) ? p : [r(p)]).forEach((run, j, arr) => {
    const last = j === arr.length - 1;
    runs.push({ text: run.text, options: Object.assign({}, run.options, last && k < paras.length - 1 ? { breakLine: true } : {}, { paraSpaceAfter: 6 }) });
  }));
  text(runs, CX[i], y, CW, h, { fontSize: pt(26), valign: "top", lineSpacingMultiple: 1.05 });
};

// 1. where clips come from
head(0, "1", "Which clips go in?");
[["1 × 20 s, 1 rec", [[0, 280, REF]]],
 ["4 × 5 s, 1 rec", [[0, 64, REF], [72, 64, REF], [144, 64, REF], [216, 64, REF]]],
 ["4 × 5 s, 3 rec", [[0, 64, REF], [72, 64, REF], [144, 64, "A9BFD9"], [216, 64, "BFD1A6"]]]]
  .forEach(([lab, bars], k) => {
    const y = 626 + k * 40;
    text(lab, CX[0], y, 190, 34, { fontSize: pt(21), color: GRAY });
    bars.forEach(([dx, w, c]) => rect(CX[0] + 200 + dx, y + 3, w, 28, { fill: { color: c }, line: { color: "999999", width: 0.75 } }));
  });
text("same 20 s budget; color = recording", CX[0] + 200, 746, 320, 26, { fontSize: pt(19), color: GRAY });
body(0, ["Same 20 s from more recordings: the person stays fixed, the conversation varies."]);

// 2. which clips to trust
head(1, "2", "Which clips to trust?");
const blocks = [["F2CACA", 64, "clean"], ["F2CACA", 58, "clean"], ["F2CACA", 62, "clean"], ["F2CACA", 22, "noisy"], ["DDDDDD", 4, "other"]];
blocks.forEach(([c, h, lab], k) => {
  const x = CX[1] + 60 + k * 86;
  rect(x + 14, 692 - h, 36, h, solid(RED));
  rect(x, 700, 64, 30, { fill: { color: c === "DDDDDD" ? "E3E3E3" : REF }, line: { color: "888888", width: 1, dashType: k >= 3 ? "dash" : "solid" } });
  text(lab, x - 6, 734, 76, 24, { fontSize: pt(19), color: GRAY, align: "center" });
});
text("α", CX[1], 640, 40, 40, { fontSize: pt(26), italic: true, color: RED });
body(1, ["Video appearance and motion agreement only set the weights; only motion is summed into z."]);

// 3. train and test
head(2, "3", "How to train and test?");
text([r("L = L"), sb("gen"), r(" + λ"), sb("id"), r(" L"), sb("NCE"), r(" + λ"), sb("q"), r(" Σ"), sb("b"), r(" α"), sb("b"), r("·1[b bad]")],
  CX[2], 630, CW, 40, { fontSize: pt(28) });
[["generate\nmotion", 52], ["same person\nis closer", 190], ["low weight on\nbad blocks", 380]].forEach(([t, x]) =>
  text(t, CX[2] + x - 70, 676, 150, 50, { fontSize: pt(19), color: GRAY, align: "center", valign: "top" }));
body(2, ["Train with injected bad blocks. Test mean / motion-only / cross-modal attention, 1 vs. 3 recordings, 0–50 % bad blocks."], 795, 190);

// ---------- footer ----------
text("TEAM 22  /  11-777 MIDTERM", 116, 1022, 600, 28, { fontSize: pt(17), color: "555555", charSpacing: 0.5 });
text("03 / 06", 1700, 1022, 124, 32, { fontSize: pt(22), color: "555555", align: "right" });

pres.writeFile({ fileName: __dirname + "/idea2_slide.pptx" }).then((f) => console.log("wrote", f));
