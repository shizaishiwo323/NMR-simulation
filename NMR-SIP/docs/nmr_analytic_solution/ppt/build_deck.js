// Builds nmr_analytic_solution.pptx: exact (eigenmode) and fast-diffusion T2 solutions with the simulation parameters.
// Run: node build_deck.js   (after python build_equations.py)
const path = require("path");
const fs = require("fs");
const pptxgen = require("pptxgenjs");
const { applyTheme } = require(process.env.PPTX_SKILL_DIR + "/scripts/apply_theme.js");

const HERE = __dirname;
const EQ_DIR = path.join(HERE, "eq");
const SIZES = JSON.parse(fs.readFileSync(path.join(EQ_DIR, "sizes.json"), "utf8"));
const OUT = path.join(HERE, process.env.DECK_NAME || "nmr_analytic_solution.pptx");

const THEME = {
  name: "NMR Porous Media",
  headFontFace: "Microsoft YaHei",
  bodyFontFace: "Microsoft YaHei",
  colors: {
    dk1: "1A1A1A", lt1: "FFFFFF", dk2: "16384F", lt2: "EEF3F6",
    accent1: "2C6E8F", accent2: "D9895B", accent3: "8FB0CC", accent4: "5B8C5A",
    accent5: "A23B3B", accent6: "6B6B6B", hlink: "2C6E8F", folHlink: "6B6B6B",
  },
};
const HEX = THEME.colors;

const pres = new pptxgen();
pres.layout = "LAYOUT_WIDE"; // 13.333" x 7.5"
pres.title = "理想二维孔隙 NMR T2 解析解与快扩散近似";
pres.theme = { headFontFace: THEME.headFontFace, bodyFontFace: THEME.bodyFontFace };
const C = pres.SchemeColor;
const W = 13.333, M = 0.6;
const EQ_SCALE = 1.9; // formulas typeset at 10 pt -> ~19 pt on the slide

// ---------- layouts ----------
pres.defineSlideMaster({
  title: "TITLE_DARK",
  background: { color: C.text2 },
  objects: [
    { placeholder: { options: { name: "title", type: "title", x: M, y: 2.0, w: 7.6, h: 1.9, fontSize: 36, bold: true, color: C.background1, valign: "bottom", align: "left", margin: 0 }, text: "" } },
    { placeholder: { options: { name: "body", type: "body", x: M, y: 4.15, w: 7.6, h: 1.2, fontSize: 18, color: C.accent3, valign: "top", align: "left", margin: 0 }, text: "" } },
  ],
});
pres.defineSlideMaster({
  title: "CONTENT",
  background: { color: C.background1 },
  margin: [0.5, 0.6, 0.6, 0.6],
  objects: [
    { placeholder: { options: { name: "title", type: "title", x: M, y: 0.35, w: W - 2 * M, h: 0.75, fontSize: 30, bold: true, color: C.text2, valign: "middle", align: "left", margin: 0 }, text: "" } },
    { text: { text: "理想二维孔隙 NMR T2：精确解与快扩散近似", options: { x: M, y: 7.0, w: 7, h: 0.3, fontSize: 10, color: C.accent6, margin: 0 } } },
  ],
  slideNumber: { x: W - M - 0.6, y: 7.0, w: 0.6, h: 0.3, fontSize: 10, color: C.accent6, align: "right" },
});

// ---------- helpers ----------
function eq(slide, key, x, y, scale = EQ_SCALE, name) {
  const [wpt, hpt] = SIZES[key];
  const w = (wpt * scale) / 72, h = (hpt * scale) / 72;
  slide.addImage({ path: path.join(EQ_DIR, key + ".png"), x, y, w, h, altText: name || key, objectName: "eq_" + key });
  return { w, h };
}
function eqHeight(key, scale = EQ_SCALE) { return (SIZES[key][1] * scale) / 72; }
function scaleToFit(keys, maxW, base = EQ_SCALE) {
  const widest = Math.max(...keys.map((k) => SIZES[k][0]));
  return Math.min(base, (maxW * 72) / widest);
}
function card(slide, x, y, w, h, name) {
  slide.addShape(pres.shapes.ROUNDED_RECTANGLE, { x, y, w, h, rectRadius: 0.12, fill: { color: C.background2 }, line: { color: C.background2 }, objectName: name });
}
function badge(slide, x, y, label, color = C.accent1, d = 0.46) {
  slide.addShape(pres.shapes.OVAL, { x, y, w: d, h: d, fill: { color }, line: { color }, objectName: "badge_" + label });
  slide.addText(String(label), { x, y, w: d, h: d, fontSize: 16, bold: true, color: C.background1, align: "center", valign: "middle", margin: 0, isTextBox: true });
}
function label(slide, text, x, y, w, h, opts = {}) {
  slide.addText(text, { x, y, w, h, fontSize: 16, color: C.text1, valign: "middle", margin: 0, isTextBox: true, ...opts });
}
function caption(slide, text, y = 6.45) {
  slide.addText(text, { x: M, y, w: W - 2 * M, h: 0.4, fontSize: 11, italic: true, color: C.accent6, margin: 0, isTextBox: true });
}
function t2(prefix = "", suffix = "") {
  // "T2" with a real subscript, as an array of runs
  return [{ text: prefix + "T" }, { text: "2", options: { subscript: true } }, { text: suffix }];
}

// ---------- 1. title ----------
pres.addSection({ title: "引言" });
let s = pres.addSlide({ masterName: "TITLE_DARK", sectionTitle: "引言" });
s.addText([{ text: "理想二维孔隙中 NMR T" }, { text: "2", options: { subscript: true } }, { text: " 的精确解析解与快扩散近似" }], { placeholder: "title" });
s.addText("以“大孔–狭长喉道–小孔”模拟案例为例：控制方程、求解过程与参数代入", { placeholder: "body" });
// geometry motif (native shapes): big pore, slit throat, small pore
s.addShape(pres.shapes.LINE, { x: 8.75, y: 3.75, w: 4.1, h: 0, line: { color: C.accent2, width: 2.5 }, objectName: "motif_throat" });
s.addShape(pres.shapes.OVAL, { x: 8.75, y: 2.45, w: 2.6, h: 2.6, fill: { color: C.accent1 }, line: { color: C.accent3, width: 1.5 }, objectName: "motif_big_pore" });
s.addShape(pres.shapes.OVAL, { x: 12.0, y: 3.42, w: 0.66, h: 0.66, fill: { color: C.accent1 }, line: { color: C.accent3, width: 1.5 }, objectName: "motif_small_pore" });
s.addNotes("本报告给出理想二维孔隙（大孔–狭长喉道–小孔）中 NMR 横向弛豫 T2 的精确本征模解与快扩散近似，并把数值模拟所用的真实参数逐步代入，便于与 pyGIMLi 数值解对照。");

// ---------- 2. model & parameters ----------
pres.addSection({ title: "模型与方程" });
s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "模型与方程" });
s.addText("物理模型与模拟参数", { placeholder: "title" });
card(s, M, 1.35, 6.0, 4.9, "geometry_card");
// to-scale schematic: period 80 um x 44 um mapped into the card
const gx = M + 0.35, gy = 1.75, gs = 5.3 / 80; // inch per um
s.addShape(pres.shapes.RECTANGLE, { x: gx, y: gy, w: 80 * gs, h: 44 * gs, fill: { color: "E7E3D8" }, line: { color: C.accent6, width: 0.75 }, objectName: "solid" });
s.addShape(pres.shapes.LINE, { x: gx, y: gy + 22 * gs, w: 80 * gs, h: 0, line: { color: C.accent2, width: 2 }, objectName: "throat_line" });
s.addShape(pres.shapes.OVAL, { x: gx + 7.5 * gs, y: gy + 2 * gs, w: 40 * gs, h: 40 * gs, fill: { color: C.accent3 }, line: { color: C.accent1, width: 1 }, objectName: "big_pore" });
s.addShape(pres.shapes.OVAL, { x: gx + 62.5 * gs, y: gy + 17 * gs, w: 10 * gs, h: 10 * gs, fill: { color: C.accent3 }, line: { color: C.accent1, width: 1 }, objectName: "small_pore" });
label(s, "大孔 R = 20 µm", gx + 7.5 * gs, gy + 30 * gs, 40 * gs, 0.35, { align: "center", fontSize: 14, bold: true, color: C.text2 });
label(s, "小孔 R = 5 µm", gx + 55 * gs, gy + 28.5 * gs, 25 * gs, 0.35, { align: "center", fontSize: 14, bold: true, color: C.text2 });
label(s, "喉道 h = 0.05 µm（w = 0.1 µm），L = 15 µm", gx, gy + 44 * gs + 0.15, 80 * gs, 0.35, { align: "center", fontSize: 14, color: C.accent2, bold: true });
label(s, "二维（每单位纵深）：圆盘 = 无限长圆柱截面，狭缝 = 平行平板。线宽未按比例。", M + 0.35, 5.6, 5.3, 0.5, { fontSize: 12, color: C.accent6 });
const hdr = { bold: true, color: C.background1, fill: { color: C.accent1 }, fontSize: 14, align: "center" };
const cell = { fontSize: 14, color: C.text1, align: "center" };
const rows = [
  [{ text: "符号", options: hdr }, { text: "含义", options: hdr }, { text: "取值", options: hdr }, { text: "计算单位", options: hdr }],
  [{ text: "D", options: { ...cell, italic: true } }, { text: "水的自扩散系数", options: cell }, { text: "2.0×10⁻⁹ m²/s", options: cell }, { text: "2.0 µm²/ms", options: cell }],
  [{ text: "ρ₂", options: { ...cell, italic: true } }, { text: "横向表面弛豫率", options: cell }, { text: "5 µm/s", options: cell }, { text: "0.005 µm/ms", options: cell }],
  [{ text: [{ text: "T", options: { italic: true } }, { text: "2B", options: { subscript: true } }], options: cell }, { text: "体弛豫时间", options: cell }, { text: "3 s", options: cell }, { text: "3000 ms", options: cell }],
  [{ text: [{ text: "R", options: { italic: true } }, { text: "B", options: { subscript: true } }], options: cell }, { text: "大孔半径", options: cell }, { text: "20 µm", options: cell }, { text: "—", options: cell }],
  [{ text: [{ text: "R", options: { italic: true } }, { text: "S", options: { subscript: true } }], options: cell }, { text: "小孔半径", options: cell }, { text: "5 µm", options: cell }, { text: "—", options: cell }],
  [{ text: "h", options: { ...cell, italic: true } }, { text: "喉道半宽", options: cell }, { text: "0.05 µm", options: cell }, { text: "—", options: cell }],
];
s.addTable(rows, { x: 7.0, y: 1.35, w: 5.73, colW: [0.8, 1.6, 1.6, 1.73], rowH: 0.52, border: { type: "solid", pt: 0.75, color: "D5DEE5" }, fill: { color: C.background1 }, objectName: "param_table" });
s.addText("参数与 Auto_NMR.py 默认值一致（Zhou & Zhang, 2025）", { x: 7.0, y: 5.15, w: 5.73, h: 0.4, fontSize: 12, italic: true, color: C.accent6, margin: 0, isTextBox: true });
s.addNotes("几何为沿 x 方向的周期链：大孔–喉道–小孔–喉道。这里给出的是孤立孔参照的解析解：大孔、小孔各为二维圆盘，喉道为二维狭缝（平行平板）。所有参数与数值模拟的输入完全一致。");

// ---------- 3. governing equations ----------
s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "模型与方程" });
s.addText("控制方程与边界条件", { placeholder: "title" });
const govern = [
  ["bt", "Bloch–Torrey 扩散–弛豫方程", "体内扩散 + 体弛豫"],
  ["robin", "Robin 表面弛豫边界", "孔壁处扩散通量 = 表面弛豫损失"],
  ["signal", "初始条件与测量信号", "均匀初始磁化，信号为孔内积分"],
];
govern.forEach(([key, head, sub], i) => {
  const y = 1.45 + i * 1.65;
  card(s, M, y, W - 2 * M, 1.4, "gov_card_" + i);
  badge(s, M + 0.3, y + 0.47, i + 1);
  label(s, head, M + 1.0, y + 0.3, 4.2, 0.45, { fontSize: 18, bold: true, color: C.text2 });
  label(s, sub, M + 1.0, y + 0.78, 4.2, 0.4, { fontSize: 14, color: C.accent6 });
  const hgt = eqHeight(key);
  eq(s, key, 6.0, y + (1.4 - hgt) / 2);
});
caption(s, "Torrey (1956)；Senturia & Robinson (1970)；Brownstein & Tarr (1979)", 6.4);
s.addNotes("无外加梯度时，横向磁化满足 Bloch–Torrey 方程。孔壁采用 Robin 边界：法向扩散通量等于表面弛豫率乘以磁化。初始磁化均匀，NMR 测得的是孔内磁化的积分。");

// ---------- 4. separation of variables ----------
s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "模型与方程" });
s.addText("分离变量：本征模展开", { placeholder: "title" });
card(s, M, 1.35, 3.6, 4.9, "kappa_card");
label(s, "无量纲表面弛豫数", M + 0.3, 1.6, 3.0, 0.4, { fontSize: 16, bold: true, color: C.text2 });
eq(s, "kappa", M + 0.3, 2.15, 2.6);
label(s, "a：特征长度\n圆盘 a = R，狭缝 a = h", M + 0.3, 3.55, 3.0, 0.8, { fontSize: 14, color: C.accent6, valign: "top" });
label(s, "κ ≪ 1：快扩散区\nκ ≫ 1：慢扩散区", M + 0.3, 4.6, 3.0, 0.9, { fontSize: 16, bold: true, color: C.accent1, valign: "top" });
const sepSteps = [["sep", "体弛豫因子分离，余下为本征值问题"], ["eig", "本征值问题（同一 Robin 边界）"], ["modes", "多指数衰减：第 n 模态速率与权重"]];
sepSteps.forEach(([key, head], i) => {
  const y = 1.35 + i * 1.65;
  badge(s, 4.65, y + 0.05, i + 1, C.accent2, 0.4);
  label(s, head, 5.2, y, 7.5, 0.5, { fontSize: 16, bold: true, color: C.text2 });
  eq(s, key, 5.2, y + 0.55, scaleToFit(["sep", "eig", "modes"], 7.4));
});
caption(s, "Brownstein & Tarr (1979)；Grebenkov (2007)", 6.4);
s.addNotes("令 M 等于体弛豫因子乘以 m，体弛豫完全分离。m 满足纯扩散方程和同样的 Robin 边界，展开为本征模之和。每个模态的弛豫速率为体弛豫速率加 D ξn²/a²，权重 An 之和为 1。κ 决定处于快扩散还是慢扩散区。");

// ---------- 5. exact solutions ----------
pres.addSection({ title: "精确解与快扩散近似" });
s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "精确解与快扩散近似" });
s.addText("精确解：圆盘（孔体）与狭缝（喉道）", { placeholder: "title" });
const colW = (W - 2 * M - 0.4) / 2;
const exactScale = scaleToFit(["disk_phi", "disk_char", "disk_A", "slab_phi", "slab_char", "slab_A"], colW - 0.6);
[["二维圆盘（大孔、小孔）", ["disk_phi", "disk_char", "disk_A"], C.accent1], ["二维狭缝（喉道，平行平板）", ["slab_phi", "slab_char", "slab_A"], C.accent2]].forEach(([head, keys, color], j) => {
  const x = M + j * (colW + 0.4);
  card(s, x, 1.35, colW, 4.95, "exact_card_" + j);
  s.addShape(pres.shapes.OVAL, { x: x + 0.3, y: 1.6, w: 0.32, h: 0.32, fill: { color }, line: { color }, objectName: "dot_" + j });
  label(s, head, x + 0.75, 1.52, colW - 1.0, 0.48, { fontSize: 18, bold: true, color: C.text2 });
  ["本征函数", "特征方程（数值求根）", "模态权重"].forEach((sub, i) => {
    const y = 2.2 + i * 1.35;
    label(s, sub, x + 0.3, y, colW - 0.6, 0.35, { fontSize: 13, color: C.accent6 });
    eq(s, keys[i], x + 0.3, y + 0.38, exactScale);
  });
});
caption(s, "Brownstein & Tarr (1979)；Carslaw & Jaeger (1959)", 6.45);
s.addNotes("圆盘的轴对称本征函数为 J0，由 J0' = −J1 得到特征方程 ξ J1/J0 = κ；狭缝的偶对称本征函数为余弦，特征方程为 ξ tan ξ = κ。两式都需数值求根，最低根 ξ0 决定主模态。");

// ---------- 6. fast diffusion ----------
s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "精确解与快扩散近似" });
s.addText("快扩散极限：κ ≪ 1", { placeholder: "title" });
label(s, "圆盘：", M, 1.45, 1.0, 0.5, { fontSize: 16, bold: true, color: C.accent1 });
eq(s, "fd_disk", M + 1.0, 1.35);
label(s, "狭缝：", M, 2.4, 1.0, 0.5, { fontSize: 16, bold: true, color: C.accent2 });
eq(s, "fd_slab", M + 1.0, 2.3);
card(s, M, 3.25, W - 2 * M, 1.5, "fd_main_card");
eq(s, "fd_main", (W - (SIZES.fd_main[0] * EQ_SCALE) / 72) / 2, 3.25 + (1.5 - eqHeight("fd_main")) / 2);
label(s, "由 T2 反算尺寸：", M, 5.05, 2.4, 0.5, { fontSize: 16, bold: true, color: C.text2 });
eq(s, "inverse", M + 2.4, 4.95);
label(s, "同一 T2：狭缝若按圆盘公式换算会得到 R = 2h", M, 5.95, W - 2 * M, 0.4, { fontSize: 14, color: C.accent2 });
caption(s, "Brownstein & Tarr (1979)；Kleinberg et al. (1994)；Coates et al. (1999)", 6.45);
s.addNotes("κ 很小时，对特征方程做级数展开：圆盘 ξ0² ≈ 2κ(1 − κ/4)，狭缝 ξ0² ≈ κ(1 − κ/3)。保留首项即经典快扩散关系 1/T2 = 1/T2B + ρ2 S/V。反过来可由 T2 换算孔隙尺寸；注意狭缝的 S/V 是 1/h，同一 T2 按圆盘公式会被放大为 2h。");

// ---------- 7-9. worked examples (same step layout) ----------
function stepSlide(title, steps, notes) {
  const sl = pres.addSlide({ masterName: "CONTENT", sectionTitle: "代入参数求解" });
  sl.addText(title, { placeholder: "title" });
  const pitch = steps.length > 4 ? 0.9 : 1.2;
  const sc = scaleToFit(steps.map((r) => r[0]), W - 2 * M - 2.8);
  steps.forEach(([key, head, isApprox], i) => {
    const y = 1.3 + i * pitch;
    badge(sl, M, y + 0.13, i + 1, isApprox ? C.accent2 : C.accent1, 0.42);
    label(sl, head, M + 0.6, y + 0.1, 2.1, 0.48, { fontSize: 15, bold: true, color: C.text2 });
    eq(sl, key, M + 2.8, y + (0.68 - eqHeight(key, sc)) / 2 + 0.03, sc);
  });
  sl.addNotes(notes);
  return sl;
}
pres.addSection({ title: "代入参数求解" });
stepSlide("代入求解：大孔 R = 20 µm",
  [["b1", "判断扩散区"], ["b2", "特征方程求根"], ["b3", "表面弛豫"], ["b4", "总弛豫"], ["b5", "模态权重"], ["b6", "快扩散近似", true]],
  "大孔：κ = 0.05，处于快扩散区。数值求解特征方程得 ξ0 = 0.314262，表面弛豫 T2S = 2025.10 ms，计入体弛豫后 T2 = 1208.99 ms。主模态权重 0.99995，信号近似单指数。快扩散近似给出 1200.0 ms，偏差 −0.74%；加一阶修正 (1 − κ/4) 后偏差降至 0.01%。");
stepSlide("代入求解：小孔 R = 5 µm",
  [["s1", "判断扩散区与求根"], ["s2", "表面弛豫"], ["s3", "总弛豫"], ["s4", "快扩散近似", true]],
  "小孔：κ = 0.0125，ξ0 = 0.157867，T2S = 501.56 ms，T2 = 429.72 ms；主模态权重 0.999997。快扩散近似 428.57 ms，偏差 −0.27%。");
stepSlide("代入求解：喉道 h = 0.05 µm",
  [["t1", "判断扩散区与求根"], ["t2", "表面弛豫"], ["t3", "总弛豫"], ["t4", "快扩散近似", true]],
  "喉道（狭缝）：κ = 1.25×10⁻⁴，几乎完全处于快扩散极限，ξ0 = 0.0111801，T2S = 10.0004 ms，T2 = 9.9672 ms。快扩散近似 9.9668 ms，偏差只有 −0.004%。注意狭缝的 S/V = 1/h。");

// ---------- 10. size back-calculation from simulated T2 ----------
s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "代入参数求解" });
s.addText("由模拟 T2 反算孔隙尺寸", { placeholder: "title" });
card(s, M, 1.3, W - 2 * M, 1.3, "inverse_formula_card");
const genScale = scaleToFit(["inv_general"], W - 2 * M - 0.8);
eq(s, "inv_general", M + 0.4, 1.3 + (1.3 - eqHeight("inv_general", genScale)) / 2, genScale);
const invRows = [["inv_b", "大孔", "单指数拟合 T2 = 1209.26 ms"], ["inv_s", "小孔", "单指数拟合 T2 = 429.83 ms"], ["inv_t", "喉道", "单指数拟合 T2 = 10.03 ms"]];
const invScale = scaleToFit(invRows.map((r) => r[0]), W - 2 * M - 3.2);
invRows.forEach(([key, head, sub], i) => {
  const y = 2.85 + i * 0.92;
  badge(s, M, y + 0.13, i + 1, i === 2 ? C.accent2 : C.accent1, 0.42);
  label(s, head, M + 0.6, y + 0.02, 2.4, 0.36, { fontSize: 15, bold: true, color: C.text2 });
  label(s, sub, M + 0.6, y + 0.38, 2.4, 0.32, { fontSize: 12, color: C.accent6 });
  eq(s, key, M + 3.2, y + (0.68 - eqHeight(key, invScale)) / 2 + 0.03, invScale);
});
card(s, M, 5.7, W - 2 * M, 1.05, "connected_card");
s.addText([
  { text: "连通链对照（L-curve 反演峰）：", options: { bold: true, color: C.accent2 } },
  { text: "1097 ms → R = 17.3 µm（−14%）；274 ms → R = 3.0 µm（−40%）。孔间扩散耦合与喉道壁弛豫汇使 T2 变短，按孤立孔公式反算会低估孔径。" },
], { x: M + 0.3, y: 5.82, w: W - 2 * M - 0.6, h: 0.8, fontSize: 14, color: C.text1, valign: "middle", margin: 0, isTextBox: true });
s.addNotes("这里的模拟 T2 是对每个孤立孔自身回波衰减做单指数拟合（M/M0 在 0.9–0.05 之间的 ln M 线性拟合）得到的，没有经过 NNLS 反演；孤立圆孔主模态权重 > 0.9999，可视为单指数。已知 T2，用快扩散关系反解尺寸：圆盘 R = 2ρ2/(1/T2 − 1/T2B)，狭缝 h = ρ2/(1/T2 − 1/T2B)，必须先扣除体弛豫。孤立孔的模拟 T2 反算得 20.26、5.017、0.0503 µm，偏差 +1.3%、+0.3%、+0.6%，其中大孔的 +1.3% 主要是快扩散近似本身的误差（κ/4 ≈ 1.25%）。连通链中两峰按同一公式反算为 17.3 和 3.0 µm，明显偏小，说明连通体系不能直接套用孤立孔公式。");

// ---------- 11. why the back-calculated sizes differ: four T2 extraction routes ----------
s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "代入参数求解" });
s.addText("反算尺寸为何不同：四种 T2 提取方式", { placeholder: "title" });
const h3 = { ...hdr, fontSize: 13 };
const c3 = { ...cell, fontSize: 13 };
const routeRows = [
  [{ text: "孔隙（真值）", options: h3 }, { text: "① 直接单指数拟合", options: h3 }, { text: "② 单独 NNLS\n主峰 / T₂lm", options: h3 }, { text: "③ 合成信号 NNLS\n主峰 / T₂lm", options: h3 }, { text: "④ 连通链 NNLS\n主峰 / T₂lm", options: h3 }],
  [{ text: "大孔（20 µm）", options: c3 }, { text: "20.26（+1.3%）", options: c3 }, { text: "20.89 / 20.24（+4.5% / +1.2%）", options: c3 }, { text: "20.89 / 19.88（+4.5% / −0.6%）", options: c3 }, { text: "17.29 / 17.71（−13.5% / −11.4%）", options: c3 }],
  [{ text: "小孔（5 µm）", options: c3 }, { text: "5.017（+0.3%）", options: c3 }, { text: "5.08 / 5.013（+1.7% / +0.3%）", options: c3 }, { text: "4.16 / 4.19（−16.8% / −16.2%）", options: c3 }, { text: "3.01 / 2.96（−39.8% / −40.9%）", options: c3 }],
  [{ text: "喉道（0.05 µm）", options: c3 }, { text: "0.0503（+0.6%）", options: c3 }, { text: "0.0508 / 0.0505（+1.5% / +1.1%）", options: c3 }, { text: "未分辨", options: c3 }, { text: "未分辨", options: c3 }],
];
s.addTable(routeRows, { x: M, y: 1.25, w: W - 2 * M, colW: [1.9, 2.05, 2.7, 2.7, 2.783], rowH: [0.62, 0.62, 0.62, 0.62], border: { type: "solid", pt: 0.75, color: "D5DEE5" }, fill: { color: C.background1 }, objectName: "route_table" });
label(s, "单位 µm；括号内为相对真值偏差。孔体按圆盘 R = 2ρ₂/(1/T2 − 1/T2B)，喉道按狭缝 h = ρ₂/(1/T2 − 1/T2B)。", M, 3.82, W - 2 * M, 0.35, { fontSize: 12, color: C.accent6 });
const routes = [
  ["① 直接拟合", "孤立孔近单指数（A₀ > 0.9999），偏差只来自快扩散近似（大孔 κ/4 ≈ 1.25%）。"],
  ["② 单独 NNLS", "T₂lm 与直接拟合一致；主峰取 T2 分箱中心（每箱约 2.4%），略偏大。"],
  ["③ 合成信号 NNLS", "小孔只占 5.9%、与大孔 T2 仅差 2.8 倍，正则化平滑把小峰推向短 T2：属反演偏差，非物理效应。"],
  ["④ 连通链 NNLS", "再叠加孔间扩散耦合与喉道壁弛豫汇（小孔区直接拟合 T2 = 312 ms），孔径被明显低估。"],
];
routes.forEach(([head, body], i) => {
  const w = (W - 2 * M - 0.9) / 4;
  const x = M + i * (w + 0.3);
  card(s, x, 4.3, w, 2.3, "route_card_" + i);
  label(s, head, x + 0.25, 4.45, w - 0.5, 0.42, { fontSize: 15, bold: true, color: i === 3 ? C.accent2 : C.text2 });
  s.addText(body, { x: x + 0.25, y: 4.95, w: w - 0.5, h: 1.55, fontSize: 13, color: C.text1, valign: "top", margin: 0, isTextBox: true });
});
s.addNotes("同一组模拟回波，用四种方式提取 T2 再反算尺寸。①直接对孤立孔回波做单指数拟合，误差只来自快扩散近似。②单独对每个孤立孔的回波做 NNLS + L-curve 反演：T2lm 与直接拟合几乎相同，主峰因取分箱中心略偏大。③把三个孤立孔的回波按面积相加后再反演：小孔分量只占 5.9%，又紧挨着占 94% 的大孔峰，正则化平滑会把小峰推向短 T2，孔径偏小约 16%，这是反演方法造成的偏差。④连通链：在反演偏差之外，还有孔间扩散耦合和喉道壁的弛豫汇，小孔区直接拟合的 T2 已降到 312 ms，反算孔径偏小约 40%。喉道在③④中占比仅 0.2%，没有被分辨出来。");

// ---------- 12. summary vs simulation ----------
s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "代入参数求解" });
s.addText("汇总：精确解、快扩散近似与数值模拟", { placeholder: "title" });
const h2 = { ...hdr, fontSize: 12 };
const c2 = { ...cell, fontSize: 13 };
const sumRows = [
  [{ text: "孔隙", options: h2 }, { text: "精确 T₂\n(ms)", options: h2 }, { text: "快扩散 T₂\n(ms，偏差)", options: h2 }, { text: "数值模拟 T₂\n单指数拟合 (ms)", options: h2 }, { text: "反算尺寸\n(µm)", options: h2 }, { text: "数值模拟 T₂\nNNLS 主峰 (ms)", options: h2 }, { text: "反算尺寸\n(µm)", options: h2 }, { text: "数值模拟 T₂\nNNLS T₂lm (ms)", options: h2 }, { text: "反算尺寸\n(µm)", options: h2 }],
  [{ text: "大孔 R = 20 µm", options: c2 }, { text: "1208.99", options: c2 }, { text: "1200.00（−0.74%）", options: c2 }, { text: "1209.26", options: c2 }, { text: "20.26（+1.3%）", options: c2 }, { text: "1231.55", options: c2 }, { text: "20.89（+4.5%）", options: c2 }, { text: "1208.73", options: c2 }, { text: "20.24（+1.2%）", options: c2 }],
  [{ text: "小孔 R = 5 µm", options: c2 }, { text: "429.72", options: c2 }, { text: "428.57（−0.27%）", options: c2 }, { text: "429.83", options: c2 }, { text: "5.017（+0.3%）", options: c2 }, { text: "434.70", options: c2 }, { text: "5.08（+1.7%）", options: c2 }, { text: "429.51", options: c2 }, { text: "5.013（+0.3%）", options: c2 }],
  [{ text: "喉道 h = 0.05 µm", options: c2 }, { text: "9.9672", options: c2 }, { text: "9.9668（−0.004%）", options: c2 }, { text: "10.03", options: c2 }, { text: "0.0503（+0.6%）", options: c2 }, { text: "10.12", options: c2 }, { text: "0.0508（+1.5%）", options: c2 }, { text: "10.07", options: c2 }, { text: "0.0505（+1.1%）", options: c2 }],
];
s.addTable(sumRows, { x: M, y: 1.25, w: W - 2 * M, colW: [1.85, 0.95, 1.55, 1.25, 1.3, 1.25, 1.3, 1.35, 1.333], rowH: [0.62, 0.62, 0.62, 0.62], margin: 0.04, border: { type: "solid", pt: 0.75, color: "D5DEE5" }, fill: { color: C.background1 }, objectName: "summary_table" });
label(s, "单指数拟合：孤立孔自身回波 ln M 线性拟合（M/M0 = 0.9–0.05）；NNLS：孤立孔自身回波单独做 L-curve NNLS 反演（SNR 1000），T₂lm 为该峰的对数平均。", M, 3.82, W - 2 * M, 0.4, { fontSize: 12, color: C.accent6 });
const stats = [["≤ 0.74%", "快扩散近似相对精确解的最大偏差（κ ≤ 0.05）"], ["≤ 0.03%", "孔体：pyGIMLi 数值解相对精确解"], ["0.6%", "喉道：主要来自隐式 Euler 时间步截断（Δt/T2 ≈ 2.5%）"]];
stats.forEach(([big, small], i) => {
  const x = M + i * ((W - 2 * M + 0.3) / 3);
  const w = (W - 2 * M - 0.6) / 3;
  card(s, x, 4.35, w, 2.1, "stat_card_" + i);
  s.addText(big, { x: x + 0.3, y: 4.5, w: w - 0.6, h: 0.85, fontSize: 40, bold: true, color: i === 2 ? C.accent2 : C.accent1, margin: 0, isTextBox: true });
  s.addText(small, { x: x + 0.3, y: 5.4, w: w - 0.6, h: 0.9, fontSize: 14, color: C.text1, valign: "top", margin: 0, isTextBox: true });
});
s.addNotes("三种孔隙的精确解、快扩散近似和数值模拟逐一对照。快扩散近似最大偏差 0.74%（大孔）；数值解与精确解在孔体上相差不超过 0.03%，喉道 0.6%，主要是 0.25 ms 时间步的隐式 Euler 截断误差。");

// ---------- 12. applicability ----------
s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "代入参数求解" });
s.addText("适用边界", { placeholder: "title" });
const limits = [
  ["孤立孔解", "连通体系中喉道的扩散交换与喉道壁“弛豫汇”会缩短小孔 T2（模拟中 430 → 312 ms），不能直接套用孤立孔公式换算尺寸。"],
  ["二维 vs 三维", "二维圆盘 = 无限长圆柱，S/V = 2/R；三维球形孔 S/V = 3/R，换算系数不同。"],
  ["快扩散判据", "κ = ρ₂a/D ≪ 1；本例 κ ≤ 0.05，快扩散近似误差 < 1%。"],
];
limits.forEach(([head, body], i) => {
  const w = (W - 2 * M - 0.6) / 3;
  const x = M + i * (w + 0.3);
  card(s, x, 1.45, w, 3.2, "limit_card_" + i);
  badge(s, x + 0.35, 1.8, i + 1, C.accent2, 0.5);
  label(s, head, x + 1.0, 1.8, w - 1.3, 0.5, { fontSize: 20, bold: true, color: C.text2 });
  s.addText(body, { x: x + 0.35, y: 2.6, w: w - 0.7, h: 1.8, fontSize: 16, color: C.text1, valign: "top", margin: 0, isTextBox: true });
});
card(s, M, 4.95, W - 2 * M, 1.3, "fd_error_card");
s.addText([
  { text: "反算误差的来源：", options: { bold: true, color: C.accent1 } },
  { text: "孤立孔反算偏差 ≤ 1.3%，大孔的 +1.3% 即快扩散近似自身误差（κ/4 ≈ 1.25%）；如需更高精度，可改用精确特征方程或一阶修正 (1 − κ/4)。" },
], { x: M + 0.35, y: 5.05, w: W - 2 * M - 0.7, h: 1.1, fontSize: 16, color: C.text1, valign: "middle", margin: 0, isTextBox: true });
s.addNotes("需要注意三点：孤立孔与连通体系不同；二维与三维的 S/V 系数不同；快扩散判据 κ ≪ 1。孤立孔反算误差不超过 1.3%，来源就是快扩散近似本身。");

// ---------- 11. references ----------
pres.addSection({ title: "参考文献" });
s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "参考文献" });
s.addText("参考文献", { placeholder: "title" });
const refs = [
  [["Brownstein, K. R., & Tarr, C. E. (1979). Importance of classical diffusion in NMR studies of water in biological cells. "], ["Physical Review A", true], [", 19(6), 2446–2453. https://doi.org/10.1103/PhysRevA.19.2446"]],
  [["Carslaw, H. S., & Jaeger, J. C. (1959). "], ["Conduction of heat in solids", true], [" (2nd ed.). Oxford, UK: Clarendon Press."]],
  [["Coates, G. R., Xiao, L., & Prammer, M. G. (1999). "], ["NMR logging: Principles and applications", true], [". Houston, TX: Halliburton Energy Services."]],
  [["Grebenkov, D. S. (2007). NMR survey of reflected Brownian motion. "], ["Reviews of Modern Physics", true], [", 79(3), 1077–1137. https://doi.org/10.1103/RevModPhys.79.1077"]],
  [["Kleinberg, R. L., Kenyon, W. E., & Mitra, P. P. (1994). Mechanism of NMR relaxation of fluids in rock. "], ["Journal of Magnetic Resonance, Series A", true], [", 108(2), 206–214. https://doi.org/10.1006/jmra.1994.1112"]],
  [["Senturia, S. D., & Robinson, J. D. (1970). Nuclear spin-lattice relaxation of liquids confined in porous solids. "], ["Society of Petroleum Engineers Journal", true], [", 10(3), 237–244. https://doi.org/10.2118/2870-PA"]],
  [["Torrey, H. C. (1956). Bloch equations with diffusion terms. "], ["Physical Review", true], [", 104(3), 563–565. https://doi.org/10.1103/PhysRev.104.563"]],
  [["Zhou, & Zhang (2025). How does pore structure affect the NMR relaxation in unsaturated porous media. "], ["Water Resources Research", true], [".（作者全名、卷期与 DOI 待核对）"]],
];
const runs = [];
refs.forEach((parts, i) => parts.forEach(([text, italic], k) => runs.push({ text, options: { italic: !!italic, breakLine: k === parts.length - 1 && i < refs.length - 1, paraSpaceAfter: 6 } })));
s.addText(runs, { x: M, y: 1.3, w: W - 2 * M, h: 5.5, fontSize: 13, color: C.text1, valign: "top", margin: 0, indentLevel: 0, isTextBox: true });
s.addNotes("参考文献按 AGU/GRL 作者–年份格式排列。Zhou & Zhang (2025) 的作者全名、卷期与 DOI 需按原文核对。");

(async () => {
  await pres.writeFile({ fileName: OUT });
  await applyTheme(OUT, THEME);
  console.log("wrote", path.basename(OUT));
})();
