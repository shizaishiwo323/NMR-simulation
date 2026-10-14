// Builds coupled_pore_backcalc.pptx: back-calculating pore sizes from NMR T2 in coupled big/small pores.
// Run: python build_equations.py && node build_deck.js   (numbers from ../numbers.json via make_numbers.py)
const path = require("path");
const fs = require("fs");
const pptxgen = require("pptxgenjs");
const { applyTheme } = require(process.env.PPTX_SKILL_DIR + "/scripts/apply_theme.js");

const HERE = __dirname;
const EQ_DIR = path.join(HERE, "eq");
const FIG_DIR = path.join(HERE, "..", "figures");
const SIZES = JSON.parse(fs.readFileSync(path.join(EQ_DIR, "sizes.json"), "utf8"));
// numbers.json is written by Python json and may contain NaN (no peak detected) -> map to null
const NUM = JSON.parse(fs.readFileSync(path.join(HERE, "..", "numbers.json"), "utf8").replace(/\bNaN\b/g, "null"));
const OUT = path.join(HERE, process.env.DECK_NAME || "coupled_pore_backcalc.pptx");

const THEME = {
  name: "NMR Porous Media",
  headFontFace: "Microsoft YaHei",
  bodyFontFace: "Microsoft YaHei",
  colors: {
    dk1: "1A1A1A", lt1: "FFFFFF", dk2: "16384F", lt2: "EEF3F6",
    accent1: "2C6E8F", accent2: "D9895B", accent3: "8FB0CC", accent4: "5B8C5A",
    accent5: "7A4F9A", accent6: "6B6B6B", hlink: "2C6E8F", folHlink: "6B6B6B",
  },
};

const pres = new pptxgen();
pres.layout = "LAYOUT_WIDE"; // 13.333" x 7.5"
pres.title = "耦合大小孔体系中 NMR T2 的孔径反算";
pres.theme = { headFontFace: THEME.headFontFace, bodyFontFace: THEME.bodyFontFace };
const C = pres.SchemeColor;
const W = 13.333, M = 0.6, CW = W - 2 * M;
const EQ_SCALE = 1.9;
const REG = { I: C.accent1, II: C.accent2, III: C.accent5 };

// ---------- layouts ----------
pres.defineSlideMaster({
  title: "TITLE_DARK",
  background: { color: C.text2 },
  objects: [
    { placeholder: { options: { name: "title", type: "title", x: M, y: 1.8, w: 7.9, h: 2.1, fontSize: 34, bold: true, color: C.background1, valign: "bottom", align: "left", margin: 0 }, text: "" } },
    { placeholder: { options: { name: "body", type: "body", x: M, y: 4.15, w: 7.9, h: 1.3, fontSize: 18, color: C.accent3, valign: "top", align: "left", margin: 0 }, text: "" } },
  ],
});
pres.defineSlideMaster({
  title: "CONTENT",
  background: { color: C.background1 },
  margin: [0.5, 0.6, 0.6, 0.6],
  objects: [
    { placeholder: { options: { name: "title", type: "title", x: M, y: 0.35, w: CW, h: 0.75, fontSize: 30, bold: true, color: C.text2, valign: "middle", align: "left", margin: 0 }, text: "" } },
    { text: { text: "耦合大小孔 NMR T2 的孔径反算", options: { x: M, y: 7.0, w: 7, h: 0.3, fontSize: 10, color: C.accent6, margin: 0 } } },
  ],
  slideNumber: { x: W - M - 0.6, y: 7.0, w: 0.6, h: 0.3, fontSize: 10, color: C.accent6, align: "right" },
});

// ---------- helpers ----------
function pngSize(file) {
  const b = fs.readFileSync(file);
  return [b.readUInt32BE(16), b.readUInt32BE(20)];
}
function fig(slide, name, x, y, w, h, alt) {
  // fit inside (w, h) keeping aspect ratio, top-left aligned at (x, y)
  const [pw, ph] = pngSize(path.join(FIG_DIR, name));
  const s = Math.min(w / pw, h / ph);
  const fw = pw * s, fh = ph * s;
  slide.addImage({ path: path.join(FIG_DIR, name), x, y, w: fw, h: fh, altText: alt || name, objectName: "fig_" + name.replace(".png", "") });
  return { w: fw, h: fh };
}
function eq(slide, key, x, y, scale = EQ_SCALE) {
  const [wpt, hpt] = SIZES[key];
  const w = (wpt * scale) / 72, h = (hpt * scale) / 72;
  slide.addImage({ path: path.join(EQ_DIR, key + ".png"), x, y, w, h, altText: key, objectName: "eq_" + key });
  return { w, h };
}
const eqH = (key, scale = EQ_SCALE) => (SIZES[key][1] * scale) / 72;
const eqW = (key, scale = EQ_SCALE) => (SIZES[key][0] * scale) / 72;
function fitScale(keys, maxW, base = EQ_SCALE) {
  return Math.min(base, (maxW * 72) / Math.max(...keys.map((k) => SIZES[k][0])));
}
function card(slide, x, y, w, h, name, color = C.background2) {
  slide.addShape(pres.shapes.ROUNDED_RECTANGLE, { x, y, w, h, rectRadius: 0.12, fill: { color }, line: { color }, objectName: name });
}
function badge(slide, x, y, label, color = C.accent1, d = 0.46) {
  slide.addShape(pres.shapes.OVAL, { x, y, w: d, h: d, fill: { color }, line: { color }, objectName: "badge_" + label });
  slide.addText(String(label), { x, y, w: d, h: d, fontSize: 16, bold: true, color: C.background1, align: "center", valign: "middle", margin: 0, isTextBox: true });
}
// "r_S", "Ψ_eff", "T_{2,f}" -> runs with real subscripts
function rt(str, base = {}) {
  if (typeof str !== "string") return str;
  const runs = [], re = /_(\{[^}]*\}|[A-Za-z0-9]+)/g;
  let last = 0, m;
  while ((m = re.exec(str))) {
    if (m.index > last) runs.push({ text: str.slice(last, m.index), options: { ...base } });
    runs.push({ text: m[1].replace(/[{}]/g, ""), options: { ...base, subscript: true } });
    last = re.lastIndex;
  }
  if (last < str.length) runs.push({ text: str.slice(last), options: { ...base } });
  return runs;
}
function label(slide, text, x, y, w, h, opts = {}) {
  slide.addText(rt(text), { x, y, w, h, fontSize: 16, color: C.text1, valign: "middle", margin: 0, isTextBox: true, ...opts });
}
function body(slide, text, x, y, w, h, opts = {}) {
  slide.addText(rt(text), { x, y, w, h, fontSize: 14, color: C.text1, valign: "top", margin: 0, isTextBox: true, ...opts });
}
function caption(slide, text, y = 6.5, sub = false) {
  slide.addText(sub ? rt(text) : text, { x: M, y, w: CW, h: 0.35, fontSize: 11, italic: true, color: C.accent6, margin: 0, isTextBox: true });
}
function content(title, section) {
  const s = pres.addSlide({ masterName: "CONTENT", sectionTitle: section });
  s.addText(title, { placeholder: "title" });
  return s;
}
function stepRows(slide, rows, y0, pitch, labelW = 2.3, stacked = false) {
  // stacked: heading on its own line, equation below at full width (for long substituted lines)
  const sc = fitScale(rows.map((r) => r[0]), stacked ? CW - 0.65 : CW - labelW - 0.65);
  rows.forEach(([key, head, color], i) => {
    const y = y0 + i * pitch;
    badge(slide, M, y + 0.12, i + 1, color || C.accent1, 0.42);
    label(slide, head, M + 0.6, y + 0.07, stacked ? 6 : labelW - 0.1, 0.52, { fontSize: 15, bold: true, color: C.text2 });
    if (stacked) eq(slide, key, M + 0.6, y + 0.66, sc);
    else eq(slide, key, M + labelW + 0.6, y + (0.66 - eqH(key, sc)) / 2, sc);
  });
}
const HDR = { bold: true, color: C.background1, fill: { color: C.accent1 }, fontSize: 13, align: "center", valign: "middle" };
const CELL = { fontSize: 13, color: C.text1, align: "center", valign: "middle" };
const cell = (t, opts = CELL) => ({ text: rt(t), options: opts });
const f1 = (x) => Number(x).toFixed(1), f2 = (x) => Number(x).toFixed(2), f4 = (x) => Number(x).toFixed(4);
const ampTxt = (x) => (Math.abs(x) >= 1e-3 ? f4(x) : Number(x).toExponential(1).replace("e-", "×10⁻").replace(/⁻(\d)$/, (m, d) => "⁻" + "⁰¹²³⁴⁵⁶⁷⁸⁹"[d]));
const CASES = ["C1", "C2", "C3", "C4", "C5"].map((k) => NUM.cases[k]);
const BACK = NUM.back;

// ---------- 1. title ----------
pres.addSection({ title: "引言" });
let s = pres.addSlide({ masterName: "TITLE_DARK", sectionTitle: "引言" });
s.addText([{ text: "耦合大小孔体系中 NMR T" }, { text: "2", options: { subscript: true } }, { text: " 的孔径反算" }], { placeholder: "title" });
s.addText("分区判据、孔–喉–孔耦合模型与反算流程（20 个二维理想算例）", { placeholder: "body" });
s.addShape(pres.shapes.LINE, { x: 9.0, y: 3.75, w: 3.6, h: 0, line: { color: C.accent2, width: 2.5 }, objectName: "motif_throat" });
s.addShape(pres.shapes.OVAL, { x: 8.75, y: 2.45, w: 2.6, h: 2.6, fill: { color: C.accent1 }, line: { color: C.accent3, width: 1.5 }, objectName: "motif_big_pore" });
s.addShape(pres.shapes.OVAL, { x: 12.0, y: 3.42, w: 0.66, h: 0.66, fill: { color: C.accent1 }, line: { color: C.accent3, width: 1.5 }, objectName: "motif_small_pore" });
s.addNotes("本报告回答三个问题：大小孔扩散耦合时如何由 T2 谱反算孔径；为什么有的算例两个峰、有的一个峰；为什么有时小孔被忽略。基准是 20 个二维理想算例的 NNLS 反演结果。");

// ---------- 2. questions ----------
s = content("三个问题与结论", "引言");
const qs = [
  ["如何反算原始孔径？", "两峰时：振幅比（区 I）或已知喉道的耦合模型反演（区 I–II），可恢复 r_S ≈ 4.6–4.8 µm、r_B ≈ 19.6–20.1 µm（真值 5 / 20）。"],
  ["为什么有时两峰、有时一峰？", "由有效交换数 Ψ_eff 决定：小孔模态保留比 η = A_f/p_S 随 Ψ_eff 增大而连续下降（快交换时 ∝ Ψ_eff⁻²），分为区 I / II / III。"],
  ["为什么小孔会被忽略？", "快交换把小孔信号并入主峰，主峰只给出等效半径 R_eff = (r_B² + r_S²)/(r_B + r_S)；低 SNR 的正则化再抹掉残余小峰。"],
];
qs.forEach(([q, a], i) => {
  const w = (CW - 0.6) / 3, x = M + i * (w + 0.3);
  card(s, x, 1.4, w, 4.9, "question_card_" + i);
  badge(s, x + 0.35, 1.7, i + 1, C.accent2, 0.5);
  label(s, q, x + 0.35, 2.4, w - 0.7, 0.9, { fontSize: 20, bold: true, color: C.text2, valign: "top" });
  body(s, a, x + 0.35, 3.45, w - 0.7, 2.7, { fontSize: 17 });
});
s.addNotes("三个问题对应三部分：分区判据、反算方法、以及小孔被忽略的物理和测量原因。每张卡片下方是本报告给出的结论。");

// ---------- 3. model & parameters ----------
pres.addSection({ title: "模型与基准" });
s = content("几何、参数与 20 个算例", "模型与基准");
card(s, M, 1.35, 5.9, 4.95, "geometry_card");
// schematic in um -> inch: big pore 0-40, throat 40-60 (drawn length, not to scale), small pore 60-70
const gx = M + 0.35, gy = 2.0, gs = 5.2 / 75;
s.addShape(pres.shapes.RECTANGLE, { x: gx + 20 * gs, y: gy + 20 * gs - 0.06, w: 42 * gs, h: 0.12, fill: { color: C.accent2 }, line: { color: C.accent2 }, objectName: "throat" });
s.addShape(pres.shapes.OVAL, { x: gx, y: gy, w: 40 * gs, h: 40 * gs, fill: { color: C.accent3 }, line: { color: C.accent1, width: 1 }, objectName: "big_pore" });
s.addShape(pres.shapes.OVAL, { x: gx + 60 * gs, y: gy + 15 * gs, w: 10 * gs, h: 10 * gs, fill: { color: C.accent3 }, line: { color: C.accent1, width: 1 }, objectName: "small_pore" });
label(s, "大孔 B：R = 20 µm", gx, gy + 40 * gs + 0.12, 40 * gs, 0.35, { align: "center", fontSize: 14, bold: true, color: C.text2 });
label(s, "小孔 S：R = 5 µm", gx + 52 * gs, gy + 26 * gs + 0.05, 26 * gs, 0.35, { align: "center", fontSize: 14, bold: true, color: C.text2 });
label(s, "喉道：宽 w、长 L", gx + 41 * gs, gy + 20 * gs - 0.85, 30 * gs, 0.35, { align: "left", fontSize: 14, bold: true, color: C.accent2 });
body(s, "闭合单喉道体系，所有固壁以同一 ρ₂ 弛豫；二维（每单位纵深）。示意图喉宽未按比例。", M + 0.35, 5.45, 5.2, 0.7, { fontSize: 12, color: C.accent6 });
const prow = (a, b, c) => [cell(a, { ...CELL, align: "left" }), cell(b), cell(c)];
s.addTable([
  [{ text: "量", options: HDR }, { text: "取值", options: HDR }, { text: "说明", options: HDR }],
  prow("D", "2.0 µm²/ms", "水的自扩散"),
  prow("ρ₂", "0.005 µm/ms", "5 µm/s"),
  prow("T_{2B}", "3000 ms", "体弛豫"),
  prow("p_S", "0.0588", "小孔面积分数"),
  prow("w", "0.1, 0.5, 1, 2, 4 µm", "喉宽"),
  prow("L", "2, 5, 15, 30 µm", "喉长（壁到壁）"),
  prow("NNLS", "L-curve，SNR 1000", "200 箱，1–10⁵ ms"),
], { x: 6.85, y: 1.35, w: 5.88, colW: [1.1, 2.6, 2.18], rowH: 0.55, border: { type: "solid", pt: 0.75, color: "D5DEE5" }, fill: { color: C.background1 }, objectName: "param_table" });
caption(s, "参数与 Auto_NMR.py 默认值一致（Zhou & Zhang, 2025）；数值解：pyGIMLi 隐式 Euler 有限元，Δt = 0.25 ms", 6.45);
s.addNotes("几何是闭合的大孔–喉道–小孔体系。喉宽 5 个、喉长 4 个，共 20 个算例。NNLS 用 L-curve 选正则化参数，SNR 1000。");

// ---------- 4. baseline ----------
s = content("基准：20 个算例的 NNLS 结果", "模型与基准");
fig(s, "sweep_overlay_grid.png", M, 1.2, 5.6, 5.6, "NNLS baseline grid");
const obs = [
  ["长而窄的喉道", "两个峰分别靠近两孔，但小孔峰偏左（偏小）。"],
  ["短而宽的喉道", "只见一个主峰，位于两孔之间的等效尺寸。"],
  ["关键问题", "小孔信号去哪了？峰位还能换算成孔径吗？"],
];
obs.forEach(([h, t], i) => {
  const y = 1.45 + i * 1.7;
  card(s, 6.6, y, 6.13, 1.45, "obs_card_" + i);
  label(s, h, 6.9, y + 0.2, 5.5, 0.45, { fontSize: 18, bold: true, color: i === 2 ? C.accent2 : C.text2 });
  body(s, t, 6.9, y + 0.72, 5.5, 0.6, { fontSize: 15 });
});
s.addNotes("这是前一阶段的基准图：行是喉宽，列是喉长。窄长喉道下两峰分离但小孔峰偏小；短宽喉道下只剩一个峰。下面用耦合模型解释这些现象并给出反算方法。");

// ---------- 5. TL model ----------
pres.addSection({ title: "耦合模型" });
s = content("孔–喉–孔一维耦合模型", "耦合模型");
stepRows(s, [["node", "孔体节点"], ["throat", "一维喉道"], ["mode", "本征模"]], 1.45, 1.35, 1.9);
card(s, M, 5.6, CW, 0.75, "assume_card");
body(s, "假设：孔内快扩散（κ = ρ₂R/D ≤ 0.05）；喉道横向快平衡（h²/D ≪ T₂）；P_i 扣除喉口弧长 2R·arcsin(h/R)。", M + 0.3, 5.75, CW - 0.6, 0.5, { fontSize: 14, valign: "middle" });
caption(s, "Brownstein & Tarr (1979)；Carslaw & Jaeger (1959)", 6.5);
s.addNotes("两个孔各用一个均匀浓度节点表示，喉道用截面平均的一维扩散–弛豫方程表示，喉道两端浓度等于两孔浓度。对 e 的负 λt 模态，喉道剖面是双曲正弦组合，由此得到流入喉道的通量。");

// ---------- 6. characteristic equation ----------
s = content("特征方程与模态权重（TL 模型）", "耦合模型");
card(s, M, 1.35, CW, 1.25, "char_card");
const scChar = fitScale(["char"], CW - 0.6);
eq(s, "char", M + 0.3, 1.35 + (1.25 - eqH("char", scChar)) / 2, scChar);
label(s, "均匀初始磁化下的模态权重：", M, 2.85, 5, 0.4, { fontSize: 16, bold: true, color: C.text2 });
eq(s, "amp_n", M, 3.3, fitScale(["amp_n"], 8));
const stats6 = [["0", "拟合参数：只用几何、D、ρ₂、T_{2B}", 0.9], ["0.65%", "20 个算例中衰减 |ΔM/M₀| 的最大偏差（相对全二维有限元）", 2.3]];
stats6.forEach(([big, small, bw], i) => {
  const x = M + i * 6.2, w = 5.9;
  card(s, x, 4.55, w, 1.75, "stat6_" + i);
  s.addText(big, { x: x + 0.35, y: 4.65, w: bw, h: 1.5, fontSize: 44, bold: true, color: C.accent1, valign: "middle", margin: 0, isTextBox: true });
  body(s, small, x + 0.6 + bw, 4.9, w - 0.9 - bw, 1.1, { fontSize: 16, valign: "middle" });
});
s.addNotes("把通量代入节点方程，非零解要求一个 2×2 行列式为零。乘以 S² 后方程没有极点，便于数值求根。λ 大于喉道弛豫速率时 q 变为虚数，对应喉道内部模态。这个模型没有拟合参数，20 个算例的衰减与全二维有限元模拟最大偏差 0.65%。");

// ---------- 7. quasi-static reduction ----------
s = content("准静态约化：带“汇”的两点交换", "耦合模型");
const sc7 = fitScale(["gx", "twosite"], CW);
eq(s, "gx", M, 1.4, sc7);
eq(s, "twosite", M, 2.4, sc7);
[["lim_small", "短喉道 qL ≪ 1", "交换导纳 = Dw/L；喉道体积的弛豫平分给两孔"], ["lim_large", "长喉道 qL ≫ 1", "两孔解耦，但喉道成为半无限“汇”，仍使小孔 T₂ 变短"]].forEach(([key, head, t], i) => {
  const x = M + i * 6.2, w = 5.9;
  card(s, x, 3.75, w, 2.55, "limit7_" + i);
  label(s, head, x + 0.3, 3.95, w - 0.6, 0.45, { fontSize: 18, bold: true, color: i ? C.accent2 : C.text2 });
  eq(s, key, x + 0.3, 4.55, fitScale([key], w - 0.6));
  body(s, t, x + 0.3, 5.35, w - 0.6, 0.8, { fontSize: 14 });
});
caption(s, "coth x − csch x = tanh(x/2)；Zimmerman & Brittin (1957)；McDonald et al. (2005)", 6.5);
s.addNotes("利用 coth 减 csch 等于 tanh(x/2)，喉道通量拆成交换项 G_x 和汇项 G_s。于是问题化为经典两点交换模型，只是两孔的弛豫速率各加上汇项。短喉道时交换导纳就是 Dw/L；长喉道时两孔解耦，但喉道仍是弛豫汇。");

// ---------- 8. closed form & dimensionless ----------
s = content("两点交换闭式解与无量纲数", "耦合模型");
const sc8 = fitScale(["lam", "ampf", "dimless"], CW - 2.6);
[["lam", "本征速率"], ["ampf", "快模态权重"], ["dimless", "无量纲数"]].forEach(([key, head], i) => {
  const y = 1.45 + i * 1.45;
  card(s, M, y, CW, 1.25, "cf_card_" + i);
  label(s, head, M + 0.3, y + 0.35, 2.2, 0.5, { fontSize: 17, bold: true, color: C.text2 });
  eq(s, key, M + 2.6, y + (1.25 - eqH(key, sc8)) / 2, sc8);
});
body(s, "快模态权重来自初始斜率：−ṡ(0) = A_f λ_f + A_s λ_s = p_S R′_S + p_B R′_B（细致平衡 p_S k_S = p_B k_B 使交换项相消）。η 为小孔信号保留比例。", M, 5.95, CW, 0.6, { fontSize: 14 });
s.addNotes("两点交换的本征速率和快模态权重都有闭式解。权重公式来自初始斜率：交换不改变总磁化，所以初始衰减速率等于面积加权的弛豫速率。Ψ 只用几何和孤立孔速率；Ψ_eff 计入喉道衰减和汇；η 是小孔模态保留下来的信号比例。");

// ---------- 9. limits ----------
s = content("两个极限：慢交换与快交换", "耦合模型");
[["slow", "慢交换 Ψ_eff → 0", "两个峰对应两个孔，权重 = 面积分数；qL ≳ 1 时汇项仍会把小孔峰推向短 T₂。", C.accent1],
 ["fast", "快交换 Ψ_eff ≫ 1", "主峰位于面积加权平均速率；快模态仍存在，但 T₂ ≈ p_B A_S / G_x 由交换决定，权重按 Ψ_eff⁻² 消失。", C.accent5]].forEach(([key, head, t, color], i) => {
  const x = M + i * 6.2, w = 5.9;
  card(s, x, 1.4, w, 3.75, "lim9_" + i);
  s.addShape(pres.shapes.OVAL, { x: x + 0.3, y: 1.7, w: 0.32, h: 0.32, fill: { color }, line: { color }, objectName: "dot9_" + i });
  label(s, head, x + 0.8, 1.6, w - 1.1, 0.5, { fontSize: 20, bold: true, color: C.text2 });
  eq(s, key, x + 0.3, 2.45, fitScale([key], w - 0.6));
  body(s, t, x + 0.3, 3.55, w - 0.6, 1.45, { fontSize: 17 });
});
card(s, M, 5.4, CW, 0.95, "limit9_takeaway", C.text2);
body(s, "η 随 Ψ_eff 从 1 连续降到 0：两峰与一峰之间没有突变，只有“小峰是否高于探测门槛”的区别。", M + 0.3, 5.5, CW - 0.6, 0.75, { fontSize: 17, bold: true, color: C.background1, valign: "middle" });
s.addNotes("慢交换时两个峰对应两个孔，权重就是面积分数。快交换时主峰位于平均速率，快模态的 T2 由交换速率决定，不再反映小孔尺寸，权重随 Ψ_eff 的平方迅速消失。");

// ---------- 10. regimes ----------
pres.addSection({ title: "分区判据" });
s = content("分区判据：小孔信号保留比 η", "分区判据");
fig(s, "fig1_regimes_and_universal_curves.png", M, 1.15, CW, 3.9, "regime figure");
const cnt = NUM.regime_counts;
[["I", "区 I 慢交换", "η > 0.8（Ψ ≲ 0.1）"], ["II", "区 II 过渡", "0.1 < η < 0.8"], ["III", "区 III 快交换", "η < 0.1（Ψ ≳ 2）"]].forEach(([k, head, t], i) => {
  const w = (CW - 0.6) / 3, x = M + i * (w + 0.3);
  card(s, x, 5.3, w, 1.15, "regime_card_" + k);
  s.addShape(pres.shapes.OVAL, { x: x + 0.25, y: 5.47, w: 0.3, h: 0.3, fill: { color: REG[k] }, line: { color: REG[k] }, objectName: "regime_dot_" + k });
  label(s, `${head}：${cnt[k]} 个算例`, x + 0.7, 5.4, w - 0.9, 0.45, { fontSize: 15, bold: true, color: C.text2 });
  body(s, t, x + 0.7, 5.88, w - 0.9, 0.45, { fontSize: 14 });
});
s.addNotes("左图是 w–L 分区图，中图是 η 对 Ψ_eff，右图是模态 T2。TL 模型的点都落在两点交换的普适曲线上。η 大于 0.8 为区 I，0.1 到 0.8 为区 II，小于 0.1 为区 III，分别对应 4、8、8 个算例。");

// ---------- 11. old figure, fixed legend ----------
s = content("“一个峰”其实是探测门槛（原图图例已修正）", "分区判据");
fig(s, "sweep_coupling_controls.png", M, 1.15, CW, 3.6, "coupling controls");
card(s, M, 5.0, CW, 1.4, "threshold_card");
body(s, [
  { text: "原判据“第二峰 > 2%”相当于 η ≈ 0.34 的门槛。", options: { bold: true, color: C.accent2, breakLine: true } },
  { text: "NNLS 在所有 20 个算例中都给出了小孔模态（0.04%–5%），位置与 TL 模型一致；它是连续衰减到探测极限以下，而不是突然消失。图例中每个符号都对应图中实际绘制的点。" },
], M + 0.3, 5.1, CW - 0.6, 1.2, { fontSize: 15, valign: "middle" });
s.addNotes("原图把第二峰面积大于 2% 作为双峰判据，相当于在 η 约 0.34 处设了门槛。实际上小孔模态从未消失，只是连续衰减。图例已改为每个符号都对应图中实际绘制的点，左图改为两类离散色块。");

// ---------- 12. validation ----------
s = content("模型验证：TL 模型 vs 全二维模拟与 NNLS", "分区判据");
fig(s, "fig2_model_validation.png", M, 1.15, CW, 4.3, "validation");
caption(s, `TL 衰减最大偏差 |ΔM/M₀| = ${(NUM.max_tl_decay_err).toFixed(4)}；NNLS 小峰在区 I、II 与模型一致，在区 III 落入噪声底（≤ 0.5%）`, 6.4);
s.addNotes("左图是 5 个代表算例的全二维模拟衰减与 TL 模型，没有拟合参数也几乎重合。中图和右图是快模态 T2 与权重对 NNLS 小峰的对比。区 III 的小峰权重已在噪声底附近，NNLS 给出的值偏离模型。");

// ---------- 13-14. worked C3 ----------
pres.addSection({ title: "代入求解" });
s = content("代入求解：C3（w = 1 µm，L = 15 µm）1/2", "代入求解");
stepRows(s, [["c3_1", "几何与孔体速率"], ["c3_2", "喉道参数"], ["c3_3", "交换 / 汇导纳"], ["c3_4", "有效参数"]], 1.3, 1.22, 2.3, true);
caption(s, "速率单位 ms⁻¹，导纳单位 µm²/ms；L_m = 15.031 µm 为网格中喉道壁到壁长度", 6.5, true);
s.addNotes("C3 位于区 II。先算两孔的面积、扣除喉口后的孔壁长度和弛豫速率；再算喉道弛豫速率、q 和 qL；然后是交换导纳和汇导纳；最后得到带汇的有效速率和交换速率。");
s = content("代入求解：C3（w = 1 µm，L = 15 µm）2/2", "代入求解");
stepRows(s, [["c3_5", "有效交换数"], ["c3_6", "本征速率"], ["c3_7", "快模态权重"], ["c3_8", "对照", C.accent2]], 1.4, 1.2, 2.3);
caption(s, "手算与 TL 精确根的差别来自准静态近似中 q 取 λ = 0", 6.4);
s.addNotes(`有效交换数 ${C3num()}，属区 II。两点交换闭式解给出快、慢模态 T2，快模态权重约 0.022，保留比 0.37。TL 精确根和 NNLS 的小峰都与手算接近。`);

// ---------- 15. representative table ----------
s = content("五个代表算例：手算、精确根与 NNLS", "代入求解");
const hdr15 = [{ text: "量", options: HDR }].concat(CASES.map((c) => ({ text: `${c.id}（${c.regime}）\nw = ${c.w}，L = ${c.L}`, options: HDR })));
const r15 = (name, fn) => [cell(name, { ...CELL, align: "left" })].concat(CASES.map((c) => cell(fn(c))));
s.addTable([
  hdr15,
  r15("Ψ_eff", (c) => Number(c.Psi_eff).toPrecision(3)),
  r15("手算 T_{2,f} / T_{2,s} (ms)", (c) => `${f1(c.T2f)} / ${f1(c.T2s)}`),
  r15("手算 A_f", (c) => ampTxt(c.A_f)),
  r15("TL T_{2,f} / T_{2,s} (ms)", (c) => `${f1(c.tl_T2f)} / ${f1(c.tl_T2s)}`),
  r15("TL A_f", (c) => ampTxt(c.tl_Af)),
  r15("TL η = A_f/p_S", (c) => ampTxt(c.eta)),
  r15("NNLS 小峰 / 主峰 (ms)", (c) => `${f1(c.nn_T2f)} / ${f1(c.nn_T2s)}`),
  r15("NNLS 小峰分数", (c) => ampTxt(c.nn_Af)),
], { x: M, y: 1.25, w: CW, colW: [2.63, 1.9, 1.9, 1.9, 1.9, 1.9], rowH: [0.68, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5], margin: 0.04, border: { type: "solid", pt: 0.75, color: "D5DEE5" }, fill: { color: C.background1 }, objectName: "rep_table" });
caption(s, "区 I、II：三者一致；区 III：快模态权重 < 0.5%，NNLS 值受噪声底影响", 6.4);
s.addNotes("五个代表算例横跨三个分区。手算的准静态解、TL 精确根和 NNLS 小峰在区 I、II 基本一致；区 III 的快模态权重已低于 0.5%，NNLS 给出的值受噪声底影响。");

// ---------- 16. fast exchange ----------
s = content("快交换（C5）：小孔峰变成“伪孔”", "代入求解");
stepRows(s, [["c5_1", "快模态权重", C.accent5], ["c5_2", "快模态 T₂", C.accent5], ["c5_3", "按孤立孔换算", C.accent2]], 1.45, 1.3, 2.3);
card(s, M, 5.45, CW, 0.95, "fake_card");
body(s, "快交换时快模态的 T₂ 由交换速率 G_x/A_S 决定，与小孔 S/V 无关：把它当作孔隙峰换算，会得到一个并不存在的亚微米“孔”。", M + 0.3, 5.55, CW - 0.6, 0.75, { fontSize: 15, valign: "middle" });
s.addNotes("C5 的有效交换数约 28，快模态权重只有 5.5 乘 10 的负 5 次方。它的 T2 约 23 ms，由交换速率决定。如果把这个峰当作孔隙峰换算，会得到一个 0.16 µm 的伪孔，而真实小孔是 5 µm。");

// ---------- 17. routes ----------
pres.addSection({ title: "反算" });
s = content("四条反算路线", "反算");
[["routeA", "A 孤立孔公式", "每个峰直接换算；耦合时偏小"], ["routeB", "B 振幅比", "区 I 适用：振幅 ≈ 面积分数，对汇不敏感"],
 ["routeC", "C 两点交换反演", "不知喉道：交换与汇无法区分，解不稳定"], ["routeD", "D TL 模型反演", "已知喉道 w、L（成像或 SIP）：区 I–II 可用"]].forEach(([key, head, t], i) => {
  const w = (CW - 0.3) / 2, x = M + (i % 2) * (w + 0.3), y = 1.4 + Math.floor(i / 2) * 2.55;
  card(s, x, y, w, 2.3, "route_card_" + i);
  label(s, head, x + 0.3, y + 0.15, w - 0.6, 0.45, { fontSize: 18, bold: true, color: i === 3 ? C.accent1 : C.text2 });
  eq(s, key, x + 0.3, y + 0.75, fitScale([key], w - 0.6, 1.7));
  body(s, t, x + 0.3, y + 1.65, w - 0.6, 0.5, { fontSize: 14 });
});
s.addNotes("四条路线：A 直接用孤立孔公式；B 用两个峰的振幅比，振幅反映面积分数，对汇不敏感，适合区 I；C 不知道喉道时用两点交换反演，但交换与汇无法区分；D 已知喉道几何时用 TL 模型反演，区 I 到 II 都可用。");

// ---------- 18. results ----------
s = content("反算结果（SNR 1000）", "反算");
const f18 = fig(s, "fig3_backcalculated_sizes.png", M, 1.1, CW, 4.05, "back-calculated sizes");
const res = [
  ["A", "小孔 0.16–3.5 µm；主孔低估 6–15%"],
  ["B", "区 I 改善到 4.3 µm；区 II、III 失效"],
  ["C", "G → 0、残差不为零：缺喉道信息"],
  ["D", "区 I–II：r_S 4.6–4.8，r_B 19.6–20.1 µm"],
];
res.forEach(([k, t], i) => {
  const w = (CW - 0.6) / 4, x = M + i * (w + 0.2), y = 1.1 + f18.h + 0.2;
  card(s, x, y, w, 1.2, "res_card_" + k);
  badge(s, x + 0.18, y + 0.39, k, k === "D" ? C.accent1 : C.accent6, 0.42);
  body(s, t, x + 0.72, y + 0.1, w - 0.85, 1.0, { fontSize: 13, valign: "middle" });
});
card(s, M + f18.w + 0.3, 1.2, CW - f18.w - 0.3, f18.h - 0.1, "back_card", C.text2);
body(s, [
  { text: "一致性检验", options: { bold: true, color: C.accent2, breakLine: true } },
  { text: "路线 D 以 TL 精确模态为输入时，五个算例全部回到 5 / 20 µm：剩余误差来自 NNLS 正则化偏差，而不是耦合模型本身。" },
], M + f18.w + 0.55, 1.4, CW - f18.w - 0.8, f18.h - 0.5, { fontSize: 16, color: C.background1, valign: "middle" });
s.addNotes("左图是五个代表算例各路线的反算结果。路线 D 以精确模态作输入时五个算例全部恢复 5 和 20 µm，说明剩余误差来自 NNLS 正则化偏差，不是模型本身。区 III 中 C4 的 0.4% 交换模态仍给出近似值（5.5 / 18.0 µm，误差约 10%），C5 已在噪声中，反算失败。");

// ---------- 19. worked back-calculation ----------
s = content("反算代入示例", "反算");
stepRows(s, [["bA", "路线 A（C3）"], ["bB", "路线 B（C1）"], ["bR", "区 III（C5）", C.accent5]], 1.5, 1.45, 2.3);
caption(s, "0.01 = 2ρ₂（µm/ms）；T₂ 取 NNLS 各峰的 T₂lm", 6.4);
s.addNotes("路线 A 直接换算 C3 的两个峰：小孔 2.31 µm、大孔 17.04 µm，都偏小。路线 B 用 C1 的小峰分数 0.0495 和主峰尺寸，得到小孔 4.28 µm。区 III 的 C5 主峰只能换算出等效半径 17.9 µm，接近两孔合并后的 17.0 µm。");

// ---------- 20. why the small pore is ignored ----------
s = content("小孔为何被“忽略”：快交换的非唯一性", "反算");
fig(s, "fig5_nonuniqueness_fast_exchange.png", M, 1.15, 4.9, 5.3, "non-uniqueness");
eq(s, "reff", 5.95, 1.5, fitScale(["reff"], 6.7));
const why = [
  "主峰只给出合并后的 S/V：R_eff = 17.0 µm（NNLS 17.7–17.9 µm）",
  "曲线上任何 (r_S, r_B) 都给出几乎相同的单峰谱：仅凭 T₂ 无法分离两种孔",
  "按“单一孔隙”解读 = 忽略小孔，并把主孔低估约 11%",
  "需要额外约束：喉道几何、CT 孔隙体积分配或 T₂–T₂ 交换谱",
];
why.forEach((t, i) => {
  badge(s, 5.95, 2.75 + i * 0.95, i + 1, i === 3 ? C.accent1 : C.accent2, 0.42);
  body(s, t, 6.55, 2.73 + i * 0.95, 6.2, 0.8, { fontSize: 15, valign: "middle" });
});
s.addNotes("区 III 中主峰速率趋于面积加权平均速率，按孤立孔公式换算得到的是合并后的等效半径。这是一条曲线，曲线上任何组合都给出几乎相同的单峰谱，所以单靠 T2 不能分离两种孔。要恢复小孔，需要喉道几何、CT 体积分配或 T2–T2 交换谱等额外约束。");

// ---------- 21. SNR ----------
s = content("信噪比的限制：SNR 1000 vs SNR 100", "反算");
const f21 = fig(s, "fig4_spectra_snr.png", M, 1.1, 8.3, 3.35, "spectra SNR");
card(s, M + f21.w + 0.3, 1.2, CW - f21.w - 0.3, f21.h - 0.1, "snr_card", C.text2);
body(s, [
  { text: "SNR 100 下", options: { bold: true, color: C.accent2, breakLine: true } },
  { text: "区 III 小峰全部消失；区 I、II 小峰被正则化拉到 58–143 ms，路线 D 也只得 1.9–2.4 µm。" },
], M + f21.w + 0.55, 1.4, CW - f21.w - 0.8, f21.h - 0.5, { fontSize: 15, color: C.background1, valign: "middle" });
const lowRows = NUM.snr100;
const h21 = ["算例", "小峰 T_{2lm} (ms)", "小峰分数", "A：r_S", "B：r_S", "D：r_S / r_B", "主峰按 A：r_B"].map((t) => cell(t, { ...HDR, fontSize: 12 }));
const rows21 = lowRows.map((r, i) => {
  const id = "C" + (i + 1);
  const has = r.fast_t2 !== null;
  const c = (t) => cell(t, { ...CELL, fontSize: 12 });
  return has ? [c(id), c(f1(r.fast_t2)), c(f4(r.frac)), c(f2(r.A_rS)), c(f2(r.B_rS)), c(`${f2(r.D_rS)} / ${f2(r.D_rB)}`), c(f2(r.slow_rB_naive))]
             : [c(id), c("无"), c("0"), c("—"), c("—"), c("—"), c(f2(r.slow_rB_naive))];
});
s.addTable([h21, ...rows21], { x: M, y: 1.1 + f21.h + 0.2, w: CW, colW: [1.0, 1.9, 1.5, 1.6, 1.6, 2.53, 2.0], rowH: 0.33, margin: 0.03, border: { type: "solid", pt: 0.75, color: "D5DEE5" }, fill: { color: C.background1 }, objectName: "snr_table" });
s.addNotes("SNR 100 下区 III 的小峰全部消失；区 I、II 的小峰保留但分数降到 1–2%，位置被正则化拉到 58–143 ms，即使路线 D 也只能给出 1.9–2.4 µm。所以小孔被忽略有两层原因：物理上快交换并入主峰，测量上低 SNR 抹掉残余小峰。");

// ---------- 22. workflow ----------
pres.addSection({ title: "总结" });
s = content("反算流程", "总结");
const flow = [
  ["NNLS 反演", "L-curve；识别高于噪声底的峰；T₂ < 4 TE 不作孔隙解释"],
  ["一个峰", "报告 R_eff，说明为全部连通孔的 S/V 等效尺寸；小孔可能被隐藏"],
  ["两个峰", "路线 B 得初值；有喉道信息时算 Ψ_eff、η 判区；区 II 用路线 D"],
  ["自洽检验", "以反算结果正演 TL 模型，确认分区并与实测谱比较"],
  ["SNR 要求", "次峰分数需远高于噪声底；SNR 100 时区 I 也严重低估小孔"],
];
flow.forEach(([h, t], i) => {
  const w = (CW - 4 * 0.25) / 5, x = M + i * (w + 0.25);
  card(s, x, 1.5, w, 4.0, "flow_card_" + i);
  badge(s, x + 0.25, 1.75, i + 1, i === 4 ? C.accent2 : C.accent1, 0.5);
  label(s, h, x + 0.25, 2.45, w - 0.5, 0.5, { fontSize: 19, bold: true, color: C.text2 });
  body(s, t, x + 0.25, 3.1, w - 0.5, 2.25, { fontSize: 16 });
  if (i < 4) s.addShape(pres.shapes.CHEVRON, { x: x + w + 0.05, y: 1.86, w: 0.15, h: 0.28, fill: { color: C.accent3 }, line: { color: C.accent3 }, objectName: "flow_arrow_" + i });
});
caption(s, "T₂ < 4 TE 的峰受回波间隔限制；区 III 的“一个峰”只能报告等效尺寸，不能据此否认小孔存在", 5.8);
s.addNotes("反算流程分五步：NNLS 反演并识别有效峰；只有一个峰时报告等效半径并说明小孔可能被隐藏；两个峰时先用振幅比得初值，再按分区选择路线；用反算结果正演 TL 模型做自洽检验；最后确认 SNR 足够。");

// ---------- 23. applicability ----------
s = content("适用边界", "总结");
const lims = [
  ["二维模型", "三维圆柱喉道截面 ∝ r_t²，汇与交换比二维狭缝弱，Ψ 的几何因子需修改。"],
  ["快扩散假设", "孔内 κ ≤ 0.05、喉道 h²/D ≪ T₂；w = 4 µm 时喉道本身像一个 T₂ ≈ 400 ms 的孔。"],
  ["准静态约化", "要求 λ ≪ R_t；宽喉道应改用精确特征方程。"],
  ["阈值依赖 p_S", "η = 0.8、0.1 对应的 Ψ 与小孔占比有关，p_S 不同需重算普适曲线。"],
];
lims.forEach(([h, t], i) => {
  const w = (CW - 0.3) / 2, x = M + (i % 2) * (w + 0.3), y = 1.4 + Math.floor(i / 2) * 2.5;
  card(s, x, y, w, 2.25, "lim_card_" + i);
  badge(s, x + 0.3, y + 0.3, i + 1, C.accent2, 0.46);
  label(s, h, x + 0.95, y + 0.3, w - 1.25, 0.46, { fontSize: 19, bold: true, color: C.text2 });
  body(s, t, x + 0.3, y + 0.95, w - 0.6, 1.15, { fontSize: 17 });
});
s.addNotes("四点适用边界：二维与三维的几何因子不同；快扩散假设；准静态约化的条件；分区阈值依赖小孔占比。扩散耦合的思想与 Ramakrishnan et al. (1999)、Anand & Hirasaki (2007) 的微–宏孔耦合分析一致。");

// ---------- 24. references ----------
pres.addSection({ title: "参考文献" });
s = content("参考文献", "参考文献");
const refs = [
  [["Anand, V., & Hirasaki, G. J. (2007). Diffusional coupling between micro and macroporosity for NMR relaxation in sandstones and grainstones. "], ["Petrophysics", true], [", 48(4), 289–307."]],
  [["Brownstein, K. R., & Tarr, C. E. (1979). Importance of classical diffusion in NMR studies of water in biological cells. "], ["Physical Review A", true], [", 19(6), 2446–2453. https://doi.org/10.1103/PhysRevA.19.2446"]],
  [["Carslaw, H. S., & Jaeger, J. C. (1959). "], ["Conduction of heat in solids", true], [" (2nd ed.). Oxford, UK: Clarendon Press."]],
  [["Fleury, M., & Soualem, J. (2009). Quantitative analysis of diffusional pore coupling from T2-store-T2 NMR experiments. "], ["Journal of Colloid and Interface Science", true], [", 336(1), 250–259.（卷期页码待核对）"]],
  [["Hansen, P. C. (1992). Analysis of discrete ill-posed problems by means of the L-curve. "], ["SIAM Review", true], [", 34(4), 561–580. https://doi.org/10.1137/1034115"]],
  [["Lawson, C. L., & Hanson, R. J. (1974). "], ["Solving least squares problems", true], [". Englewood Cliffs, NJ: Prentice-Hall."]],
  [["McDonald, P. J., Korb, J.-P., Mitchell, J., & Monteilhet, L. (2005). Surface relaxation and chemical exchange in hydrating cement pastes: A two-dimensional NMR relaxation study. "], ["Physical Review E", true], [", 72(1), 011409. https://doi.org/10.1103/PhysRevE.72.011409"]],
  [["Ramakrishnan, T. S., Schwartz, L. M., Fordham, E. J., Kenyon, W. E., & Wilkinson, D. J. (1999). Forward models for nuclear magnetic resonance in carbonate rocks. "], ["The Log Analyst", true], [", 40(4), 260–270."]],
  [["Torrey, H. C. (1956). Bloch equations with diffusion terms. "], ["Physical Review", true], [", 104(3), 563–565. https://doi.org/10.1103/PhysRev.104.563"]],
  [["Zhou, & Zhang (2025). How does pore structure affect the NMR relaxation in unsaturated porous media. "], ["Water Resources Research", true], [".（作者全名、卷期与 DOI 待核对）"]],
  [["Zimmerman, J. R., & Brittin, W. E. (1957). Nuclear magnetic resonance studies in multiple phase systems: Lifetime of a water molecule in an adsorbing phase on silica gel. "], ["The Journal of Physical Chemistry", true], [", 61(10), 1328–1333. https://doi.org/10.1021/j150556a015"]],
];
const runs = [];
refs.forEach((parts, i) => parts.forEach(([text, italic], k) => runs.push({ text, options: { italic: !!italic, breakLine: k === parts.length - 1 && i < refs.length - 1, paraSpaceAfter: 5 } })));
s.addText(runs, { x: M, y: 1.25, w: CW, h: 5.6, fontSize: 12, color: C.text1, valign: "top", margin: 0, isTextBox: true });
s.addNotes("参考文献按 AGU/GRL 作者–年份格式。Fleury & Soualem (2009) 的卷期页码与 Zhou & Zhang (2025) 的作者全名、卷期、DOI 需按原文核对。");

function C3num() { return Number(NUM.cases.C3.Psi_eff).toFixed(3); }

(async () => {
  await pres.writeFile({ fileName: OUT });
  await applyTheme(OUT, THEME);
  console.log("wrote", path.basename(OUT));
})();
