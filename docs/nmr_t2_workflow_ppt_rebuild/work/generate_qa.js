const fs = require('fs');
const path = require('path');
const sharp = require('sharp');

const ROOT = 'C:/Users/imgw/Documents/Codex/NMR模拟/docs/nmr_t2_workflow_ppt_rebuild';
const SRC = 'C:/Users/imgw/Documents/Codex/NMR模拟/docs/nmr_t2_workflow_figures';
const RENDER = path.join(ROOT, 'work', 'rendered');
const QA = path.join(ROOT, 'final', 'qa');
const SLIDES = path.join(QA, 'slides');
fs.mkdirSync(SLIDES, { recursive: true });

const names = [
  '01_workflow_overview_16x9.png',
  '02_model_reduction_16x9.png',
  '03_ct_to_fem_16x9.png',
  '04_magnetization_to_decay_16x9.png',
  '05_weighted_inversion_16x9.png',
];

async function normalized(file) {
  return sharp(file).resize(1600, 900, { fit: 'fill' }).removeAlpha().raw().toBuffer();
}

async function saveRaw(data, file) {
  await sharp(data, { raw: { width: 1600, height: 900, channels: 3 } }).png().toFile(file);
}

async function contact(files, out, title) {
  const panelW = 500, panelH = 281;
  const composites = [];
  for (let i = 0; i < files.length; i++) {
    const input = await sharp(files[i]).resize(panelW, panelH, { fit: 'fill' }).png().toBuffer();
    const x = 20 + (i % 3) * 525;
    const y = 105 + Math.floor(i / 3) * 375;
    composites.push({ input, left: x, top: y });
    const label = Buffer.from(`<svg width="500" height="40"><rect width="500" height="40" fill="white"/><text x="8" y="28" font-family="Arial" font-size="22" font-weight="700" fill="#07184d">Slide ${i + 1}</text></svg>`);
    composites.push({ input: label, left: x, top: y - 40 });
  }
  const heading = Buffer.from(`<svg width="1600" height="50"><rect width="1600" height="50" fill="white"/><text x="20" y="35" font-family="Arial" font-size="28" font-weight="700" fill="#07184d">${title}</text></svg>`);
  composites.push({ input: heading, left: 0, top: 0 });
  await sharp({ create: { width: 1600, height: 850, channels: 3, background: 'white' } }).composite(composites).png().toFile(out);
}

async function main() {
  const previews = [], sources = [], overlays = [], diffs = [];
  const metrics = [];
  for (let i = 0; i < names.length; i++) {
    const srcPath = path.join(SRC, names[i]);
    const renderPath = path.join(RENDER, `slide-${i + 1}.png`);
    const previewPath = path.join(SLIDES, `slide-${String(i + 1).padStart(2, '0')}-preview.png`);
    const sourcePath = path.join(SLIDES, `slide-${String(i + 1).padStart(2, '0')}-source.png`);
    const overlayPath = path.join(SLIDES, `slide-${String(i + 1).padStart(2, '0')}-overlay.png`);
    const diffPath = path.join(SLIDES, `slide-${String(i + 1).padStart(2, '0')}-difference.png`);
    await sharp(renderPath).resize(1600, 900, { fit: 'fill' }).png().toFile(previewPath);
    await sharp(srcPath).resize(1600, 900, { fit: 'fill' }).png().toFile(sourcePath);
    const a = await normalized(srcPath);
    const b = await normalized(renderPath);
    const blend = Buffer.alloc(a.length);
    const diff = Buffer.alloc(a.length);
    let sum = 0;
    for (let p = 0; p < a.length; p++) {
      blend[p] = Math.round((a[p] + b[p]) / 2);
      const delta = Math.abs(a[p] - b[p]);
      sum += delta;
      diff[p] = Math.min(255, delta * 2);
    }
    await saveRaw(blend, overlayPath);
    await saveRaw(diff, diffPath);
    metrics.push({ slide: i + 1, mean_absolute_pixel_difference_0_255: Number((sum / a.length).toFixed(3)) });
    previews.push(previewPath); sources.push(sourcePath); overlays.push(overlayPath); diffs.push(diffPath);
  }
  await contact(previews, path.join(QA, 'preview.png'), 'Editable rebuild preview');
  await contact(sources, path.join(QA, 'source-preview-contact-sheet.png'), 'Source reference contact sheet');
  await contact(overlays, path.join(QA, 'overlay.png'), 'Source / rebuild 50% overlay');
  await contact(diffs, path.join(QA, 'difference.png'), 'Absolute pixel difference (2x amplified)');
  fs.writeFileSync(path.join(QA, 'visual-qa.json'), JSON.stringify({
    renderer: 'WPS Presentation 12.1 COM slide.Export',
    slide_size_px: [1600, 900],
    source_normalization: 'resize-to-fill 1600x900 for comparison only',
    difference_visualization: 'absolute RGB difference amplified 2x',
    metrics,
  }, null, 2));
}

main().catch(err => { console.error(err); process.exitCode = 1; });
