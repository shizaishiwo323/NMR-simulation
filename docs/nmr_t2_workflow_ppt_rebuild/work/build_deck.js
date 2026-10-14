const pptxgen = require('pptxgenjs');
const sharp = require('sharp');
const path = require('path');
const fs = require('fs');

const ROOT = path.resolve(__dirname, '..');
const FINAL = path.join(ROOT, 'final');
const ASSET = path.join(FINAL, 'independent-assets', 'reconstructed');
const SRC = path.resolve(ROOT, '..', 'nmr_t2_workflow_figures');

const pptx = new pptxgen();
pptx.layout = 'LAYOUT_WIDE';
pptx.author = 'NMR simulation project';
pptx.subject = 'Editable pore-scale NMR T2 workflow figures';
pptx.title = 'Pore-scale NMR T2 Workflow';
pptx.company = 'NMR simulation project';
pptx.lang = 'en-US';
pptx.theme = {
  headFontFace: 'Arial', bodyFontFace: 'Arial', lang: 'en-US'
};

const C = {
  navy: '07164A', blue: '0789C7', cyan: '08A9E0', red: 'E31A1C',
  green: '148B2B', yellow: 'FFD95A', pale: 'F6FAFD', gray: '6B7280',
  line: 'D7E2EC', black: '111111', white: 'FFFFFF'
};
const SH = pptx.ShapeType;
const CT = pptx.ChartType;

function addTitle(slide, text) {
  slide.addText(text, {x:0.4,y:0.18,w:12.53,h:0.62,fontFace:'Arial',fontSize:31,bold:true,color:C.navy,align:'center',margin:0,breakLine:false});
}
function addText(slide, text, x,y,w,h, opts={}) {
  slide.addText(text, {x,y,w,h,fontFace:opts.fontFace||'Arial',fontSize:opts.fontSize||14,color:opts.color||C.navy,bold:!!opts.bold,italic:!!opts.italic,align:opts.align||'center',valign:opts.valign||'mid',margin:opts.margin===undefined?0.03:opts.margin,fit:'shrink',breakLine:false, ...opts});
}
function card(slide,x,y,w,h) {
  slide.addShape(SH.roundRect,{x,y,w,h,rectRadius:0.05,fill:{color:C.white,transparency:0},line:{color:C.navy,width:1.2}});
}
function arrow(slide,x,y,w=0.32,h=0.22,color=C.navy) {
  slide.addShape(SH.chevron,{x,y,w,h,fill:{color},line:{color},adjustPoint:0.35});
}
function lineLegend(slide,x,y,color,dash,label) {
  slide.addShape(SH.line,{x,y:y+0.08,w:0.28,h:0,line:{color,width:2,dashType:dash}});
  addText(slide,label,x+0.34,y,1.35,0.18,{fontSize:9,align:'left'});
}
function addImage(slide,file,x,y,w,h,opts={}) {
  slide.addImage({path:file,x,y,w,h,transparency:opts.transparency||0});
}
function addLineChart(slide, series, x,y,w,h, opts={}) {
  slide.addChart(CT.line, series, {
    x,y,w,h,showLegend:opts.showLegend||false,legendPos:'tr',showTitle:false,
    showValue:false,showCatName:false,showSerName:false,
    catAxisLabelFontFace:'Arial',catAxisLabelFontSize:8,
    valAxisLabelFontFace:'Arial',valAxisLabelFontSize:8,
    catAxisTitle:opts.catAxisTitle||'',valAxisTitle:opts.valAxisTitle||'',
    showCatName:false,showValAxisTitle:!!opts.valAxisTitle,showCatAxisTitle:!!opts.catAxisTitle,
    showValAxis:true,showCatAxis:true,showGridLines:false,
    valAxisMinVal:opts.min===undefined?0:opts.min,valAxisMaxVal:opts.max===undefined?1:opts.max,
    valAxisLogScaleBase:opts.logVal?10:undefined,
    chartColors:opts.colors||[C.blue,C.green],
    lineSize:2.2,showMarker:false,
    showBorder:false,showPlotArea:false,showChartArea:false
  });
}
function decayData(scale=1) {
  const labels=['10⁻³','10⁻²','10⁻¹','10⁰','10¹','10²'];
  const values=labels.map((_,i)=>Math.exp(-Math.pow(10,i-3)*3.2)*scale);
  return {labels,values};
}
function t2Data(shift=0) {
  const labels=['10⁻³','10⁻²','10⁻¹','10⁰','10¹','10²'];
  const values=labels.map((_,i)=>0.04+0.9*Math.exp(-Math.pow(i-(2+shift),2)/0.6)+0.35*Math.exp(-Math.pow(i-3.7,2)/0.35));
  return {labels,values};
}
function t2SmoothData(shift=0) {
  const labels=[]; const values=[];
  for(let i=0;i<=25;i++) {
    const x=-2+i*0.2;
    labels.push(i%5===0 ? ['10⁻²','10⁻¹','10⁰','10¹','10²','10³'][i/5] : '');
    values.push(1e-5 + 0.18*Math.exp(-Math.pow(x-shift,2)/0.22) + 0.0013*Math.exp(-Math.pow(x-(1.65+shift),2)/0.11));
  }
  return {labels,values};
}
function phaseLegend(slide,x,y) {
  const items=[['water',C.red],['solid',C.yellow],['air',C.white]];
  items.forEach((it,i)=>{
    slide.addShape(SH.rect,{x:x+i*1.8,y,w:0.25,h:0.25,fill:{color:it[1]},line:{color:C.black,width:0.7}});
    addText(slide,it[0],x+0.34+i*1.8,y-0.01,0.72,0.27,{fontSize:11,align:'left'});
  });
}
function addSimpleMeshOverlay(slide,x,y,w,h,color='FFFFFF') {
  for(let i=0;i<9;i++) slide.addShape(SH.line,{x:x+(i/9)*w,y,w:0,h,line:{color,width:0.35,transparency:20}});
  for(let j=0;j<9;j++) slide.addShape(SH.line,{x,y:y+(j/9)*h,w,h:0,line:{color,width:0.35,transparency:20}});
  for(let i=-5;i<10;i++) slide.addShape(SH.line,{x:x+Math.max(0,i)*w/9,y:y+Math.max(0,-i)*h/9,w:w-Math.abs(i)*w/9,h:h-Math.abs(i)*h/9,line:{color,width:0.28,transparency:25}});
}

async function prepareFallbacks() {
  const s3=path.join(SRC,'03_ct_to_fem_16x9.png');
  const s4=path.join(SRC,'04_magnetization_to_decay_16x9.png');
  // Crop only the classified domain.  The slide-level title and legend are
  // rebuilt as editable objects, so keeping them in the raster duplicates text.
  await sharp(s3).extract({left:875,top:280,width:330,height:340}).toFile(path.join(ASSET,'interface_authoritative_fallback.png'));
  const crops=[
    ['mag_t0_authoritative_fallback.png',45,270,305,380],
    ['mag_t1_authoritative_fallback.png',405,270,305,380],
    ['mag_t2_authoritative_fallback.png',765,270,305,380]
  ];
  for (const [name,left,top,width,height] of crops) {
    await sharp(s4).extract({left,top,width,height}).toFile(path.join(ASSET,name));
  }
}

function slide1() {
  const s=pptx.addSlide(); s.background={color:C.white}; addTitle(s,'Pore-scale NMR T₂ Workflow');
  const xs=[0.18,2.02,3.95,5.88,7.82,9.74,11.32];
  const titles=['3D CT Stack','Representative\n2D Slice','Water Domain','2D FEM','M(x,t)','Decay S(t)','T₂ Spectrum'];
  titles.forEach((t,i)=>addText(s,t,xs[i],1.35,i===6?1.72:1.45,0.42,{fontSize:13,bold:true}));
  addImage(s,path.join(ASSET,'ct_stack.png'),0.20,2.00,1.42,2.35);
  addImage(s,path.join(ASSET,'segmented_slice.png'),2.03,2.13,1.48,1.85);
  addImage(s,path.join(ASSET,'interface_authoritative_fallback.png'),3.99,2.10,1.48,1.91);
  addImage(s,path.join(ASSET,'fem_mesh.png'),5.92,2.05,1.50,1.95);
  addImage(s,path.join(ASSET,'magnetization_field.png'),7.85,2.05,1.55,1.95);
  for(let i=0;i<5;i++) arrow(s,xs[i]+1.56,2.82,0.28,0.25);
  arrow(s,9.46,2.82,0.25,0.25);
  const d=decayData(); addLineChart(s,[{name:'S(t)',labels:d.labels,values:d.values}],9.72,2.18,1.43,1.78,{catAxisTitle:'t',valAxisTitle:'S(t)',colors:[C.black]});
  arrow(s,11.12,2.82,0.22,0.25);
  const a=t2Data(0),b=t2Data(0.25); addLineChart(s,[{name:'Simulated',labels:a.labels,values:a.values},{name:'Experimental',labels:b.labels,values:b.values}],11.36,2.18,1.78,1.78,{catAxisTitle:'T₂ (s)',valAxisTitle:'f(T₂)',colors:[C.blue,C.green],showLegend:false});
  lineLegend(s,11.62,1.82,C.blue,'solid','Simulated');
  lineLegend(s,11.62,2.04,C.green,'dash','Experimental');
  lineLegend(s,4.02,4.15,C.red,'solid','Solid-fluid boundary'); lineLegend(s,4.02,4.43,C.navy,'dash','Water-air boundary');
  s.addShape(SH.roundRect,{x:3.93,y:5.40,w:5.5,h:0.66,fill:{color:C.white},line:{color:C.navy,width:1.2}});
  addText(s,'Geometry   →   M(x,t)   →   S(t)   →   f(T₂)',4.05,5.51,5.25,0.36,{fontFace:'Cambria Math',fontSize:21,bold:true});
  addText(s,'representative-slice approximation — not a full 3D solve',3.35,6.47,6.65,0.34,{fontSize:14,bold:true,italic:true,color:C.blue});
  return s;
}

function slide2() {
  const s=pptx.addSlide(); s.background={color:C.white};
  addText(s,'Full Physical Model',0.45,0.23,4.8,0.48,{fontSize:25,bold:true});
  addText(s,'Current T₂ Solver',8.18,0.23,4.65,0.48,{fontSize:25,bold:true});
  card(s,0.25,0.88,5.18,5.95); card(s,7.48,0.88,5.58,5.95);
  const ox=1.45,oy=1.35;
  s.addShape(SH.line,{x:ox,y:oy+1.65,w:0,h:-1.35,line:{color:C.navy,width:1.5,beginArrowType:'none',endArrowType:'triangle'}});
  s.addShape(SH.line,{x:ox,y:oy+1.65,w:1.15,h:0,line:{color:C.navy,width:1.5,endArrowType:'triangle'}});
  s.addShape(SH.line,{x:ox,y:oy+1.65,w:-0.62,h:0.65,line:{color:C.navy,width:1.5,endArrowType:'triangle'}});
  s.addShape(SH.line,{x:ox,y:oy+1.65,w:0.62,h:-1.05,line:{color:C.navy,width:2,endArrowType:'triangle'}});
  s.addText([{text:'m = [mₓ, mᵧ, m_z]',options:{}},{text:'T',options:{superscript:true}}],{x:2.75,y:1.25,w:2.15,h:0.35,fontFace:'Cambria Math',fontSize:19,color:C.navy,margin:0,breakLine:false});
  const wave=[]; for(let i=0;i<=40;i++) wave.push({x:2.8+i*0.055,y:3.0+0.32*Math.sin(i*1.2)*Math.exp(-Math.pow(i-20,2)/150)});
  for(let i=1;i<wave.length;i++) s.addShape(SH.line,{x:wave[i-1].x,y:wave[i-1].y,w:wave[i].x-wave[i-1].x,h:wave[i].y-wave[i-1].y,line:{color:C.red,width:1.3}});
  addText(s,'RF pulse',3.40,3.28,1.35,0.28,{fontSize:15,bold:true,color:C.red});
  s.addShape(SH.rect,{x:0.62,y:4.02,w:3.45,h:1.95,fill:{color:'FFEAA4'},line:{color:C.black,width:1}});
  const blobs=[[1.05,4.45,1.45,0.75],[1.9,4.25,1.35,0.85],[1.45,5.05,1.35,0.72],[2.65,4.85,1.1,0.68]];
  blobs.forEach((b,i)=>s.addShape(SH.ellipse,{x:b[0],y:b[1],w:b[2],h:b[3],fill:{color:C.white},line:{color:i===3?C.navy:C.red,width:1.4}}));
  s.addShape(SH.rect,{x:4.18,y:4.38,w:0.24,h:0.24,fill:{color:'FFEAA4'},line:{color:C.black,width:0.8}}); addText(s,'solid',4.48,4.37,0.70,0.26,{fontSize:12,align:'left'});
  s.addShape(SH.rect,{x:4.18,y:4.78,w:0.24,h:0.24,fill:{color:C.white},line:{color:C.red,width:1.1}}); addText(s,'water',4.48,4.77,0.70,0.26,{fontSize:12,align:'left'});
  s.addShape(SH.rect,{x:4.18,y:5.18,w:0.24,h:0.24,fill:{color:C.white},line:{color:C.navy,width:1.1}}); addText(s,'air',4.48,5.17,0.70,0.26,{fontSize:12,align:'left'});
  s.addShape(SH.roundRect,{x:5.55,y:1.65,w:1.78,h:0.95,fill:{color:C.white},line:{color:C.blue,width:1.2}});
  addText(s,'homogeneous field\n+\npost-excitation',5.68,1.75,1.52,0.72,{fontSize:13,color:C.blue});
  s.addShape(SH.chevron,{x:5.66,y:3.13,w:1.60,h:0.45,fill:{color:C.navy},line:{color:C.navy}});
  s.addShape(SH.roundRect,{x:7.84,y:1.27,w:4.18,h:1.35,fill:{color:C.white},line:{color:C.blue,width:1.4}});
  addText(s,'∂M/∂t = D∇²M − M/T₂B',8.15,1.62,3.58,0.58,{fontFace:'Cambria Math',fontSize:26});
  s.addShape(SH.rect,{x:7.66,y:3.02,w:3.10,h:2.50,fill:{color:'FFEAA4'},line:{color:C.black,width:1}});
  [[8.05,3.45,1.55,0.9],[8.95,3.3,1.45,1.0],[8.2,4.15,1.7,0.95],[9.25,4.15,1.2,0.8]].forEach(b=>s.addShape(SH.ellipse,{x:b[0],y:b[1],w:b[2],h:b[3],fill:{color:'74D0F5',transparency:10},line:{color:C.red,width:1.2}}));
  s.addShape(SH.line,{x:10.45,y:3.34,w:1.08,h:-0.2,line:{color:C.red,width:1.4}}); addText(s,'solid–fluid',11.55,3.02,1.05,0.30,{fontSize:12,bold:true,color:C.red});
  addText(s,'Robin relaxation',11.2,3.35,1.65,0.30,{fontSize:12,bold:true}); addText(s,'D∂ₙM + ρₛM = 0',11.10,3.73,1.70,0.42,{fontFace:'Cambria Math',fontSize:15,color:C.black});
  s.addShape(SH.line,{x:10.43,y:4.86,w:1.08,h:0.0,line:{color:C.blue,width:1.4}}); addText(s,'water–air',11.55,4.67,1.05,0.30,{fontSize:12,bold:true,color:C.blue});
  addText(s,'zero flux',11.3,4.97,1.45,0.30,{fontSize:12,bold:true}); addText(s,'D∂ₙM = 0',11.28,5.31,1.40,0.40,{fontFace:'Cambria Math',fontSize:15,color:C.black});
  s.addShape(SH.roundRect,{x:8.22,y:5.84,w:2.50,h:0.55,fill:{color:C.white},line:{color:C.blue,width:1.2}}); addText(s,'M(x,0) = 1',8.45,5.92,2.05,0.36,{fontFace:'Cambria Math',fontSize:21});
  return s;
}

function slide3() {
  const s=pptx.addSlide(); s.background={color:C.white}; addTitle(s,'CT Geometry → FEM Domain');
  s.addShape(SH.line,{x:0.55,y:1.34,w:6.42,h:0,line:{color:C.blue,width:1.4}});
  s.addShape(SH.line,{x:0.55,y:1.34,w:0,h:0.27,line:{color:C.blue,width:1.4}}); s.addShape(SH.line,{x:6.97,y:1.34,w:0,h:0.27,line:{color:C.blue,width:1.4}});
  s.addShape(SH.roundRect,{x:1.18,y:1.08,w:5.15,h:0.43,fill:{color:C.white},line:{color:C.blue,width:1.2}}); addText(s,'contiguous porosity groups:  max(ϕ) − min(ϕ) ≤ 0.01',1.28,1.14,4.95,0.28,{fontSize:14,color:C.blue});
  const xs=[0.28,3.45,6.88,10.15]; const ws=[2.88,3.00,2.90,2.88]; const heads=['3D CT stack','Representative 2D slice','Interface classification','Boundary-refined mesh'];
  xs.forEach((x,i)=>{card(s,x,1.72,ws[i],4.07); addText(s,heads[i],x+0.18,1.90,ws[i]-0.36,0.35,{fontSize:14,bold:true});});
  addImage(s,path.join(ASSET,'ct_stack.png'),0.60,2.25,2.18,2.98); addImage(s,path.join(ASSET,'segmented_slice.png'),3.64,2.38,2.62,2.62); addImage(s,path.join(ASSET,'interface_authoritative_fallback.png'),7.13,2.28,2.34,2.75); addImage(s,path.join(ASSET,'fem_mesh.png'),10.37,2.30,2.42,2.78);
  arrow(s,3.13,3.45); arrow(s,6.56,3.45); arrow(s,9.83,3.45);
  lineLegend(s,7.18,5.03,C.red,'solid','solid–fluid edge'); lineLegend(s,7.18,5.30,C.blue,'dash','water–air edge');
  phaseLegend(s,4.35,6.10); addText(s,'Ωw = water-occupied domain',4.00,6.67,5.45,0.45,{fontFace:'Cambria Math',fontSize:24,bold:true});
  return s;
}

function slide4() {
  const s=pptx.addSlide(); s.background={color:C.white}; addTitle(s,'Magnetization Evolution → NMR Decay');
  const heads=[['diffusion','↝'],['bulk relaxation','↑'],['surface relaxation','↓↓↓']];
  heads.forEach((h,i)=>{addText(s,h[1],0.95+i*2.92,0.95,1.2,0.52,{fontSize:27,bold:true,color:C.blue}); addText(s,h[0],0.72+i*2.90,1.55,1.65,0.35,{fontSize:15});});
  const imgs=['mag_t0_authoritative_fallback.png','mag_t1_authoritative_fallback.png','mag_t2_authoritative_fallback.png'];
  imgs.forEach((im,i)=>{addImage(s,path.join(ASSET,im),0.38+i*2.88,2.05,2.32,2.88); addText(s,`t${['₀','₁','₂'][i]}`,1.16+i*2.88,4.98,0.72,0.38,{fontFace:'Cambria Math',fontSize:22,italic:true}); if(i<2) arrow(s,2.78+i*2.88,3.23,0.35,0.27);});
  const d=decayData(); addLineChart(s,[{name:'normalized decay',labels:d.labels,values:d.values}],9.55,1.92,3.20,3.20,{catAxisTitle:'t',valAxisTitle:'normalized decay',colors:[C.black]});
  addText(s,'S(t) =',4.55,5.65,1.20,0.48,{fontFace:'Cambria Math',fontSize:24,align:'right'});
  addText(s,'∫',5.82,5.58,0.48,0.62,{fontFace:'Cambria Math',fontSize:31});
  addText(s,'Ωw',5.92,5.98,0.48,0.25,{fontFace:'Cambria Math',fontSize:13,italic:true});
  addText(s,'M(x,t) dΩ',6.28,5.65,2.25,0.48,{fontFace:'Cambria Math',fontSize:24,align:'left'});
  addText(s,'(M + Δt DK + Δt M/T₂B + Δt ρₛRₛ) mⁿ⁺¹ = M mⁿ',2.72,6.45,7.98,0.42,{fontFace:'Cambria Math',fontSize:20});
  return s;
}

function slide5() {
  const s=pptx.addSlide(); s.background={color:C.white}; addTitle(s,'Representative Slices → T₂ Spectrum');
  addText(s,'Porosity groups',0.28,1.13,3.48,0.35,{fontSize:15,bold:true}); addText(s,'Representative decays',4.05,1.13,2.55,0.35,{fontSize:15,bold:true}); addText(s,'Weighted decay',7.33,1.13,2.1,0.35,{fontSize:15,bold:true}); addText(s,'NNLS / L-curve',10.35,1.13,2.35,0.35,{fontSize:15,bold:true});
  addImage(s,path.join(ASSET,'porosity_groups.png'),0.18,1.55,3.72,3.85); arrow(s,3.95,3.12);
  const labels=['10⁻¹','10⁰','10¹','10²'];
  for(let i=0;i<3;i++) {
    const vals=[1,0.98,0.32-i*0.03,0.03]; addLineChart(s,[{name:`Sg${i+1}`,labels,values:vals}],4.30,1.64+i*1.15,1.85,0.90,{catAxisTitle:'t (ms)',colors:[C.blue]});
    addText(s,`weight w${['₁','₂','₃'][i]}\n(N${['₁','₂','₃'][i]} slices)`,6.12,1.77+i*1.15,1.15,0.48,{fontSize:11,color:C.blue,align:'left'});
  }
  arrow(s,7.27,3.08); addLineChart(s,[{name:'S̄(t)',labels,values:[1,0.96,0.36,0.04]}],7.63,2.02,2.12,2.58,{catAxisTitle:'t (ms)',valAxisTitle:'S̄(t)',colors:[C.blue]});
  arrow(s,9.89,3.08); const a=t2SmoothData(0),b=t2SmoothData(-0.08); addLineChart(s,[{name:'Simulation',labels:a.labels,values:a.values},{name:'Experiment',labels:b.labels,values:b.values}],10.25,1.88,2.80,2.85,{catAxisTitle:'T₂ (ms)',valAxisTitle:'Amplitude (a.u.)',colors:[C.blue,C.green],showLegend:false,logVal:true,min:0.00001,max:1});
  lineLegend(s,10.98,1.53,C.blue,'solid','Simulation');
  lineLegend(s,11.95,1.53,C.green,'solid','Experiment');
  addText(s,'2D representative-slice approximation — not a full 3D solve',4.05,5.72,5.35,0.30,{fontSize:13,bold:true});
  s.addShape(SH.roundRect,{x:0.98,y:6.06,w:11.35,h:0.92,fill:{color:C.white},line:{color:C.blue,width:1.2}}); s.addShape(SH.line,{x:6.66,y:6.20,w:0,h:0.64,line:{color:C.blue,width:1.2}});
  s.addText([{text:'S̄(t) = Σ',options:{}},{text:'g',options:{subscript:true}},{text:' w',options:{}},{text:'g',options:{subscript:true}},{text:' S',options:{}},{text:'g',options:{subscript:true}},{text:'(t) / Σ',options:{}},{text:'g',options:{subscript:true}},{text:' w',options:{}},{text:'g',options:{subscript:true}}],{x:1.22,y:6.25,w:5.10,h:0.46,fontFace:'Cambria Math',fontSize:22,color:C.blue,margin:0,breakLine:false});
  addText(s,'min',7.10,6.18,0.72,0.33,{fontFace:'Cambria Math',fontSize:21,color:C.blue});
  addText(s,'f ≥ 0',7.12,6.52,0.68,0.24,{fontFace:'Cambria Math',fontSize:12,color:C.blue});
  addText(s,'‖Kf − s‖² + α²‖Lf‖²',7.82,6.25,3.90,0.46,{fontFace:'Cambria Math',fontSize:21,color:C.blue});
  return s;
}

async function main(){
  fs.mkdirSync(FINAL,{recursive:true}); await prepareFallbacks();
  slide1(); slide2(); slide3(); slide4(); slide5();
  const out=path.join(FINAL,'nmr_t2_workflow_editable.pptx');
  await pptx.writeFile({fileName:out}); console.log(out);
}
main().catch(e=>{console.error(e);process.exit(1);});
