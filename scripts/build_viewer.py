"""Build figures/axion_wavepacket_viewer.html, a self-contained interactive
viewer of data/run_main.npz: the field is resampled to 640 x-points, quantized
to 16-bit integers, base64-encoded, and inlined into the HTML below (~1 MB)."""
import base64
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent    # the project directory
sys.path.insert(0, str(ROOT / "code"))           # so the solver imports work
DATA, FIGS = ROOT / "data", ROOT / "figures"

from axion_solver import Params

p = Params()
d = np.load(DATA / "run_main.npz")
x, t, E = d["x"], d["t"], d["E"] / p.gB0a0

XMIN, XMAX = -3.0, 55.0
m = (x >= XMIN) & (x <= XMAX)
xs = x[m]
# resample x to ~640 columns
NX = 640
xg = np.linspace(xs[0], xs[-1], NX)
Eg = np.array([np.interp(xg, xs, row) for row in E[:, m]], dtype=np.float32)

scale = float(np.abs(Eg).max())
q = np.clip(np.round(Eg / scale * 32000.0), -32767, 32767).astype("<i2")
b64 = base64.b64encode(q.tobytes()).decode()

wp = (p.omega_p(xg) / p.m_a).astype(np.float32)
meta = dict(
    nt=int(Eg.shape[0]), nx=NX, xmin=float(xg[0]), xmax=float(xg[-1]),
    scale=scale, t=[round(float(v), 2) for v in t],
    wp=[round(float(v), 4) for v in wp],
    edges=[i * p.L_region for i in range(6)],
    wlab=[1.2, 1.1, 1.0, 0.9, 0.8],
    Lbox=p.L_box, x0=p.x0, va=p.v_a, sigma=p.sigma_x,
)

HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">
<title>Axion wavepacket in a density-graded plasma</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500&family=IBM+Plex+Sans:wght@400;500;600&display=swap" rel="stylesheet">
<style>
:root{
  --paper:#FBFAF7; --ink:#141A21; --grid:#DFDAD0; --rule:#C8C2B6;
  --trace:#1F3F7A; --trace2:#6E86B8; --res:#B23A2E; --packet:#D98A1F;
  --muted:#6C6960; --chip:#EFEBE3; --accent:#1F3F7A;
  box-sizing:border-box;
  padding-top:env(safe-area-inset-top,0px); padding-bottom:env(safe-area-inset-bottom,0px);
}
@media (prefers-color-scheme: dark){
  :root:not([data-theme="light"]){
    --paper:#0E1216; --ink:#E6E9EC; --grid:#232A32; --rule:#39424D;
    --trace:#7FC4E8; --trace2:#3D6A8A; --res:#FF7A66; --packet:#F0B45C;
    --muted:#8A949E; --chip:#19202708; --accent:#7FC4E8;
  }
}
:root[data-theme="dark"]{
  --paper:#0E1216; --ink:#E6E9EC; --grid:#232A32; --rule:#39424D;
  --trace:#7FC4E8; --trace2:#3D6A8A; --res:#FF7A66; --packet:#F0B45C;
  --muted:#8A949E; --chip:#192027; --accent:#7FC4E8;
}
*,*::before,*::after{box-sizing:inherit}
html{height:100%}
body{
  margin:0; min-height:100%; background:var(--paper); color:var(--ink);
  font-family:"IBM Plex Sans",-apple-system,BlinkMacSystemFont,"Segoe UI",Helvetica,Arial,sans-serif;
  font-size:15px; line-height:1.5; -webkit-font-smoothing:antialiased;
}
.wrap{max-width:1180px; margin:0 auto; padding:22px 18px 30px}
header{display:flex; flex-wrap:wrap; align-items:baseline; gap:10px 16px; margin-bottom:6px}
h1{font-size:19px; font-weight:600; letter-spacing:-0.01em; margin:0}
.sub{color:var(--muted); font-size:13.5px; margin:2px 0 18px; max-width:72ch}
canvas{display:block; width:100%; height:auto; touch-action:pan-y}
.stage{border-top:1px solid var(--rule); border-bottom:1px solid var(--rule); padding:6px 0 2px}
.bar{display:flex; flex-wrap:wrap; align-items:center; gap:12px; margin-top:14px}
button{
  font:inherit; font-size:14px; font-weight:500; color:var(--paper); background:var(--accent);
  border:0; border-radius:3px; padding:7px 16px; cursor:pointer; min-width:84px;
}
button:hover{opacity:.88}
button:focus-visible,input:focus-visible{outline:2px solid var(--res); outline-offset:2px}
button.ghost{background:transparent; color:var(--muted); border:1px solid var(--rule); min-width:0; padding:6px 11px}
button.ghost[aria-pressed="true"]{color:var(--accent); border-color:var(--accent)}
input[type=range]{flex:1 1 240px; min-width:180px; accent-color:var(--accent); height:22px}
.readout{font-family:"IBM Plex Mono",ui-monospace,monospace; font-size:14px; font-variant-numeric:tabular-nums; white-space:nowrap}
.readout b{font-weight:500}
.legend{display:flex; flex-wrap:wrap; gap:6px 20px; margin-top:16px; color:var(--muted); font-size:12.5px}
.legend i{display:inline-block; width:22px; height:2px; vertical-align:middle; margin-right:7px}
.note{margin-top:20px; padding-top:14px; border-top:1px solid var(--grid); color:var(--muted); font-size:13px; max-width:78ch}
.note b{color:var(--ink); font-weight:500}
@media (prefers-reduced-motion: reduce){button{transition:none}}
</style>
</head>
<body>
<div class="wrap">
  <header>
    <h1>Axion wavepacket crossing a density-graded plasma</h1>
    <span class="readout" style="color:var(--muted)">1D &middot; O-mode &middot; no backreaction</span>
  </header>
  <p class="sub">A Gaussian axion packet moves right at <span class="readout">v<sub>a</sub>=0.1</span> through five plasma slabs of falling density, in a fixed <span class="readout">B<sub>0</sub>&#374;</span>. Watch the field grow as the packet reaches the slab where <span class="readout">&omega;<sub>p</sub>=m<sub>a</sub></span>, then shed a wave that outruns it into the low-density side.</p>

  <div class="stage"><canvas id="c" width="1160" height="520"></canvas></div>

  <div class="bar">
    <button id="play">Pause</button>
    <input type="range" id="scrub" min="0" value="0" step="1">
    <span class="readout">t = <b id="tv">0.0</b></span>
    <button class="ghost" id="trail" aria-pressed="true">Running max</button>
    <button class="ghost" id="slow" aria-pressed="false">Half speed</button>
  </div>

  <div class="legend">
    <span><i style="background:var(--trace)"></i>E<sub>y</sub>(x)</span>
    <span><i style="background:var(--res)"></i>running max of |E<sub>y</sub>|</span>
    <span><i style="background:var(--packet)"></i>axion packet envelope</span>
    <span><i style="background:var(--rule)"></i>slab boundaries</span>
  </div>

  <p class="note">Field in units of <span class="readout">g<sub>a&gamma;&gamma;</sub>B<sub>0</sub>a<sub>0</sub></span>. The shaded band past <span class="readout">x=24.6</span> is the extension added so outgoing radiation leaves without reflecting. The largest field ends up in the <span class="readout">&omega;<sub>p</sub>=0.9</span> slab rather than the resonant one: the driven response <b>flips sign</b> across the crossing, and the old field is shed as a free wave that appears downstream.</p>
</div>

<script>
const META = __META__;
const B64  = "__DATA__";

function decode(b64){
  const bin = atob(b64), n = bin.length, u8 = new Uint8Array(n);
  for(let i=0;i<n;i++) u8[i]=bin.charCodeAt(i);
  return new Int16Array(u8.buffer);
}
const RAW = decode(B64);
const {nt,nx,xmin,xmax,scale,t,wp,edges,wlab,Lbox,x0,va,sigma} = META;
const frame = j => RAW.subarray(j*nx,(j+1)*nx);
const S = 32000.0;

let EMAX = 0;
for(let i=0;i<RAW.length;i++){const v=Math.abs(RAW[i]); if(v>EMAX) EMAX=v;}
EMAX = EMAX/S*scale*1.08;

const cv = document.getElementById("c"), ctx = cv.getContext("2d");
const scrub = document.getElementById("scrub"); scrub.max = nt-1;
const tv = document.getElementById("tv"), playBtn = document.getElementById("play");
const trailBtn = document.getElementById("trail"), slowBtn = document.getElementById("slow");

let j = 0, playing = true, showTrail = true, half = false;
let running = new Float32Array(nx);

const css = k => getComputedStyle(document.documentElement).getPropertyValue(k).trim();

function layout(){
  const w = cv.clientWidth || 1160;
  const dpr = Math.min(window.devicePixelRatio||1, 2);
  const h = Math.max(360, Math.min(560, w*0.45));
  cv.width = w*dpr; cv.height = h*dpr; cv.style.height = h+"px";
  ctx.setTransform(dpr,0,0,dpr,0,0);
  return {w,h};
}

const PADL = 54, PADR = 14;
function draw(){
  const {w,h} = layout();
  const P = css("--paper"), INK = css("--ink"), GR = css("--grid"), RU = css("--rule");
  const TR = css("--trace"), RS = css("--res"), PK = css("--packet"), MU = css("--muted");
  ctx.clearRect(0,0,w,h);

  const topH = h*0.17, midT = topH+14, midH = h*0.50, botT = midT+midH+26, botH = h-botT-24;
  const X = v => PADL + (v-xmin)/(xmax-xmin)*(w-PADL-PADR);

  ctx.font = '11px "IBM Plex Mono", monospace';
  ctx.textBaseline = "middle";

  // ---------- slab shading (all panels) ----------
  for(let i=0;i<5;i++){
    const a = X(edges[i]), b = X(edges[i+1]);
    const near = 1 - Math.min(Math.abs(wlab[i]-1.0)/0.2, 1);
    ctx.fillStyle = RS; ctx.globalAlpha = 0.05*near;
    ctx.fillRect(a, 0, b-a, h-20); ctx.globalAlpha = 1;
  }
  ctx.fillStyle = GR; ctx.globalAlpha=0.45;
  ctx.fillRect(X(Lbox), 0, X(xmax)-X(Lbox), h-20); ctx.globalAlpha=1;

  // ---------- density strip ----------
  ctx.strokeStyle = RU; ctx.lineWidth = 1;
  ctx.setLineDash([3,3]);
  const yr = topH - (1.0-0.72)/(1.30-0.72)*topH;
  ctx.beginPath(); ctx.moveTo(PADL,yr); ctx.lineTo(w-PADR,yr); ctx.strokeStyle=RS; ctx.stroke();
  ctx.setLineDash([]);
  ctx.strokeStyle = INK; ctx.lineWidth = 1.5; ctx.beginPath();
  for(let i=0;i<nx;i++){
    const xx = xmin + (xmax-xmin)*i/(nx-1);
    const yy = topH - (wp[i]-0.72)/(1.30-0.72)*topH;
    i? ctx.lineTo(X(xx),yy) : ctx.moveTo(X(xx),yy);
  }
  ctx.stroke();
  ctx.fillStyle = MU; ctx.textAlign="right";
  ctx.fillText("ωp/ma", PADL-8, topH*0.5);
  ctx.fillStyle = RS; ctx.textAlign="left";
  ctx.fillText("ωp = ma", X(xmax)-72, yr-9);

  // ---------- slab dividers + labels ----------
  ctx.strokeStyle = RU; ctx.lineWidth=1; ctx.setLineDash([2,4]);
  for(const e of edges){ ctx.beginPath(); ctx.moveTo(X(e),0); ctx.lineTo(X(e),h-20); ctx.stroke(); }
  ctx.setLineDash([]);
  ctx.fillStyle = MU; ctx.textAlign="center";
  for(let i=0;i<5;i++) ctx.fillText(wlab[i].toFixed(1), (X(edges[i])+X(edges[i+1]))/2, h-9);
  ctx.fillText("extension", (X(Lbox)+X(xmax))/2, h-9);

  // ---------- packet envelope ----------
  const xc = x0 + va*t[j];
  const Y0 = midT + midH/2;
  ctx.fillStyle = PK; ctx.globalAlpha = 0.14; ctx.beginPath();
  for(let i=0;i<nx;i++){
    const xx = xmin + (xmax-xmin)*i/(nx-1);
    const en = Math.exp(-0.5*Math.pow((xx-xc)/sigma,2));
    i? ctx.lineTo(X(xx), Y0-en*midH/2) : ctx.moveTo(X(xx), Y0-en*midH/2);
  }
  for(let i=nx-1;i>=0;i--){
    const xx = xmin + (xmax-xmin)*i/(nx-1);
    const en = Math.exp(-0.5*Math.pow((xx-xc)/sigma,2));
    ctx.lineTo(X(xx), Y0+en*midH/2);
  }
  ctx.closePath(); ctx.fill(); ctx.globalAlpha=1;
  ctx.strokeStyle = PK; ctx.lineWidth=1; ctx.setLineDash([4,4]);
  ctx.beginPath(); ctx.moveTo(X(xc),midT); ctx.lineTo(X(xc),midT+midH); ctx.stroke();
  ctx.setLineDash([]);

  // ---------- axis + trace ----------
  ctx.strokeStyle = GR; ctx.lineWidth=1;
  ctx.beginPath(); ctx.moveTo(PADL,Y0); ctx.lineTo(w-PADR,Y0); ctx.stroke();
  const f = frame(j);
  ctx.strokeStyle = TR; ctx.lineWidth=1.3; ctx.beginPath();
  for(let i=0;i<nx;i++){
    const v = f[i]/S*scale;
    const yy = Y0 - v/EMAX*(midH/2);
    i? ctx.lineTo(PADL+i/(nx-1)*(w-PADL-PADR), yy) : ctx.moveTo(PADL, yy);
  }
  ctx.stroke();
  ctx.fillStyle = MU; ctx.textAlign="right";
  ctx.fillText("+"+EMAX.toFixed(1), PADL-8, midT+6);
  ctx.fillText("0", PADL-8, Y0);
  ctx.fillText("-"+EMAX.toFixed(1), PADL-8, midT+midH-6);
  ctx.save(); ctx.translate(15, Y0); ctx.rotate(-Math.PI/2);
  ctx.textAlign="center"; ctx.fillStyle=MU;
  ctx.font='11px "IBM Plex Sans", sans-serif';
  ctx.fillText("Ey / (gaγγ B0 a0)", 0, 0); ctx.restore();

  // ---------- envelope panel ----------
  ctx.font = '11px "IBM Plex Mono", monospace';
  const B0 = botT+botH;
  ctx.strokeStyle = GR; ctx.beginPath(); ctx.moveTo(PADL,B0); ctx.lineTo(w-PADR,B0); ctx.stroke();
  if(showTrail){
    ctx.strokeStyle = RS; ctx.lineWidth=1.2; ctx.setLineDash([5,3]); ctx.beginPath();
    for(let i=0;i<nx;i++){
      const yy = B0 - running[i]/EMAX*botH;
      i? ctx.lineTo(PADL+i/(nx-1)*(w-PADL-PADR), yy) : ctx.moveTo(PADL, yy);
    }
    ctx.stroke(); ctx.setLineDash([]);
  }
  ctx.strokeStyle = TR; ctx.lineWidth=1.1; ctx.beginPath();
  for(let i=0;i<nx;i++){
    const yy = B0 - Math.abs(f[i]/S*scale)/EMAX*botH;
    i? ctx.lineTo(PADL+i/(nx-1)*(w-PADL-PADR), yy) : ctx.moveTo(PADL, yy);
  }
  ctx.stroke();
  ctx.fillStyle = MU; ctx.textAlign="right"; ctx.fillText("|Ey|", PADL-8, botT+botH/2);
  ctx.textAlign="center"; ctx.fillStyle=MU;
  ctx.font='11px "IBM Plex Sans", sans-serif';
  ctx.fillText("x", (PADL+w-PADR)/2, h-9+0.001);
}

function setFrame(k, resetTrail){
  j = k;
  if(resetTrail){ running = new Float32Array(nx); for(let q=0;q<=j;q++){const f=frame(q);
      for(let i=0;i<nx;i++){const v=Math.abs(f[i]/S*scale); if(v>running[i]) running[i]=v;} } }
  else { const f=frame(j); for(let i=0;i<nx;i++){const v=Math.abs(f[i]/S*scale); if(v>running[i]) running[i]=v;} }
  scrub.value = j; tv.textContent = t[j].toFixed(1);
  draw();
}

let last = 0;
function loop(ts){
  if(playing){
    const step = half ? 80 : 40;
    if(ts-last > step){
      last = ts;
      let k = j+1; if(k>=nt){ k=0; running=new Float32Array(nx); }
      setFrame(k,false);
    }
  }
  requestAnimationFrame(loop);
}

playBtn.onclick = ()=>{ playing=!playing; playBtn.textContent = playing?"Pause":"Play"; };
scrub.oninput = ()=>{ playing=false; playBtn.textContent="Play"; setFrame(+scrub.value,true); };
trailBtn.onclick = ()=>{ showTrail=!showTrail; trailBtn.setAttribute("aria-pressed",showTrail); draw(); };
slowBtn.onclick  = ()=>{ half=!half; slowBtn.setAttribute("aria-pressed",half); };
window.addEventListener("resize", draw);
matchMedia("(prefers-color-scheme: dark)").addEventListener("change", draw);
document.fonts && document.fonts.ready.then(draw);

setFrame(0,true);
requestAnimationFrame(loop);
</script>
</body>
</html>
"""

out = HTML.replace("__META__", json.dumps(meta)).replace("__DATA__", b64)
open(FIGS / "axion_wavepacket_viewer.html", "w").write(out)
print("bytes:", len(out) / 1e6, "MB")
