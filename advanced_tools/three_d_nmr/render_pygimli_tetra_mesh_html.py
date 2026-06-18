#!/usr/bin/env python3
"""Render exported voxel-conformal pyGIMLi tetra mesh boundaries as HTML.

The companion ``pygimli_3d_tetra_nmr.py`` script writes
``pygimli_tetra_mesh_geometry.npz`` with mesh nodes and boundary triangles.
This renderer intentionally avoids pyvista so the mesh can still be visualized
in environments where VTK/PyVista is unavailable.
"""

from __future__ import annotations

import argparse
import html
import json
import sys
from pathlib import Path

import numpy as np


def _sample_triangles(triangles: np.ndarray, max_triangles: int) -> np.ndarray:
    if triangles.shape[0] <= max_triangles:
        return triangles
    pick = np.linspace(0, triangles.shape[0] - 1, max_triangles, dtype=np.int64)
    return triangles[pick]


def _unit_normals(vertices: np.ndarray) -> np.ndarray:
    tri = vertices.reshape(-1, 3, 3)
    normals = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    length = np.linalg.norm(normals, axis=1)
    normals = normals / np.maximum(length[:, None], 1e-30)
    return np.repeat(normals, 3, axis=0)


def _hex_rgb01(color: str) -> tuple[float, float, float]:
    clean = color.lstrip("#")
    return (
        int(clean[0:2], 16) / 255.0,
        int(clean[2:4], 16) / 255.0,
        int(clean[4:6], 16) / 255.0,
    )


def load_render_arrays(
    geometry_npz: Path,
    *,
    max_triangles: int,
    include_external: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, object]]:
    data = np.load(geometry_npz)
    nodes = np.asarray(data["nodes_xyz_um"], dtype=np.float32)
    solid = np.asarray(data["solid_boundary_triangles"], dtype=np.int64)
    external = np.asarray(data["external_boundary_triangles"], dtype=np.int64)

    groups: list[tuple[str, np.ndarray, str]] = [("solid_boundary", solid, "#9099a3")]
    if include_external and external.size:
        groups.append(("external_boundary", external, "#2f8f9d"))

    total_available = sum(group[1].shape[0] for group in groups)
    if total_available == 0:
        raise ValueError(f"No boundary triangles are available in {geometry_npz}")

    rendered_vertices: list[np.ndarray] = []
    rendered_colors: list[np.ndarray] = []
    sample_summary = []
    for name, triangles, color in groups:
        group_limit = max(1, int(round(max_triangles * triangles.shape[0] / total_available)))
        sampled = _sample_triangles(triangles, group_limit)
        verts = nodes[sampled].reshape(-1, 3)
        rgb = np.asarray(_hex_rgb01(color), dtype=np.float32)
        colors = np.tile(rgb, (verts.shape[0], 1))
        rendered_vertices.append(verts)
        rendered_colors.append(colors)
        sample_summary.append(
            {
                "name": name,
                "available_triangles": int(triangles.shape[0]),
                "rendered_triangles": int(sampled.shape[0]),
                "color": color,
            }
        )

    vertices_um = np.vstack(rendered_vertices).astype(np.float32, copy=False)
    colors = np.vstack(rendered_colors).astype(np.float32, copy=False)
    mins = np.min(nodes, axis=0)
    maxs = np.max(nodes, axis=0)
    center = (mins + maxs) * 0.5
    span = np.maximum(maxs - mins, 1e-9)
    scale = float(np.max(span))
    vertices = (vertices_um - center) / scale
    normals = _unit_normals(vertices).astype(np.float32, copy=False)
    metadata = {
        "geometry_npz": str(geometry_npz.resolve()),
        "node_count": int(nodes.shape[0]),
        "bbox_min_um": [float(v) for v in mins],
        "bbox_max_um": [float(v) for v in maxs],
        "bbox_span_um": [float(v) for v in span],
        "rendered_vertex_count": int(vertices.shape[0]),
        "rendered_triangle_count": int(vertices.shape[0] // 3),
        "max_triangles": int(max_triangles),
        "include_external": bool(include_external),
        "sample_summary": sample_summary,
    }
    return vertices, normals, colors, metadata


def _json_float_array(array: np.ndarray) -> str:
    return json.dumps(np.asarray(array, dtype=float).reshape(-1).round(6).tolist(), separators=(",", ":"))


def write_html(
    output_html: Path,
    *,
    title: str,
    vertices: np.ndarray,
    normals: np.ndarray,
    colors: np.ndarray,
    metadata: dict[str, object],
) -> None:
    escaped_title = html.escape(title)
    metadata_json = json.dumps(metadata, ensure_ascii=False, indent=2)
    page = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{escaped_title}</title>
  <style>
    html, body {{
      margin: 0;
      width: 100%;
      height: 100%;
      overflow: hidden;
      font-family: Arial, Helvetica, sans-serif;
      background: #f5f6f7;
      color: #1f2933;
    }}
    #canvas {{
      width: 100vw;
      height: 100vh;
      display: block;
    }}
    #panel {{
      position: fixed;
      right: 16px;
      top: 16px;
      width: min(360px, calc(100vw - 32px));
      padding: 12px 14px;
      border: 1px solid rgba(31, 41, 51, 0.18);
      border-radius: 8px;
      background: rgba(255, 255, 255, 0.92);
      box-shadow: 0 10px 28px rgba(31, 41, 51, 0.16);
      font-size: 13px;
      line-height: 1.35;
    }}
    #panel h1 {{
      margin: 0 0 8px;
      font-size: 15px;
      font-weight: 700;
      letter-spacing: 0;
    }}
    #panel label {{
      display: grid;
      grid-template-columns: 74px 1fr 44px;
      align-items: center;
      gap: 8px;
      margin-top: 8px;
    }}
    #meta {{
      max-height: 230px;
      overflow: auto;
      margin-top: 10px;
      padding: 8px;
      border-radius: 6px;
      background: #eef1f4;
      white-space: pre-wrap;
      font-size: 11px;
    }}
  </style>
</head>
<body>
<canvas id="canvas"></canvas>
<div id="panel">
  <h1>{escaped_title}</h1>
  <div>Drag to rotate, wheel to zoom. Boundary triangles are sampled for interactive rendering.</div>
  <label><span>Opacity</span><input id="opacity" type="range" min="0.05" max="1" step="0.01" value="0.70"><span id="opacityValue">0.70</span></label>
  <label><span>Light</span><input id="light" type="range" min="0.15" max="1" step="0.01" value="0.78"><span id="lightValue">0.78</span></label>
  <pre id="meta">{html.escape(metadata_json)}</pre>
</div>
<script>
const positions = new Float32Array({_json_float_array(vertices)});
const normals = new Float32Array({_json_float_array(normals)});
const colors = new Float32Array({_json_float_array(colors)});
const canvas = document.getElementById('canvas');
const gl = canvas.getContext('webgl', {{ alpha: false, antialias: true }});
if (!gl) {{
  document.body.innerHTML = '<div style="padding:24px;font-family:Arial">WebGL is unavailable in this browser.</div>';
}}
const vertexSource = `
attribute vec3 aPosition;
attribute vec3 aNormal;
attribute vec3 aColor;
uniform mat4 uMatrix;
uniform float uLight;
varying vec3 vColor;
void main() {{
  vec3 lightDir = normalize(vec3(0.4, 0.7, 0.55));
  float diffuse = max(dot(normalize(aNormal), lightDir), 0.0);
  float shade = 0.26 + uLight * diffuse;
  vColor = aColor * shade;
  gl_Position = uMatrix * vec4(aPosition, 1.0);
}}
`;
const fragmentSource = `
precision mediump float;
uniform float uOpacity;
varying vec3 vColor;
void main() {{
  gl_FragColor = vec4(vColor, uOpacity);
}}
`;
function shader(type, source) {{
  const s = gl.createShader(type);
  gl.shaderSource(s, source);
  gl.compileShader(s);
  if (!gl.getShaderParameter(s, gl.COMPILE_STATUS)) throw new Error(gl.getShaderInfoLog(s));
  return s;
}}
const program = gl.createProgram();
gl.attachShader(program, shader(gl.VERTEX_SHADER, vertexSource));
gl.attachShader(program, shader(gl.FRAGMENT_SHADER, fragmentSource));
gl.linkProgram(program);
if (!gl.getProgramParameter(program, gl.LINK_STATUS)) throw new Error(gl.getProgramInfoLog(program));
gl.useProgram(program);
function bindAttribute(name, data) {{
  const buffer = gl.createBuffer();
  gl.bindBuffer(gl.ARRAY_BUFFER, buffer);
  gl.bufferData(gl.ARRAY_BUFFER, data, gl.STATIC_DRAW);
  const loc = gl.getAttribLocation(program, name);
  gl.enableVertexAttribArray(loc);
  gl.vertexAttribPointer(loc, 3, gl.FLOAT, false, 0, 0);
}}
bindAttribute('aPosition', positions);
bindAttribute('aNormal', normals);
bindAttribute('aColor', colors);
const matrixLoc = gl.getUniformLocation(program, 'uMatrix');
const opacityLoc = gl.getUniformLocation(program, 'uOpacity');
const lightLoc = gl.getUniformLocation(program, 'uLight');
let rotX = -0.72, rotY = 0.64, zoom = 2.4;
let dragging = false, lastX = 0, lastY = 0;
function multiply(a, b) {{
  const out = new Float32Array(16);
  for (let r = 0; r < 4; r++) for (let c = 0; c < 4; c++) {{
    out[c * 4 + r] = a[0 * 4 + r] * b[c * 4 + 0] + a[1 * 4 + r] * b[c * 4 + 1] + a[2 * 4 + r] * b[c * 4 + 2] + a[3 * 4 + r] * b[c * 4 + 3];
  }}
  return out;
}}
function perspective(fovy, aspect, near, far) {{
  const f = 1 / Math.tan(fovy / 2);
  const nf = 1 / (near - far);
  return new Float32Array([f / aspect,0,0,0, 0,f,0,0, 0,0,(far + near) * nf,-1, 0,0,2 * far * near * nf,0]);
}}
function translate(z) {{
  return new Float32Array([1,0,0,0, 0,1,0,0, 0,0,1,0, 0,0,z,1]);
}}
function rotateX(a) {{
  const c = Math.cos(a), s = Math.sin(a);
  return new Float32Array([1,0,0,0, 0,c,s,0, 0,-s,c,0, 0,0,0,1]);
}}
function rotateY(a) {{
  const c = Math.cos(a), s = Math.sin(a);
  return new Float32Array([c,0,-s,0, 0,1,0,0, s,0,c,0, 0,0,0,1]);
}}
function resize() {{
  const dpr = Math.min(window.devicePixelRatio || 1, 2);
  const w = Math.floor(canvas.clientWidth * dpr);
  const h = Math.floor(canvas.clientHeight * dpr);
  if (canvas.width !== w || canvas.height !== h) {{
    canvas.width = w;
    canvas.height = h;
  }}
  gl.viewport(0, 0, canvas.width, canvas.height);
}}
function draw() {{
  resize();
  gl.clearColor(0.96, 0.97, 0.98, 1);
  gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);
  gl.enable(gl.DEPTH_TEST);
  gl.enable(gl.BLEND);
  gl.blendFunc(gl.SRC_ALPHA, gl.ONE_MINUS_SRC_ALPHA);
  const aspect = canvas.width / Math.max(canvas.height, 1);
  let m = perspective(Math.PI / 4, aspect, 0.01, 20);
  m = multiply(m, translate(-zoom));
  m = multiply(m, rotateX(rotX));
  m = multiply(m, rotateY(rotY));
  gl.uniformMatrix4fv(matrixLoc, false, m);
  gl.uniform1f(opacityLoc, Number(document.getElementById('opacity').value));
  gl.uniform1f(lightLoc, Number(document.getElementById('light').value));
  gl.drawArrays(gl.TRIANGLES, 0, positions.length / 3);
  requestAnimationFrame(draw);
}}
canvas.addEventListener('mousedown', (event) => {{ dragging = true; lastX = event.clientX; lastY = event.clientY; }});
window.addEventListener('mouseup', () => {{ dragging = false; }});
window.addEventListener('mousemove', (event) => {{
  if (!dragging) return;
  rotY += (event.clientX - lastX) * 0.008;
  rotX += (event.clientY - lastY) * 0.008;
  lastX = event.clientX;
  lastY = event.clientY;
}});
canvas.addEventListener('wheel', (event) => {{
  event.preventDefault();
  zoom = Math.max(0.8, Math.min(8.0, zoom * Math.exp(event.deltaY * 0.001)));
}}, {{ passive: false }});
for (const id of ['opacity', 'light']) {{
  const input = document.getElementById(id);
  const value = document.getElementById(id + 'Value');
  input.addEventListener('input', () => {{ value.textContent = Number(input.value).toFixed(2); }});
}}
draw();
</script>
</body>
</html>
"""
    output_html.parent.mkdir(parents=True, exist_ok=True)
    output_html.write_text(page, encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--geometry-npz", type=Path, required=True)
    parser.add_argument("--output-html", type=Path, required=True)
    parser.add_argument("--title", default="pyGIMLi tetra mesh boundary")
    parser.add_argument("--max-triangles", type=int, default=60000)
    parser.add_argument("--include-external", action="store_true")
    return parser.parse_args()


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    args = parse_args()
    vertices, normals, colors, metadata = load_render_arrays(
        args.geometry_npz,
        max_triangles=int(args.max_triangles),
        include_external=bool(args.include_external),
    )
    write_html(
        args.output_html,
        title=str(args.title),
        vertices=vertices,
        normals=normals,
        colors=colors,
        metadata=metadata,
    )
    print(json.dumps({"output_html": str(args.output_html.resolve()), **metadata}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
