"""Bounded, qualitative point clouds from relative depth; never metric reconstruction."""
import html
import json
from pathlib import Path

import numpy as np
from PIL import Image


def project_points(rgb, depth, intrinsics, max_points=10000):
    rgb, depth = np.asarray(rgb), np.asarray(depth, dtype=np.float64)
    matrix = np.asarray(intrinsics, dtype=np.float64)
    if rgb.ndim != 3 or rgb.shape[2] != 3 or rgb.dtype != np.uint8 or not rgb.size:
        raise ValueError("Point cloud expects nonempty uint8 RGB")
    if depth.ndim != 2 or not depth.size or not isinstance(max_points, int) or not 1 <= max_points <= 10000:
        raise ValueError("Point cloud expects 2D depth and a 1-10000 point budget")
    if matrix.shape != (3, 3) or not np.isfinite(matrix).all() or matrix[0, 0] <= 0 or matrix[1, 1] <= 0 or not np.allclose(matrix[2], [0, 0, 1]):
        raise ValueError("Point cloud requires finite pinhole intrinsics at depth-map resolution")
    # 1e6 is the reciprocal of the shared backend's inverse-depth floor, not measured range.
    valid = np.isfinite(depth) & (depth > 0) & (depth < 1e6)
    if not valid.any():
        raise ValueError("No usable depth for a relative point cloud")
    median = float(np.median(depth[valid]))
    normalized = depth / median
    retained = valid & (normalized <= 20)
    indices = np.flatnonzero(retained)
    if indices.size > max_points:
        indices = indices[np.linspace(0, indices.size - 1, max_points, dtype=np.int64)]
    rows, columns = np.unravel_index(indices, depth.shape)
    pixels = np.stack((columns, rows, np.ones(indices.size)))
    try:
        rays = np.linalg.solve(matrix, pixels)
    except np.linalg.LinAlgError:
        raise ValueError("Point cloud intrinsics are singular")
    points = (rays * normalized.flat[indices]).T.astype(np.float32)
    if not np.isfinite(points).all() or (points[:, 2] <= 0).any():
        raise ValueError("Point cloud projection overflowed or produced invalid geometry")
    aligned_rgb = np.asarray(Image.fromarray(rgb).resize((depth.shape[1], depth.shape[0]), Image.Resampling.BILINEAR))
    colours = aligned_rgb[rows, columns].copy()
    metadata = dict(point_count=len(points), max_points=max_points, map_shape=list(depth.shape),
                    intrinsics=matrix.tolist(), raw_depth_median=median,
                    coordinates="x right, y down, z forward; depth divided by valid scene median",
                    units="median-normalized relative units; not metres",
                    invalid_or_floor_pixels=int((~valid).sum()), far_pixels_filtered=int((valid & ~retained).sum()),
                    retained_before_sampling=int(retained.sum()), max_normalized_depth=20,
                    scope="Single-frame estimated surface projection; not calibrated 3D, occluded geometry or SLAM")
    return points, colours, metadata


VIEWER = r'''<!doctype html>
<html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Relative point-cloud snapshot</title>
<style>body{margin:0;background:#101923;color:#edf3fa;font:16px system-ui}header{padding:16px 22px}h1{font-size:22px;margin:0 0 8px}p{margin:6px 0;color:#bfcedd}button{padding:8px 16px;cursor:pointer}canvas{width:100%;height:70vh;display:block;touch-action:none;background:#080e16}footer{padding:12px 22px;color:#bfcedd}</style>
<header><h1>Relative point-cloud snapshot — __MODEL__</h1>
<p>Estimated from one RGB frame. Assumed/provided camera intrinsics; median scene depth = 1. No metre distances.</p>
<p>Drag to rotate · Scroll to zoom · or use the buttons:</p>
<p><button id="left" type="button">Rotate left</button> <button id="right" type="button">Rotate right</button>
<button id="in" type="button">Zoom in</button> <button id="out" type="button">Zoom out</button> <button id="reset" type="button">Reset view</button></p></header>
<canvas id="cloud" aria-label="Interactive coloured point cloud from estimated relative depth"></canvas>
<footer>__COUNT__ coloured points, at most 10,000. Invalid/floor values and depths beyond 20 scene medians are excluded.
This is a static first-frame snapshot, not a live 3D stream or calibrated reconstruction. Original colours are sampled from RGB.</footer>
<script>
const points=__DATA__,canvas=document.getElementById('cloud'),ctx=canvas.getContext('2d');
let yaw=-.35,pitch=.12,zoom=1,last=null;
const low=[0,1,2].map(i=>Math.min(...points.map(p=>p[i]))),high=[0,1,2].map(i=>Math.max(...points.map(p=>p[i])));
const centre=low.map((v,i)=>(v+high[i])/2),span=Math.max(...high.map((v,i)=>v-low[i]),.001);
function draw(){const dpr=window.devicePixelRatio||1,w=canvas.clientWidth,h=canvas.clientHeight;
 canvas.width=Math.round(w*dpr);canvas.height=Math.round(h*dpr);ctx.scale(dpr,dpr);ctx.clearRect(0,0,w,h);
 const cy=Math.cos(yaw),sy=Math.sin(yaw),cp=Math.cos(pitch),sp=Math.sin(pitch),scale=Math.min(w,h)*.85*zoom/span;
 const projected=points.map(p=>{const x=p[0]-centre[0],y=-(p[1]-centre[1]),z=p[2]-centre[2],rx=x*cy+z*sy,rz=-x*sy+z*cy;
 return [w/2+rx*scale,h/2-(y*cp-rz*sp)*scale,y*sp+rz*cp,p];}).sort((a,b)=>b[2]-a[2]);
 for(const [x,y,z,p] of projected){ctx.fillStyle=`rgb(${p[3]},${p[4]},${p[5]})`;ctx.fillRect(x,y,2,2);}}
canvas.addEventListener('pointerdown',e=>{last=[e.clientX,e.clientY];canvas.setPointerCapture(e.pointerId);});
canvas.addEventListener('pointermove',e=>{if(!last)return;yaw+=(e.clientX-last[0])*.008;pitch=Math.max(-1.5,Math.min(1.5,pitch+(e.clientY-last[1])*.008));last=[e.clientX,e.clientY];draw();});
for(const event of ['pointerup','pointercancel'])canvas.addEventListener(event,()=>last=null);
canvas.addEventListener('wheel',e=>{e.preventDefault();zoom=Math.max(.2,Math.min(8,zoom*Math.exp(-e.deltaY*.001)));draw();},{passive:false});
document.getElementById('reset').addEventListener('click',()=>{yaw=-.35;pitch=.12;zoom=1;draw();});
const actions={left:()=>yaw-=.2,right:()=>yaw+=.2,in:()=>zoom=Math.min(8,zoom*1.2),out:()=>zoom=Math.max(.2,zoom/1.2)};
for(const id of Object.keys(actions))document.getElementById(id).addEventListener('click',()=>{actions[id]();draw();});
new ResizeObserver(draw).observe(canvas);draw();
</script></html>'''


def export_point_cloud(output, rgb, depth, intrinsics, details):
    points, colours, metadata = project_points(rgb, depth, intrinsics)
    metadata.update(model_name=details["model_name"], backend=details["backend"],
                    intrinsics_source=details.get("intrinsics_source", "Assumed 90-degree FOV; not camera-calibrated"))
    folder = Path(output) / "point_cloud"
    folder.mkdir(exist_ok=False)
    records = np.empty(len(points), dtype=[("x", "<f4"), ("y", "<f4"), ("z", "<f4"), ("red", "u1"), ("green", "u1"), ("blue", "u1")])
    for index, name in enumerate(("x", "y", "z")):
        records[name] = points[:, index]
    for index, name in enumerate(("red", "green", "blue")):
        records[name] = colours[:, index]
    header = ("ply\nformat binary_little_endian 1.0\ncomment Estimated relative cloud; not metres\nelement vertex {}\n"
              "property float x\nproperty float y\nproperty float z\nproperty uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n").format(len(points))
    (folder / "cloud.ply").write_bytes(header.encode("ascii") + records.tobytes())
    data = np.column_stack((points.astype(np.float64).round(6), colours)).tolist()
    viewer = VIEWER.replace("__MODEL__", html.escape(metadata["model_name"])).replace("__COUNT__", str(len(points))).replace("__DATA__", json.dumps(data, separators=(",", ":"), allow_nan=False))
    (folder / "viewer.html").write_text(viewer, encoding="utf-8")
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    figure = Figure(figsize=(8, 5), facecolor="#101923")
    FigureCanvasAgg(figure)
    axis = figure.add_subplot(111, projection="3d", facecolor="#101923")
    axis.scatter(points[:, 0], points[:, 2], -points[:, 1], c=colours / 255.0, s=1, linewidths=0)
    ranges = np.maximum(np.ptp(points[:, [0, 2, 1]], axis=0), .01)
    axis.set_box_aspect(ranges)
    axis.set_axis_off()
    axis.view_init(elev=15, azim=-65)
    figure.suptitle("Estimated relative point cloud — not metric 3D", color="white", fontsize=13)
    figure.text(.5, .03, metadata["model_name"] + " | first-frame snapshot | assumed/provided intrinsics", ha="center", color="white", fontsize=9)
    figure.savefig(folder / "preview.png", dpi=130, facecolor=figure.get_facecolor())
    (folder / "metadata.json").write_text(json.dumps(metadata, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return dict(point_count=len(points), units=metadata["units"], scope=metadata["scope"],
                ply="point_cloud/cloud.ply", viewer="point_cloud/viewer.html", preview="point_cloud/preview.png",
                metadata="point_cloud/metadata.json")
