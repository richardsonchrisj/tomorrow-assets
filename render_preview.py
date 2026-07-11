"""Offline preview of monorail.glb — shaded orthographic views, no browser.
Renders the coupled 5-car set + a nose close-up to preview.png for visual
iteration on the generator."""
import numpy as np
import trimesh
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection

SPACING = 5.7
LIGHT = np.array([-0.45, 0.75, 0.5])
LIGHT = LIGHT / np.linalg.norm(LIGHT)

scene = trimesh.load('monorail.glb')

def car_meshes(prefix):
    out = []
    for name, geom in scene.geometry.items():
        if name.startswith(prefix):
            mat = geom.visual.material
            col = np.array(mat.baseColorFactor[:3], dtype=float) if mat.baseColorFactor is not None else np.array([0.8, 0.8, 0.8])
            if col.max() > 1.001: col = col / 255.0          # GLB round-trip stores 0-255
            emis = np.array(mat.emissiveFactor[:3], dtype=float) if getattr(mat, 'emissiveFactor', None) is not None else np.zeros(3)
            if emis.max() > 1.001: emis = emis / 255.0
            out.append((geom, col, emis))
    return out

CARS = [('lead_', 2 * SPACING), ('mid_', SPACING), ('mid_', 0), ('mid_', -SPACING), ('tail_', -2 * SPACING)]

def gather(rot):
    """Return (faces2d, depth, shade colors) for the whole set under rotation."""
    polys, depths, cols = [], [], []
    for prefix, xoff in CARS:
        for geom, col, emis in car_meshes(prefix):
            V = geom.vertices.copy()
            V[:, 0] += xoff
            Vr = V @ rot.T
            F = geom.faces
            tri = Vr[F]                       # (n,3,3)
            e1 = tri[:, 1] - tri[:, 0]
            e2 = tri[:, 2] - tri[:, 0]
            nrm = np.cross(e1, e2)
            nl = np.linalg.norm(nrm, axis=1); nl[nl == 0] = 1
            nrm = nrm / nl[:, None]
            vis = nrm[:, 2] > 0               # facing camera (+z out of screen)
            lam = np.clip(nrm @ (rot @ LIGHT), 0, 1)
            shade = 0.35 + 0.65 * lam
            c = np.clip(col[None, :] * shade[:, None] + emis[None, :] * 0.8, 0, 1)
            polys.append(tri[vis][:, :, :2])
            depths.append(tri[vis][:, :, 2].mean(axis=1))
            cols.append(c[vis])
    P = np.concatenate(polys); D = np.concatenate(depths); C = np.concatenate(cols)
    order = np.argsort(D)
    return P[order], C[order]

def rotmat(yaw, pitch):
    cy, sy = np.cos(yaw), np.sin(yaw)
    cp, sp = np.cos(pitch), np.sin(pitch)
    Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    Rx = np.array([[1, 0, 0], [0, cp, -sp], [0, sp, cp]])
    return Rx @ Ry

views = [
    ('side', rotmat(0.0, 0.06), None),
    ('three-quarter', rotmat(0.6, 0.22), None),
    ('nose close', rotmat(0.85, 0.18), (6.0, 18.0, -4.5, 3.5)),
]

fig, axes = plt.subplots(len(views), 1, figsize=(16, 13))
fig.patch.set_facecolor('#dfeaf2')
for ax, (title, rot, clip) in zip(axes, views):
    P, C = gather(rot)
    ax.add_collection(PolyCollection(P, facecolors=C, edgecolors='none'))
    ax.set_aspect('equal'); ax.set_facecolor('#dfeaf2')
    if clip:
        ax.set_xlim(clip[0], clip[1]); ax.set_ylim(clip[2], clip[3])
    else:
        ax.autoscale()
    ax.set_title(title, fontsize=10); ax.axis('off')
plt.tight_layout()
plt.savefig('preview.png', dpi=110)
print('wrote preview.png')
