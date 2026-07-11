"""
Chrome & Sky monorail trainset generator → monorail.glb

Authors three car variants (lead / mid / tail) as named node groups in one GLB,
sized to drop into the Tomorrow hero scene (hero3d.js) at scale 1:
  - car local +X = direction of travel, +Y up (three.js convention)
  - hull spans y -1.42..+1.06 (origin at hull center; the set straddles the
    beam exactly like the procedural cars it replaces, lift 0.74)
  - mid car length 5.05; lead/tail carry a 1.85-long drooping streamlined nose

Construction: every part is a superellipse tube lofted along X — hull, glass
band, crimson beltline, skirt, chrome roof spine — and the nose taper scales
all of them coherently, so the glass wraps into a windshield and the beltline
rings the nose by construction. ~23k tris for the whole set.

License of output: CC0 1.0 (public domain). Generated art for the Genentech
DDA "Tomorrow" series prototype.
"""
import numpy as np
import trimesh
from trimesh.visual.material import PBRMaterial
from trimesh.visual import TextureVisuals

M = 64          # segments around each ring
OUT = 'monorail.glb'

PARTS = {
    #        a      b      c      n    material
    'hull':  (0.99, 1.24, -0.18, 2.6),
    'glass': (0.945, 0.31, 0.42, 2.2),
    'belt':  (1.005, 0.07, -0.24, 2.0),
    'skirt': (0.72, 0.20, -1.22, 2.2),
    'spine': (0.20, 0.075, 1.02, 2.0),
}

MATS = {
    'hull':  PBRMaterial(name='hull',  baseColorFactor=[0.956, 0.972, 0.984, 1.0], metallicFactor=0.55, roughnessFactor=0.28),
    'glass': PBRMaterial(name='glass', baseColorFactor=[0.055, 0.23, 0.39, 1.0], metallicFactor=0.25, roughnessFactor=0.12,
                         emissiveFactor=[0.10, 0.34, 0.50]),
    'belt':  PBRMaterial(name='belt',  baseColorFactor=[0.847, 0.20, 0.18, 1.0], metallicFactor=0.35, roughnessFactor=0.42,
                         emissiveFactor=[0.14, 0.02, 0.02]),
    'skirt': PBRMaterial(name='skirt', baseColorFactor=[0.70, 0.76, 0.81, 1.0], metallicFactor=0.60, roughnessFactor=0.45),
    'spine': PBRMaterial(name='spine', baseColorFactor=[0.87, 0.91, 0.94, 1.0], metallicFactor=0.95, roughnessFactor=0.20),
}

L = 5.05          # car length
HALF = L / 2.0
NOSE = 1.85       # nose length on lead/tail cars


def smoothstep(t):
    return t * t * (3.0 - 2.0 * t)


def ring(a, b, c, n, scale, ydrop):
    th = np.linspace(0, 2 * np.pi, M, endpoint=False)
    ct, st = np.cos(th), np.sin(th)
    z = a * np.sign(ct) * np.abs(ct) ** (2.0 / n)
    y = b * np.sign(st) * np.abs(st) ** (2.0 / n)
    pts = np.zeros((M, 3))
    pts[:, 1] = y * scale + c + ydrop
    pts[:, 2] = z * scale
    return pts


def stations_straight():
    return [(-HALF + t * L, 1.0, 0.0) for t in np.linspace(0, 1, 4)]


def stations_nose():
    """Straight body, then a drooping taper to a rounded tip at +X."""
    sts = [(-HALF + t * (L - NOSE), 1.0, 0.0) for t in np.linspace(0, 1, 4)]
    for t in np.linspace(0.0, 1.0, 14)[1:]:
        e = smoothstep(t)
        scale = 1.0 - 0.90 * e ** 1.35
        ydrop = -0.36 * e ** 1.2
        sts.append((HALF - NOSE + t * NOSE, max(scale, 0.09), ydrop))
    return sts


def loft(sts, a, b, c, n):
    rings = []
    for (x, s, yd) in sts:
        r = ring(a, b, c, n, s, yd)
        r[:, 0] = x
        rings.append(r)
    V = np.vstack(rings)
    F = []
    S = len(sts)
    for i in range(S - 1):
        p, q = i * M, (i + 1) * M
        for j in range(M):
            j2 = (j + 1) % M
            F.append([p + j, q + j, q + j2])
            F.append([p + j, q + j2, p + j2])
    # end caps: fan to ring centroid
    verts = [V]
    extra = len(V)
    for (ring_start, flip) in ((0, True), ((S - 1) * M, False)):
        centroid = V[ring_start:ring_start + M].mean(axis=0, keepdims=True)
        verts.append(centroid)
        ci = extra
        extra += 1
        for j in range(M):
            j2 = (j + 1) % M
            tri = [ring_start + j, ring_start + j2, ci]
            if flip:
                tri = tri[::-1]
            F.append(tri)
    V = np.vstack(verts)
    mesh = trimesh.Trimesh(vertices=V, faces=np.array(F), process=False)
    # orient outward without scipy: a closed mesh wound inward has negative
    # signed volume
    if mesh.volume < 0:
        mesh.invert()
    return mesh


def build_car(kind):
    """kind: 'lead' (nose at +x), 'mid' (flat), 'tail' (nose at -x)"""
    sts = stations_nose() if kind in ('lead', 'tail') else stations_straight()
    meshes = {}
    for part, (a, b, c, n) in PARTS.items():
        m = loft(sts, a, b, c, n)
        if kind == 'tail':
            m.vertices[:, 0] *= -1.0
            m.invert()
        m.visual = TextureVisuals(material=MATS[part])
        meshes[part] = m
    return meshes


scene = trimesh.Scene()
for kind in ('lead', 'mid', 'tail'):
    for part, mesh in build_car(kind).items():
        name = f'{kind}_{part}'
        scene.add_geometry(mesh, node_name=name, geom_name=name)

scene.export(OUT)

check = trimesh.load(OUT)
tris = sum(g.faces.shape[0] for g in check.geometry.values())
print(f'wrote {OUT}: {len(check.geometry)} parts, {tris} tris')
print('bounds:', np.round(check.bounds, 3).tolist())
print('nodes:', sorted(check.graph.nodes_geometry))
