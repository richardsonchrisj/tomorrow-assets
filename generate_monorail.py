"""
Chrome & Sky monorail trainset generator v2 → monorail.glb

v2 craft pass: the tube-train becomes an industrial design.
  - hull: flatter sides (n=3.0) with tumblehome (width tapers above the belt)
  - glass: DISCRETE window panes with rhythm + a wraparound nose windshield
    (the full-length glass tube of v1 is gone)
  - livery: slim crimson beltline + navy pinstripe, crimson nose bib
  - detail: doors with seams, roof equipment pods, chrome spine, dark
    underframe + bogie masses, real headlight / tail-light lenses
Node naming unchanged (lead_* / mid_* / tail_*), same footprint (L=5.05,
hull y -1.42..+1.06, +X = travel) — drop-in for hero3d.js.

License of output: CC0 1.0. Generated art for the Genentech DDA "Tomorrow"
series prototype.
"""
import numpy as np
import trimesh
from trimesh.creation import box as make_box, cylinder as make_cyl
from trimesh.visual.material import PBRMaterial
from trimesh.visual import TextureVisuals

M = 72
OUT = 'monorail.glb'

L = 5.05
HALF = L / 2.0
NOSE = 1.85

# hull section params
HULL = dict(a=0.99, b=1.24, c=-0.18, n=3.0)
BELT_Y = -0.24          # tumblehome starts here
TUMBLE = 0.10           # width loss at the roofline

MATS = {
    'hull':   PBRMaterial(name='hull', baseColorFactor=[0.956, 0.972, 0.984, 1.0], metallicFactor=0.5, roughnessFactor=0.32),
    'glass':  PBRMaterial(name='glass', baseColorFactor=[0.045, 0.20, 0.35, 1.0], metallicFactor=0.3, roughnessFactor=0.10,
                          emissiveFactor=[0.09, 0.32, 0.48]),
    'belt':   PBRMaterial(name='belt', baseColorFactor=[0.847, 0.20, 0.18, 1.0], metallicFactor=0.35, roughnessFactor=0.42,
                          emissiveFactor=[0.13, 0.02, 0.02]),
    'stripe': PBRMaterial(name='stripe', baseColorFactor=[0.04, 0.17, 0.34, 1.0], metallicFactor=0.4, roughnessFactor=0.4),
    'skirt':  PBRMaterial(name='skirt', baseColorFactor=[0.663, 0.714, 0.753, 1.0], metallicFactor=0.55, roughnessFactor=0.5),
    'spine':  PBRMaterial(name='spine', baseColorFactor=[0.87, 0.91, 0.94, 1.0], metallicFactor=0.95, roughnessFactor=0.20),
    'door':   PBRMaterial(name='door', baseColorFactor=[0.898, 0.925, 0.945, 1.0], metallicFactor=0.45, roughnessFactor=0.4),
    'seam':   PBRMaterial(name='seam', baseColorFactor=[0.42, 0.48, 0.53, 1.0], metallicFactor=0.3, roughnessFactor=0.6),
    'pods':   PBRMaterial(name='pods', baseColorFactor=[0.855, 0.894, 0.914, 1.0], metallicFactor=0.35, roughnessFactor=0.55),
    'under':  PBRMaterial(name='under', baseColorFactor=[0.235, 0.275, 0.306, 1.0], metallicFactor=0.2, roughnessFactor=0.85),
    'bib':    PBRMaterial(name='bib', baseColorFactor=[0.847, 0.20, 0.18, 1.0], metallicFactor=0.35, roughnessFactor=0.42),
    'lamp':   PBRMaterial(name='lamp', baseColorFactor=[0.95, 0.97, 1.0, 1.0], metallicFactor=0.1, roughnessFactor=0.15,
                          emissiveFactor=[0.9, 0.9, 0.85]),
    'tlamp':  PBRMaterial(name='tlamp', baseColorFactor=[0.75, 0.1, 0.08, 1.0], metallicFactor=0.1, roughnessFactor=0.2,
                          emissiveFactor=[0.85, 0.06, 0.04]),
}


def smoothstep(t):
    return t * t * (3.0 - 2.0 * t)


def tumble_factor(y):
    """Width multiplier: 1 below the belt, tapering to 1-TUMBLE at the roof."""
    top = HULL['c'] + HULL['b']
    if y <= BELT_Y:
        return 1.0
    return 1.0 - TUMBLE * smoothstep((y - BELT_Y) / (top - BELT_Y))


def ring(a, b, c, n, scale, ydrop, tumble):
    th = np.linspace(0, 2 * np.pi, M, endpoint=False)
    ct, st = np.cos(th), np.sin(th)
    z = a * np.sign(ct) * np.abs(ct) ** (2.0 / n)
    y = b * np.sign(st) * np.abs(st) ** (2.0 / n)
    pts = np.zeros((M, 3))
    yy = y * scale + c + ydrop
    if tumble:
        z = z * np.array([tumble_factor(v) for v in yy])
    pts[:, 1] = yy
    pts[:, 2] = z * scale
    return pts


def hull_halfwidth(y):
    """Numeric half-width of the hull section at height y (tumblehome incl.)."""
    t = (y - HULL['c']) / HULL['b']
    t = max(-0.999, min(0.999, t))
    z = HULL['a'] * (1 - abs(t) ** HULL['n']) ** (1.0 / HULL['n'])
    return z * tumble_factor(y)


def stations_straight():
    return [(-HALF + t * L, 1.0, 0.0) for t in np.linspace(0, 1, 4)]


def stations_nose(from_x=None):
    sts = [(-HALF + t * (L - NOSE), 1.0, 0.0) for t in np.linspace(0, 1, 4)]
    for t in np.linspace(0.0, 1.0, 16)[1:]:
        e = smoothstep(t)
        scale = 1.0 - 0.90 * e ** 1.35
        ydrop = -0.36 * e ** 1.2
        sts.append((HALF - NOSE + t * NOSE, max(scale, 0.09), ydrop))
    if from_x is not None:
        sts = [s for s in sts if s[0] >= from_x]
    return sts


def loft(sts, a, b, c, n, tumble=False, cap_start=True, cap_end=True):
    rings = []
    for (x, s, yd) in sts:
        r = ring(a, b, c, n, s, yd, tumble)
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
    verts = [V]
    extra = len(V)
    caps = []
    if cap_start:
        caps.append((0, True))
    if cap_end:
        caps.append(((S - 1) * M, False))
    for (ring_start, flip) in caps:
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
    if mesh.is_volume and mesh.volume < 0:
        mesh.invert()
    elif not mesh.is_volume and mesh.volume < 0:
        mesh.invert()
    return mesh


def boxed(ext, at):
    m = make_box(extents=ext)
    m.apply_translation(at)
    return m


def side_panel(ext, x, y, proud=0.012):
    """A pane placed flush-proud on BOTH hull sides at (x, y)."""
    z = hull_halfwidth(y) + proud
    return [boxed(ext, [x, y, z]), boxed(ext, [x, y, -z])]


def lens(r, x, y, z, axis_x=True):
    m = make_cyl(radius=r, height=0.07, sections=18)
    if axis_x:
        m.apply_transform(trimesh.transformations.rotation_matrix(np.pi / 2, [0, 1, 0]))
    m.apply_translation([x, y, z])
    return m


def build_car(kind):
    """kind: 'lead' (nose at +x, built then mirrored for 'tail'), 'mid'."""
    nose = kind != 'mid'
    sts = stations_nose() if nose else stations_straight()
    parts = {k: [] for k in MATS}

    # hull with tumblehome
    parts['hull'].append(loft(sts, HULL['a'], HULL['b'], HULL['c'], HULL['n'], tumble=True))

    # windows: discrete panes with rhythm, laid out AROUND the doors — doors
    # near the car ends, a clean run of panes between them.
    win_y = 0.42
    win_ext = [0.48, 0.48, 0.05]   # width < spacing, so panes read as panes
    if nose:
        win_xs = np.linspace(-1.15, 0.25, 3)
    else:
        win_xs = np.linspace(-1.3, 1.3, 5)
    for wx in win_xs:
        parts['glass'] += side_panel(win_ext, wx, win_y)
    if nose:
        # windshield: front two-thirds of the nose only — a raked visor, not
        # a glass hood over the whole taper
        shield = loft(stations_nose(from_x=HALF - NOSE + 0.35), 0.95, 0.26, 0.48, 2.2,
                      cap_start=False, cap_end=True)
        parts['glass'].append(shield)

    # livery: slim crimson belt + navy pinstripe, full length (they taper
    # through the nose with the loft)
    parts['belt'].append(loft(sts, 1.005, 0.05, -0.24, 2.0))
    parts['stripe'].append(loft(sts, 1.007, 0.018, -0.345, 2.0))
    if nose:
        parts['bib'].append(loft(stations_nose(from_x=HALF - NOSE + 0.12), 0.78, 0.46, -0.88, 2.3,
                                 cap_start=False, cap_end=True))

    # skirt + a dark shadow strip along its base (bogies on a straddle
    # monorail wrap the beam INSIDE the skirt — nothing may hang below it,
    # or it clips the guideway at the close pass)
    parts['skirt'].append(loft(sts, 0.72, 0.20, -1.22, 2.2))
    for sz in (0.62, -0.62):
        parts['under'].append(boxed([L * 0.55, 0.09, 0.12], [(-0.2 if nose else 0.0), -1.36, sz]))

    # doors: recessed-reading panels + center seam, near the car ends
    door_ext = [0.55, 1.72, 0.03]
    seam_ext = [0.025, 1.72, 0.032]
    door_xs = [-1.95] if nose else [-1.95, 1.95]
    for dx in door_xs:
        parts['door'] += side_panel(door_ext, dx, -0.38, proud=0.008)
        parts['seam'] += side_panel(seam_ext, dx, -0.38, proud=0.010)

    # roof: chrome spine + equipment pods, seated INTO the roof curve so no
    # corner floats above the shoulder radius
    parts['spine'].append(loft(sts, 0.20, 0.075, 1.02, 2.0))
    pod_xs = [-0.9] if nose else [-0.9, 0.9]
    for px_ in pod_xs:
        parts['pods'].append(boxed([1.0, 0.14, 0.68], [px_, 1.045, 0]))

    # lamps: white head lenses on the nose face, set into the bib
    if nose:
        tip_x = HALF - 0.16
        parts['lamp'].append(lens(0.085, tip_x, -0.62, 0.30))
        parts['lamp'].append(lens(0.085, tip_x, -0.62, -0.30))

    out = {}
    for key, meshes in parts.items():
        if not meshes:
            continue
        m = trimesh.util.concatenate(meshes) if len(meshes) > 1 else meshes[0]
        out[key] = m
    return out


def finish(kind, meshes):
    """Apply mirroring for tail (incl. lamp swap to red) and materials."""
    final = {}
    for key, m in meshes.items():
        mm = m.copy()
        mat_key = key
        if kind == 'tail':
            mm.vertices[:, 0] *= -1.0
            mm.invert()
            if key == 'lamp':
                mat_key = 'tlamp'
        mm.visual = TextureVisuals(material=MATS[mat_key])
        final[f'{kind}_{key}'] = mm
    return final


scene = trimesh.Scene()
lead_parts = build_car('lead')
mid_parts = build_car('mid')
for kind, src in (('lead', lead_parts), ('mid', mid_parts), ('tail', lead_parts)):
    for name, mesh in finish(kind, src).items():
        scene.add_geometry(mesh, node_name=name, geom_name=name)

scene.export(OUT)

check = trimesh.load(OUT)
tris = sum(g.faces.shape[0] for g in check.geometry.values())
print(f'wrote {OUT}: {len(check.geometry)} parts, {tris} tris')
print('bounds:', np.round(check.bounds, 3).tolist())
print('nodes:', sorted(check.graph.nodes_geometry))
