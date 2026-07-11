# tomorrow-assets

Generated 3D assets (glTF/GLB) for the **Tomorrow** project — the Genentech DDA
AI Lunch & Learn series web prototype. Served to the web app via jsDelivr:

```
https://cdn.jsdelivr.net/gh/richardsonchrisj/tomorrow-assets@master/monorail.glb
```

| Asset | Contents |
|---|---|
| `monorail.glb` | Chrome & Sky monorail trainset: three car variants as named node groups (`lead_*`, `mid_*`, `tail_*`), each five PBR parts (hull, glass band, crimson beltline, skirt, chrome roof spine). Lead/tail carry drooping streamlined noses. ~24k tris, sized for hero3d.js at scale 1 (car length 5.05, +X = travel). |

Every asset is **generated, not downloaded** — see `generate_monorail.py` for
full provenance. Regenerate with `python generate_monorail.py`
(needs `trimesh` + `numpy`).

## License

CC0 1.0 Universal (public domain) — see `LICENSE`. No attribution required.
