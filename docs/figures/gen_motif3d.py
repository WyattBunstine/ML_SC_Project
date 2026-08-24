#!/usr/bin/env python3
"""Generate the 3-D structure-motif row for graph_construction.tex.

Oblique projection: P(x,y,z) = (x + KY*y, z + KZ*y); larger y = farther.
Painter's algorithm: primitives sorted by depth (descending y), atoms get a
small bias so they draw over bonds that end at them.
"""
import numpy as np

KY, KZ = 0.42, 0.30
TEX = "graph_construction.tex"  # run from docs/figures


def proj(p):
    x, y, z = p
    return (x + KY * y, z + KZ * y)


def fmt(v):
    return f"{v:.3f}"


class Panel:
    def __init__(self, post=None):
        self.post = post if post else (lambda xy: xy)
        self.prims = []  # (depth, bias, tikz_line_template_with_{dx}{dy} placeholders)
        self.pts = []    # projected points w/ radii for bbox

    def atom(self, p, style, r):
        px, py = self.post(proj(p))
        self.prims.append((p[1], 0.02, ("node", style, px, py)))
        self.pts.append((px, py, r))

    def bond(self, p, q, style="mbond", depth=None):
        (px, py), (qx, qy) = self.post(proj(p)), self.post(proj(q))
        depth = (p[1] + q[1]) / 2 if depth is None else depth
        self.prims.append((depth, 0.0, ("line", style, px, py, qx, qy)))
        self.pts += [(px, py, 0.0), (qx, qy, 0.0)]

    def raw(self, depth, bias, line):
        self.prims.append((depth, bias, ("raw", line)))

    def emit(self, dx, dy):
        out = []
        for depth, bias, prim in sorted(self.prims, key=lambda t: (-t[0], t[1])):
            if prim[0] == "node":
                _, style, px, py = prim
                out.append(f"  \\node[{style}] at ({fmt(px+dx)},{fmt(py+dy)}) {{}};")
            elif prim[0] == "line":
                _, style, px, py, qx, qy = prim
                out.append(f"  \\draw[{style}] ({fmt(px+dx)},{fmt(py+dy)}) -- ({fmt(qx+dx)},{fmt(qy+dy)});")
            else:
                out.append(prim[1].replace("@DX", fmt(dx)).replace("@DY", fmt(dy)))
        return out

    def center_offset(self, cx=1.6, cy=1.2):
        xs0 = min(px - r for px, py, r in self.pts); xs1 = max(px + r for px, py, r in self.pts)
        ys0 = min(py - r for px, py, r in self.pts); ys1 = max(py + r for px, py, r in self.pts)
        print(f"  extent {xs1-xs0:.2f} x {ys1-ys0:.2f}")
        return cx - (xs0 + xs1) / 2, cy - (ys0 + ys1) / 2


RA, RB, RO = 0.25, 0.19, 0.14
A = 1.55  # cubic cell edge


def perovskite(tilt=0.0, polar=0.0):
    """B-corner cell: 8 B, 12 edge O, A center. tilt = tangential O shift,
    polar = +z shift of all B."""
    pan = Panel()
    corners = [(i, j, k) for i in (0, A) for j in (0, A) for k in (0, A)]
    # edge O + the two half-bonds of each edge
    for axis in range(3):
        for u in (0, A):
            for v in (0, A):
                base = [0.0, 0.0, 0.0]
                base[axis] = A / 2
                base[(axis + 1) % 3] = u
                base[(axis + 2) % 3] = v
                s = 1 if ((u + v) / A) % 2 == 0 else -1
                # bond ends anchor on the TRUE corners; only the O is displaced,
                # so tilted edges render as kinked B-O-B polylines
                e0 = list(base); e0[axis] = 0.0
                e1 = list(base); e1[axis] = A
                e0[2] += polar; e1[2] += polar     # bond ends live on displaced B
                mid = list(base)
                mid[(axis + 1) % 3] += s * tilt   # tangential (a-a-a rotation, 1st order)
                pan.atom(tuple(mid), "matomO", RO)
                pan.bond(tuple(e0), tuple(mid))
                pan.bond(tuple(mid), tuple(e1))
    for c in corners:
        pan.atom((c[0], c[1], c[2] + polar), "matomB", RB)
    pan.atom((A / 2, A / 2, A / 2), "matomA", RA)
    return pan


# ilmenite primitive cell (mp-19417, R-3), pre-rotated so the 3-fold axis
# (= body diagonal) is +z; scale 0.178 cm/Angstrom.  10 cell atoms plus the
# two O3-triangle periodic images (t~0.25/0.75) that complete the Fe-Ti
# face-sharing pairs -- without them the Fe sit bondless in a strict cell.
ILM_LVEC = [[0.5227, 0.0, 0.8157], [-0.2614, -0.4527, 0.8157], [-0.2614, 0.4527, 0.8157]]
ILM_ATOMS = [
    ("Ti", (0.0, 0.0, 0.8561)),        # t=0.350
    ("Ti", (0.0, 0.0, 1.5908)),        # t=0.650
    ("Fe", (0.0, 0.0, 0.3650)),        # t=0.149
    ("Fe", (0.0, 0.0, 2.0820)),        # t=0.851
    ("O", (-0.0229, -0.3012, 1.4346)), # t=0.586
    ("O", (0.2723, 0.1308, 1.4346)),
    ("O", (-0.2494, 0.1704, 1.4346)),
    ("O", (0.0229, 0.3012, 1.0123)),   # t=0.414
    ("O", (0.2494, -0.1704, 1.0123)),
    ("O", (-0.2723, -0.1308, 1.0123)),
    ("O", (0.2385, 0.1515, 0.6190)),   # t=0.253 (image)
    ("O", (-0.2505, 0.1308, 0.6190)),
    ("O", (0.0120, -0.2823, 0.6190)),
    ("O", (-0.2385, -0.1515, 1.8280)), # t=0.747 (image)
    ("O", (-0.0120, 0.2823, 1.8280)),
    ("O", (0.2505, -0.1308, 1.8280)),
]
ILM_BONDS = [(0, 7), (0, 8), (0, 9), (0, 10), (0, 11), (0, 12),
             (1, 4), (1, 5), (1, 6), (1, 13), (1, 14), (1, 15),
             (2, 10), (2, 11), (2, 12), (3, 13), (3, 14), (3, 15)]
ILM_FACE = [13, 14, 15]  # shared O3 face of the upper Ti-Fe pair


def ilmenite(K=1.0, phi=-15.0, tip=50.0, spin=15.0):
    """One primitive rhombohedral cell, oblique view ("option Q" of the
    2026-08-17 orientation sweep): spin = pre-rotation about the 3-fold axis
    (= body diagonal, +z; staggers the O3 triangles), tip = rotation of that
    axis toward the viewer about the horizontal screen axis (0 = axis vertical
    in the page, 90 = straight out of the page; 50 shows the face-sharing
    Fe-Ti pairs with real depth), phi = in-screen lean."""
    cph, sph = np.cos(np.radians(phi)), np.sin(np.radians(phi))
    post = lambda xy: (cph * xy[0] - sph * xy[1], sph * xy[0] + cph * xy[1])
    T, S = np.radians(tip), np.radians(spin)

    def rx(p):
        x, y, z = p
        x, y = x * np.cos(S) - y * np.sin(S), x * np.sin(S) + y * np.cos(S)
        return (x, y * np.cos(T) - z * np.sin(T), y * np.sin(T) + z * np.cos(T))

    pan = Panel(post=post)
    sp = lambda p: post(proj(p))
    a1, a2, a3 = (K * np.array(v) for v in ILM_LVEC)
    corners = [rx(c) for c in
               [np.zeros(3), a1, a2, a3, a1 + a2, a1 + a3, a2 + a3, a1 + a2 + a3]]
    E = [(0, 1), (0, 2), (0, 3), (1, 4), (1, 5), (2, 4), (2, 6),
         (3, 5), (3, 6), (4, 7), (5, 7), (6, 7)]
    for i, j in E:  # frame sits behind the structure -- outline only
        pan.bond(corners[i], corners[j], style="medge", depth=99)
    pos = {i: rx(tuple(K * c for c in p)) for i, (el, p) in enumerate(ILM_ATOMS)}
    for i, j in ILM_BONDS:
        pan.bond(pos[i], pos[j])
    # Fe drawn at Ti's radius (keeping the A-site colour): unlike the
    # perovskite A-site, ilmenite's Fe2+ is octahedral with a radius similar
    # to Ti4+ -- a big A-site ball would misrepresent the structure.
    STYLE = {"Fe": ("matomA, minimum size=3.8mm", RB),
             "Ti": ("matomB", RB), "O": ("matomO", RO)}
    for i, (el, p) in enumerate(ILM_ATOMS):
        sty, r = STYLE[el]
        pan.atom(pos[i], sty, r)
    for idx, txt, anc, ox, oy in [(3, "Fe", "west", 0.40, 0.06),
                                  (1, "Ti", "east", -0.39, -0.34)]:
        px, py = sp(pos[idx])
        pan.raw(-99, 0.1,
                f"  \\node[psub, anchor={anc}] at ({fmt(px+ox)}+@DX,{fmt(py+oy)}+@DY) {{{txt}}};")
    return pan


def fit_ilmenite(wmax=3.15, hmax=2.48, **kw):
    K = 1.0
    for _ in range(3):
        p = ilmenite(K, **kw)
        w = max(px + r for px, py, r in p.pts) - min(px - r for px, py, r in p.pts)
        h = max(py + r for px, py, r in p.pts) - min(py - r for px, py, r in p.pts)
        K *= min(wmax / w, hmax / h)
    return ilmenite(K, **kw)


def render_panel(pan, shift, comment, extra=()):
    dx, dy = pan.center_offset()
    lines = [comment, f"\\begin{{scope}}[shift={{({shift},11.9)}}]",
             "  \\draw[panelbox] (-0.35,-0.15) rectangle (3.55,2.55);"]
    lines += pan.emit(dx, dy)
    lines += list(extra)
    lines.append("\\end{scope}")
    return "\n".join(lines)


print("SrTiO3");  p1 = perovskite()
print("GdFeO3");  p2 = perovskite(tilt=0.20)
print("FeTiO3");  p4 = fit_ilmenite()   # option Q: tip 50, phi -15, spin 15

blocks = [
    render_panel(p1, "0", "% SrTiO3: cubic cell, straight B-O-B cage, A at body centre"),
    render_panel(p2, "4.3", "% GdFeO3: edge O tangentially displaced (octahedral tilt)"),
    render_panel(p4, "8.6",
                 "% FeTiO3: primitive rhombohedral cell, oblique view (orientation\n"
                 "% sweep option Q): the two face-sharing Fe-Ti pairs along the body\n"
                 "% diagonal with the shared O3 triangles shown in depth",
                 ),
]

header = """% ---------------------------------------------------------------------
%  Row 1 — structure motifs (3-D ball-and-stick, oblique projection;
%  GENERATED by gen_motif3d.py (this directory) — edit that script, not this block)
% ---------------------------------------------------------------------
"""
generated = header + "\n".join(blocks) + "\n\n"

src = open(TEX).read()
i0 = src.index("%  Row 1 —"); i0 = src.rfind("% ----", 0, i0)
i1 = src.index("%  Row 2 —"); i1 = src.rfind("% ----", 0, i1)
open(TEX, "w").write(src[:i0] + generated + src[i1:])
print("patched", TEX)
