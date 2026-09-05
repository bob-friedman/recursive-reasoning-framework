"""grid_algebra.py — generic (non-overfitted) compression + DSL + verifier.

Small LLM proposes hypotheses as short Python using these ops only.
Python does exact checking. No per-game templates allowed here.
Stdlib only (matches requirements.txt).
"""
from collections import deque


def find_objects(grid, background=0):
    """4-connected components. Returns [{id,color,cells,bbox,area}]."""
    h = len(grid)
    w = len(grid[0]) if h else 0
    seen = [[False] * w for _ in range(h)]
    objs = []
    for y in range(h):
        for x in range(w):
            if seen[y][x] or grid[y][x] == background:
                continue
            color = grid[y][x]
            cells, q = [], deque([(x, y)])
            seen[y][x] = True
            while q:
                cx, cy = q.popleft()
                cells.append((cx, cy))
                for nx, ny in ((cx + 1, cy), (cx - 1, cy), (cx, cy + 1), (cx, cy - 1)):
                    if 0 <= nx < w and 0 <= ny < h and not seen[ny][nx] and grid[ny][nx] == color:
                        seen[ny][nx] = True
                        q.append((nx, ny))
            xs = [c[0] for c in cells]
            ys = [c[1] for c in cells]
            objs.append({
                "id": len(objs), "color": color, "cells": sorted(cells),
                "bbox": (min(xs), min(ys), max(xs), max(ys)), "area": len(cells),
            })
    return objs


def compress(grid, background=0):
    """Algebraic summary: the 'Opus/Astra trick' done deterministically.

    Returns dict with size, color histogram, RLE rows, objects, relations.
    Typical token saving: full grid listing vs. this summary.
    """
    objs = find_objects(grid, background)
    hist = {}
    for row in grid:
        for v in row:
            hist[v] = hist.get(v, 0) + 1
    rle = []
    for row in grid:
        run, prev, n = [], row[0] if row else None, 0
        for v in row:
            if v == prev:
                n += 1
            else:
                run.append((prev, n))
                prev, n = v, 1
        run.append((prev, n))
        rle.append(run[1:] if run and run[0][1] == 0 else run)
    rels = []
    for i, a in enumerate(objs):
        for b in objs[i + 1:]:
            ax0, ay0, ax1, ay1 = a["bbox"]
            bx0, by0, bx1, by1 = b["bbox"]
            if ax1 < bx0:
                rels.append((a["id"], "left-of", b["id"]))
            elif bx1 < ax0:
                rels.append((b["id"], "left-of", a["id"]))
            if ay1 < by0:
                rels.append((a["id"], "above", b["id"]))
            elif by1 < ay0:
                rels.append((b["id"], "above", a["id"]))
            if a["color"] == b["color"]:
                rels.append((a["id"], "same-color", b["id"]))
    return {
        "size": (len(grid[0]) if grid else 0, len(grid)),
        "hist": hist, "n_objects": len(objs), "objects": objs,
        "relations": rels, "rle": rle,
    }


# --- DSL ops (generic only; add new op only if 2+ games need it) ---
def translate(grid, dx, dy, color=None, background=0):
    """Shift cells by (dx, dy); out-of-bounds cells are dropped.

    Non-moving cells (background or other colors when filtered) are preserved
    in place; moving cells vacate their origin. Whole-grid (color=None) is lossless
    except at edges.
    """
    h, w = len(grid), len(grid[0])
    out = [[background] * w for _ in range(h)]
    for y in range(h):
        for x in range(w):
            v = grid[y][x]
            if v == background or (color is not None and v != color):
                if out[y][x] == background:
                    out[y][x] = v
                continue
            nx, ny = x + dx, y + dy
            if 0 <= nx < w and 0 <= ny < h:
                out[ny][nx] = v
    return out


def recolor(grid, mapping):
    return [[mapping.get(v, v) for v in row] for row in grid]


def rotate_cw(grid):
    return [list(r) for r in zip(*grid[::-1])]


def reflect_h(grid):
    return [row[::-1] for row in grid]


def count_color(grid, color):
    return sum(v == color for row in grid for v in row)


def bbox_of(obj):
    """(x0, y0, x1, y1) for a find_objects()/compress() object dict."""
    return obj["bbox"]


def area_of(obj):
    return obj["area"]


def smallest(objs, key="area"):
    return min(objs, key=lambda o: o[key])


def largest(objs, key="area"):
    return max(objs, key=lambda o: o[key])


def in_contact(obj_a, obj_b):
    """True iff any cell of A is orthogonally adjacent to (or overlaps) any cell of B."""
    cells_b = set(obj_b["cells"])
    for x, y in obj_a["cells"]:
        for nx, ny in ((x, y), (x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)):
            if (nx, ny) in cells_b:
                return True
    return False


def is_adjacent(grid, obj, color, direction=None):
    """True iff `obj` touches `color` in `direction` (or any direction if None).

    direction ∈ {LEFT, RIGHT, UP, DOWN}: side of the object's bbox to check.
    """
    h, w = len(grid), len(grid[0])
    x0, y0, x1, y1 = obj["bbox"]
    if direction in (None, "RIGHT"):
        if any(grid[y][x1 + 1] == color for y in range(y0, y1 + 1) if x1 + 1 < w):
            return True
    if direction in (None, "LEFT"):
        if any(grid[y][x0 - 1] == color for y in range(y0, y1 + 1) if x0 - 1 >= 0):
            return True
    if direction in (None, "DOWN"):
        if any(grid[y1 + 1][x] == color for x in range(x0, x1 + 1) if y1 + 1 < h):
            return True
    if direction in (None, "UP"):
        if any(grid[y0 - 1][x] == color for x in range(x0, x1 + 1) if y0 - 1 >= 0):
            return True
    return False


def flood_fill_interior(grid, color=1, background=0, wall=2):
    """Paint zero-regions NOT connected to the border (enclosed interiors).

    Generic BFS, no coordinates. Returns (new_grid, filled_count). Outdoor zeros
    (connected to any edge) are never touched — only walled-in interiors fill.
    """
    h, w = len(grid), len(grid[0])
    seen = [[False] * w for _ in range(h)]
    q = deque()
    for x in range(w):
        for y in (0, h - 1):
            if grid[y][x] == background and not seen[y][x]:
                seen[y][x] = True
                q.append((x, y))
    for y in range(h):
        for x in (0, w - 1):
            if grid[y][x] == background and not seen[y][x]:
                seen[y][x] = True
                q.append((x, y))
    while q:  # mark all outdoor background
        cx, cy = q.popleft()
        for nx, ny in ((cx + 1, cy), (cx - 1, cy), (cx, cy + 1), (cx, cy - 1)):
            if 0 <= nx < w and 0 <= ny < h and not seen[ny][nx] and grid[ny][nx] == background:
                seen[ny][nx] = True
                q.append((nx, ny))
    out = [row[:] for row in grid]
    filled = 0
    for y in range(h):
        for x in range(w):
            if grid[y][x] == background and not seen[y][x]:
                out[y][x] = color
                filled += 1
    return out, filled


def enclosed_zeros(grid, background=0):
    """Count of background cells NOT connected to the border (for win predicates)."""
    h, w = len(grid), len(grid[0])
    painted, _ = flood_fill_interior(grid, color=-1, background=background)
    return sum(v == -1 for row in painted for v in row)


def reflect_object(grid, color=1):
    """Mirror all `color` cells left-right in place about their bbox center.

    Returns (new_grid, moved). No-op (original copy, False) if the mirrored shape
    would hit a wall (color 2). Bounds cannot be violated (same bbox by construction).
    """
    h, w = len(grid), len(grid[0])
    cells = [(x, y) for y in range(h) for x in range(w) if grid[y][x] == color]
    if not cells:
        return [row[:] for row in grid], False
    xs = [c[0] for c in cells]
    x0, x1 = min(xs), max(xs)
    targets = {(x0 + x1 - x, y) for x, y in cells}
    for tx, ty in targets:
        if grid[ty][tx] == 2:
            return [row[:] for row in grid], False
    out = [row[:] for row in grid]
    for x, y in cells:
        out[y][x] = 0
    for tx, ty in targets:
        out[ty][tx] = color
    return out, True


def rotate_object(grid, color=1):
    """Rigid 90° cw rotation of all `color` cells about their bbox center.

    Returns (new_grid, moved). No-op (original copy, False) if the rotated shape
    would leave the bounds or hit a wall (color 2). Generic; self-overlap ignored.
    """
    h, w = len(grid), len(grid[0])
    cells = [(x, y) for y in range(h) for x in range(w) if grid[y][x] == color]
    if not cells:
        return [row[:] for row in grid], False
    xs = [c[0] for c in cells]
    ys = [c[1] for c in cells]
    x0, y0, x1, y1 = min(xs), min(ys), max(xs), max(ys)
    cx, cy = (x0 + x1) / 2.0, (y0 + y1) / 2.0
    targets = set()
    for x, y in cells:
        # 90° cw about center: (dx, dy) -> (-dy, dx), rounded to grid
        rx = int(round(cx - (y - cy)))
        ry = int(round(cy + (x - cx)))
        targets.add((rx, ry))
    for tx, ty in targets:
        if not (0 <= tx < w and 0 <= ty < h):
            return [row[:] for row in grid], False
        if grid[ty][tx] == 2:
            return [row[:] for row in grid], False
    out = [row[:] for row in grid]
    for x, y in cells:
        out[y][x] = 0
    for tx, ty in targets:
        out[ty][tx] = color
    return out, True


# --- Verifier: score a hypothesis fn predict(state, action) -> state ---
def score_hypothesis(predict_fn, history, code_len=1, lam=0.1):
    """history: list of (state, action, next_state). Exact match scoring + MDL penalty."""
    fits = 0
    for s, a, ns in history:
        try:
            if predict_fn(s, a) == ns:
                fits += 1
        except Exception:
            pass
    n = max(1, len(history))
    return fits / n - lam * code_len / 100.0, fits


def demo():
    g = [
        [0, 0, 0, 0, 0],
        [0, 1, 1, 0, 2],
        [0, 1, 1, 0, 2],
        [0, 0, 0, 0, 0],
    ]
    c = compress(g)
    print("size:", c["size"], "hist:", c["hist"], "n_obj:", c["n_objects"])
    print("bbox:", [o["bbox"] for o in c["objects"]], "rels:", c["relations"])
    hyp = lambda s, a: translate(s, 1, 0, color=1)  # "red square moves right"
    score, fits = score_hypothesis(hyp, [(g, "RIGHT", translate(g, 1, 0, color=1))], code_len=20)
    print(f"hypothesis score={score:.2f} fits={fits}/1")
    # exercise the remaining admitted ops (fails loudly if any regresses)
    o1, o2 = c["objects"]
    assert bbox_of(o1) == (1, 1, 2, 2) and area_of(o1) == 4
    assert smallest(c["objects"])["area"] == 2 and largest(c["objects"])["area"] == 4
    assert count_color(g, 1) == 4 and in_contact(o1, o1)
    assert not in_contact(o1, o2) and not is_adjacent(g, o1, 2)
    moved_g = translate(g, 1, 0, color=1)  # block now touches wall: adjacency True
    moved_blk = [o for o in find_objects(moved_g) if o["color"] == 1][0]
    assert moved_blk["bbox"] == (2, 1, 3, 2) and is_adjacent(moved_g, moved_blk, 2, "RIGHT")
    r, moved = rotate_object([[0, 0, 0], [0, 1, 0], [0, 1, 0], [0, 1, 0], [0, 0, 0]], color=1)
    assert moved and count_color(r, 1) == 3
    L = [[0, 0, 0], [0, 0, 1], [0, 0, 1], [0, 1, 1]]
    f, moved = reflect_object(L, color=1)
    assert moved and count_color(f, 1) == 4 and f[3] == [0, 1, 1]
    print("admitted-ops self-test OK (12/12)")


if __name__ == "__main__":
    demo()
