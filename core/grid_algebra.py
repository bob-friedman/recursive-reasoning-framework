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


def tile_pattern(grid, color=1, background=0, wall=2):
    """Extend 1D periodic color runs row-wise until a wall or edge.

    Period per row = min gap between consecutive `color` cells (seed must show >=2 to
    establish the beat). Copies extend from the last seed cell by the period; stops at
    walls/edges; never overwrites non-background. Returns (new_grid, added_count).
    """
    h, w = len(grid), len(grid[0])
    out = [row[:] for row in grid]
    added = 0
    for y in range(h):
        xs = [x for x in range(w) if grid[y][x] == color]
        if len(xs) < 2:
            continue
        gaps = [b - a for a, b in zip(xs, xs[1:]) if b - a > 0]
        if not gaps:
            continue
        period = min(gaps)
        x = xs[-1] + period
        while x < w:
            if out[y][x] == wall:
                break
            if out[y][x] == background:
                out[y][x] = color
                added += 1
            x += period
    return out, added


def tile_beat(grid, color=1, background=0):
    """Detect (row, x0, period) of the first row with >=2 `color` cells. Helper."""
    h, w = len(grid), len(grid[0])
    for y in range(h):
        xs = [x for x in range(w) if grid[y][x] == color]
        if len(xs) >= 2:
            gaps = [b - a for a, b in zip(xs, xs[1:]) if b - a > 0]
            if gaps:
                return y, xs[0], min(gaps)
    return None


def symmetry_complete(grid, axis="vertical", color=1, background=0):
    """Paint the missing symmetric counterparts of `color` cells across `axis`.

    axis='vertical': mirror left-right about bbox center (like reflect, but UNION:
    original cells kept, counterparts added). axis='horizontal': same top-bottom.
    Returns (new_grid, added_count). Generic, no coordinates; walls never overwritten.
    """
    h, w = len(grid), len(grid[0])
    cells = [(x, y) for y in range(h) for x in range(w) if grid[y][x] == color]
    if not cells:
        return [row[:] for row in grid], 0
    out = [row[:] for row in grid]
    added = 0
    if axis == "vertical":
        xs = [c[0] for c in cells]
        x0, x1 = min(xs), max(xs)
        for x, y in cells:
            tx = x0 + x1 - x
            if 0 <= tx < w and out[y][tx] == background:
                out[y][tx] = color
                added += 1
    else:
        ys = [c[1] for c in cells]
        y0, y1 = min(ys), max(ys)
        for x, y in cells:
            ty = y0 + y1 - y
            if 0 <= ty < h and out[ty][x] == background:
                out[ty][x] = color
                added += 1
    return out, added


def is_symmetric(grid, axis="vertical", background=0):
    """True iff every non-background cell has its mirror counterpart (any color match)."""
    h, w = len(grid), len(grid[0])
    live = [(x, y) for y in range(h) for x in range(w) if grid[y][x] != background]
    if not live:
        return False
    if axis == "vertical":
        xs = [c[0] for c in live]
        x0, x1 = min(xs), max(xs)
        return all(grid[y][x0 + x1 - x] == grid[y][x] for x, y in live if 0 <= x0 + x1 - x < w)
    ys = [c[1] for c in live]
    y0, y1 = min(ys), max(ys)
    return all(grid[y0 + y1 - y][x] == grid[y][x] for x, y in live if 0 <= y0 + y1 - y < h)


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


# --- Batch queries (B/F/G/K/D/E helpers; no env action needed) ---
def filter_by_size(objs, min_area=1, max_area=10**9):
    """Objects with min_area <= area <= max_area. Generic query helper."""
    return [o for o in objs if min_area <= o["area"] <= max_area]


def merge_objects(objs):
    """Union of cells/bboxes over an object list. Returns one pseudo-object dict."""
    cells = sorted({c for o in objs for c in o["cells"]})
    if not cells:
        return {"id": -1, "color": 0, "cells": [], "bbox": (0, 0, 0, 0), "area": 0}
    xs = [c[0] for c in cells]
    ys = [c[1] for c in cells]
    colors = {o["color"] for o in objs}
    return {"id": -1, "color": next(iter(colors)) if len(colors) == 1 else -1,
            "cells": cells, "bbox": (min(xs), min(ys), max(xs), max(ys)), "area": len(cells)}


def contains(outer, inner):
    """True iff inner's bbox lies fully inside outer's bbox."""
    ax0, ay0, ax1, ay1 = outer["bbox"]
    bx0, by0, bx1, by1 = inner["bbox"]
    return ax0 <= bx0 and ay0 <= by0 and ax1 >= bx1 and ay1 >= by1


def touches_border(obj, grid):
    """True iff any cell of obj touches the grid edge."""
    h, w = len(grid), len(grid[0])
    x0, y0, x1, y1 = obj["bbox"]
    return x0 == 0 or y0 == 0 or x1 == w - 1 or y1 == h - 1


def sort_objects(objs, key="area", reverse=False):
    """Objects ordered by key (area default). Generic query helper."""
    return sorted(objs, key=lambda o: o[key], reverse=reverse)


def count_compare(grid, color_a, color_b):
    """-1/0/+1: count(color_a) vs count(color_b). Generic query helper."""
    ca = sum(v == color_a for row in grid for v in row)
    cb = sum(v == color_b for row in grid for v in row)
    return (ca > cb) - (ca < cb)


def top_k_by_area(objs, k=1):
    """k largest objects by area. Generic query helper."""
    return sort_objects(objs, key="area", reverse=True)[:k]


def most_common_color(grid, background=0):
    """Most frequent non-background color (ties -> smallest value)."""
    hist = {}
    for row in grid:
        for v in row:
            if v != background:
                hist[v] = hist.get(v, 0) + 1
    return min(hist, key=lambda c: (-hist[c], c)) if hist else background


def rotational_symmetric_4(grid, background=0):
    """True iff grid equals itself under 90° rotations (order-4 symmetry)."""
    g = grid
    for _ in range(3):
        g = [list(r) for r in zip(*g[::-1])]
        if g != grid:
            return False
    return True


def paint_object_at(grid, obj, dx, dy, background=0):
    """Copy of grid with obj's cells painted shifted by (dx,dy); collisions block those cells."""
    h, w = len(grid), len(grid[0])
    out = [row[:] for row in grid]
    for x, y in obj["cells"]:
        nx, ny = x + dx, y + dy
        if 0 <= nx < w and 0 <= ny < h and out[ny][nx] == background:
            out[ny][nx] = obj["color"]
    return out


# --- Batch actions (E/G/H/I/J/K; each with env level + win path) ---
def hollow_object(grid, color=1, background=0):
    """Clear interior cells of `color` objects, keeping border cells. Returns (grid, cleared)."""
    h, w = len(grid), len(grid[0])
    out = [row[:] for row in grid]
    cleared = 0
    for y in range(h):
        for x in range(w):
            if grid[y][x] != color:
                continue
            if all(0 <= x + dx < w and 0 <= y + dy < h and grid[y + dy][x + dx] == color
                   for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1))):
                out[y][x] = background
                cleared += 1
    return out, cleared


def cover_color(grid, color=1, background=0):
    """Erase all `color` cells to background. Returns (grid, erased)."""
    out = [row[:] for row in grid]
    erased = 0
    for y in range(len(grid)):
        for x in range(len(grid[0])):
            if out[y][x] == color:
                out[y][x] = background
                erased += 1
    return out, erased


def recolor_by_rank(grid, rank=0, to_color=3, background=0, wall=2):
    """Recolor the rank-th smallest non-bg non-wall object. Returns (grid, ok)."""
    objs = sort_objects([o for o in find_objects(grid, background) if o["color"] != wall])
    if not objs or rank >= len(objs):
        return [row[:] for row in grid], False
    tgt = objs[rank]
    out = [row[:] for row in grid]
    for x, y in tgt["cells"]:
        out[y][x] = to_color
    return out, True


def repeat_until_wall(grid, color=1, background=0, wall=2):
    """Contiguous +x fill from each row's last seed until wall/edge. Returns (grid, added)."""
    h, w = len(grid), len(grid[0])
    out = [row[:] for row in grid]
    added = 0
    for y in range(h):
        xs = [x for x in range(w) if grid[y][x] == color]
        if not xs:
            continue
        x = xs[-1] + 1
        while x < w:
            if out[y][x] == wall:
                break
            if out[y][x] == background:
                out[y][x] = color
                added += 1
            x += 1
    return out, added


def mirror_repeat(grid, color=1, background=0, wall=2):
    """Palindromic extension: mirror the seed run (bounce) until wall/edge. Returns (grid, added)."""
    h, w = len(grid), len(grid[0])
    out = [row[:] for row in grid]
    added = 0
    for y in range(h):
        xs = [x for x in range(w) if grid[y][x] == color]
        if len(xs) < 2:
            continue
        lo, hi = min(xs), max(xs)
        seed = [grid[y][x] for x in range(lo, hi + 1)]
        x, i, direction = hi + 1, len(seed) - 2, -1
        while x < w:
            if out[y][x] == wall:
                break
            if out[y][x] == background and seed[i] == color:
                out[y][x] = color
                added += 1
            x += 1
            i += direction
            if i < 0 or i >= len(seed):
                direction *= -1
                i += 2 * direction
        # note: bounce walk; stops at wall/edge
    return out, added


def upscale_integer(grid, factor=2, background=0):
    """Scale grid by integer factor; fails (None) if it would exceed 10x10 bounds."""
    h, w = len(grid), len(grid[0])
    if h * factor > 10 or w * factor > 10:
        return None
    out = []
    for row in grid:
        big = [v for v in row for _ in range(factor)]
        for _ in range(factor):
            out.append(big[:])
    return out


def downscale_integer(grid, factor=2, background=0):
    """Shrink by factor; block maps to its top-left value iff uniform else background."""
    h, w = len(grid), len(grid[0])
    if h % factor or w % factor:
        return None
    out = []
    for y in range(0, h, factor):
        row = []
        for x in range(0, w, factor):
            block = [grid[y + dy][x + dx] for dy in range(factor) for dx in range(factor)]
            row.append(block[0] if all(v == block[0] for v in block) else background)
        out.append(row)
    return out


def crop_to_bbox(grid, background=0):
    """Crop to bbox of all non-background cells. Returns (grid, ok)."""
    h, w = len(grid), len(grid[0])
    live = [(x, y) for y in range(h) for x in range(w) if grid[y][x] != background]
    if not live:
        return [row[:] for row in grid], False
    x0, y0 = min(x for x, _ in live), min(y for _, y in live)
    x1, y1 = max(x for x, _ in live), max(y for _, y in live)
    return [[grid[y][x] for x in range(x0, x1 + 1)] for y in range(y0, y1 + 1)], True


def center_object(grid, color=1, background=0, wall=2):
    """Translate color bbox center onto grid center if clear; else no-op. Returns (grid, moved)."""
    h, w = len(grid), len(grid[0])
    cells = [(x, y) for y in range(h) for x in range(w) if grid[y][x] == color]
    if not cells:
        return [row[:] for row in grid], False
    xs = [c[0] for c in cells]
    ys = [c[1] for c in cells]
    cx, cy = (min(xs) + max(xs)) / 2.0, (min(ys) + max(ys)) / 2.0
    dx = round((w - 1) / 2.0 - cx)
    dy = round((h - 1) / 2.0 - cy)
    targets = {(x + dx, y + dy) for x, y in cells}
    for tx, ty in targets:
        if not (0 <= tx < w and 0 <= ty < h) or grid[ty][tx] == wall:
            return [row[:] for row in grid], False
    out = [row[:] for row in grid]
    for x, y in cells:
        out[y][x] = background
    for tx, ty in targets:
        out[ty][tx] = color
    return out, True


def fall_rigid(grid, color=1, background=0, wall=2):
    """Rigid gravity: whole color body falls max dy until wall/floor/other. Returns (grid, fell)."""
    h, w = len(grid), len(grid[0])
    cells = [(x, y) for y in range(h) for x in range(w) if grid[y][x] == color]
    if not cells:
        return [row[:] for row in grid], False
    others = {(x, y) for y in range(h) for x in range(w)
              if grid[y][x] not in (background, color)}
    dy = 0
    while True:
        trial = {(x, y + dy + 1) for x, y in cells}
        if any(ny >= h or (tx, ny) in others for tx, ny in trial):
            break
        dy += 1
    if dy == 0:
        return [row[:] for row in grid], False
    out = [row[:] for row in grid]
    for x, y in cells:
        out[y][x] = background
    for x, y in cells:
        out[y + dy][x] = color
    return out, True


def settle_sand(grid, background=0, wall=2):
    """Per-cell sand fall per column (walls/floor stop). Returns (grid, moved_any)."""
    h, w = len(grid), len(grid[0])
    out = [row[:] for row in grid]
    moved = False
    for x in range(w):
        write = h - 1
        for y in range(h - 1, -1, -1):
            v = out[y][x]
            if v == wall:
                write = y - 1
            elif v != background:
                if write != y:
                    out[write][x] = v
                    out[y][x] = background
                    moved = True
                write -= 1
    return out, moved


def align_tops(grid, background=0, wall=2):
    """Shift every non-wall color body up so tops share the global min top. Returns (grid, moved)."""
    h, w = len(grid), len(grid[0])
    objs = [o for o in find_objects(grid, background) if o["color"] != wall]
    if not objs:
        return [row[:] for row in grid], False
    top = min(o["bbox"][1] for o in objs)
    out = [row[:] for row in grid]
    moved = False
    for o in objs:
        dy = top - o["bbox"][1]
        if dy == 0:
            continue
        targets = [(x, y + dy) for x, y in o["cells"]]
        if any(ny < 0 or out[ny][nx] == wall for nx, ny in targets):
            continue
        for x, y in o["cells"]:
            out[y][x] = background
        for nx, ny in targets:
            out[ny][nx] = o["color"]
        moved = True
    return out, moved


def swap_colors(grid, color_a=1, color_b=4):
    """Exchange two colors grid-wide. Returns new grid."""
    return [[color_b if v == color_a else color_a if v == color_b else v for v in row] for row in grid]


# --- Closing batch (op-29..38 actions; sim-win uniform) ---
def erase_by_size(grid, min_area=0, max_area=10**9, background=0, wall=2):
    """Erase non-wall objects with area outside [min_area, max_area]. Returns (grid, erased)."""
    objs = [o for o in find_objects(grid, background) if o["color"] != wall]
    kill = {c for o in objs if not (min_area <= o["area"] <= max_area) for c in o["cells"]}
    out = [row[:] for row in grid]
    for x, y in kill:
        out[y][x] = background
    return out, len(kill)


def frame_merged_bbox(grid, background=0, wall=2):
    """Paint bbox border of merged non-wall objects with most common color. Returns (grid, painted)."""
    objs = [o for o in find_objects(grid, background) if o["color"] != wall]
    if not objs:
        return [row[:] for row in grid], 0
    m = merge_objects(objs)
    x0, y0, x1, y1 = m["bbox"]
    h, w = len(grid), len(grid[0])
    color = most_common_color(grid, background)
    out = [row[:] for row in grid]
    painted = 0
    for x in range(x0, x1 + 1):
        for y in (y0, y1):
            if 0 <= x < w and 0 <= y < h and out[y][x] == background:
                out[y][x] = color
                painted += 1
    for y in range(y0, y1 + 1):
        for x in (x0, x1):
            if 0 <= x < w and 0 <= y < h and out[y][x] == background:
                out[y][x] = color
                painted += 1
    return out, painted


def sort_row_by_area(grid, background=0, wall=2):
    """Re-seat non-wall objects along their min-y row ordered by area asc. Returns (grid, moved)."""
    h, w = len(grid), len(grid[0])
    objs = sort_objects([o for o in find_objects(grid, background) if o["color"] != wall])
    if len(objs) < 2:
        return [row[:] for row in grid], False
    out = [[background if v != wall else wall for v in row] for row in grid]
    y = min(o["bbox"][1] for o in objs)
    x = 0
    for o in objs:
        ow = o["bbox"][2] - o["bbox"][0] + 1
        oh = o["bbox"][3] - o["bbox"][1] + 1
        if x + ow > w or y + oh > h:
            return [row[:] for row in grid], False
        for cx, cy in o["cells"]:
            out[y + (cy - o["bbox"][1])][x + (cx - o["bbox"][0])] = o["color"]
        x += ow + 1
    return out, True


def topk_keep(grid, k=2, background=0, wall=2):
    """Keep k largest non-wall objects, erase rest. Returns (grid, erased)."""
    objs = [o for o in find_objects(grid, background) if o["color"] != wall]
    keep = {o["id"] for o in top_k_by_area(objs, k)}
    out = [row[:] for row in grid]
    erased = 0
    for o in objs:
        if o["id"] not in keep:
            for x, y in o["cells"]:
                out[y][x] = background
                erased += 1
    return out, erased


def beat_mark(grid, color=1, background=0):
    """Paint the detected beat origin cell color 3. Returns (grid, ok)."""
    beat = tile_beat(grid, color, background)
    if beat is None:
        return [row[:] for row in grid], False
    y, x0, _ = beat
    out = [row[:] for row in grid]
    out[y][x0] = 3
    return out, True


def stamp_copy(grid, dx=2, dy=0, color=1, background=0):
    """Stamp a shifted copy of the color object via paint_object_at. Returns (grid, ok)."""
    objs = [o for o in find_objects(grid, background) if o["color"] == color]
    if not objs:
        return [row[:] for row in grid], False
    out = paint_object_at(grid, objs[0], dx, dy, background)
    return out, out != grid


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
