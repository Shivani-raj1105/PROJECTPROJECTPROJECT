"""
quick_test.py  --  Batch alignment + hole verification

Coordinate chain (preserves full DXF/blueprint integrity):
  DXF mm
    -> blueprint PNG px          (M_dxf2bp  : scale from content bbox)
    -> resized edge-map px       (M_cad2edge: replicates _validate_inputs exactly)
    -> real image px             (M_align   : from alignment pipeline)
"""

import math
import cv2
import numpy as np
from pathlib import Path
from cad_image_alignment import align, match_best_template


# ── Directories ───────────────────────────────────────────────────────────────
INPUTS_DIR     = Path("inputs")
BLUEPRINTS_DIR = Path("blueprints")
DXF_DIR        = Path("dxf")
OUTPUTS_DIR    = Path("outputs")

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".tif"}

DXF_CENTER_CX  = 148.5
DXF_CENTER_CY  = 105.0
DXF_CENTER_TOL = 2.0


# ── Preprocessing ─────────────────────────────────────────────────────────────
def preprocess_cad(path: Path) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Cannot load blueprint: {path}")
    inv   = cv2.bitwise_not(img)
    blur  = cv2.GaussianBlur(inv, (3, 3), 0)
    edges = cv2.Canny(blur, 20, 80)
    edges = cv2.dilate(edges, np.ones((2, 2), np.uint8))
    return edges


def preprocess_real(img: np.ndarray) -> tuple:
    blur = cv2.GaussianBlur(img, (5, 5), 0)
    _, mask = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k, iterations=3)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  k, iterations=2)
    real_masked = cv2.bitwise_and(img, img, mask=mask)
    k3 = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    gradient = cv2.morphologyEx(real_masked, cv2.MORPH_GRADIENT, k3)
    _, internal = cv2.threshold(gradient, 20, 255, cv2.THRESH_BINARY)
    internal = cv2.morphologyEx(
        internal, cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    )
    outer = cv2.Canny(mask, 50, 150)
    real_edges = cv2.bitwise_or(internal, outer)
    return real_edges, mask


# ── DXF parsing ───────────────────────────────────────────────────────────────
def parse_dxf_circles(path: Path) -> list[dict]:
    circles = []
    with open(str(path), "r") as f:
        raw = f.readlines()
    pairs = []
    i = 0
    while i + 1 < len(raw):
        try:
            pairs.append((int(raw[i].strip()), raw[i + 1].strip()))
        except ValueError:
            pass
        i += 2
    ent_start = ent_end = None
    for idx, (code, val) in enumerate(pairs):
        if code == 2 and val == "ENTITIES":
            ent_start = idx + 1
        if ent_start and code == 0 and val == "ENDSEC":
            ent_end = idx
            break
    if ent_start is None:
        return circles
    blocks = []
    current = None
    for code, val in pairs[ent_start:ent_end]:
        if code == 0:
            if current is not None:
                blocks.append(current)
            current = {"type": val, "data": {}}
        elif current is not None:
            current["data"][code] = val
    if current is not None:
        blocks.append(current)
    for block in blocks:
        if block["type"] != "CIRCLE":
            continue
        d = block["data"]
        try:
            circles.append({"cx": float(d[10]), "cy": float(d[20]), "r": float(d[40])})
        except (KeyError, ValueError):
            pass
    return circles


def get_holes_for_view(circles: list[dict], view: str) -> list[dict]:
    holes = []
    center_hole = None
    for c in circles:
        dist = math.hypot(c["cx"] - DXF_CENTER_CX, c["cy"] - DXF_CENTER_CY)
        if dist < DXF_CENTER_TOL:
            if abs(c["r"] - 14.5) < 0.1:
                center_hole = c
            continue
        if view in ("front", "top"):
            if abs(c["cx"] - 152.16) < 0.5 and abs(c["cy"] - 63.16) < 0.5:
                continue
        elif view == "rear":
            if (abs(c["cx"] - 152.16) < 0.5 and abs(c["cy"] - 146.84) < 0.5
                    and abs(c["r"] - 2.0) < 0.1):
                continue
        if c["r"] <= 10.0:
            holes.append(c)
    if center_hole is not None:
        holes.append(center_hole)
    return holes


# ── Coordinate mapping ────────────────────────────────────────────────────────
def compute_M_dxf2bp(blueprint_path: Path) -> tuple[np.ndarray, float]:
    """
    M_dxf2bp: DXF mm -> blueprint PNG pixels.

    Finds the drawn content bounding box in the blueprint PNG and maps
    the DXF geometry (center=DXF_CENTER, span=106mm) onto it.

    Returns (M, scale_px_per_mm).
    """
    img = cv2.imread(str(blueprint_path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(str(blueprint_path))

    dark = img < 200
    row_idx = np.where(np.any(dark, axis=1))[0]
    col_idx = np.where(np.any(dark, axis=0))[0]
    if len(row_idx) == 0:
        raise ValueError(f"No content in {blueprint_path}")

    rmin, rmax = int(row_idx[0]),  int(row_idx[-1])
    cmin, cmax = int(col_idx[0]),  int(col_idx[-1])
    bp_cx = (cmin + cmax) / 2.0
    bp_cy = (rmin + rmax) / 2.0
    bp_w  = cmax - cmin
    bp_h  = rmax - rmin

    scale = min(bp_w, bp_h) / 106.0   # 106mm = 2 * 53mm outer radius

    #  px  = (dxf_x - DXF_CX) * scale + bp_cx
    #  py  = (DXF_CY - dxf_y) * scale + bp_cy   (Y flipped)
    M = np.array([
        [ scale,      0,  bp_cx - DXF_CENTER_CX * scale          ],
        [     0, -scale,  bp_cy + DXF_CENTER_CY * scale          ],
        [     0,      0,  1.0                                     ],
    ], dtype=np.float64)
    return M, scale


def compute_M_cad2edge(cad_edge_map: np.ndarray,
                       real_shape: tuple[int, int]) -> np.ndarray:
    """
    M_cad2edge: blueprint PNG pixels -> resized edge-map pixels.

    Replicates _validate_inputs from alignment.py exactly:
      1. Find part bounding box in CAD edge map
      2. Crop with 5% margin
      3. Scale to fit real image
      4. Center on canvas

    This guarantees the DXF geometry goes through the identical transform
    that the CAD image went through — no integrity loss.
    """
    rh, rw = real_shape
    ch, cw = cad_edge_map.shape

    contours, _ = cv2.findContours(
        cad_edge_map, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    x1, y1, cw_crop, ch_crop = 0, 0, cw, ch
    if contours:
        largest = max(contours, key=cv2.contourArea)
        bx, by, bw, bh = cv2.boundingRect(largest)
        margin = int(max(bw, bh) * 0.05)
        x1 = max(0, bx - margin)
        y1 = max(0, by - margin)
        x2 = min(cw, bx + bw + margin)
        y2 = min(ch, by + bh + margin)
        cw_crop = x2 - x1
        ch_crop = y2 - y1

    fit_scale = min(rw / cw_crop, rh / ch_crop)
    new_w = int(round(cw_crop * fit_scale))
    new_h = int(round(ch_crop * fit_scale))
    x_off = (rw - new_w) // 2
    y_off = (rh - new_h) // 2

    #  edge_px = (bp_px - crop_origin) * fit_scale + canvas_offset
    M = np.array([
        [fit_scale, 0,          -x1 * fit_scale + x_off],
        [0,         fit_scale,  -y1 * fit_scale + y_off],
        [0,         0,          1.0                     ],
    ], dtype=np.float64)
    return M


# ── Hole verification ─────────────────────────────────────────────────────────
def check_hole_in_image(real_gray: np.ndarray,
                        cx_px: int, cy_px: int, r_px: int,
                        search_margin: int = 6) -> tuple[bool, float]:
    """
    Check for a hole at (cx_px, cy_px) with radius r_px.
    Samples mean intensity inside vs surrounding annulus.
    Holes appear bright (light through) or dark (shadow).
    Returns (found, ratio).  ratio > 1.20 or < 0.80 => hole.
    """
    h, w = real_gray.shape
    if cx_px < 0 or cy_px < 0 or cx_px >= w or cy_px >= h:
        return False, 1.0

    r_inner = max(2, r_px)
    r_outer = r_inner + search_margin

    Y, X = np.ogrid[:h, :w]
    dist_sq = (X - cx_px) ** 2 + (Y - cy_px) ** 2
    inner_mask   = dist_sq <= r_inner ** 2
    annulus_mask = (dist_sq > r_inner ** 2) & (dist_sq <= r_outer ** 2)

    inner_px   = real_gray[inner_mask]
    annulus_px = real_gray[annulus_mask]

    if inner_px.size == 0 or annulus_px.size == 0:
        return False, 1.0

    mean_inner   = float(inner_px.mean())
    mean_annulus = float(annulus_px.mean())
    if mean_annulus < 1.0:
        return False, 1.0

    ratio = mean_inner / mean_annulus
    return (ratio > 1.20 or ratio < 0.80), ratio


def verify_holes(real_gray: np.ndarray,
                 holes_dxf: list[dict],
                 M_dxf2bp: np.ndarray,
                 M_cad2edge: np.ndarray,
                 M_align: np.ndarray,
                 scale_px_per_mm: float) -> list[dict]:
    """
    Full chain for each DXF hole (no approximations, no Hough):
      DXF mm -> blueprint px -> edge-map px -> real image px

    scale_px_per_mm is used only for the hole radius in pixels.
    The alignment scale is extracted from M_align.
    """
    M_combined  = M_align @ M_cad2edge @ M_dxf2bp
    align_scale = math.sqrt(M_align[0, 0] ** 2 + M_align[1, 0] ** 2)
    cad2edge_sc = math.sqrt(M_cad2edge[0, 0] ** 2 + M_cad2edge[1, 0] ** 2)
    total_scale = scale_px_per_mm * cad2edge_sc * align_scale

    results = []
    for i, hole in enumerate(holes_dxf, start=1):
        pt     = np.array([hole["cx"], hole["cy"], 1.0], dtype=np.float64)
        mapped = M_combined @ pt
        if abs(mapped[2]) > 1e-9:
            mapped /= mapped[2]
        cx_px = int(round(mapped[0]))
        cy_px = int(round(mapped[1]))
        r_px  = max(3, int(round(hole["r"] * total_scale)))

        found, ratio = check_hole_in_image(real_gray, cx_px, cy_px, r_px)
        results.append({
            "idx":            i,
            "cx_dxf":         hole["cx"],
            "cy_dxf":         hole["cy"],
            "r_dxf":          hole["r"],
            "cx_px":          cx_px,
            "cy_px":          cy_px,
            "r_px":           r_px,
            "found":          found,
            "ratio":          ratio,
        })
    return results


def draw_hole_verification(real_gray: np.ndarray,
                           hole_results: list[dict],
                           title: str) -> np.ndarray:
    vis = cv2.cvtColor(real_gray, cv2.COLOR_GRAY2BGR)
    FOUND   = (0, 220, 80)
    MISSING = (0, 60, 255)
    WHITE   = (255, 255, 255)

    for r in hole_results:
        cx, cy = r["cx_px"], r["cy_px"]
        rp     = r["r_px"]
        color  = FOUND if r["found"] else MISSING

        cv2.circle(vis, (cx, cy), rp,     color, 2, cv2.LINE_AA)
        cv2.circle(vis, (cx, cy), rp + 5, color, 1, cv2.LINE_AA)
        overlay = vis.copy()
        cv2.circle(overlay, (cx, cy), rp, color, -1)
        cv2.addWeighted(overlay, 0.25, vis, 0.75, 0, vis)

        label = f"#{r['idx']}"
        font  = cv2.FONT_HERSHEY_SIMPLEX
        fs, th = 0.4, 1
        (tw, _), _ = cv2.getTextSize(label, font, fs, th)
        lx = max(2, min(vis.shape[1] - tw - 2, cx - tw // 2))
        ly = max(12, min(vis.shape[0] - 2, cy - rp - 6))
        cv2.putText(vis, label, (lx, ly), font, fs, WHITE, th, cv2.LINE_AA)

    found_n   = sum(1 for r in hole_results if r["found"])
    missing_n = len(hole_results) - found_n
    cv2.putText(vis, title, (10, 24),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, WHITE, 2, cv2.LINE_AA)
    cv2.putText(vis, f"Found: {found_n}  Missing: {missing_n}", (10, 48),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                FOUND if missing_n == 0 else MISSING, 1, cv2.LINE_AA)
    return vis


# ── Helpers ───────────────────────────────────────────────────────────────────
def collect_images(folder: Path) -> list[Path]:
    return sorted(p for p in folder.iterdir() if p.suffix.lower() in IMAGE_EXTS)


def print_result(name: str, result, rank: int = None):
    prefix = f"  {name}" if rank == 1 else f"  {rank}. {name}" if rank else f"   {name}"
    print(f"\n{prefix}")
    print(f"   Coverage:        {result.coverage:.4f}  ({result.coverage:.1%})")
    print(f"   Alignment Score: {result.alignment_score:.4f}")
    print(f"   Strategy:        {result.strategy}")
    if result.inlier_ratio:
        print(f"   Inlier Ratio:    {result.inlier_ratio:.2%}")


def save_outputs(out_dir: Path, blueprint_stem: str,
                 real_edges: np.ndarray, mask: np.ndarray,
                 result, is_best: bool = False,
                 debug_written: bool = False) -> bool:
    prefix = f"{blueprint_stem}_"
    cv2.imwrite(str(out_dir / f"{prefix}aligned.png"), result.aligned_image)
    overlay = np.zeros((*real_edges.shape, 3), dtype=np.uint8)
    overlay[:, :, 2] = result.aligned_image
    overlay[:, :, 1] = real_edges
    cv2.imwrite(str(out_dir / f"{prefix}overlay.png"), overlay)
    if not debug_written:
        cv2.imwrite(str(out_dir / "debug_mask.png"),       mask)
        cv2.imwrite(str(out_dir / "debug_real_edges.png"), real_edges)
    if is_best:
        cv2.imwrite(str(out_dir / "best_aligned.png"), result.aligned_image)
        cv2.imwrite(str(out_dir / "best_overlay.png"),  overlay)
    return True


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    print("=" * 70)
    print("Quick Alignment Test  --  Batch Mode")
    print("=" * 70)

    if not INPUTS_DIR.exists():
        print(f"\n[FAIL] Inputs folder not found: {INPUTS_DIR.resolve()}"); return
    input_images = collect_images(INPUTS_DIR)
    if not input_images:
        print(f"\n[FAIL] No images found in {INPUTS_DIR.resolve()}"); return

    if not BLUEPRINTS_DIR.exists():
        print(f"\n[FAIL] Blueprints folder not found: {BLUEPRINTS_DIR.resolve()}"); return
    blueprint_images = collect_images(BLUEPRINTS_DIR)
    if not blueprint_images:
        print(f"\n[FAIL] No images found in {BLUEPRINTS_DIR.resolve()}"); return

    print(f"\nFound {len(input_images)} input image(s)  in '{INPUTS_DIR}'")
    print(f"Found {len(blueprint_images)} blueprint(s)  in '{BLUEPRINTS_DIR}'")

    print(f"\nLoading blueprints...")
    templates: list[tuple[str, np.ndarray]] = []
    for bp in blueprint_images:
        try:
            edges = preprocess_cad(bp)
            templates.append((bp.stem, edges))
            print(f"   [OK] {bp.name}  ({np.count_nonzero(edges)} edge px)")
        except FileNotFoundError as e:
            print(f"   [FAIL] {e}")

    if not templates:
        print("\n[FAIL] No valid blueprints loaded."); return

    for idx, inp in enumerate(input_images, start=1):
        print(f"\n{'=' * 70}")
        print(f"[{idx}/{len(input_images)}]  Input: {inp.name}")
        print(f"{'=' * 70}")

        real = cv2.imread(str(inp), cv2.IMREAD_GRAYSCALE)
        if real is None:
            print(f"   [FAIL] Could not load image -- skipping."); continue
        print(f"   Shape: {real.shape}")

        print(f"   Preprocessing...")
        real_edges, mask = preprocess_real(real)
        print(f"   Real edges: {np.count_nonzero(real_edges)} pixels")

        out_dir = OUTPUTS_DIR / inp.stem
        out_dir.mkdir(parents=True, exist_ok=True)
        print(f"   Output folder: {out_dir.resolve()}")

        print(f"\n   Aligning against {len(templates)} blueprint(s)...")
        matches = match_best_template(templates, real_edges)

        print(f"\n   Results -- ranked by coverage:")
        for m in matches:
            print_result(m.name, m.result, rank=m.rank)

        best = matches[0]
        print(f"\n   Identification:")
        if best.result.identified:
            print(f"   [OK]   '{best.name}'  (coverage {best.result.coverage:.1%})")
        else:
            print(f"   [FAIL] Unknown -- best '{best.name}' only {best.result.coverage:.1%}")

        print(f"\n   Saving outputs...")
        debug_written = False
        for m in matches:
            is_best = (m.rank == 1)
            debug_written = save_outputs(
                out_dir=out_dir, blueprint_stem=m.name,
                real_edges=real_edges, mask=mask,
                result=m.result, is_best=is_best,
                debug_written=debug_written,
            )
            tag = " <- best" if is_best else ""
            print(f"   [OK] {m.name}_aligned.png  |  {m.name}_overlay.png{tag}")

        # ── Hole verification ─────────────────────────────────────────────────
        best_name  = best.name
        name_lower = best_name.lower()
        if "front" in name_lower or "top" in name_lower:
            view = "front"
        elif "rear" in name_lower or "back" in name_lower:
            view = "rear"
        else:
            view = None

        dxf_path = DXF_DIR / f"{best_name}.dxf"
        blueprint_path = None
        for ext in IMAGE_EXTS:
            c = BLUEPRINTS_DIR / f"{best_name}{ext}"
            if c.exists():
                blueprint_path = c
                break

        if view is None:
            print(f"\n   [SKIP] Hole check: cannot determine view from '{best_name}'")
        elif not dxf_path.exists():
            print(f"\n   [SKIP] Hole check: no DXF at {dxf_path}")
        elif blueprint_path is None:
            print(f"\n   [SKIP] Hole check: no blueprint PNG for '{best_name}'")
        else:
            print(f"\n   Hole verification  ({view} view, DXF: {dxf_path.name})")

            circles = parse_dxf_circles(dxf_path)
            holes   = get_holes_for_view(circles, view)
            print(f"   DXF holes loaded: {len(holes)}")

            # Build the three matrices
            M_dxf2bp,  scale_px_per_mm = compute_M_dxf2bp(blueprint_path)
            cad_edges_for_bp = preprocess_cad(blueprint_path)
            M_cad2edge = compute_M_cad2edge(cad_edges_for_bp, real_edges.shape)
            M_align    = best.result.transform_matrix

            print(f"   Scale: {scale_px_per_mm:.3f} px/mm (blueprint)")

            hole_results = verify_holes(
                real_gray=real,
                holes_dxf=holes,
                M_dxf2bp=M_dxf2bp,
                M_cad2edge=M_cad2edge,
                M_align=M_align,
                scale_px_per_mm=scale_px_per_mm,
            )

            found_count   = sum(1 for r in hole_results if r["found"])
            missing_count = len(hole_results) - found_count

            print(f"   Results: {found_count}/{len(hole_results)} holes found")
            for r in hole_results:
                status = "[OK]  " if r["found"] else "[MISS]"
                print(f"     {status} #{r['idx']:2d}  "
                      f"dxf=({r['cx_dxf']:.1f},{r['cy_dxf']:.1f})  "
                      f"px=({r['cx_px']},{r['cy_px']})  "
                      f"r={r['r_px']}px  ratio={r['ratio']:.2f}")

            vis = draw_hole_verification(
                real, hole_results,
                title=f"{inp.stem} -> {best_name} | holes {found_count}/{len(hole_results)}"
            )
            cv2.imwrite(str(out_dir / "hole_verification.png"), vis)
            print(f"   [OK] hole_verification.png saved")

            if missing_count == 0:
                print(f"   [PASS] All {len(hole_results)} holes present")
            else:
                missing = [r["idx"] for r in hole_results if not r["found"]]
                print(f"   [WARN] Missing holes: {missing}")

    print(f"\n{'=' * 70}")
    print(f"[OK] Done!  All outputs saved under '{OUTPUTS_DIR.resolve()}'")
    print(f"{'=' * 70}\n")


if __name__ == "__main__":
    main()
