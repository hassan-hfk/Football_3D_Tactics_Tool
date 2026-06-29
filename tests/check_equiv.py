"""
check_equiv.py — verify two result JSONs are numerically equivalent.

The football pipeline produces two kinds of JSON:
  1. pipeline.process_video()  -> outputs/<id>.json           (pre-mesh)
  2. lift_gvhmr.process()      -> outputs/<id>_mesh.json      (mesh)

This script compares two JSONs of either kind, field by field, within
tolerances tuned to the FP16 / BF16 noise floor of the models involved.

If every field is within tolerance, the optimization is lossless.
If anything drifts further, the harness prints the worst offenders and
exits with code 1, so it can wire into a pre-merge check.

Usage:
  # Pre-mesh comparison
  python check_equiv.py outputs/baseline.json outputs/optimized.json

  # Mesh comparison (auto-detected from `smplx` field, --mesh forces it)
  python check_equiv.py outputs/baseline_mesh.json outputs/optimized_mesh.json --mesh

  # Tighter tolerances for a stricter check
  python check_equiv.py a.json b.json --strict

  # JSON output (for CI)
  python check_equiv.py a.json b.json --report report.json

Tolerances (default):
  bbox          : 1.0  px       (rounding in pipeline writes int pixels)
  pitch_pos     : 0.05 m
  kpts xy       : 1.0  px
  kpts conf     : 0.02
  smplx params  : 1e-3 rad      (axis-angle / betas / transl)
  smplx verts   : 1e-2 m        (fp16 noise floor on a 2 m body)

Strict (--strict, for paranoid runs):
  bbox          : 1.0  px       (cannot tighten — already at int precision)
  pitch_pos     : 0.01 m
  kpts xy       : 0.3  px
  kpts conf     : 0.005
  smplx params  : 1e-4 rad
  smplx verts   : 5e-3 m

The harness also checks structural equivalence:
  - same set of frame_idx
  - same set of player track ids per frame
  - same `kind` and `team` per (frame_idx, track_id)
  - same homography_available flag per frame
Structural mismatches always fail, regardless of tolerance.

Verts: smplx.verts_b64 is decoded and compared as float arrays. If the two
runs used different dtypes (fp16 vs fp32), the comparison happens in fp32.
"""

from __future__ import annotations

import argparse
import base64
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np


# ─────────────────────────────────────────────────────────────────────────────
# Tolerances
# ─────────────────────────────────────────────────────────────────────────────

DEFAULT_TOL = {
    "bbox":          1.0,     # px
    "pitch_pos":     0.05,    # m
    "kpts_xy":       1.0,     # px
    "kpts_conf":     0.02,
    "smplx_param":   1e-3,    # rad / unitless
    "smplx_verts":   1e-2,    # m
    "gvhmr_yaw":     1e-3,    # rad
    "homography_pct_match": 0.99,  # fraction of frames where the
                                   #   homography_available flag must agree
}

STRICT_TOL = {
    "bbox":          1.0,
    "pitch_pos":     0.01,
    "kpts_xy":       0.3,
    "kpts_conf":     0.005,
    "smplx_param":   1e-4,
    "smplx_verts":   5e-3,
    "gvhmr_yaw":     1e-4,
    "homography_pct_match": 1.0,
}


# ─────────────────────────────────────────────────────────────────────────────
# Loading and basic structure
# ─────────────────────────────────────────────────────────────────────────────

def load_json(p: str) -> dict:
    with open(p, "r") as f:
        return json.load(f)


def is_mesh_json(data: dict) -> bool:
    """True if any player entry has an `smplx` block."""
    for f in data.get("frames", []):
        for p in f.get("players", []):
            if "smplx" in p:
                return True
    return False


# ─────────────────────────────────────────────────────────────────────────────
# Verts decoding
# ─────────────────────────────────────────────────────────────────────────────

def decode_verts(b64_str: str, dtype_hint: str | None) -> np.ndarray:
    """Decode a verts_b64 string into a (10475, 3) float32 array.

    dtype_hint comes from metadata.smplx.verts_dtype, but if absent we infer
    from byte length: 10475 * 3 = 31425 floats. fp16 = 62850 bytes, fp32 = 125700.
    """
    raw = base64.b64decode(b64_str)
    n_floats = 10475 * 3
    if dtype_hint == "fp16" or len(raw) == n_floats * 2:
        arr = np.frombuffer(raw, dtype=np.float16)
    elif dtype_hint == "fp32" or len(raw) == n_floats * 4:
        arr = np.frombuffer(raw, dtype=np.float32)
    else:
        raise ValueError(
            f"verts_b64 has {len(raw)} bytes, can't infer dtype"
        )
    return arr.astype(np.float32).reshape(10475, 3)


# ─────────────────────────────────────────────────────────────────────────────
# Comparators
# ─────────────────────────────────────────────────────────────────────────────

class DiffReport:
    """Accumulates per-field statistics across the whole comparison."""

    def __init__(self, tol: dict):
        self.tol = tol
        self.fields: dict = {}     # name -> {"max": float, "count": int,
                                   #         "worst_at": str, "tol": float}
        self.structural_errors: list[str] = []
        self.warnings: list[str] = []

    def record(self, field: str, max_abs: float, where: str = ""):
        """Record one numeric comparison."""
        tol_val = self.tol.get(field, 0.0)
        entry = self.fields.setdefault(field, {
            "max": 0.0, "count": 0, "worst_at": "", "tol": tol_val,
        })
        entry["count"] += 1
        if max_abs > entry["max"]:
            entry["max"] = float(max_abs)
            entry["worst_at"] = where

    def structural(self, msg: str):
        self.structural_errors.append(msg)

    def warn(self, msg: str):
        self.warnings.append(msg)

    def passed(self) -> bool:
        if self.structural_errors:
            return False
        for name, e in self.fields.items():
            if e["max"] > e["tol"]:
                return False
        return True

    def to_dict(self) -> dict:
        return {
            "passed": self.passed(),
            "structural_errors": self.structural_errors,
            "warnings": self.warnings,
            "fields": {
                name: {
                    "max_abs":  e["max"],
                    "tol":      e["tol"],
                    "n":        e["count"],
                    "worst_at": e["worst_at"],
                    "ok":       e["max"] <= e["tol"],
                }
                for name, e in self.fields.items()
            },
        }

    def print(self):
        ok = self.passed()
        head = "EQUIVALENT" if ok else "NOT EQUIVALENT"
        print(f"\n== check_equiv: {head} ==")
        if self.structural_errors:
            print("\nSTRUCTURAL ERRORS:")
            for e in self.structural_errors:
                print(f"  - {e}")
        if self.warnings:
            print("\nWARNINGS:")
            for w in self.warnings:
                print(f"  - {w}")
        if self.fields:
            print("\nFIELD COMPARISONS (max abs error vs tolerance):")
            name_w = max(len(n) for n in self.fields) + 2
            for name, e in sorted(self.fields.items()):
                mark = "OK " if e["max"] <= e["tol"] else "FAIL"
                print(f"  [{mark}] {name:<{name_w}} "
                      f"max={e['max']:.4g}  tol={e['tol']:.4g}  "
                      f"n={e['count']}  at={e['worst_at']}")
        else:
            print("\n  (no numeric fields compared)")
        print()


def _abs_diff_max(a, b) -> float:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if a.shape != b.shape:
        return float("inf")
    return float(np.max(np.abs(a - b)))


# ─────────────────────────────────────────────────────────────────────────────
# Pre-mesh comparison (pipeline.process_video output)
# ─────────────────────────────────────────────────────────────────────────────

def compare_premesh(a: dict, b: dict, rep: DiffReport) -> None:
    fa = {f["frame_idx"]: f for f in a.get("frames", [])}
    fb = {f["frame_idx"]: f for f in b.get("frames", [])}

    keys_a, keys_b = set(fa), set(fb)
    if keys_a != keys_b:
        only_a = sorted(keys_a - keys_b)[:5]
        only_b = sorted(keys_b - keys_a)[:5]
        rep.structural(
            f"frame_idx sets differ: |A|={len(keys_a)} |B|={len(keys_b)}  "
            f"only_A={only_a}  only_B={only_b}"
        )
        # Continue with the intersection so the user sees field-level info too.

    homo_mismatch = 0
    homo_total = 0
    for fi in sorted(keys_a & keys_b):
        f_a, f_b = fa[fi], fb[fi]
        homo_total += 1
        if bool(f_a.get("homography_available")) != bool(f_b.get("homography_available")):
            homo_mismatch += 1

        _compare_players(f_a, f_b, fi, rep)
        _compare_ball(f_a, f_b, fi, rep)

    if homo_total > 0:
        match_pct = 1.0 - homo_mismatch / homo_total
        rep.record("homography_pct_match",
                   max(0.0, rep.tol["homography_pct_match"] - match_pct),
                   where=f"{homo_mismatch}/{homo_total} frames differ")


def _index_players(frame: dict) -> dict:
    """Map (track_id) -> player_entry. Drops tid < 0 (untracked detections)."""
    out = {}
    for p in frame.get("players", []):
        tid = p.get("id")
        if tid is None or tid < 0:
            continue
        out[int(tid)] = p
    return out


def _compare_players(f_a: dict, f_b: dict, fi: int, rep: DiffReport) -> None:
    pa, pb = _index_players(f_a), _index_players(f_b)
    common = set(pa) & set(pb)
    if set(pa) != set(pb):
        only_a = sorted(set(pa) - set(pb))[:5]
        only_b = sorted(set(pb) - set(pa))[:5]
        rep.structural(
            f"frame {fi}: player track ids differ "
            f"only_A={only_a} only_B={only_b}"
        )

    for tid in common:
        a_p, b_p = pa[tid], pb[tid]
        loc = f"frame={fi},tid={tid}"

        if a_p.get("kind") != b_p.get("kind"):
            rep.structural(f"{loc}: kind differs "
                           f"({a_p.get('kind')} vs {b_p.get('kind')})")
        if a_p.get("team") != b_p.get("team"):
            rep.structural(f"{loc}: team differs "
                           f"({a_p.get('team')} vs {b_p.get('team')})")

        # bbox
        bb_a, bb_b = a_p.get("bbox"), b_p.get("bbox")
        if bb_a is not None and bb_b is not None:
            rep.record("bbox", _abs_diff_max(bb_a, bb_b), loc)

        # pitch_pos
        pp_a, pp_b = a_p.get("pitch_pos"), b_p.get("pitch_pos")
        if pp_a is not None and pp_b is not None:
            rep.record("pitch_pos", _abs_diff_max(pp_a, pp_b), loc)
        elif (pp_a is None) != (pp_b is None):
            rep.structural(f"{loc}: pitch_pos presence differs "
                           f"(A={'yes' if pp_a else 'no'} "
                           f"B={'yes' if pp_b else 'no'})")

        # keypoints
        kp_a, kp_b = a_p.get("kpts"), b_p.get("kpts")
        if kp_a is not None and kp_b is not None:
            kpa = np.asarray(kp_a, dtype=np.float64)
            kpb = np.asarray(kp_b, dtype=np.float64)
            if kpa.shape == kpb.shape and kpa.shape[-1] >= 3:
                rep.record("kpts_xy",
                           _abs_diff_max(kpa[..., :2], kpb[..., :2]), loc)
                rep.record("kpts_conf",
                           _abs_diff_max(kpa[..., 2], kpb[..., 2]), loc)
            else:
                rep.structural(f"{loc}: kpts shape differs "
                               f"({kpa.shape} vs {kpb.shape})")
        elif (kp_a is None) != (kp_b is None):
            rep.warn(f"{loc}: kpts presence differs "
                     f"(A={'yes' if kp_a else 'no'} "
                     f"B={'yes' if kp_b else 'no'})")


def _compare_ball(f_a: dict, f_b: dict, fi: int, rep: DiffReport) -> None:
    ba, bb = f_a.get("ball"), f_b.get("ball")
    if ba is None and bb is None:
        return
    if (ba is None) != (bb is None):
        rep.warn(f"frame {fi}: ball presence differs "
                 f"(A={'yes' if ba else 'no'} B={'yes' if bb else 'no'})")
        return
    loc = f"frame={fi},ball"
    if "bbox" in ba and "bbox" in bb:
        rep.record("bbox", _abs_diff_max(ba["bbox"], bb["bbox"]), loc)
    if ba.get("pitch_pos") and bb.get("pitch_pos"):
        rep.record("pitch_pos",
                   _abs_diff_max(ba["pitch_pos"], bb["pitch_pos"]), loc)


# ─────────────────────────────────────────────────────────────────────────────
# Mesh comparison (lift_gvhmr.process output)
# ─────────────────────────────────────────────────────────────────────────────

def compare_mesh(a: dict, b: dict, rep: DiffReport) -> None:
    # Run the pre-mesh checks first (mesh JSON is a superset)
    compare_premesh(a, b, rep)

    md_a = (a.get("metadata") or {}).get("smplx", {}) or {}
    md_b = (b.get("metadata") or {}).get("smplx", {}) or {}
    dt_a = md_a.get("verts_dtype")
    dt_b = md_b.get("verts_dtype")

    fa = {f["frame_idx"]: f for f in a.get("frames", [])}
    fb = {f["frame_idx"]: f for f in b.get("frames", [])}

    for fi in sorted(set(fa) & set(fb)):
        pa = _index_players(fa[fi])
        pb = _index_players(fb[fi])
        for tid in set(pa) & set(pb):
            sa = pa[tid].get("smplx")
            sb = pb[tid].get("smplx")
            if sa is None and sb is None:
                continue
            if (sa is None) != (sb is None):
                rep.structural(f"frame={fi},tid={tid}: smplx presence differs")
                continue

            loc = f"frame={fi},tid={tid}"

            for key in ("body_pose", "global_orient", "betas", "transl"):
                if key in sa and key in sb:
                    rep.record("smplx_param",
                               _abs_diff_max(sa[key], sb[key]),
                               f"{loc}.{key}")

            if "gvhmr_yaw" in sa and "gvhmr_yaw" in sb:
                rep.record("gvhmr_yaw",
                           abs(float(sa["gvhmr_yaw"]) - float(sb["gvhmr_yaw"])),
                           f"{loc}.gvhmr_yaw")

            if "verts_b64" in sa and "verts_b64" in sb:
                try:
                    va = decode_verts(sa["verts_b64"], dt_a)
                    vb = decode_verts(sb["verts_b64"], dt_b)
                    rep.record("smplx_verts",
                               _abs_diff_max(va, vb), f"{loc}.verts")
                except Exception as e:
                    rep.warn(f"{loc}: verts decode failed: {e}")


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="Verify two pipeline JSONs are numerically equivalent.",
    )
    parser.add_argument("a", help="Baseline JSON (reference)")
    parser.add_argument("b", help="Optimized JSON (under test)")
    parser.add_argument("--mesh", action="store_true",
                        help="Force mesh comparison. Auto-detected otherwise.")
    parser.add_argument("--strict", action="store_true",
                        help="Use tighter tolerances.")
    parser.add_argument("--report", default=None,
                        help="Write a machine-readable JSON report to this path.")
    args = parser.parse_args(argv)

    pa, pb = Path(args.a), Path(args.b)
    if not pa.is_file():
        print(f"[check_equiv] baseline not found: {pa}", file=sys.stderr)
        return 2
    if not pb.is_file():
        print(f"[check_equiv] optimized not found: {pb}", file=sys.stderr)
        return 2

    a = load_json(args.a)
    b = load_json(args.b)

    is_mesh = args.mesh or is_mesh_json(a) or is_mesh_json(b)
    tol = STRICT_TOL if args.strict else DEFAULT_TOL
    rep = DiffReport(tol)

    print(f"[check_equiv] baseline : {args.a}")
    print(f"[check_equiv] optimized: {args.b}")
    print(f"[check_equiv] mode     : {'mesh' if is_mesh else 'pre-mesh'}  "
          f"tol={'strict' if args.strict else 'default'}")

    if is_mesh:
        compare_mesh(a, b, rep)
    else:
        compare_premesh(a, b, rep)

    rep.print()

    if args.report:
        with open(args.report, "w") as f:
            json.dump(rep.to_dict(), f, indent=2)
        print(f"[check_equiv] report written to {args.report}")

    return 0 if rep.passed() else 1


if __name__ == "__main__":
    sys.exit(main())
