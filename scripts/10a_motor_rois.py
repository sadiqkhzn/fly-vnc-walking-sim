"""Fetch per-motor-neuron VNC-ROI membership and cache to data/.

For each motor-neuron body ID in the cached subset, query neuPrint for which
LegNp(T1/T2/T3)(L/R) ROIs it has synapses in. Produces:

  data/motor_leg_assignment.parquet
    body_id   int64
    leg       str   one of {lf, lm, lh, rf, rm, rh, unknown}
    pre_T1L, pre_T1R, pre_T2L, pre_T2R, pre_T3L, pre_T3R   int  (presynapse counts)
    post_T1L, post_T1R, post_T2L, post_T2R, post_T3L, post_T3R  int

Assignment rule: `leg` is the LegNp with the maximum POSTsynaptic site count
(motor neurons receive their drive post-synaptically; their dendrites sit in
the leg neuropil of the leg they innervate). If no LegNp dominates (e.g. a
non-leg motor like a wing MN that slipped past the filter), `leg` = 'unknown'.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
load_dotenv(Path(__file__).resolve().parent.parent / ".env")

from neuprint import Client, fetch_custom

from src.data.neuprint_client import fetch_vnc_subset

DATASET = "male-cns:v1.0"
OUT_PATH = Path(__file__).resolve().parent.parent / "data" / "motor_leg_assignment.parquet"

ROI_TO_LEG = {
    ("T1", "L"): "lf",
    ("T2", "L"): "lm",
    ("T3", "L"): "lh",
    ("T1", "R"): "rf",
    ("T2", "R"): "rm",
    ("T3", "R"): "rh",
}

ROI_COLS = [f"{t}{s}" for t in ("T1", "T2", "T3") for s in ("L", "R")]  # T1L, T1R, T2L, T2R, T3L, T3R


def main():
    s = fetch_vnc_subset()
    motor_ids = sorted({int(b) for ids in s.motor_ids_by_type.values() for b in ids})
    print(f"motor neurons to characterize: {len(motor_ids)}")

    c = Client(
        os.environ["NEUPRINT_SERVER"],
        dataset=DATASET,
        token=os.environ["NEUPRINT_TOKEN"],
    )

    # Fetch roiInfo for just the motor neurons. roiInfo is JSON:
    #   {"ROIName": {"pre": N, "post": M}, ...}
    ids_literal = ",".join(str(b) for b in motor_ids)
    q = f"""
    MATCH (n:Neuron)
    WHERE n.bodyId IN [{ids_literal}]
    RETURN n.bodyId AS body_id, n.roiInfo AS roi_info
    """
    df = fetch_custom(q, client=c)
    print(f"fetched roiInfo for {len(df)} neurons")

    rows = []
    for _, r in df.iterrows():
        info = r["roi_info"]
        if isinstance(info, str):
            info = json.loads(info)
        row = {"body_id": int(r["body_id"])}
        for t in ("T1", "T2", "T3"):
            for side in ("L", "R"):
                roi_name = f"LegNp({t})({side})"
                sub = info.get(roi_name, {}) if isinstance(info, dict) else {}
                row[f"pre_{t}{side}"] = int(sub.get("pre", 0))
                row[f"post_{t}{side}"] = int(sub.get("post", 0))
        rows.append(row)
    out = pd.DataFrame(rows).set_index("body_id").sort_index()

    # Assign leg: argmax of post counts across the six LegNp ROIs.
    post_cols = [f"post_{c}" for c in ROI_COLS]
    post_mat = out[post_cols].to_numpy()
    best = post_mat.argmax(axis=1)
    best_val = post_mat.max(axis=1)
    legs = []
    for i, (b, v) in enumerate(zip(best, best_val)):
        if v == 0:
            legs.append("unknown")
            continue
        t, side = ROI_COLS[b][:2], ROI_COLS[b][2]
        legs.append(ROI_TO_LEG[(t, side)])
    out["leg"] = legs

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(OUT_PATH)
    print(f"wrote {OUT_PATH.relative_to(Path.cwd())}  ({len(out)} rows)")

    print("\nleg assignment counts:")
    print(out["leg"].value_counts().to_string())
    print(f"\nunassigned (unknown): {(out['leg'] == 'unknown').sum()}")


if __name__ == "__main__":
    main()
