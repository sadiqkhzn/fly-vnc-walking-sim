"""Pre-flight probe (pass 2): drill into male-cns:v1.0.

Resolves:
  - ROI list (via Meta node, since fetch_roi_hierarchy isn't in neuprint-python 0.6.4)
  - VNC-ish ROI candidates
  - Descending-neuron population (type starts with 'DN')
  - Neuron counts inside VNC-ish ROIs
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

from neuprint import Client, fetch_custom

DATASET = "male-cns:v1.0"


def main():
    server = os.environ["NEUPRINT_SERVER"]
    token = os.environ["NEUPRINT_TOKEN"]
    c = Client(server, dataset=DATASET, token=token)

    # --- ROIs via Meta node ---
    print("\n[Meta.primaryRois (first 80)]")
    q = "MATCH (m:Meta) RETURN m.primaryRois AS r"
    df = fetch_custom(q, client=c)
    primary = json.loads(df["r"][0]) if isinstance(df["r"][0], str) else df["r"][0]
    for r in primary[:80]:
        print(f"  {r}")
    print(f"  ... total primary ROIs: {len(primary)}")

    # VNC-ish filter
    print("\n[VNC-ish primary ROIs (substring: VNC / nerve cord / ganglion / mesothoracic / prothoracic / metathoracic / abdominal)]")
    needles = ["VNC", "nerve", "Prothoracic", "Mesothoracic", "Metathoracic", "Abdominal", "Ganglion", "T1", "T2", "T3"]
    vnc_rois = [r for r in primary if any(n.lower() in r.lower() for n in needles)]
    for r in vnc_rois:
        print(f"  {r}")

    # --- Descending-neuron population ---
    print("\n[descending neurons: type matches '^DN' and status=Traced]")
    q = """
    MATCH (n:Neuron)
    WHERE n.type =~ '^DN.*' AND n.status = 'Traced'
    RETURN count(DISTINCT n) AS n_dn, count(DISTINCT n.type) AS n_dn_types
    """
    df = fetch_custom(q, client=c)
    print(f"  DN cells: {int(df['n_dn'][0]):,}   distinct DN types: {int(df['n_dn_types'][0])}")

    # Sanity sample: top 20 DN types by cell count
    print("\n[top 20 DN types by cell count]")
    q = """
    MATCH (n:Neuron)
    WHERE n.type =~ '^DN.*' AND n.status = 'Traced'
    RETURN n.type AS type, count(n) AS n
    ORDER BY n DESC
    LIMIT 20
    """
    df = fetch_custom(q, client=c)
    for _, r in df.iterrows():
        print(f"  {r['type']:20s} {int(r['n']):4d}")

    # --- Neurons innervating each VNC-ish ROI ---
    if vnc_rois:
        print("\n[neuron counts per VNC-ish ROI]")
        for roi in vnc_rois[:12]:
            q = f"""
            MATCH (n:Neuron)
            WHERE n.`{roi}` IS NOT NULL AND n.status = 'Traced'
            RETURN count(DISTINCT n) AS n
            """
            try:
                df = fetch_custom(q, client=c)
                print(f"  {roi:40s} {int(df['n'][0]):,}")
            except Exception as e:
                print(f"  {roi:40s} query failed → {type(e).__name__}: {e}")

    # --- Any motor-neuron-ish types? ---
    print("\n[motor-ish types: name contains 'MN' or 'motor']")
    q = """
    MATCH (n:Neuron)
    WHERE (n.type =~ '(?i).*MN.*' OR n.type =~ '(?i).*motor.*')
      AND n.status = 'Traced'
    RETURN n.type AS type, count(n) AS n
    ORDER BY n DESC
    LIMIT 20
    """
    df = fetch_custom(q, client=c)
    for _, r in df.iterrows():
        print(f"  {r['type']:30s} {int(r['n']):4d}")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"FATAL: {type(e).__name__}: {e}", file=sys.stderr)
        sys.exit(1)
