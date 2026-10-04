"""Last pre-flight: how big is the VNC + DN subset?"""
from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

from neuprint import Client, fetch_custom

DATASET = "male-cns:v1.0"


def main():
    c = Client(os.environ["NEUPRINT_SERVER"], dataset=DATASET, token=os.environ["NEUPRINT_TOKEN"])

    print("[A] Traced neurons with any VNC presence")
    q = """
    MATCH (n:Neuron)
    WHERE n.status = 'Traced' AND n.VNC IS NOT NULL
    RETURN count(n) AS n
    """
    print(f"  = {int(fetch_custom(q, client=c)['n'][0]):,}")

    print("[B] Traced DN neurons (type ^DN)")
    q = """
    MATCH (n:Neuron)
    WHERE n.status = 'Traced' AND n.type =~ '^DN.*'
    RETURN count(n) AS n
    """
    print(f"  = {int(fetch_custom(q, client=c)['n'][0]):,}")

    print("[C] Union of A and B")
    q = """
    MATCH (n:Neuron)
    WHERE n.status = 'Traced'
      AND (n.VNC IS NOT NULL OR n.type =~ '^DN.*')
    RETURN count(n) AS n
    """
    print(f"  = {int(fetch_custom(q, client=c)['n'][0]):,}")

    print("[D] Edges with both endpoints in the union set (ConnectsTo)")
    q = """
    MATCH (a:Neuron)-[r:ConnectsTo]->(b:Neuron)
    WHERE a.status = 'Traced' AND b.status = 'Traced'
      AND (a.VNC IS NOT NULL OR a.type =~ '^DN.*')
      AND (b.VNC IS NOT NULL OR b.type =~ '^DN.*')
    RETURN count(r) AS n_edges, sum(r.weight) AS tot_syn
    """
    df = fetch_custom(q, client=c)
    print(f"  edges = {int(df['n_edges'][0]):,}   total synapses = {int(df['tot_syn'][0]):,}")

    print("[E] Edges filtered to weight >= 3 (common denoising threshold)")
    q = """
    MATCH (a:Neuron)-[r:ConnectsTo]->(b:Neuron)
    WHERE a.status = 'Traced' AND b.status = 'Traced'
      AND r.weight >= 3
      AND (a.VNC IS NOT NULL OR a.type =~ '^DN.*')
      AND (b.VNC IS NOT NULL OR b.type =~ '^DN.*')
    RETURN count(r) AS n_edges
    """
    print(f"  = {int(fetch_custom(q, client=c)['n_edges'][0]):,}")


if __name__ == "__main__":
    main()
