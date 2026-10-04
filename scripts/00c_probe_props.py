"""Final probe: what properties are available on a Neuron node?"""
from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

from neuprint import Client, fetch_custom

DATASET = "male-cns:v1.0"


def main():
    c = Client(os.environ["NEUPRINT_SERVER"], dataset=DATASET, token=os.environ["NEUPRINT_TOKEN"])

    q = """
    MATCH (n:Neuron)
    WHERE n.type = 'MDN' AND n.status = 'Traced'
    RETURN keys(n) AS k
    LIMIT 1
    """
    df = fetch_custom(q, client=c)
    print("neuron property keys:")
    for k in sorted(df["k"][0]):
        print(f"  {k}")

    q = """
    MATCH (n:Neuron)
    WHERE n.type = 'MDN' AND n.status = 'Traced'
    RETURN n.bodyId, n.somaLocation, n.somaRadius, n.somaSide, n.size, n.pre, n.post
    LIMIT 2
    """
    df = fetch_custom(q, client=c)
    print("\nsample MDN rows:")
    print(df.to_string())


if __name__ == "__main__":
    main()
