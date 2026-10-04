"""neuPrint fetch layer for the Sept 2026 male CNS release (male-cns:v1.0).

Scope restriction: all Traced neurons that either
  (a) innervate any VNC ROI (`n.VNC IS NOT NULL`), or
  (b) are descending neurons from brain (`n.type =~ '^DN.*'`).

Yields ~24,115 neurons and ~1.77M edges after a weight>=3 denoising threshold.

Edges are directed and weighted by raw synapse count. Signing (excitatory vs
inhibitory) is derived from the postsynaptic neuron's `predictedNt` prediction
later, in the LIF builder — not here. This module is purely data retrieval.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent.parent.parent / ".env")

DATASET_DEFAULT = "male-cns:v1.0"
WEIGHT_THRESHOLD_DEFAULT = 3

# VNC primary ROIs we care about for walking.
# Wings (WTct) and halteres (HTct) are excluded on purpose — they are flight,
# not walking. We can add them later if we extend to flight.
VNC_WALKING_ROIS = (
    "LegNp(T1)(L)", "LegNp(T1)(R)",
    "LegNp(T2)(L)", "LegNp(T2)(R)",
    "LegNp(T3)(L)", "LegNp(T3)(R)",
    "NTct(UTct-T1)(L)", "NTct(UTct-T1)(R)",
    "mVAC(T1)(L)", "mVAC(T1)(R)",
    "mVAC(T2)(L)", "mVAC(T2)(R)",
    "mVAC(T3)(L)", "mVAC(T3)(R)",
    "VNC-unspecified",
)

# Named descending neurons the validation experiments depend on.
# Keys are canonical names; values are the `type` strings to match (case-insensitive).
# Aliases from male-cns:v1.0 instance naming are handled by regex later.
COMMAND_DN_TYPES = {
    "MDN":   r"^MDN$",                                  # moonwalker / backward
    "DNp09": r"^DNp09$|^DNp71$",                         # stop (DNp71 is an alias)
    "DNa01": r"^DNa01$|^DNae001$",                       # turning
    "DNa02": r"^DNa02$",                                 # turning
    "DNb01": r"^DNb01$|^DNb09$",                         # forward walk candidate
}


@dataclass
class ConnectomeSubset:
    """In-memory representation of the fetched subset.

    `neurons` is indexed by bodyId. `edges` is a long-form table (pre, post, weight).
    All queries, filters, and sim steps downstream work off these two tables.
    """
    dataset: str
    weight_threshold: int
    neurons: pd.DataFrame          # cols: type, instance, soma_side, x, y, z, nt, nt_conf, pre_count, post_count
    edges: pd.DataFrame            # cols: pre, post, weight
    command_ids: dict[str, list[int]] = field(default_factory=dict)
    motor_ids_by_type: dict[str, list[int]] = field(default_factory=dict)
    descending_ids: list[int] = field(default_factory=list)

    def summary(self) -> str:
        lines = [
            f"dataset           : {self.dataset}",
            f"weight threshold  : >= {self.weight_threshold}",
            f"neurons           : {len(self.neurons):,}",
            f"  descending       : {len(self.descending_ids):,}",
            f"  motor (by type) : {sum(len(v) for v in self.motor_ids_by_type.values()):,} across {len(self.motor_ids_by_type)} types",
            f"edges             : {len(self.edges):,}",
            f"total synapses    : {int(self.edges['weight'].sum()):,}",
            f"command neurons   :",
        ]
        for name, ids in self.command_ids.items():
            lines.append(f"  {name:8s} {len(ids)} cell(s): {ids}")
        return "\n".join(lines)


def _client(dataset: str):
    from neuprint import Client

    server = os.environ["NEUPRINT_SERVER"]
    token = os.environ["NEUPRINT_TOKEN"]
    return Client(server, dataset=dataset, token=token)


def _parse_soma(loc) -> tuple[float, float, float]:
    """Soma location comes back as a dict or JSON string. Return (x, y, z) or (nan, nan, nan)."""
    if loc is None:
        return (float("nan"),) * 3
    if isinstance(loc, str):
        try:
            loc = json.loads(loc)
        except Exception:
            return (float("nan"),) * 3
    coords = loc.get("coordinates") if isinstance(loc, dict) else None
    if not coords or len(coords) < 3:
        return (float("nan"),) * 3
    return float(coords[0]), float(coords[1]), float(coords[2])


def _fetch_neurons(c) -> pd.DataFrame:
    from neuprint import fetch_custom

    q = """
    MATCH (n:Neuron)
    WHERE n.status = 'Traced'
      AND (n.VNC IS NOT NULL OR n.type =~ '^DN.*')
    RETURN n.bodyId        AS body_id,
           n.type          AS type,
           n.instance      AS instance,
           n.somaSide      AS soma_side,
           n.somaLocation  AS soma_loc,
           n.predictedNt   AS nt,
           n.predictedNtConfidence AS nt_conf,
           n.pre           AS pre_count,
           n.post          AS post_count
    """
    df = fetch_custom(q, client=c)
    xyz = df["soma_loc"].map(_parse_soma)
    df["x"] = xyz.map(lambda t: t[0])
    df["y"] = xyz.map(lambda t: t[1])
    df["z"] = xyz.map(lambda t: t[2])
    df = df.drop(columns=["soma_loc"])
    df = df.set_index("body_id").sort_index()
    return df


def _fetch_edges(c, weight_threshold: int) -> pd.DataFrame:
    """Fetch edges in 4 shards keyed on `pre.bodyId % 4` to stay under query limits."""
    from neuprint import fetch_custom

    shards = []
    for mod in range(4):
        q = f"""
        MATCH (a:Neuron)-[r:ConnectsTo]->(b:Neuron)
        WHERE a.status = 'Traced' AND b.status = 'Traced'
          AND r.weight >= {weight_threshold}
          AND (a.VNC IS NOT NULL OR a.type =~ '^DN.*')
          AND (b.VNC IS NOT NULL OR b.type =~ '^DN.*')
          AND a.bodyId % 4 = {mod}
        RETURN a.bodyId AS pre, b.bodyId AS post, r.weight AS weight
        """
        shards.append(fetch_custom(q, client=c))
    return pd.concat(shards, ignore_index=True)


def _identify_commands(neurons: pd.DataFrame) -> dict[str, list[int]]:
    out: dict[str, list[int]] = {}
    types = neurons["type"].fillna("").astype(str)
    for canonical, pattern in COMMAND_DN_TYPES.items():
        mask = types.str.match(pattern, case=False, na=False)
        hits = neurons.index[mask].tolist()
        out[canonical] = hits
    return out


def _identify_motor(neurons: pd.DataFrame) -> dict[str, list[int]]:
    """Motor neurons: type contains 'MN' and the cell is in a leg neuropil."""
    types = neurons["type"].fillna("").astype(str)
    # Motor neuron naming is systematic in male-cns:v1.0. Three families seen in probe:
    #   (a) muscle-named with trailing " MN" (e.g. "Ti flexor MN")
    #   (b) systematic MNad*/MNxx* (e.g. "MNad01")
    #   (c) tergopleural: TPMN1/2
    is_mn = types.str.contains(r"(?:\bMN$)|(?:^MN[a-z])|(?:^TPMN\d)", case=False, regex=True, na=False)
    mn = neurons[is_mn]
    groups: dict[str, list[int]] = {}
    for t, grp in mn.groupby("type"):
        groups[str(t)] = grp.index.tolist()
    return groups


def _identify_descending(neurons: pd.DataFrame) -> list[int]:
    mask = neurons["type"].fillna("").str.match(r"^DN", na=False)
    return neurons.index[mask].tolist()


def fetch_vnc_subset(
    cache_dir: Path | None = None,
    dataset: str = DATASET_DEFAULT,
    weight_threshold: int = WEIGHT_THRESHOLD_DEFAULT,
    force_refetch: bool = False,
) -> ConnectomeSubset:
    """Fetch VNC + descending-neuron subset. Caches to parquet so re-runs are free."""
    if cache_dir is None:
        cache_dir = Path(__file__).resolve().parent.parent.parent / "data"
    cache_dir.mkdir(parents=True, exist_ok=True)
    neurons_cache = cache_dir / f"vnc_neurons__{dataset.replace(':', '_')}__w{weight_threshold}.parquet"
    edges_cache = cache_dir / f"vnc_edges__{dataset.replace(':', '_')}__w{weight_threshold}.parquet"

    if not force_refetch and neurons_cache.exists() and edges_cache.exists():
        neurons = pd.read_parquet(neurons_cache)
        edges = pd.read_parquet(edges_cache)
    else:
        c = _client(dataset)
        neurons = _fetch_neurons(c)
        edges = _fetch_edges(c, weight_threshold)
        neurons.to_parquet(neurons_cache)
        edges.to_parquet(edges_cache)

    # Drop edges whose endpoints didn't make it into the neuron table (defensive —
    # should be empty given the query filters, but worth asserting).
    valid = set(neurons.index.to_numpy().tolist())
    before = len(edges)
    edges = edges[edges["pre"].isin(valid) & edges["post"].isin(valid)].reset_index(drop=True)
    assert len(edges) == before, (
        f"orphan edges in cache: {before - len(edges)}. Rebuild cache with force_refetch=True."
    )

    subset = ConnectomeSubset(
        dataset=dataset,
        weight_threshold=weight_threshold,
        neurons=neurons,
        edges=edges,
        command_ids=_identify_commands(neurons),
        motor_ids_by_type=_identify_motor(neurons),
        descending_ids=_identify_descending(neurons),
    )
    return subset


def identify_command_neurons(subset: ConnectomeSubset) -> dict[str, list[int]]:
    """Already resolved inside fetch_vnc_subset(). Kept for the public API shape."""
    return subset.command_ids
