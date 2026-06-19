"""Shared pytest fixtures for Phase 1 tests.

The 4-node test topology is encoded directly as a dict (no nx.node_link_data
call needed) so the fixtures work even before requirements.txt is installed.
"""
import json

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Minimal 4-node ring + cross topology (bidirectional directed edges)
# Ring:   0↔1, 1↔2, 2↔3, 3↔0  (1 000 Mbps)
# Cross:  0↔2, 1↔3             (2 000 Mbps)
# Every ordered pair has at least 3 vertex-disjoint paths ⟹ k=3 is satisfied.
# ---------------------------------------------------------------------------
_TOPO_4 = {
    "directed": True,
    "multigraph": False,
    "graph": {},
    "nodes": [{"id": 0}, {"id": 1}, {"id": 2}, {"id": 3}],
    "links": [
        {"source": 0, "target": 1, "capacity_mbps": 1000, "weight": 1},
        {"source": 1, "target": 0, "capacity_mbps": 1000, "weight": 1},
        {"source": 1, "target": 2, "capacity_mbps": 1000, "weight": 1},
        {"source": 2, "target": 1, "capacity_mbps": 1000, "weight": 1},
        {"source": 2, "target": 3, "capacity_mbps": 1000, "weight": 1},
        {"source": 3, "target": 2, "capacity_mbps": 1000, "weight": 1},
        {"source": 3, "target": 0, "capacity_mbps": 1000, "weight": 1},
        {"source": 0, "target": 3, "capacity_mbps": 1000, "weight": 1},
        {"source": 0, "target": 2, "capacity_mbps": 2000, "weight": 1},
        {"source": 2, "target": 0, "capacity_mbps": 2000, "weight": 1},
        {"source": 1, "target": 3, "capacity_mbps": 2000, "weight": 1},
        {"source": 3, "target": 1, "capacity_mbps": 2000, "weight": 1},
    ],
}


@pytest.fixture
def topo_file(tmp_path):
    """Write the 4-node topology JSON to a temp file and return its path."""
    p = tmp_path / "topology.json"
    p.write_text(json.dumps(_TOPO_4))
    return str(p)


@pytest.fixture
def base_cfg(topo_file):
    return {
        "topology_file": topo_file,
        "num_nodes": 4,
        "k_shortest_paths": 3,
        "arrival_rate": 2.0,
        "slice_duration_mean": 5,
        "bandwidth_range": [10, 100],
        "price_scale": 1.0,
        "inelastic_prob": 0.5,
        "seed": 42,
        "penalty_weight": 0.5,
    }


@pytest.fixture
def env_unified(base_cfg):
    from src.env.network_env import NetworkEnv

    return NetworkEnv(base_cfg, mode="unified")


@pytest.fixture
def env_separated(base_cfg):
    from src.env.network_env import NetworkEnv

    return NetworkEnv(base_cfg, mode="separated")
