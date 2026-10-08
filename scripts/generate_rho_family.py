"""Generate a CONTROLLED family of topologies that varies routing headroom rho.

Why this exists
---------------
The paper claims rho (Definition: fraction of arrivals admissible on some
candidate path but NOT on the shortest one) predicts whether joint AC+RA beats
admission-only control. That claim is currently supported by two topologies
which differ in size, core model AND capacity layout -- so rho is confounded
with everything else. This script removes the confound.

The mechanism
-------------
With degree-1 endpoints every candidate path between two endpoints traverses the
SAME two access links. So:
  * access links are the bottleneck  -> all K paths fail together -> rho = 0
  * core links are the bottleneck    -> paths differ in which core links they
                                        use -> rho > 0
Hence the single structural knob is the ACCESS-TO-CORE CAPACITY RATIO.
Measured on the existing substrates: operator ratio 2.00 -> rho 11.0%;
Waxman ratio 0.41 -> rho 0.0%.

What is held fixed
------------------
Same core graph (fixed seed), same endpoint count and attachment, same traffic
model. Only the access/core capacity ratio varies.

Matched operating point
-----------------------
The joint-vs-AC-only gap is itself load dependent (the paper's load study), so
each family member is calibrated -- by binary search on capacity_scale -- to the
SAME greedy acceptance (default 0.47). Without this, rho would be confounded
with offered load and the sweep would prove nothing.

Usage
-----
    python scripts/generate_rho_family.py                # build + calibrate all
    python scripts/generate_rho_family.py --quick        # fewer eval episodes
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import networkx as nx
import numpy as np
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.env.network_env import NetworkEnv
from src.baselines.greedy_admission import GreedyAdmission

CORE_SEED = 11
N_ACCESS = 15
N_CORE = 24
CORE_CAP_MEAN = 600.0          # base core capacity before global scaling
TARGET_GREEDY_ACCEPT = 0.47    # matched operating point across the family
OUT_DIR = "data/rho_family"

# access/core capacity ratios spanning "access-limited" -> "core-limited".
# Operator sits at 2.00 (rho 11%), Waxman at 0.41 (rho 0%).
RATIOS = [0.4, 0.8, 1.3, 2.0, 3.0]


def build_core(rng: np.random.Generator) -> nx.Graph:
    """One fixed core graph, reused for every family member."""
    for _ in range(200):
        g = nx.waxman_graph(N_CORE, beta=0.50, alpha=0.30,
                            seed=int(rng.integers(1 << 30)))
        if nx.is_connected(g) and min(dict(g.degree()).values()) >= 2:
            return nx.relabel_nodes(g, {i: f"C{i}" for i in g.nodes()})
    raise RuntimeError("could not build a connected core")


def build_topology(core: nx.Graph, ratio: float, rng: np.random.Generator) -> nx.Graph:
    """Attach access leaves; set capacities from the access/core ratio."""
    G = nx.Graph()
    for c in core.nodes():
        G.add_node(c, tier="core")
    for u, v in core.edges():
        # mild heterogeneity around the mean, deterministic per edge
        jitter = 0.8 + 0.4 * ((hash((u, v)) % 1000) / 1000.0)
        G.add_edge(u, v, capacity=round(CORE_CAP_MEAN * jitter, 1))
    by_deg = sorted(core.nodes(), key=lambda c: core.degree(c), reverse=True)
    for a in range(N_ACCESS):
        aid = f"A{a}"
        G.add_node(aid, tier="access")
        G.add_edge(aid, by_deg[a % len(by_deg)],
                   capacity=round(CORE_CAP_MEAN * ratio, 1))
    return G


def topology_from_existing(ref_path: str, ratio: float) -> nx.Graph:
    """New family member from a SAVED member: identical core, new access caps.

    build_topology() jitters core capacities with Python's hash(), which is
    salted per process (PYTHONHASHSEED), so re-running it would silently give
    a new level a DIFFERENT core. Extending the family must reuse the core
    that the existing members were built with.
    """
    d = json.load(open(ref_path))
    G = nx.Graph()
    for n in d["nodes"]:
        G.add_node(n["id"], tier=n["tier"])
    for l in d["links"]:
        u, v = l["source"], l["target"]
        access = G.nodes[u]["tier"] == "access" or G.nodes[v]["tier"] == "access"
        G.add_edge(u, v, capacity=round(CORE_CAP_MEAN * ratio, 1) if access
                   else l["capacity"])
    return G


def to_nodelink(G: nx.Graph) -> dict:
    return {
        "directed": False, "multigraph": False, "graph": {},
        "nodes": [{"tier": G.nodes[n]["tier"], "id": n} for n in G.nodes()],
        "links": [{"capacity": G[u][v]["capacity"], "source": u, "target": v}
                  for u, v in G.edges()],
    }


def measure(topo_path: str, cap_scale: float, base: dict, episodes: int,
            seed: int = 142):
    """Return (greedy_acceptance, rho) for a topology at a given capacity scale."""
    cfg = {**base, "seed": seed, "topology_file": topo_path,
           "capacity_scale": cap_scale}
    env = NetworkEnv(cfg, mode="unified")
    agent = GreedyAdmission(mode="unified", V=env.V, K=env.K)
    K = env.K
    arrivals = admitted = need_routing = 0
    for _ in range(episodes):
        s, _ = env.reset()
        for _ in range(base.get("max_steps_per_episode", 500)):
            req = env.current_request
            bw = req["bandwidth"]
            conns = [(i, j) for i in range(env.V) for j in range(env.V)
                     if req["Mt"][i, j] == 1]
            B = env.topo.bottleneck_tensor()
            feas = []
            for k in range(K):
                ok = bool(conns)
                for (i, j) in conns:
                    pl = env.topo.paths.get(
                        (env.topo.nodes[i], env.topo.nodes[j]), [])
                    if not pl or k >= len(pl) or B[i, j, k] < bw:
                        ok = False
                        break
                feas.append(ok)
            arrivals += 1
            if any(feas) and not feas[0]:
                need_routing += 1
            s, _, _, _, info = env.step(agent.select_action(s))
            admitted += int(info["admitted"])
    return admitted / arrivals, need_routing / arrivals


def calibrate(topo_path: str, base: dict, episodes: int,
              lo: float = 0.15, hi: float = 6.0, iters: int = 9):
    """Binary-search capacity_scale so greedy acceptance hits the target."""
    best = None
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        acc, rho = measure(topo_path, mid, base, episodes)
        if best is None or abs(acc - TARGET_GREEDY_ACCEPT) < abs(best[1] - TARGET_GREEDY_ACCEPT):
            best = (mid, acc, rho)
        if acc < TARGET_GREEDY_ACCEPT:
            lo = mid           # too congested -> more capacity
        else:
            hi = mid
    return best                 # (scale, greedy_acceptance, rho)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--quick", action="store_true",
                   help="fewer episodes during calibration")
    p.add_argument("--extend", type=float, nargs="+", default=None,
                   help="add these ratios to the EXISTING family (same core, "
                        "read from topo_r2.0.json); appends to family.json")
    args = p.parse_args()
    cal_eps = 8 if args.quick else 14
    fin_eps = 20 if args.quick else 40

    base = yaml.safe_load(open("configs/base.yaml"))
    os.makedirs(OUT_DIR, exist_ok=True)
    if args.extend:
        fam_path = os.path.join(OUT_DIR, "family.json")
        fam = json.load(open(fam_path))
        have = {f["ratio"] for f in fam}
        for ratio in args.extend:
            if ratio in have:
                print(f"ratio {ratio} already in family, skipped")
                continue
            G = topology_from_existing(os.path.join(OUT_DIR, "topo_r2.0.json"), ratio)
            path = os.path.join(OUT_DIR, f"topo_r{ratio:0.1f}.json")
            with open(path, "w") as f:
                json.dump(to_nodelink(G), f, indent=2)
            scale, acc, _ = calibrate(path, base, cal_eps)
            acc, rho = measure(path, scale, base, fin_eps)
            fam.append({"ratio": ratio, "capacity_scale": scale,
                        "greedy_acceptance": acc, "rho": rho, "topology": path})
            print(f"ratio={ratio:4.1f}  cap_scale={scale:5.3f}  "
                  f"greedy_accept={acc:.4f}  rho={rho:6.2%}  -> {path}", flush=True)
        fam.sort(key=lambda f: f["ratio"])
        with open(fam_path, "w") as f:
            json.dump(fam, f, indent=1)
        print(f"wrote {fam_path}")
        return
    rng = np.random.default_rng(CORE_SEED)
    core = build_core(rng)
    print(f"core: {core.number_of_nodes()} nodes, {core.number_of_edges()} edges, "
          f"avg degree {2*core.number_of_edges()/core.number_of_nodes():.2f}\n")

    rows = []
    for ratio in RATIOS:
        G = build_topology(core, ratio, rng)
        path = os.path.join(OUT_DIR, f"topo_r{ratio:0.1f}.json")
        with open(path, "w") as f:
            json.dump(to_nodelink(G), f, indent=2)
        scale, acc, _ = calibrate(path, base, cal_eps)
        acc, rho = measure(path, scale, base, fin_eps)   # final, longer run
        rows.append((ratio, scale, acc, rho, path))
        print(f"ratio={ratio:4.1f}  cap_scale={scale:5.3f}  "
              f"greedy_accept={acc:.4f}  rho={rho:6.2%}  -> {path}")

    with open(os.path.join(OUT_DIR, "family.json"), "w") as f:
        json.dump([{"ratio": r, "capacity_scale": s, "greedy_acceptance": a,
                    "rho": rho, "topology": p} for r, s, a, rho, p in rows],
                  f, indent=2)
    print(f"\nwrote {OUT_DIR}/family.json")
    print("\nrho range: "
          f"{min(r[3] for r in rows):.2%} .. {max(r[3] for r in rows):.2%}")
    print("greedy acceptance spread (should be ~flat): "
          f"{min(r[2] for r in rows):.4f} .. {max(r[2] for r in rows):.4f}")


if __name__ == "__main__":
    main()
