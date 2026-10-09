"""Build the WP2.3 out-of-sample topologies from SNDlib, exactly per the rules fixed in
paper/preregistration_wp2.md section 6, then calibrate and measure rho.

Rules (registered before any rho was computed):
  * every SNDlib node is a core router; parallel links collapsed into one undirected
    link with unit routing weight;
  * each core node gets exactly one endpoint on its own degree-1 access link (V = N);
  * capacity, load-proportional: L_e = number of ordered endpoint pairs whose k = 0
    candidate path traverses e (either direction); C_e = max(L_e, 0.1 * max L);
    the same formula for access and core links;
  * ONE global capacity_scale per network, calibrated to greedy acceptance 0.47.
Capacities are rescaled to mean 600 before calibration: a constant factor, absorbed by
the global scale, so that the binary search starts in the same range as the rho family.

rho is then measured with the fixed WP2 estimator (scripts/measure_rho.py settings:
greedy, eval seeds 142-146, 100 episodes each).

Usage:  python3 scripts/build_sndlib.py [names...]     (default: the registered 9)
Output: data/sndlib_topo/<name>.json, data/sndlib_topo/networks.json
"""
from __future__ import annotations

import json
import os
import re
import sys
import tempfile
from multiprocessing import Pool

import networkx as nx
import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))
os.chdir(ROOT)

import generate_rho_family as g  # noqa: E402
from measure_rho import EPISODES, SEEDS  # noqa: E402
from src.env.topology import NetworkTopology  # noqa: E402

REGISTERED = ["abilene", "polska", "nobel-us", "atlanta", "newyork", "nobel-germany",
              "geant", "janos-us", "nobel-eu"]
RESERVE = ["france", "norway", "cost266"]
SRC = os.path.join("data", "sndlib")
OUT = os.path.join("data", "sndlib_topo")
FLOOR = 0.1
MEAN_CAP = 600.0


def parse(name: str) -> nx.Graph:
    t = open(os.path.join(SRC, f"{name}.txt")).read()
    nodes = re.search(r"NODES \((.*?)\n\)", t, re.S).group(1)
    links = re.search(r"LINKS \((.*?)\n\)", t, re.S).group(1)
    G = nx.Graph()
    for line in nodes.strip().splitlines():
        if line.strip():
            G.add_node(line.split()[0])
    for line in links.strip().splitlines():
        m = re.match(r"\s*\S+ \( (\S+) (\S+) \)", line)
        if m and m.group(1) != m.group(2):
            G.add_edge(m.group(1), m.group(2))          # parallel links collapse here
    return G


def to_json(G: nx.Graph, cap: dict) -> dict:
    return {
        "directed": False, "multigraph": False, "graph": {},
        "nodes": [{"id": n, "tier": G.nodes[n]["tier"]} for n in G.nodes()],
        "links": [{"source": u, "target": v, "capacity": cap[frozenset((u, v))]}
                  for u, v in G.edges()],
    }


def build(name: str) -> dict:
    core = parse(name)
    if not nx.is_connected(core):
        return {"name": name, "status": "disconnected"}
    G = nx.Graph()
    for n in core.nodes():
        G.add_node(n, tier="core")
    G.add_edges_from(core.edges())
    for n in list(core.nodes()):
        e = f"E_{n}"
        G.add_node(e, tier="access")
        G.add_edge(e, n)

    # k = 0 paths depend only on unit weights, not on capacity
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
        json.dump(to_json(G, {frozenset(e): 1.0 for e in G.edges()}), f)
        tmp = f.name
    k = yaml.safe_load(open("configs/base.yaml"))["k_shortest_paths"]
    topo = NetworkTopology(tmp, k=k)
    os.unlink(tmp)
    load = {frozenset(e): 0 for e in G.edges()}
    for plist in topo.paths.values():
        if plist:
            for (u, v) in plist[0]:
                load[frozenset((u, v))] += 1
    mx = max(load.values())
    cap = {e: max(l, FLOOR * mx) for e, l in load.items()}
    norm = MEAN_CAP / (sum(cap.values()) / len(cap))
    cap = {e: round(c * norm, 3) for e, c in cap.items()}

    os.makedirs(OUT, exist_ok=True)
    path = os.path.join(OUT, f"{name}.json")
    json.dump(to_json(G, cap), open(path, "w"), indent=1)

    base = yaml.safe_load(open("configs/base.yaml"))
    scale, acc_cal, _ = g.calibrate(path, base, 14)
    accs, rhos = [], []
    for s in SEEDS:
        a, r = g.measure(path, scale, base, EPISODES, seed=s)
        accs.append(a)
        rhos.append(r)
    acc_cap = [c for e, c in cap.items() if any(n.startswith("E_") for n in e)]
    core_cap = [c for e, c in cap.items() if not any(n.startswith("E_") for n in e)]
    res = {"name": name, "status": "ok", "topology": path,
           "N": core.number_of_nodes(), "core_links": core.number_of_edges(),
           "avg_degree": 2 * core.number_of_edges() / core.number_of_nodes(),
           "access_core_ratio": (sum(acc_cap) / len(acc_cap)) / (sum(core_cap) / len(core_cap)),
           "capacity_scale": scale,
           "greedy_acceptance_wp2": sum(accs) / len(accs),
           "rho_wp2": sum(rhos) / len(rhos), "rho_per_seed": rhos}
    print(f"{name:14s} N={res['N']:2d} ratio={res['access_core_ratio']:.2f} "
          f"scale={scale:.3f} greedy={res['greedy_acceptance_wp2']:.4f} "
          f"rho={100*res['rho_wp2']:.2f}%", flush=True)
    return res


def main() -> None:
    names = sys.argv[1:] or REGISTERED
    with Pool(min(len(names), os.cpu_count() or 4)) as p:
        results = p.map(build, names)
    out = os.path.join(OUT, "networks.json")
    old = {r["name"]: r for r in json.load(open(out))} if os.path.exists(out) else {}
    old.update({r["name"]: r for r in results})
    json.dump(list(old.values()), open(out, "w"), indent=1)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
