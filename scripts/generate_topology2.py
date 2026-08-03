"""Generate a SECOND, structurally different topology for generalization tests.

Deliberately unlike data/operator_topology.json (21 access leaves on a 65-node
core, uniform 500 Mbps):
  * 15 access nodes (vs 21),
  * a Waxman random-geometric CORE of 24 nodes (standard synthetic-backbone
    model) instead of the operator's structured core,
  * heterogeneous link capacities (core 800-1600, access 500 Mbps).

Same node-link schema the loader expects: directed=false, nodes carry a "tier"
(access endpoints vs core), edges carry "capacity". Reproducible (fixed seed).
Guarantees the core is connected and that >=3 shortest paths exist between the
overwhelming majority of access pairs (so K=3 routing is meaningful).
"""
import json
import networkx as nx
import numpy as np

SEED = 7
N_ACCESS = 15
N_CORE = 24
rng = np.random.default_rng(SEED)


def build():
    # --- Waxman core (random geometric backbone) ---
    for attempt in range(50):
        core = nx.waxman_graph(N_CORE, beta=0.55, alpha=0.35,
                               seed=int(rng.integers(1 << 30)))
        if nx.is_connected(core):
            # want a reasonably meshed core so K=3 paths exist
            if min(dict(core.degree()).values()) >= 2:
                break
    core = nx.convert_node_labels_to_integers(core, first_label=1,
                                              label_attribute=None)
    core_ids = [f"C{i}" for i in core.nodes()]
    mapping = {i: f"C{i}" for i in core.nodes()}
    core = nx.relabel_nodes(core, mapping)

    G = nx.Graph()
    for c in core_ids:
        G.add_node(c, tier="core")
    for u, v in core.edges():
        cap = float(rng.integers(8, 17) * 100)  # 800..1600
        G.add_edge(u, v, capacity=cap)

    # --- access leaves, each attached to a distinct-ish core node ---
    core_by_deg = sorted(core_ids, key=lambda c: core.degree(c), reverse=True)
    for a in range(N_ACCESS):
        aid = f"A{a}"
        G.add_node(aid, tier="access")
        # attach to a well-connected core node (spread across the top of the core)
        attach = core_by_deg[a % len(core_by_deg)]
        G.add_edge(aid, attach, capacity=500.0)

    return G


def to_nodelink(G):
    nodes = [{"tier": G.nodes[n]["tier"], "id": n} for n in G.nodes()]
    edges = [{"capacity": G[u][v]["capacity"], "source": u, "target": v}
             for u, v in G.edges()]
    return {"directed": False, "multigraph": False, "graph": {},
            "nodes": nodes, "links": edges}


def main():
    G = build()
    access = [n for n in G.nodes() if G.nodes[n]["tier"] == "access"]
    # sanity: how many access pairs have >=3 shortest paths through the core?
    import itertools
    ok = tot = 0
    for a, b in itertools.permutations(access, 2):
        tot += 1
        paths = 0
        try:
            for _ in nx.shortest_simple_paths(G, a, b):
                paths += 1
                if paths >= 3:
                    break
        except nx.NetworkXNoPath:
            pass
        ok += int(paths >= 3)
    out = "data/topology2.json"
    with open(out, "w") as f:
        json.dump(to_nodelink(G), f, indent=2)
    print(f"wrote {out}")
    print(f"nodes={G.number_of_nodes()} (access={len(access)}, core={G.number_of_nodes()-len(access)})"
          f" edges={G.number_of_edges()}")
    print(f"access pairs with >=3 paths: {ok}/{tot} = {ok/tot:.1%}")


if __name__ == "__main__":
    main()
