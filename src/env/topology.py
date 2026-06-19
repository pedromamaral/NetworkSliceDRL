import json
from itertools import islice

import networkx as nx
import numpy as np


class NetworkTopology:
    """IP/MPLS core network topology with k-shortest-path routing support."""

    def __init__(self, graph_file: str, k: int = 3) -> None:
        with open(graph_file) as f:
            data = json.load(f)
        self.G: nx.DiGraph = self._load_graph(data)
        self.k = k

        # Endpoints (tier="access") are the slice attachment points — base stations,
        # cloud sites, MEC servers.  Core/dist nodes (S1–S65) are routing
        # infrastructure only and must not appear in the MDP state.
        # Test fixtures have no "tier" field, so all their nodes default to "access".
        node_tiers = {n["id"]: n.get("tier", "access") for n in data["nodes"]}
        endpoint_nodes = [n for n in self.G.nodes if node_tiers.get(n, "access") == "access"]
        if not endpoint_nodes:
            endpoint_nodes = list(self.G.nodes)

        self.nodes: list = endpoint_nodes
        self.V: int = len(endpoint_nodes)
        self.node_idx: dict = {n: i for i, n in enumerate(endpoint_nodes)}

        # Capacity is tracked for ALL edges; routing paths traverse core/dist links.
        self._cap: dict = {
            (u, v): attrs["capacity_mbps"]
            for u, v, attrs in self.G.edges(data=True)
        }
        self.avail: dict = dict(self._cap)
        self._B_cache: np.ndarray | None = None  # invalidated by reserve/release/reset
        self._precompute_paths()

    @staticmethod
    def _load_graph(data: dict) -> nx.DiGraph:
        """Parse node-link JSON format without relying on nx.node_link_graph
        so the loader is robust across NetworkX 3.2–3.4+ (where the default
        edge-list key changed from 'links' to 'edges').
        Undirected topologies (``"directed": false``) have both orientations
        of each link added so the DiGraph supports routing in both directions."""
        G = nx.DiGraph()
        for node in data["nodes"]:
            G.add_node(node["id"])
        links = data.get("links", data.get("edges", []))
        directed = data.get("directed", True)
        for lnk in links:
            cap = float(lnk.get("capacity_mbps") or lnk.get("capacity", 0.0))
            w = float(lnk.get("weight", 1.0))
            G.add_edge(lnk["source"], lnk["target"], capacity_mbps=cap, weight=w)
            if not directed:
                G.add_edge(lnk["target"], lnk["source"], capacity_mbps=cap, weight=w)
        return G

    def _precompute_paths(self) -> None:
        """Precompute up to *k* shortest simple paths for every ordered endpoint pair."""
        self.paths: dict = {}
        for src in self.nodes:
            for dst in self.nodes:
                if src == dst:
                    continue
                try:
                    gen = nx.shortest_simple_paths(self.G, src, dst, weight="weight")
                    node_paths = list(islice(gen, self.k))
                    self.paths[(src, dst)] = [
                        list(zip(p[:-1], p[1:])) for p in node_paths
                    ]
                except (nx.NetworkXNoPath, nx.NodeNotFound):
                    self.paths[(src, dst)] = []

    # ------------------------------------------------------------------
    # Derived state
    # ------------------------------------------------------------------

    def bottleneck_tensor(self) -> np.ndarray:
        """Return B of shape (V, V, k) where B[i,j,p] is the minimum available
        capacity (Mbps) along the p-th shortest path from endpoint i to endpoint j.

        Result is cached and only recomputed when avail changes (reserve/release/reset).
        Typical cache hit rate is high: a rejected step never touches avail.
        """
        if self._B_cache is not None:
            return self._B_cache
        B = np.zeros((self.V, self.V, self.k), dtype=np.float32)
        for (src, dst), path_list in self.paths.items():
            i = self.node_idx[src]
            j = self.node_idx[dst]
            for p_idx, path in enumerate(path_list):
                if path:
                    B[i, j, p_idx] = min(self.avail.get(e, 0.0) for e in path)
        self._B_cache = B
        return B

    # ------------------------------------------------------------------
    # Resource management
    # ------------------------------------------------------------------

    def reserve(self, path_edges: list, bw: float) -> None:
        """Deduct *bw* Mbps from every edge in *path_edges*."""
        for e in path_edges:
            self.avail[e] -= bw
        self._B_cache = None

    def release(self, path_edges: list, bw: float) -> None:
        """Restore *bw* Mbps on every edge, capped at original capacity."""
        for e in path_edges:
            self.avail[e] = min(self.avail[e] + bw, self._cap[e])
        self._B_cache = None

    def reset_capacities(self) -> None:
        """Restore all links to their original capacities."""
        self.avail = dict(self._cap)
        self._B_cache = None

    def avg_link_utilization(self) -> float:
        """Mean fraction of capacity in use across all links (0.0–1.0)."""
        if not self._cap:
            return 0.0
        return float(
            np.mean([1.0 - self.avail[e] / self._cap[e] for e in self._cap])
        )
