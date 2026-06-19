"""Unit tests for NetworkTopology (src/env/topology.py)."""
import numpy as np
import pytest

from src.env.topology import NetworkTopology


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _first_path(topo: NetworkTopology):
    """Return the first non-empty path found in the topology."""
    for path_list in topo.paths.values():
        if path_list and path_list[0]:
            return path_list[0]
    pytest.skip("no usable path found")


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


class TestConstruction:
    def test_node_count(self, topo_file):
        topo = NetworkTopology(topo_file, k=3)
        assert topo.V == 4

    def test_node_idx_complete(self, topo_file):
        topo = NetworkTopology(topo_file, k=3)
        assert len(topo.node_idx) == topo.V

    def test_avail_equals_cap_on_init(self, topo_file):
        topo = NetworkTopology(topo_file, k=3)
        assert topo.avail == topo._cap

    def test_edges_loaded(self, topo_file):
        topo = NetworkTopology(topo_file, k=3)
        assert len(topo.G.edges()) == 12  # 4-node fixture has 12 directed edges


# ---------------------------------------------------------------------------
# k-shortest paths
# ---------------------------------------------------------------------------


class TestPaths:
    def test_paths_cover_all_pairs(self, topo_file):
        topo = NetworkTopology(topo_file, k=3)
        expected_pairs = topo.V * (topo.V - 1)  # ordered pairs, excl. self
        assert len(topo.paths) == expected_pairs

    def test_at_most_k_paths_per_pair(self, topo_file):
        topo = NetworkTopology(topo_file, k=3)
        for path_list in topo.paths.values():
            assert len(path_list) <= 3

    def test_paths_are_edge_lists(self, topo_file):
        topo = NetworkTopology(topo_file, k=3)
        for path_list in topo.paths.values():
            for path in path_list:
                assert isinstance(path, list)
                for edge in path:
                    assert isinstance(edge, tuple) and len(edge) == 2

    def test_most_pairs_have_max_paths(self, topo_file):
        """Dense topology: most pairs should have k=3 paths."""
        topo = NetworkTopology(topo_file, k=3)
        counts = [len(pl) for pl in topo.paths.values()]
        # At least half of ordered pairs should have 3 paths in a dense graph
        assert sum(c == 3 for c in counts) >= len(counts) // 2


# ---------------------------------------------------------------------------
# Bottleneck tensor
# ---------------------------------------------------------------------------


class TestBottleneckTensor:
    def test_shape(self, topo_file):
        topo = NetworkTopology(topo_file, k=3)
        B = topo.bottleneck_tensor()
        assert B.shape == (4, 4, 3)

    def test_dtype(self, topo_file):
        topo = NetworkTopology(topo_file, k=3)
        assert topo.bottleneck_tensor().dtype == np.float32

    def test_diagonal_is_zero(self, topo_file):
        """Paths from a node to itself don't exist ⟹ diagonal stays zero."""
        topo = NetworkTopology(topo_file, k=3)
        B = topo.bottleneck_tensor()
        for i in range(topo.V):
            assert np.all(B[i, i, :] == 0)

    def test_off_diagonal_positive(self, topo_file):
        """Connected topology ⟹ at least path 0 has positive bottleneck."""
        topo = NetworkTopology(topo_file, k=3)
        B = topo.bottleneck_tensor()
        for i in range(topo.V):
            for j in range(topo.V):
                if i != j:
                    assert B[i, j, 0] > 0, f"B[{i},{j},0] should be > 0"


# ---------------------------------------------------------------------------
# Reserve / release
# ---------------------------------------------------------------------------


class TestReserveRelease:
    def test_reserve_decreases_avail(self, topo_file):
        topo = NetworkTopology(topo_file, k=3)
        path = _first_path(topo)
        before = {e: topo.avail[e] for e in path}
        topo.reserve(path, 100.0)
        for e in path:
            assert topo.avail[e] == before[e] - 100.0

    def test_release_restores_avail(self, topo_file):
        topo = NetworkTopology(topo_file, k=3)
        path = _first_path(topo)
        original = {e: topo.avail[e] for e in path}
        topo.reserve(path, 100.0)
        topo.release(path, 100.0)
        for e in path:
            assert topo.avail[e] == pytest.approx(original[e])

    def test_release_capped_at_original_capacity(self, topo_file):
        topo = NetworkTopology(topo_file, k=3)
        path = _first_path(topo)
        cap_before = {e: topo._cap[e] for e in path}
        topo.reserve(path, 50.0)
        topo.release(path, 9999.0)  # over-release
        for e in path:
            assert topo.avail[e] == cap_before[e]

    def test_reset_capacities(self, topo_file):
        topo = NetworkTopology(topo_file, k=3)
        path = _first_path(topo)
        cap_before = dict(topo._cap)
        topo.reserve(path, 300.0)
        topo.reset_capacities()
        assert topo.avail == cap_before

    def test_bottleneck_reflects_reservation(self, topo_file):
        topo = NetworkTopology(topo_file, k=3)
        # Get nodes for pair (0→1), path index 0
        src, dst = topo.nodes[0], topo.nodes[1]
        path = topo.paths[(src, dst)][0]
        B_before = topo.bottleneck_tensor()[0, 1, 0]
        topo.reserve(path, 200.0)
        B_after = topo.bottleneck_tensor()[0, 1, 0]
        assert B_after < B_before
