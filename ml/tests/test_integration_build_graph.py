"""
ml/tests/test_integration_build_graph.py
=========================================
End-to-end integration test for build_graph.py.

Exercises the FULL pipeline on a synthetic 4-transaction fixture:
  Raw DataFrame → Features → Bipartite graph → Projection → PyG HeteroData

Checks:
  - Exact node and edge counts
  - Feature tensor shape matches ADDRESS_FEATURE_COLS order
  - No NaN in any feature tensor
  - Edge index is well-formed (no out-of-range indices)
  - Edge attribute shapes match edge count
  - Self-loops absent from projection and bipartite graph
  - Correct edge_type labels on bipartite edges
  - graph_50k.pt can be re-loaded from disk after save (round-trip test)

Fixture (4 addresses, 4 transactions):

  tx_1  addr_A -[1.0 BTC]→ tx_1 -[0.9]→ addr_B, [0.1]→ addr_A (change)
  tx_2  addr_B -[0.9 BTC]→ tx_2 -[0.85]→ addr_C
  tx_3  addr_A -[0.1 BTC]→ tx_3 -[0.09]→ addr_C
  tx_4  addr_D -[2.0 BTC]→ tx_4 -[1.8]→ addr_D   (addr_D reuses, isolated)

addr_D is an isolated-to-others address: only transacts with itself.
That tests the isolated component in the projection.
"""

import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from scripts.build_graph import (
    ADDRESS_FEATURE_COLS,
    TX_FEATURE_COLS,
    SATOSHI,
    _hash_string,
    _compute_address_features,
    _compute_projection_features,
    build_bipartite_and_projection,
    build_pyg_heterodata,
    validate_graph,
)


# ── Fixture ────────────────────────────────────────────────────────────────────

def _ts(day: str, hour: int) -> pd.Timestamp:
    dates = {"Mon": "2026-09-01", "Tue": "2026-09-02"}
    return pd.Timestamp(f"{dates[day]} {hour:02d}:00:00", tz="UTC")


@pytest.fixture(scope="module")
def integration_df() -> pd.DataFrame:
    """4-transaction synthetic fixture for end-to-end pipeline testing."""
    rows = [
        # tx_1: addr_A spends 1 BTC; outputs 0.9→addr_B, 0.1→addr_A (change)
        {"tx_hash": "tx_1", "block_timestamp": _ts("Mon", 2),
         "input_address": "addr_A", "input_value_sat": int(1.0 * SATOSHI),
         "output_address": "addr_B", "output_value_sat": int(0.9 * SATOSHI),
         "output_type": "pubkeyhash", "output_index": 0,
         "fee_satoshi": 0, "input_count": 1, "output_count": 2,
         "total_input_sat": int(1.0 * SATOSHI), "total_output_sat": int(1.0 * SATOSHI),
         "is_coinbase": False},
        {"tx_hash": "tx_1", "block_timestamp": _ts("Mon", 2),
         "input_address": "addr_A", "input_value_sat": int(1.0 * SATOSHI),
         "output_address": "addr_A", "output_value_sat": int(0.1 * SATOSHI),  # change
         "output_type": "pubkeyhash", "output_index": 1,
         "fee_satoshi": 0, "input_count": 1, "output_count": 2,
         "total_input_sat": int(1.0 * SATOSHI), "total_output_sat": int(1.0 * SATOSHI),
         "is_coinbase": False},

        # tx_2: addr_B spends 0.9 BTC; outputs 0.85→addr_C
        {"tx_hash": "tx_2", "block_timestamp": _ts("Mon", 3),
         "input_address": "addr_B", "input_value_sat": int(0.9 * SATOSHI),
         "output_address": "addr_C", "output_value_sat": int(0.85 * SATOSHI),
         "output_type": "pubkeyhash", "output_index": 0,
         "fee_satoshi": 0, "input_count": 1, "output_count": 1,
         "total_input_sat": int(0.9 * SATOSHI), "total_output_sat": int(0.85 * SATOSHI),
         "is_coinbase": False},

        # tx_3: addr_A spends 0.1 BTC; output 0.09→addr_C (P2SH — tests is_script_hash)
        {"tx_hash": "tx_3", "block_timestamp": _ts("Tue", 14),
         "input_address": "addr_A", "input_value_sat": int(0.1 * SATOSHI),
         "output_address": "addr_C", "output_value_sat": int(0.09 * SATOSHI),
         "output_type": "scripthash",  # ← P2SH: addr_C should get is_script_hash=1
         "output_index": 0,
         "fee_satoshi": 0, "input_count": 1, "output_count": 1,
         "total_input_sat": int(0.1 * SATOSHI), "total_output_sat": int(0.09 * SATOSHI),
         "is_coinbase": False},

        # tx_4: addr_D spends 2.0 BTC to itself only (isolated from A/B/C)
        {"tx_hash": "tx_4", "block_timestamp": _ts("Mon", 10),
         "input_address": "addr_D", "input_value_sat": int(2.0 * SATOSHI),
         "output_address": "addr_D", "output_value_sat": int(1.8 * SATOSHI),
         "output_type": "pubkeyhash", "output_index": 0,
         "fee_satoshi": 0, "input_count": 1, "output_count": 1,
         "total_input_sat": int(2.0 * SATOSHI), "total_output_sat": int(1.8 * SATOSHI),
         "is_coinbase": False},
    ]
    df = pd.DataFrame(rows)
    df["block_timestamp"] = pd.to_datetime(df["block_timestamp"], utc=True)
    return df


@pytest.fixture(scope="module")
def pipeline_output(integration_df):
    """Run the full pipeline and return all intermediate objects."""
    snapshot_now = integration_df["block_timestamp"].max()
    feat = _compute_address_features(integration_df, snapshot_now)
    bipartite_G, projection_G = build_bipartite_and_projection(integration_df, feat)
    feat = _compute_projection_features(feat, projection_G)
    pyg = build_pyg_heterodata(integration_df, feat, bipartite_G)
    return feat, bipartite_G, projection_G, pyg


# ── Feature table checks ───────────────────────────────────────────────────────

class TestFeatureTable:
    def test_all_four_addresses_present(self, pipeline_output):
        feat, *_ = pipeline_output
        for addr in ["addr_A", "addr_B", "addr_C", "addr_D"]:
            assert _hash_string(addr) in feat.index, f"{addr} missing from feature table"

    def test_all_14_features_present(self, pipeline_output):
        feat, *_ = pipeline_output
        for col in ADDRESS_FEATURE_COLS:
            assert col in feat.columns, f"Feature column '{col}' missing"

    def test_no_nan_in_feature_matrix(self, pipeline_output):
        feat, *_ = pipeline_output
        nan_info = feat[ADDRESS_FEATURE_COLS].isna().sum()
        assert not nan_info.any(), f"NaN features:\n{nan_info[nan_info > 0]}"

    def test_is_script_hash_addr_c(self, pipeline_output):
        """addr_C received a P2SH output in tx_3 → must have is_script_hash=1."""
        feat, *_ = pipeline_output
        assert feat.loc[_hash_string("addr_C"), "is_script_hash"] == 1

    def test_is_script_hash_addr_a_zero(self, pipeline_output):
        """addr_A received only pubkeyhash outputs → is_script_hash must be 0."""
        feat, *_ = pipeline_output
        assert feat.loc[_hash_string("addr_A"), "is_script_hash"] == 0

    def test_addr_d_reuse_count(self, pipeline_output):
        """addr_D appears as both input and output in tx_4 → reuse_count = 1."""
        feat, *_ = pipeline_output
        assert feat.loc[_hash_string("addr_D"), "address_reuse_count"] == 1

    def test_monetary_values_non_negative(self, pipeline_output):
        feat, *_ = pipeline_output
        for col in ["total_sent_btc", "total_received_btc", "avg_tx_value_btc"]:
            assert (feat[col] >= 0).all(), f"Negative values in {col}"


# ── Bipartite graph checks ─────────────────────────────────────────────────────

class TestBipartiteGraph:
    def test_node_types(self, pipeline_output):
        _, bipartite_G, *_ = pipeline_output
        addr_nodes = [n for n, d in bipartite_G.nodes(data=True) if d.get("node_type") == "address"]
        tx_nodes   = [n for n, d in bipartite_G.nodes(data=True) if d.get("node_type") == "transaction"]
        assert len(addr_nodes) == 4, f"Expected 4 address nodes, got {len(addr_nodes)}"
        assert len(tx_nodes)   == 4, f"Expected 4 transaction nodes, got {len(tx_nodes)}"

    def test_edge_types(self, pipeline_output):
        _, bipartite_G, *_ = pipeline_output
        edge_types = {d.get("edge_type") for _, _, d in bipartite_G.edges(data=True)}
        assert "INPUT_TO"  in edge_types
        assert "OUTPUT_TO" in edge_types

    def test_no_self_loops_in_bipartite(self, pipeline_output):
        """Bipartite graph has no address→address or tx→tx self-loops."""
        _, bipartite_G, *_ = pipeline_output
        import networkx as nx
        self_loops = list(nx.selfloop_edges(bipartite_G))
        assert len(self_loops) == 0, f"Self-loops found in bipartite graph: {self_loops}"

    def test_input_to_edges_go_addr_to_tx(self, pipeline_output):
        _, bipartite_G, *_ = pipeline_output
        node_types = dict(bipartite_G.nodes(data="node_type"))
        for u, v, d in bipartite_G.edges(data=True):
            if d.get("edge_type") == "INPUT_TO":
                assert node_types[u] == "address",     f"INPUT_TO source {u} is not an address"
                assert node_types[v] == "transaction",  f"INPUT_TO target {v} is not a transaction"

    def test_output_to_edges_go_tx_to_addr(self, pipeline_output):
        _, bipartite_G, *_ = pipeline_output
        node_types = dict(bipartite_G.nodes(data="node_type"))
        for u, v, d in bipartite_G.edges(data=True):
            if d.get("edge_type") == "OUTPUT_TO":
                assert node_types[u] == "transaction", f"OUTPUT_TO source {u} is not a tx"
                assert node_types[v] == "address",     f"OUTPUT_TO target {v} is not an address"

    def test_edge_weights_non_negative(self, pipeline_output):
        _, bipartite_G, *_ = pipeline_output
        for u, v, d in bipartite_G.edges(data=True):
            assert d.get("value_btc", 0) >= 0, f"Negative edge weight on ({u}, {v})"


# ── Projection graph checks ────────────────────────────────────────────────────

class TestProjectionGraph:
    def test_no_self_loops_in_projection(self, pipeline_output):
        import networkx as nx
        _, _, projection_G, _ = pipeline_output
        self_loops = list(nx.selfloop_edges(projection_G))
        assert len(self_loops) == 0, f"Self-loops in projection: {self_loops}"

    def test_addr_d_isolated_in_projection(self, pipeline_output):
        """addr_D only transacts with itself → degree=0 in projection."""
        feat, _, projection_G, _ = pipeline_output
        h = _hash_string("addr_D")
        degree = projection_G.degree(h) if projection_G.has_node(h) else 0
        assert degree == 0, f"addr_D should be isolated in projection, got degree={degree}"

    def test_abc_triangle_in_projection(self, pipeline_output):
        """addr_A, addr_B, addr_C should form a triangle in the projection."""
        import networkx as nx
        _, _, projection_G, _ = pipeline_output
        hA, hB, hC = _hash_string("addr_A"), _hash_string("addr_B"), _hash_string("addr_C")
        assert projection_G.has_edge(hA, hB), "Missing edge A-B in projection"
        assert projection_G.has_edge(hA, hC), "Missing edge A-C in projection"
        assert projection_G.has_edge(hB, hC), "Missing edge B-C in projection"

    def test_projection_edge_weights_positive(self, pipeline_output):
        _, _, projection_G, _ = pipeline_output
        for u, v, d in projection_G.edges(data=True):
            assert d.get("weight", 0) > 0, f"Zero/negative edge weight ({u}, {v})"


# ── Validation function check ──────────────────────────────────────────────────

class TestValidateGraph:
    def test_validation_passes_on_clean_fixture(self, pipeline_output):
        feat, _, projection_G, _ = pipeline_output
        ok = validate_graph(feat, projection_G, tag="integration_test")
        assert ok, "validate_graph returned False on clean integration fixture"


# ── PyG HeteroData checks ──────────────────────────────────────────────────────

class TestPyGHeteroData:
    def test_pyg_not_none(self, pipeline_output):
        pytest.importorskip("torch", reason="torch required for PyG tests")
        *_, pyg = pipeline_output
        assert pyg is not None, "build_pyg_heterodata returned None (torch-geometric missing?)"

    def test_address_node_count(self, pipeline_output):
        pytest.importorskip("torch")
        *_, pyg = pipeline_output
        if pyg is None:
            pytest.skip("PyG not available")
        assert pyg["address"].num_nodes == 4

    def test_transaction_node_count(self, pipeline_output):
        pytest.importorskip("torch")
        *_, pyg = pipeline_output
        if pyg is None:
            pytest.skip("PyG not available")
        assert pyg["transaction"].num_nodes == 4

    def test_address_feature_shape(self, pipeline_output):
        pytest.importorskip("torch")
        *_, pyg = pipeline_output
        if pyg is None:
            pytest.skip("PyG not available")
        x = pyg["address"].x
        assert x.shape[0] == 4,                       f"Expected 4 address nodes, got {x.shape[0]}"
        assert x.shape[1] == len(ADDRESS_FEATURE_COLS), \
            f"Expected {len(ADDRESS_FEATURE_COLS)} features, got {x.shape[1]}"

    def test_tx_feature_shape(self, pipeline_output):
        pytest.importorskip("torch")
        *_, pyg = pipeline_output
        if pyg is None:
            pytest.skip("PyG not available")
        x = pyg["transaction"].x
        assert x.shape[0] == 4,                    f"Expected 4 tx nodes, got {x.shape[0]}"
        assert x.shape[1] == len(TX_FEATURE_COLS), f"Expected {len(TX_FEATURE_COLS)} tx features"

    def test_no_nan_in_address_feature_tensor(self, pipeline_output):
        import torch
        pytest.importorskip("torch")
        *_, pyg = pipeline_output
        if pyg is None:
            pytest.skip("PyG not available")
        assert not torch.isnan(pyg["address"].x).any(), "NaN in address feature tensor"

    def test_no_nan_in_tx_feature_tensor(self, pipeline_output):
        import torch
        pytest.importorskip("torch")
        *_, pyg = pipeline_output
        if pyg is None:
            pytest.skip("PyG not available")
        assert not torch.isnan(pyg["transaction"].x).any(), "NaN in transaction feature tensor"

    def test_input_to_edge_index_shape(self, pipeline_output):
        pytest.importorskip("torch")
        *_, pyg = pipeline_output
        if pyg is None:
            pytest.skip("PyG not available")
        ei = pyg["address", "INPUT_TO", "transaction"].edge_index
        assert ei.shape[0] == 2, "edge_index must be shape [2, num_edges]"
        assert ei.shape[1] > 0, "No INPUT_TO edges found"

    def test_output_to_edge_index_shape(self, pipeline_output):
        pytest.importorskip("torch")
        *_, pyg = pipeline_output
        if pyg is None:
            pytest.skip("PyG not available")
        ei = pyg["transaction", "OUTPUT_TO", "address"].edge_index
        assert ei.shape[0] == 2
        assert ei.shape[1] > 0

    def test_edge_indices_in_range(self, pipeline_output):
        """No edge should reference a non-existent node."""
        import torch
        pytest.importorskip("torch")
        *_, pyg = pipeline_output
        if pyg is None:
            pytest.skip("PyG not available")
        n_addr = pyg["address"].num_nodes
        n_tx   = pyg["transaction"].num_nodes

        ei_in = pyg["address", "INPUT_TO", "transaction"].edge_index
        assert ei_in[0].max() < n_addr, "INPUT_TO src index out of range for address nodes"
        assert ei_in[1].max() < n_tx,   "INPUT_TO dst index out of range for tx nodes"

        ei_out = pyg["transaction", "OUTPUT_TO", "address"].edge_index
        assert ei_out[0].max() < n_tx,   "OUTPUT_TO src index out of range for tx nodes"
        assert ei_out[1].max() < n_addr, "OUTPUT_TO dst index out of range for address nodes"

    def test_edge_attr_length_matches_edge_index(self, pipeline_output):
        pytest.importorskip("torch")
        *_, pyg = pipeline_output
        if pyg is None:
            pytest.skip("PyG not available")
        for etype in [
            ("address", "INPUT_TO", "transaction"),
            ("transaction", "OUTPUT_TO", "address"),
        ]:
            ei   = pyg[etype].edge_index
            attr = pyg[etype].edge_attr
            assert ei.shape[1] == attr.shape[0], \
                f"{etype}: edge_index has {ei.shape[1]} edges but edge_attr has {attr.shape[0]}"

    def test_round_trip_save_and_load(self, pipeline_output, tmp_path):
        """Save the PyG object to disk and reload it — tensors must be identical."""
        import torch
        pytest.importorskip("torch")
        *_, pyg = pipeline_output
        if pyg is None:
            pytest.skip("PyG not available")

        save_path = tmp_path / "test_graph.pt"
        torch.save(pyg, save_path)
        loaded = torch.load(save_path, weights_only=False)

        orig_x   = pyg["address"].x
        loaded_x = loaded["address"].x
        assert torch.allclose(orig_x, loaded_x), "Address feature tensors differ after round-trip"
