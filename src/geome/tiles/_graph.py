from __future__ import annotations

import numpy as np
from scipy import sparse
from scipy.spatial import KDTree

__all__ = ["expand", "gcn_normalize", "knn_adjacency", "morton_order", "spatial_graph"]


def knn_adjacency(coords: np.ndarray, n_neighbors: int = 6, groups: np.ndarray | None = None) -> sparse.csr_array:
    """Binary, symmetric kNN adjacency over ``coords``; with ``groups``, neighbours never cross groups."""
    n = coords.shape[0]
    codes = np.zeros(n, dtype=np.int64) if groups is None else groups
    rows, cols = [], []
    for g in np.unique(codes):
        idx = np.flatnonzero(codes == g)
        k = min(n_neighbors, idx.size - 1)
        if k < 1:
            continue
        _, nn = KDTree(coords[idx]).query(coords[idx], k=k + 1, workers=-1)
        rows.append(np.repeat(idx, k))
        cols.append(idx[nn[:, 1:].ravel()])
    rows, cols = (np.concatenate(v) if v else np.empty(0, dtype=np.int64) for v in (rows, cols))
    a = sparse.csr_array((np.ones(rows.size, dtype=np.float32), (rows, cols)), shape=(n, n))
    return ((a + a.T) > 0).astype(np.float32)


def gcn_normalize(adjacency) -> sparse.csr_array:
    """:math:`D^{-1/2} (A + I) D^{-1/2}` of a binarized adjacency, as a GCN normalizes it."""
    a = (sparse.csr_array(adjacency) != 0).astype(np.float32)
    a = sparse.csr_array(a.maximum(a.T)) + sparse.eye_array(a.shape[0], dtype=np.float32, format="csr")
    a = sparse.csr_array(a > 0, dtype=np.float32)
    d = 1.0 / np.sqrt(np.asarray(a.sum(axis=1)).ravel())
    return sparse.csr_array(sparse.diags_array(d) @ a @ sparse.diags_array(d), dtype=np.float32)


def spatial_graph(coords: np.ndarray, n_neighbors: int = 6, groups: np.ndarray | None = None) -> sparse.csr_array:
    """Symmetric kNN graph with self-loops, normalized as in a GCN: :math:`D^{-1/2} (A + I) D^{-1/2}`.

    The weights are computed once on the whole slide, so a tile's subgraph keeps its cells' global degrees and a
    GCN layer with ``normalize=False`` gives a tile's cells exactly their full-graph output.
    """
    return gcn_normalize(knn_adjacency(coords, n_neighbors, groups))


def _spread(v: np.ndarray) -> np.ndarray:
    """Interleave zero bits into the low 32 bits of ``v`` (one half of a 2-D Morton code)."""
    v = v & np.uint64(0x00000000FFFFFFFF)
    for shift, mask in ((16, 0x0000FFFF0000FFFF), (8, 0x00FF00FF00FF00FF), (4, 0x0F0F0F0F0F0F0F0F),
                        (2, 0x3333333333333333), (1, 0x5555555555555555)):  # fmt: skip
        v = (v | (v << np.uint64(shift))) & np.uint64(mask)
    return v


def morton_order(coords: np.ndarray) -> np.ndarray:
    """Permutation sorting cells along a Z-order curve, so cells close in space are close in row order."""
    q = coords[:, :2] - coords[:, :2].min(axis=0)
    q = (q / max(float(q.max()), 1e-12) * (2**16 - 1)).astype(np.uint64)
    return np.argsort(_spread(q[:, 0]) | (_spread(q[:, 1]) << np.uint64(1)), kind="stable")


def expand(graph: sparse.csr_array, core: np.ndarray, n_hops: int) -> np.ndarray:
    """Sorted rows of ``core`` plus every cell within ``n_hops`` edges of it."""
    seen = np.zeros(graph.shape[0], dtype=bool)
    seen[core] = True
    frontier = core
    for _ in range(n_hops):
        nbrs = np.unique(graph[frontier].indices)
        frontier = nbrs[~seen[nbrs]]
        seen[frontier] = True
    return np.flatnonzero(seen)
