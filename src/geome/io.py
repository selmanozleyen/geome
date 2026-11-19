"""Rewrite a store with its cells grouped by obs categories, out of core."""

from __future__ import annotations

import os
from collections.abc import Sequence

import anndata as ad
import numpy as np
import zarr
from anndata.acc import A, LayerAcc, MultiAcc
from scipy import sparse
from zarr.codecs import BloscCodec

__all__ = ["write_grouped"]

Rep = LayerAcc | MultiAcc


def _path(ref: Rep) -> tuple[str | None, str]:
    """``(parent group, name)`` of a rep in an AnnData store."""
    if isinstance(ref, LayerAcc):
        return (None, "X") if ref.k is None else ("layers", ref.k)
    if isinstance(ref, MultiAcc) and ref.dim == "obs":
        return "obsm", ref.k
    raise ValueError(f"{ref!r} is not A.X, A.layers[k] or A.obsm[k].")


def _runs(rows: np.ndarray, max_gap: int) -> list[np.ndarray]:
    """``rows`` (ascending) split where consecutive rows are more than ``max_gap`` apart."""
    return np.split(rows, np.flatnonzero(np.diff(rows) > max_gap) + 1)


def _read(x, rows: np.ndarray, max_gap: int):
    """Rows ``rows`` of ``x`` in that order: each run of nearby rows read as one slice, then reordered."""
    srt = np.argsort(rows, kind="stable")
    parts = []
    for run in _runs(rows[srt], max_gap):
        lo, hi = int(run[0]), int(run[-1]) + 1
        block = x[lo:hi]
        parts.append(block[run - lo])
    out = sparse.vstack(parts, format="csr") if sparse.issparse(parts[0]) else np.concatenate(parts)
    back = np.empty_like(srt)
    back[srt] = np.arange(srt.size)
    return out[back]


def _copy_sparse(x, group: zarr.Group, name: str, order: np.ndarray, chunk_rows: int, shard_bytes: int, gap: int):
    indptr = np.asarray(x.group["indptr"][:], dtype=np.int64)
    new_indptr = np.concatenate([[0], np.cumsum(np.diff(indptr)[order])])
    nnz, (n, m) = int(new_indptr[-1]), x.shape
    per_row = max(1, nnz // max(n, 1))
    chunk = max(1, chunk_rows * per_row)
    g = group.create_group(name)
    g.attrs.update({"encoding-type": "csr_matrix", "encoding-version": "0.1.0", "shape": [n, m]})
    blosc = (BloscCodec(cname="lz4", clevel=3, shuffle="shuffle"),)
    arrays = {}
    for key, dtype in (("data", x.dtype), ("indices", x.group["indices"].dtype)):
        shard = max(chunk, shard_bytes // np.dtype(dtype).itemsize // chunk * chunk)
        arrays[key] = g.create_array(key, shape=(nnz,), dtype=dtype, chunks=(min(chunk, max(nnz, 1)),),
                                     shards=(min(shard, max(nnz, 1)),), compressors=blosc)  # fmt: skip
    g.create_array("indptr", data=new_indptr)
    # ponytail: blocks end on row boundaries, not shard boundaries, so each block rewrites at most two partial
    # shards; align blocks to shards if that shows up in the write time
    block = max(1, (arrays["data"].shards[0]) // per_row)
    for a in range(0, n, block):
        b = min(n, a + block)
        sub = _read(x, order[a:b], gap)
        s, e = int(new_indptr[a]), int(new_indptr[b])
        arrays["data"][s:e] = sub.data
        arrays["indices"][s:e] = sub.indices


def _copy_dense(x, group: zarr.Group, name: str, order: np.ndarray, chunk_rows: int, shard_bytes: int, gap: int):
    n, rest = x.shape[0], tuple(x.shape[1:])
    row_bytes = max(1, int(np.prod(rest, dtype=np.int64)) * np.dtype(x.dtype).itemsize)
    chunk = min(chunk_rows, max(n, 1))
    shard = min(max(chunk, shard_bytes // row_bytes // chunk * chunk), max(n, 1))
    out = group.create_array(name, shape=x.shape, dtype=x.dtype, chunks=(chunk, *rest), shards=(shard, *rest),
                             compressors=(BloscCodec(cname="lz4", clevel=3, shuffle="shuffle"),))  # fmt: skip
    out.attrs.update({"encoding-type": "array", "encoding-version": "0.2.0"})
    for a in range(0, n, shard):
        out[a : a + shard] = _read(x, order[a : a + shard], gap)


def write_grouped(
    src: str | os.PathLike,
    out: str | os.PathLike,
    groupby: Sequence[str],
    reps: Sequence[Rep] = (A.X,),
    *,
    n_obs_per_chunk: int = 64,
    shard_bytes: int = 2**30,
    max_gap: int = 4096,
) -> np.ndarray:
    """Copy the zarr store ``src`` to ``out`` with its cells stably sorted by the ``obs`` columns ``groupby``.

    ``obs`` and ``var`` are read into memory; the ``reps`` are opened with :func:`scfit.data.open_adata` and streamed in
    shard-sized blocks of rows, so memory holds about one shard. Within a group cells keep their order, and ``obs_names``
    are kept, so results can be mapped back to the source. ``out`` is zarr v3, sharded and chunked like annbatch's
    ``write_sharded`` (``n_obs_per_chunk`` rows per chunk). Returns the order: row ``i`` of ``out`` is row
    ``order[i]`` of ``src``.
    """
    from scfit.data import open_adata

    groupby = list(groupby)
    g = zarr.open_group(src, mode="r")
    obs, var = ad.io.read_elem(g["obs"]), ad.io.read_elem(g["var"])
    order = obs[groupby].reset_index(drop=True).sort_values(groupby, kind="stable").index.to_numpy()
    backed = open_adata(src, *reps)

    with ad.settings.override(zarr_write_format=3):
        ad.AnnData(obs=obs.iloc[order], var=var).write_zarr(out)  # everything but the reps; they are streamed in
    root = zarr.open_group(out, mode="r+")
    for ref in reps:
        parent, name = _path(ref)
        group = root if parent is None else root[parent]
        x = ref.get(backed)
        copy = _copy_sparse if hasattr(x, "group") and "indptr" in x.group else _copy_dense
        copy(x, group, name, order, n_obs_per_chunk, shard_bytes, max_gap)
    return order
