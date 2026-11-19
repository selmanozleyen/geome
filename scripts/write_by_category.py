"""Reorder a zarr store by obs categories, out of core, and write it sharded with annbatch: how long, how much memory.

    python scripts/write_by_category.py SRC.zarr OUT.zarr --groupby domain --mode lazy
    python scripts/write_by_category.py SRC.zarr OUT.zarr --groupby domain --mode annbatch

Modes:

- ``lazy``: :func:`anndata.experimental.read_lazy` (dask), one global stable order by ``--groupby``, then
  :func:`annbatch.io.write_sharded` on the lazy, reordered view. Nothing beyond dask's working set should be in memory.
- ``annbatch``: :meth:`annbatch.DatasetCollection.add_adatas` with ``groupby=`` and ``shuffle=False``, annbatch's own
  path: lazy load, then each ``--dataset-size`` chunk is loaded into memory, sorted and written.

Then the result is opened with :func:`scfit.data.open_adata` and checked: every category one contiguous run of rows,
and the rows of one category equal to the source's.
"""

from __future__ import annotations

import argparse
import json
import resource
import shutil
import sys
import time

import anndata as ad
import numpy as np
import pandas as pd
import zarr


def peak_rss_gb() -> float:
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return rss / 1e9 if sys.platform == "darwin" else rss / 1e6


def stable_order(obs: pd.DataFrame, groupby: list[str]) -> np.ndarray:
    """Rows sorted by ``groupby``, original order kept within each group."""
    return obs[groupby].reset_index(drop=True).sort_values(groupby, kind="stable").index.to_numpy()


def write_lazy(src: str, out: str, groupby: list[str], args) -> ad.AnnData:
    from annbatch.io import write_sharded

    adata = ad.experimental.read_lazy(src)
    adata.obs = adata.obs.to_memory()  # obs is small; X / layers / obsm stay dask
    adata = adata[stable_order(adata.obs, groupby)]
    write_sharded(
        zarr.open_group(out, mode="w"), adata, n_obs_per_chunk=args.n_obs_per_chunk, shard_size=args.shard_size
    )
    return adata


def write_annbatch(src: str, out: str, groupby: list[str], args) -> None:
    from annbatch import DatasetCollection

    DatasetCollection(out).add_adatas(
        [src],
        groupby=groupby,
        shuffle=False,
        n_obs_per_chunk=args.n_obs_per_chunk,
        shard_size=args.shard_size,
        dataset_size=args.dataset_size,
    )


def check(src: str, out: str, groupby: list[str], mode: str) -> dict:
    """Categories contiguous in the output, and one category's rows equal to the source's (read with scfit)."""
    from anndata.acc import A
    from scfit.data import open_adata

    if mode == "annbatch":  # a collection: its datasets are the stores
        g = zarr.open_group(out, mode="r")
        out = f"{out}/{sorted(k for k in g.group_keys())[0]}"
    cols = [A.obs[c] for c in groupby]
    got = open_adata(out, A.X, *cols)
    want = open_adata(src, A.X, *cols)
    codes = pd.MultiIndex.from_frame(got.obs[groupby].astype(str)).factorize()[0]
    contiguous = bool((np.diff(codes) != 0).sum() == len(np.unique(codes)) - 1)
    key = tuple(got.obs[groupby].astype(str).iloc[0])
    sel_got = np.flatnonzero((got.obs[groupby].astype(str) == key).all(axis=1).to_numpy())
    sel_want = np.flatnonzero((want.obs[groupby].astype(str) == key).all(axis=1).to_numpy())

    def rows(x, idx):
        sub = x[int(idx[0]) : int(idx[-1]) + 1]
        sub = sub.toarray() if hasattr(sub, "toarray") else np.asarray(sub)
        return sub[idx - idx[0]]

    same = bool(np.array_equal(rows(got.X, sel_got), rows(want.X, sel_want)))
    return {"contiguous": contiguous, "first_category_rows_equal": same, "checked_category": list(key)}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("src")
    p.add_argument("out")
    p.add_argument("--groupby", nargs="+", required=True)
    p.add_argument("--mode", choices=["lazy", "annbatch"], default="lazy")
    p.add_argument("--n-obs-per-chunk", type=int, default=64)
    p.add_argument("--shard-size", default="1GB")
    p.add_argument("--dataset-size", default="20GB", help="annbatch mode: rows loaded into memory at once")
    p.add_argument("--no-check", action="store_true")
    args = p.parse_args()

    shutil.rmtree(args.out, ignore_errors=True)
    t0 = time.perf_counter()
    if args.mode == "lazy":
        write_lazy(args.src, args.out, args.groupby, args)
    else:
        write_annbatch(args.src, args.out, args.groupby, args)
    result = {"mode": args.mode, "groupby": args.groupby, "write_seconds": time.perf_counter() - t0}
    result["peak_rss_gb_write"] = peak_rss_gb()
    if not args.no_check:
        result.update(check(args.src, args.out, args.groupby, args.mode))
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
