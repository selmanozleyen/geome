"""Reorder a zarr store by obs categories, out of core, and write it sharded with annbatch: how long, how much memory.

    python scripts/write_by_category.py SRC.zarr OUT.zarr --groupby domain

The rewrite is :func:`geome.io.write_permuted` with the grouped order: ``obs`` read eagerly, arrays opened with
:func:`scfit.data.open_adata` and streamed into place in shard-sized batches. (A dask baseline, ``read_lazy`` plus
annbatch's ``write_sharded``, does not run: ``write_sharded`` reads ``nnz`` off the sparse matrix, which a lazy dask
view does not have.)

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


def write_custom(src: str, out: str, groupby: list[str], args) -> None:
    from anndata.acc import A

    from geome.io import write_permuted

    obs = ad.io.read_elem(zarr.open_group(src, mode="r")["obs"])
    order = stable_order(obs, groupby) if args.order == "grouped" else np.random.default_rng(0).permutation(len(obs))
    reps = (A.X, A.obsm["spatial"])
    write_permuted(src, out, order, reps=reps, n_obs_per_chunk=args.n_obs_per_chunk)


def check(src: str, out: str, groupby: list[str]) -> dict:
    """Categories contiguous in the output, and one category's rows equal to the source's (read with scfit)."""
    from anndata.acc import A
    from scfit.data import open_adata

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
    try:  # annbatch's recommendation for local disks: zarrs' Rust codec pipeline
        import zarrs  # noqa: F401

        zarr.config.set({"codec_pipeline.path": "zarrs.ZarrsCodecPipeline"})
    except ImportError:
        pass
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("src")
    p.add_argument("out")
    p.add_argument("--groupby", nargs="+", required=True)
    p.add_argument("--order", choices=["grouped", "shuffled"], default="grouped", help="shuffled: the worst case")
    p.add_argument("--n-obs-per-chunk", type=int, default=64)
    p.add_argument("--shard-size", default="1GB")
    p.add_argument("--no-check", action="store_true")
    args = p.parse_args()

    shutil.rmtree(args.out, ignore_errors=True)
    t0 = time.perf_counter()
    write_custom(args.src, args.out, args.groupby, args)
    result = {"order": args.order, "groupby": args.groupby, "write_seconds": time.perf_counter() - t0}
    result["peak_rss_gb_write"] = peak_rss_gb()
    print(json.dumps(result), flush=True)  # the timing stands even if the check below fails
    if not args.no_check and args.order == "grouped":
        print(json.dumps(check(args.src, args.out, args.groupby)), flush=True)


if __name__ == "__main__":
    main()
