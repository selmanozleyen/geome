import anndata as ad
import numpy as np
import pytest
import torch
from scipy import sparse

from geome import ann2data, iterables, transforms


def _a2d():
    return ann2data.Ann2DataBasic(
        fields={"x": ["X"], "edge_index": ["uns/edge_index"], "obs_names": ["obs_names"]},
        adata2iter=iterables.ToCategoryIterator("sample", axis="obs"),
        transform=transforms.AddEdgeIndex(
            edge_index_key="edge_index",
            func_args={"radius": 20.0, "coord_type": "generic"},
            spatial_key="spatial",
            key_added="adj",
        ),
    )


def _adata(n=300, seed=0):
    rng = np.random.default_rng(seed)
    sample = rng.choice(["s1", "s2", "s3"], size=n)
    adata = ad.AnnData(
        X=sparse.csr_matrix(rng.poisson(1.0, size=(n, 6)).astype(np.float32)),
        obs={"sample": sample, "split": np.where(sample == "s3", "val", "train")},
        obsm={"spatial": rng.uniform(0, 100, size=(n, 2))},
    )
    adata.obs["sample"] = adata.obs["sample"].astype("category")
    adata.obs_names = [str(i) for i in range(1, n + 1)]  # as InterScale names its cells
    return adata


def _assert_same(eager, lazy):
    assert len(eager) == len(lazy)
    for e, i in zip(eager, range(len(lazy)), strict=True):
        got = lazy[i]
        assert got.num_nodes == e.num_nodes == lazy.num_nodes(i)
        for key in ("x", "edge_index", "obs_names"):
            torch.testing.assert_close(getattr(got, key), getattr(e, key))


@pytest.mark.parametrize("backed", [False, True])
def test_lazy_windows_equal_eager_windows(tmp_path, backed):
    adata = _adata()
    eager, _ = _a2d()(adata.copy())
    if backed:
        adata.write_h5ad(tmp_path / "a.h5ad")
        adata = ad.read_h5ad(tmp_path / "a.h5ad", backed="r")
    _assert_same(eager, ann2data.LazyCategoryData(adata, _a2d()))


def test_rows_restrict_like_subsetting():
    adata = _adata()
    train = (adata.obs["split"] == "train").to_numpy()
    eager, _ = _a2d()(adata[train].copy())  # s3 is all val: subsetting drops it, so neither has its window
    _assert_same(eager, ann2data.LazyCategoryData(adata, _a2d(), rows=train))


def test_write_grouped_matches_an_in_memory_reorder(tmp_path):
    pytest.importorskip("scfit.data")
    from anndata.acc import A

    from geome.io import write_grouped

    adata = _adata(n=500)
    adata.layers["dense"] = adata.X.toarray()
    adata.write_zarr(tmp_path / "src.zarr")
    order = write_grouped(
        tmp_path / "src.zarr",
        tmp_path / "out.zarr",
        groupby=["sample"],
        reps=(A.X, A.layers["dense"], A.obsm["spatial"]),
        n_obs_per_chunk=16,
        shard_bytes=2_000,  # many small shards and blocks
    )
    want = adata[np.argsort(adata.obs["sample"].cat.codes.to_numpy(), kind="stable")]
    got = ad.read_zarr(tmp_path / "out.zarr")
    np.testing.assert_array_equal(order, np.argsort(adata.obs["sample"].cat.codes.to_numpy(), kind="stable"))
    assert list(got.obs_names) == list(want.obs_names)
    np.testing.assert_array_equal(got.X.toarray(), want.X.toarray())
    np.testing.assert_array_equal(got.layers["dense"], want.layers["dense"])
    np.testing.assert_array_equal(got.obsm["spatial"], want.obsm["spatial"])
