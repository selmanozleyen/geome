from __future__ import annotations

import numpy as np
from anndata import AnnData
from torch.utils.data import Dataset
from torch_geometric.data import Data

from geome.iterables import ToCategoryIterator

from .base.abstract import Ann2DataAbstract

__all__ = ["LazyCategoryData"]


class LazyCategoryData(Dataset):
    """The ``Data`` of ``a2d(adata)`` for an ``a2d`` iterating by category, one category read per index.

    ``a2d(adata)`` copies every category into memory and keeps every ``Data``. This reads only category ``i``'s rows
    of a backed ``adata`` (e.g. ``anndata.read_h5ad(path, backed="r")``) when indexed, applies ``a2d``'s
    ``transform`` and builds the same ``Data``, then lets it go: memory holds one category at a time.

    ``a2d``'s ``preprocess`` runs once, on the backed object, so it must only touch ``obs``.

    Parameters
    ----------
    adata
        The data, backed or in memory.
    a2d
        A converter whose ``adata2iter`` is a :class:`~geome.iterables.ToCategoryIterator`.
    rows
        Restrict to these rows (e.g. one split), as a boolean mask over ``adata``'s cells; categories with no rows
        left stay, empty, as with ``a2d(adata[rows])``.
    """

    def __init__(self, adata: AnnData, a2d: Ann2DataAbstract, rows: np.ndarray | None = None) -> None:
        it = a2d._adata2iterable
        if not isinstance(it, ToCategoryIterator) or it.axis != "obs":
            raise ValueError("LazyCategoryData needs an Ann2Data iterating with ToCategoryIterator(axis='obs').")
        if a2d._preprocess is not None:
            adata = a2d._preprocess(adata)
        col = adata.obs[it.category]
        keep = np.ones(adata.n_obs, dtype=bool) if rows is None else np.asarray(rows, dtype=bool)
        self.rows = [np.flatnonzero((col == c).to_numpy() & keep) for c in col.cat.categories]
        self._preserved = {
            k: adata.obs[k].cat.categories
            for k in (it.preserve_categories or [])
            if k in adata.obs and adata.obs[k].dtype.name == "category"
        }
        self._adata, self._a2d = adata, a2d

    def __len__(self) -> int:
        return len(self.rows)

    @property
    def n_vars(self) -> int:
        """Genes in the data, known without reading any window."""
        return int(self._adata.n_vars)

    def num_nodes(self, i: int) -> int:
        """Cells in category ``i``, known without reading it."""
        return int(self.rows[i].size)

    def __getitem__(self, i: int) -> Data:
        sub = self._adata[self.rows[i]]
        sub = sub.to_memory() if self._adata.isbacked else sub.copy()
        for k, cats in self._preserved.items():  # as ToCategoryIterator(preserve_categories=...)
            if k in sub.obs:
                sub.obs[k] = sub.obs[k].cat.set_categories(cats)
        if self._a2d._transform is not None:
            sub = self._a2d._transform(sub)
        return self._a2d.create_data_obj(sub)
