from __future__ import annotations

import numpy as np

from imgaug2.augmenters._typing import Array
from imgaug2.compat.markers import legacy


@legacy(version="0.4.0")
class _QuantizeUniformCenterizedLUTTableSingleton:
    _INSTANCE = None

    @legacy(version="0.4.0")
    @classmethod
    def get_instance(cls) -> _QuantizeUniformLUTTable:
        """Get singleton instance of `_QuantizeUniformLUTTable`.


        Returns
        -------
        _QuantizeUniformLUTTable
            The global instance of `_QuantizeUniformLUTTable`.

        """
        if cls._INSTANCE is None:
            cls._INSTANCE = _QuantizeUniformLUTTable(centerize=True)
        return cls._INSTANCE


@legacy(version="0.4.0")
class _QuantizeUniformNotCenterizedLUTTableSingleton:
    """Table for `quantize_uniform()` with ``to_bin_centers=False``."""

    _INSTANCE = None

    @legacy(version="0.4.0")
    @classmethod
    def get_instance(cls) -> _QuantizeUniformLUTTable:
        """Get singleton instance of `_QuantizeUniformLUTTable`.


        Returns
        -------
        _QuantizeUniformLUTTable
            The global instance of `_QuantizeUniformLUTTable`.

        """
        if cls._INSTANCE is None:
            cls._INSTANCE = _QuantizeUniformLUTTable(centerize=False)
        return cls._INSTANCE


@legacy(version="0.4.0")
class _QuantizeUniformLUTTable:
    def __init__(self, centerize: bool) -> None:
        self.table = self._generate_quantize_uniform_table(centerize)

    @legacy(version="0.4.0")
    def get_for_nb_bins(self, nb_bins: int) -> Array:
        """Get LUT ndarray for a provided number of bins."""
        return self.table[nb_bins, :]

    @legacy(version="0.4.0")
    @classmethod
    def _generate_quantize_uniform_table(cls, centerize: bool) -> Array:
        # For simplicity, we generate here the tables for nb_bins=0 (results
        # in all zeros) and nb_bins=256 too, even though these should usually
        # not be requested.
        table = np.arange(0, 256).astype(np.float32)
        table_all_nb_bins = np.zeros((256, 256), dtype=np.float32)

        # This loop could be done a little bit faster by vectorizing it.
        # It is expected to be run exactly once per run of a whole script,
        # making the difference negligible.
        for nb_bins in np.arange(1, 255).astype(np.uint8):
            binsize = 256 / nb_bins
            table_q_f32 = np.floor(table / binsize) * binsize
            if centerize:
                table_q_f32 = table_q_f32 + binsize / 2
            table_all_nb_bins[nb_bins] = table_q_f32
        table_all_nb_bins = np.clip(np.round(table_all_nb_bins), 0, 255).astype(np.uint8)
        return table_all_nb_bins
