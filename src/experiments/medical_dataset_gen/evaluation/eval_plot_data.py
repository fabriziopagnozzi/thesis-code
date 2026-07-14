"""Cached result slices used by evaluation figures.

The evaluation result table contains one row per query and retrieval setting.
Several figures reuse the same setting-specific rows to draw uncertainty
intervals, so retaining these slices avoids repeatedly scanning the complete
table without changing the plotted estimators.
"""

import polars as pl

from experiments.medical_dataset_gen.evaluation.retrieval_utils import ci_half_width

type ResultSetting = tuple[str, int, float | None]
type PairedMetricSetting = tuple[str, int, float | None, str]


class EvaluationResultLookup:
    """Memoize query-level result slices and paired confidence intervals."""

    def __init__(self, results_df: pl.DataFrame) -> None:
        self.results_df = results_df
        self._strategy_k_slices: dict[tuple[str, int], pl.DataFrame] = {}
        self._lambda_slices: dict[tuple[str, int], dict[float, pl.DataFrame]] = {}
        self._result_slices: dict[ResultSetting, pl.DataFrame] = {}
        self._paired_ci: dict[PairedMetricSetting, float] = {}

    def query_values(
        self,
        strategy: str,
        k: int,
        lam: float | None,
        metric: str,
    ) -> list[float]:
        if metric not in self.results_df.columns:
            return []
        return [float(value) for value in self.result_slice(strategy, k, lam)[metric].drop_nulls()]

    def paired_delta_ci(
        self,
        strategy: str,
        k: int,
        lam: float | None,
        metric: str,
    ) -> float:
        cache_key = (strategy, k, lam, metric)
        cached = self._paired_ci.get(cache_key)
        if cached is not None:
            return cached
        if metric not in self.results_df.columns or 'query_id' not in self.results_df.columns:
            return float('nan')

        topk = self.result_slice('top_k', k, None).select(
            'query_id', pl.col(metric).alias('topk_val')
        )
        compared = self.result_slice(strategy, k, lam).select(
            'query_id', pl.col(metric).alias('strategy_val')
        )
        paired = topk.join(compared, on='query_id', how='inner')
        if paired.height < 2:
            value = float('nan')
        else:
            value = ci_half_width(
                [float(delta) for delta in (paired['strategy_val'] - paired['topk_val'])]
            )
        self._paired_ci[cache_key] = value
        return value

    def result_slice(self, strategy: str, k: int, lam: float | None) -> pl.DataFrame:
        cache_key = (strategy, k, lam)
        cached = self._result_slices.get(cache_key)
        if cached is not None:
            return cached

        result = (
            self._lambda_slices_for(strategy, k).get(lam, pl.DataFrame())
            if lam is not None
            else self._strategy_k_slice(strategy, k)
        )
        self._result_slices[cache_key] = result
        return result

    def _lambda_slices_for(self, strategy: str, k: int) -> dict[float, pl.DataFrame]:
        cache_key = (strategy, k)
        cached = self._lambda_slices.get(cache_key)
        if cached is not None:
            return cached
        partitions = self._strategy_k_slice(strategy, k).partition_by('lam', as_dict=True)
        result = {float(lam_key[0]): frame for lam_key, frame in partitions.items()}
        self._lambda_slices[cache_key] = result
        return result

    def _strategy_k_slice(self, strategy: str, k: int) -> pl.DataFrame:
        cache_key = (strategy, k)
        cached = self._strategy_k_slices.get(cache_key)
        if cached is not None:
            return cached
        result = self.results_df.filter(
            (pl.col('strategy') == strategy) & (pl.col('k') == k)
        )
        self._strategy_k_slices[cache_key] = result
        return result
