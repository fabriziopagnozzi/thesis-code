import json
from typing import cast

import numpy as np
import polars as pl
from numpy.typing import NDArray
from tqdm import tqdm

from experiments.mimic.evaluation.schemas_evaluation import RetrievalResult
from experiments.mimic.global_configs import (
    get_table_path,
    global_cfg,
    read_parquet,
    setup_logging,
)
from experiments.mimic.queries.schemas_queries import (
    GoldAnnotationRow,
    QueryModifier,
    QueryRowPostFiltering,
)
from experiments.mimic.utils.candidate_pools import (
    ChunkPool,
    ChunkPoolBuilder,
)
from experiments.mimic.utils.utils import load_filtered_queries
from helpers.metrics import avg_cos, fac_cov_score, jaccard
from helpers.query_algorithms import ScoringFunction, select

from .metrics import (
    aspect_recall,
    gold_precision,
    gold_recall,
    weighted_aspect_recall,
)
from .plots import store_eval_figures
from .schemas_evaluation import (
    EvaluateCfg,
    EvaluationMetrics,
)

evaluate_cfg = EvaluateCfg.load()


def run_evaluate(cfg: EvaluateCfg | None = None) -> pl.DataFrame:
    global evaluate_cfg
    if cfg is not None:
        evaluate_cfg = cfg

    builder = ChunkPoolBuilder(model_name=global_cfg.embedding_model)

    if evaluate_cfg.gold_mode == 'structural':
        results = evaluate_structural(builder)
    else:
        annotations_df = read_parquet('gold_annotations')
        annotations_df = annotations_df.filter(pl.col('n_facets') > 0)
        print(f'Loaded {len(annotations_df):,} annotated queries with facets')
        results = evaluate_llm(annotations_df, builder)

    out_path = get_table_path('evaluation_results')
    results.write_parquet(out_path)
    print(f'\nSaved {len(results):,} result rows to {out_path}')

    store_eval_stats(results)
    store_best_per_metric(results)
    store_eval_figures(evaluate_cfg)
    return results


def evaluate_llm(
    annotations_df: pl.DataFrame,
    builder: ChunkPoolBuilder,
) -> pl.DataFrame:
    all_rows = []

    for row in tqdm(
        annotations_df.iter_rows(named=True),
        total=len(annotations_df),
        desc='Evaluating',
        dynamic_ncols=True,
    ):
        row = cast(GoldAnnotationRow, row)
        icd10_3char = row['icd10_3char']
        query_text = row['query_text']
        facets = json.loads(row['facets_json'])
        if not facets:
            continue

        query_vec = builder.embed_query(query_text)

        cosine_pool = builder.topk_cosine(query_vec, n=global_cfg.prefilter_n)
        all_gold_ids = {cid for cids in facets.values() for cid in cids}
        gold_pool = builder.topk_cosine_for_chunk_ids(ids=all_gold_ids)
        pool = ChunkPool.merge([cosine_pool, gold_pool])

        query_metrics = evaluate_query(
            pool,
            query_vec,
            facets,
            strategies=evaluate_cfg.strategies,
            k_values=evaluate_cfg.k_values,
            lam_values=evaluate_cfg.lam_values,
        )

        for m in query_metrics:
            all_rows.append({
                'query_id': row['query_id'],
                'icd10_3char': icd10_3char,
                'n_facets': row['n_facets'],
                **m,
            })

    return pl.DataFrame(all_rows)


def evaluate_structural(builder: ChunkPoolBuilder) -> pl.DataFrame:
    queries_df = load_filtered_queries(global_cfg.embedding_model)
    print(f'Loaded {len(queries_df):,} queries for structural evaluation')

    all_rows = []
    for curr_query in tqdm(
        queries_df.iter_rows(named=True),
        total=len(queries_df),
        desc='Evaluating (structural)',
        dynamic_ncols=True,
    ):
        curr_query = cast(QueryRowPostFiltering, curr_query)
        modifiers: list[QueryModifier] = QueryModifier.parse_list(
            curr_query.get('modifiers_json', '') or ''
        )
        if not modifiers:
            continue

        query_vec = builder.embed_query(curr_query['query_text'])
        pool = builder.topk_cosine_for_condition(
            query_vec, curr_query['icd10_3char'], n=global_cfg.prefilter_n
        )

        facets = build_structural_facets(pool, curr_query['icd10_3char'], modifiers, builder)
        if not facets:
            print(
                f'  [skip] query_id={curr_query["query_id"]} ({curr_query["icd10_3char"]}): '
                f'no modifier chunks found in pool'
            )
            continue

        query_metrics = evaluate_query(
            pool,
            query_vec,
            facets,
            strategies=evaluate_cfg.strategies,
            k_values=evaluate_cfg.k_values,
            lam_values=evaluate_cfg.lam_values,
        )

        for m in query_metrics:
            all_rows.append({
                'query_id': curr_query['query_id'],
                'icd10_3char': curr_query['icd10_3char'],
                'stratum': curr_query.get('stratum'),
                'n_facets': len(facets),
                **m,
            })

    return pl.DataFrame(all_rows)


def build_structural_facets(
    pool: ChunkPool,
    icd10_3char: str,
    modifiers: list[QueryModifier],
    builder: ChunkPoolBuilder,
) -> dict[str, list[str]]:
    """Build gold facets structurally: pool chunks whose hadm_id ∈ condition_hadm_ids ∩ modifier_hadm_ids."""
    facets: dict[str, list[str]] = {}

    for m in modifiers:
        qualifying = builder.get_hadm_ids_by_condition_modifier(icd10_3char, m)
        chunk_ids = [
            cid
            for hid, cid in zip(pool.hadm_ids.tolist(), pool.chunk_ids, strict=False)
            if hid in qualifying
        ]
        if chunk_ids:
            facets[m.comorb_label] = chunk_ids

    return facets


def evaluate_query(
    pool: ChunkPool,
    query_vec: np.ndarray,
    facets: dict[str, list[str]],
    strategies: list[ScoringFunction],
    k_values: list[int],
    lam_values: list[float],
) -> list[EvaluationMetrics]:
    """Evaluate all strategy x k x λ combos for a single query.
    Returns list of metric dicts.
    """
    retrieval_results = run_retrieval(
        pool,
        query_vec,
        strategies=strategies,
        k_values=k_values,
        lam_values=lam_values,
        prefilter_n=None,
    )

    # Find top_k results for Jaccard comparison
    topk_by_k: dict[int, RetrievalResult] = {}
    for result in retrieval_results:
        if result.strategy == 'top_k':
            topk_by_k[result.k] = result

    sim_to_query = pool.sim_scores(query_vec)
    sim_matrix = pool.sim_matrix()

    chunk_id_to_idx: dict[str, int] = {cid: i for i, cid in enumerate(pool.chunk_ids)}
    pool_id_set = set(pool.chunk_ids)
    all_gold_ids = {cid for cids in facets.values() for cid in cids}

    metrics = []
    for result in retrieval_results:
        selected_set = set(result.selected_chunk_ids)
        ar = aspect_recall(selected_set, facets)
        war = weighted_aspect_recall(selected_set, facets)
        gp = gold_precision(selected_set, all_gold_ids)
        gr = gold_recall(selected_set, all_gold_ids, pool_id_set)

        eval_indices = np.array(
            [chunk_id_to_idx[cid] for cid in result.selected_chunk_ids if cid in chunk_id_to_idx],
            dtype=np.intp,
        )

        fac = fac_cov_score(eval_indices, sim_matrix) if len(eval_indices) > 0 else 0.0
        ac = avg_cos(eval_indices, sim_to_query) if len(eval_indices) > 0 else 0.0

        topk_ref = topk_by_k.get(result.k)
        if topk_ref is not None and result.strategy != 'top_k':
            topk_eval_idx = np.array(
                [
                    chunk_id_to_idx[cid]
                    for cid in topk_ref.selected_chunk_ids
                    if cid in chunk_id_to_idx
                ],
                dtype=np.intp,
            )
            jac = jaccard(eval_indices, topk_eval_idx)
        else:
            jac = 1.0

        metrics.append({
            'strategy': result.strategy,
            'k': result.k,
            'lam': result.lam,
            'aspect_recall': ar,
            'weighted_aspect_recall': war,
            'gold_precision': gp,
            'gold_recall': gr,
            'fac_cov_score': fac,
            'avg_cos': ac,
            'jaccard_vs_topk': jac,
            'n_unique_hadms': len(set(result.selected_hadm_ids)),
        })

    return metrics


def run_retrieval(
    pool: ChunkPool,
    query_vec: NDArray[np.float32],
    strategies: list[ScoringFunction],
    k_values: list[int],
    lam_values: list[float],
    prefilter_n: int | None = 500,
) -> list[RetrievalResult]:
    sim_to_query = pool.sim_scores(query_vec)

    if prefilter_n is not None and pool.n > prefilter_n:
        top_indices = np.argsort(sim_to_query)[::-1][:prefilter_n].copy()
        pool = pool.slice(top_indices)
        sim_to_query = sim_to_query[top_indices]

    sim_matrix = pool.sim_matrix()

    valid_k_values = [k for k in k_values if k <= pool.n]
    if not valid_k_values:
        return []
    max_k = max(valid_k_values)

    results = []
    for strategy in strategies:
        needs_lambda = strategy in ('mmr', 'gmmr', 'fac_loc')
        lams = lam_values if needs_lambda else [None]

        for lam in lams:
            all_selected = select(
                strategy=strategy,
                sim_to_query=sim_to_query,
                k=max_k,
                sim_matrix=sim_matrix,
                embeddings=pool.vectors,
                query_embedding=query_vec,
                lam=lam if lam is not None else 0.5,
            )

            for k in valid_k_values:
                selected = all_selected[:k]
                results.append(
                    RetrievalResult(
                        strategy=strategy,
                        k=k,
                        lam=lam,
                        selected_indices=selected,
                        selected_chunk_ids=[pool.chunk_ids[i] for i in selected],
                        selected_hadm_ids=[int(pool.hadm_ids[i]) for i in selected],
                        sim_to_query=sim_to_query[selected],
                    )
                )

    return results


def store_eval_stats(results_df: pl.DataFrame) -> None:
    print('\n=== Evaluation Summary ===\n')

    summaries = []
    for k in sorted(results_df['k'].unique().to_list()):
        print(f'--- k = {k} ---')
        subset = results_df.filter(pl.col('k') == k)

        summary = (
            subset
            .group_by('strategy', 'lam')
            .agg(
                pl.col('aspect_recall').mean().alias('AR'),
                pl.col('weighted_aspect_recall').mean().alias('WAR'),
                pl.col('gold_precision').mean().alias('GP'),
                pl.col('gold_recall').mean().alias('GR'),
                pl.col('fac_cov_score').mean().alias('fac'),
                pl.col('avg_cos').mean().alias('cos'),
                pl.col('jaccard_vs_topk').mean().alias('jac'),
                pl.col('aspect_recall').count().alias('n'),
            )
            .sort('strategy', 'lam')
        )
        summaries.append(summary.with_columns(pl.lit(k).alias('k')))
        print(summary)
        print()

    stats_df = pl.concat(summaries)
    stats_path = get_table_path('evaluation_stats')
    stats_df.write_parquet(stats_path)
    print(f'Saved summary to {stats_path}')

    if 'stratum' in results_df.columns:
        stratum_summaries = []
        for k in sorted(results_df['k'].unique().to_list()):
            subset = results_df.filter(pl.col('k') == k)
            stratum_summaries.append(
                subset
                .group_by('strategy', 'lam', 'stratum')
                .agg(
                    pl.col('aspect_recall').mean().alias('AR'),
                    pl.col('weighted_aspect_recall').mean().alias('WAR'),
                    pl.col('gold_precision').mean().alias('GP'),
                    pl.col('gold_recall').mean().alias('GR'),
                    pl.col('fac_cov_score').mean().alias('fac'),
                    pl.col('avg_cos').mean().alias('cos'),
                    pl.col('aspect_recall').count().alias('n'),
                )
                .sort('stratum', 'strategy', 'lam')
                .with_columns(pl.lit(k).alias('k'))
            )
        stratum_stats_df = pl.concat(stratum_summaries)
        stratum_stats_path = get_table_path('evaluation_stats_by_stratum')
        stratum_stats_df.write_parquet(stratum_stats_path)
        print(f'Saved stratum summary to {stratum_stats_path}')

        strata = sorted(results_df['stratum'].drop_nulls().unique().to_list())
        for stratum in strata:
            print(f'\n=== Stratum {stratum} ===\n')
            stratum_rows = results_df.filter(pl.col('stratum') == stratum)
            for k in sorted(stratum_rows['k'].unique().to_list()):
                print(f'--- k = {k} ---')
                print(
                    stratum_rows
                    .filter(pl.col('k') == k)
                    .group_by('strategy', 'lam')
                    .agg(
                        pl.col('aspect_recall').mean().alias('AR'),
                        pl.col('weighted_aspect_recall').mean().alias('WAR'),
                        pl.col('gold_precision').mean().alias('GP'),
                        pl.col('gold_recall').mean().alias('GR'),
                        pl.col('fac_cov_score').mean().alias('fac'),
                        pl.col('avg_cos').mean().alias('cos'),
                        pl.col('aspect_recall').count().alias('n'),
                    )
                    .sort('strategy', 'lam')
                )
                print()


def store_best_per_metric(results_df: pl.DataFrame) -> None:
    """For each (k, lam) pair and each metric, find the best strategy among top_k, mmr, fl."""
    metric_cols = ['AR', 'WAR', 'GP', 'GR']

    summary = results_df.group_by('k', 'lam', 'strategy').agg(
        pl.col('aspect_recall').mean().alias('AR'),
        pl.col('weighted_aspect_recall').mean().alias('WAR'),
        pl.col('gold_precision').mean().alias('GP'),
        pl.col('gold_recall').mean().alias('GR'),
    )

    # top_k has lam=null - it must compete against mmr/fl at every lambda value
    top_k_s = summary.filter(pl.col('strategy') == 'top_k').drop('lam')
    others_s = summary.filter(pl.col('strategy') != 'top_k')

    # best_per_metric: at each (k, lam), top_k@k vs mmr@(k,lam) vs fl@(k,lam)
    top_k_expanded = others_s.select('k', 'lam').unique().join(top_k_s, on='k', how='left')
    per_k_lam = pl.concat([others_s, top_k_expanded], how='diagonal_relaxed')

    best_rows = []
    for (k, lam), group in per_k_lam.group_by('k', 'lam'):
        for col in metric_cols:
            best_val = group[col].max()
            tied = group.filter(pl.col(col) == best_val)
            best_rows.append({
                'k': k,
                'lam': lam,
                'best_for': col,
                'strategy': tied['strategy'].to_list(),
                **{c: best_val if c == col else tied[c][0] for c in metric_cols},
            })

    best_df = pl.DataFrame(best_rows).sort('k', 'lam', 'best_for')
    best_path = get_table_path('evaluation_best_per_metric')
    best_df.write_parquet(best_path)
    print(f'Saved best-per-metric summary to {best_path}')

    # best_fixed_lam: for each lam, best (strategy, k) - top_k competes at all k values
    best_fixed_lam_rows = []
    for (lam,), group in others_s.group_by('lam'):
        combined = pl.concat([group, top_k_s], how='diagonal_relaxed')
        for col in metric_cols:
            best_val = combined[col].max()
            tied = combined.filter(pl.col(col) == best_val)
            best_fixed_lam_rows.append({
                'lam': lam,
                'best_for': col,
                'strategy': tied['strategy'].to_list(),
                'k': tied['k'].to_list(),
                **{c: best_val if c == col else tied[c][0] for c in metric_cols},
            })

    best_fixed_lam_df = pl.DataFrame(best_fixed_lam_rows).sort('lam', 'best_for')
    best_fixed_lam_path = get_table_path('evaluation_best_per_metric_fixed_lam')
    best_fixed_lam_df.write_parquet(best_fixed_lam_path)
    print(f'Saved best-per-metric (fixed lam) summary to {best_fixed_lam_path}')


if __name__ == '__main__':
    setup_logging()
    run_evaluate(cfg=EvaluateCfg.load())
