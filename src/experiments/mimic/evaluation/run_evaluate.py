import json
from typing import cast

import numpy as np
import polars as pl
from numpy.typing import NDArray
from tqdm import tqdm

from experiments.mimic.global_configs import (
    get_table_path,
    global_cfg,
    read_parquet,
    setup_logging,
)
from experiments.mimic.queries.schemas_queries import (
    GoldAnnotationRow,
    QueryModifier,
    QueryModifierLabelId,
    QueryRowPostFiltering,
)
from experiments.mimic.utils.chunk_pools import (
    ChunkPool,
    ChunkPoolBuilder,
    ChunkPoolRetrievalResult,
)
from experiments.mimic.utils.utils import load_filtered_queries
from helpers.metrics import avg_cos, fac_cov_score, jaccard
from helpers.query_algorithms import ScoringFunction

from .metrics import (
    compute_answer_support_metrics,
    compute_chunk_support_metrics,
)
from .plots import store_eval_figures
from .schemas_evaluation import (
    EvaluateCfg,
    QueryEvalResult,
)

evaluate_cfg = EvaluateCfg.load()


def run_evaluate(cfg: EvaluateCfg | None = None) -> pl.DataFrame:
    global evaluate_cfg
    if cfg is not None:
        evaluate_cfg = cfg

    pool_builder = ChunkPoolBuilder(model_name=global_cfg.embedding_model)

    results: pl.DataFrame
    match evaluate_cfg.gold_mode:
        case 'structural':
            results = evaluate_structural(pool_builder)
        case 'llm':
            annotations_df = read_parquet('gold_annotations').filter(pl.col('n_facets') > 0)
            print(f'Loaded {len(annotations_df):,} annotated queries with facets')
            results = evaluate_llm(annotations_df, pool_builder)
        case _:
            raise RuntimeError(f'[ERROR] Unsupported case: {evaluate_cfg.gold_mode=}')

    out_path = get_table_path('evaluation_results')
    results.write_parquet(out_path)
    print(f'\nSaved {len(results):,} result rows to {out_path}')

    store_eval_stats(results)
    store_best_per_metric(results)
    store_eval_figures(evaluate_cfg)  # reads for convenience the results on disk
    return results


def evaluate_llm(annotations_df: pl.DataFrame, pool_builder: ChunkPoolBuilder) -> pl.DataFrame:
    all_rows = []

    for row in tqdm(
        annotations_df.iter_rows(named=True),
        total=len(annotations_df),
        desc='Evaluating',
        dynamic_ncols=True,
    ):
        row = cast(GoldAnnotationRow, row)
        mod_label_to_chunk_ids: dict[QueryModifierLabelId, list[str]] = json.loads(
            row['facets_json']
        )
        if not mod_label_to_chunk_ids:
            continue

        query_vec: NDArray[np.float32] = pool_builder.embed_query(row['query_text'])

        cosine_pool: ChunkPool
        match evaluate_cfg.pool_preretrieval_mode:
            case 'full_corpus':
                cosine_pool = pool_builder.topk_cosine(query_vec, k=global_cfg.prefilter_n)
            case 'primary_condition_restricted':
                cosine_pool = pool_builder.topk_cosine_for_condition(
                    query_vec,
                    condition_icd10_prefix=row['icd10_3char'],
                    k=global_cfg.prefilter_n,
                )
            case _:
                raise RuntimeError(f'[ERROR] Unsupported: {evaluate_cfg.pool_preretrieval_mode=}')

        all_gold_ids = {chunk_id for ids in mod_label_to_chunk_ids.values() for chunk_id in ids}
        gold_pool: ChunkPool = pool_builder.get_by_chunk_ids(all_gold_ids)

        pool = ChunkPool.merge([gold_pool, cosine_pool])

        query_metrics = evaluate_query(
            pool,
            query_vec,
            mod_label_to_chunk_ids,
            strategies=evaluate_cfg.strategies,
            k_values=evaluate_cfg.k_values,
            lam_values=evaluate_cfg.lam_values,
            answer_text=row['answer_text'],
            chunk_id_to_text=dict(zip(pool.chunk_ids, pool.texts, strict=True)),
        )

        for m in query_metrics:
            all_rows.append({
                'query_id': row['query_id'],
                'icd10_3char': row['icd10_3char'],
                'n_facets': row['n_facets'],
                **m,
            })

    return pl.DataFrame(all_rows)


def evaluate_structural(builder: ChunkPoolBuilder) -> pl.DataFrame:
    queries_df = load_filtered_queries(global_cfg.embedding_model)
    print(f'Loaded {len(queries_df):,} queries for structural evaluation')

    all_rows = []
    for query in tqdm(
        queries_df.iter_rows(named=True),
        total=len(queries_df),
        desc='Evaluating (structural)',
        dynamic_ncols=True,
    ):
        query = cast(QueryRowPostFiltering, query)
        modifiers: list[QueryModifier] = QueryModifier.parse_list(query.get('modifiers_json', ''))
        if not modifiers:
            continue

        query_vec = builder.embed_query(query['query_text'])
        pool = builder.topk_cosine_for_condition(
            query_vec, query['icd10_3char'], k=global_cfg.prefilter_n
        )
        mod_to_chunk_ids = build_structural_facets(pool, query['icd10_3char'], modifiers, builder)
        if not mod_to_chunk_ids:
            print(
                f'  [skip] query_id={query["query_id"]} ({query["icd10_3char"]}): '
                f'no modifier chunks found in pool'
            )
            continue

        query_result: list[QueryEvalResult] = evaluate_query(
            pool,
            query_vec,
            mod_to_chunk_ids,
            strategies=evaluate_cfg.strategies,
            k_values=evaluate_cfg.k_values,
            lam_values=evaluate_cfg.lam_values,
        )

        for m in query_result:
            all_rows.append({
                'query_id': query['query_id'],
                'icd10_3char': query['icd10_3char'],
                'stratum': query.get('stratum'),
                'n_facets': len(mod_to_chunk_ids),
                **m,
            })

    return pl.DataFrame(all_rows)


def build_structural_facets(
    pool: ChunkPool,
    icd10_3char: str,
    modifiers: list[QueryModifier],
    builder: ChunkPoolBuilder,
) -> dict[QueryModifierLabelId, list[str]]:
    mod_to_chunk_ids: dict[QueryModifierLabelId, list[str]] = {}

    for mod in modifiers:
        qualifying_hadm_ids = builder.get_hadm_ids_by_condition_modifier(icd10_3char, mod)
        facet_pool = pool.filter_by_hadm_ids(qualifying_hadm_ids)
        if facet_pool.n > 0:
            mod_to_chunk_ids[mod.label] = facet_pool.chunk_ids

    return mod_to_chunk_ids


def evaluate_query(
    pool: ChunkPool,
    query_vec: NDArray,
    modifier_to_chunk_ids: dict[QueryModifierLabelId, list[str]],
    strategies: list[ScoringFunction],
    k_values: list[int],
    lam_values: list[float],
    answer_text: str | None = None,
    chunk_id_to_text: dict[str, str] | None = None,
) -> list[QueryEvalResult]:
    """Evaluate all strategy x k x λ combos for a single query.
    Returns list of metric dicts.
    """
    retrieval_results = pool.run_retrieval(
        query_vec,
        strategies=strategies,
        k_values=k_values,
        lam_values=lam_values,
    )

    # Find top_k results for Jaccard comparison
    topk_by_k: dict[int, ChunkPoolRetrievalResult] = {}
    for result in retrieval_results:
        if result.strategy == 'top_k':
            topk_by_k[result.k] = result

    sim_to_query = pool.sim_scores(query_vec)
    sim_matrix = pool.sim_matrix()

    chunk_id_to_pool_index: dict[str, int] = {cid: i for i, cid in enumerate(pool.chunk_ids)}
    pool_id_set = set(pool.chunk_ids)
    all_gold_ids = {cid for cids in modifier_to_chunk_ids.values() for cid in cids}

    metrics = []
    for result in retrieval_results:
        selected_set = set(result.selected_chunk_ids)
        chunk_metrics = compute_chunk_support_metrics(
            selected_set, modifier_to_chunk_ids, all_gold_ids, pool_id_set
        )

        eval_indices = np.array(
            [
                chunk_id_to_pool_index[cid]
                for cid in result.selected_chunk_ids
                if cid in chunk_id_to_pool_index
            ],
            dtype=np.intp,
        )

        fac = fac_cov_score(eval_indices, sim_matrix) if len(eval_indices) > 0 else 0.0
        ac = avg_cos(eval_indices, sim_to_query) if len(eval_indices) > 0 else 0.0

        topk_ref = topk_by_k.get(result.k)
        if topk_ref is not None and result.strategy != 'top_k':
            topk_eval_idx = np.array(
                [
                    chunk_id_to_pool_index[cid]
                    for cid in topk_ref.selected_chunk_ids
                    if cid in chunk_id_to_pool_index
                ],
                dtype=np.intp,
            )
            jac = jaccard(eval_indices, topk_eval_idx)
        else:
            jac = 1.0

        row_metrics: dict = {
            'strategy': result.strategy,
            'k': result.k,
            'lam': result.lam,
            **chunk_metrics,
            'fac_cov_score': fac,
            'avg_cos': ac,
            'jaccard_vs_topk': jac,
            'n_unique_hadms': len(set(result.selected_hadm_ids)),
        }
        if answer_text is not None and chunk_id_to_text is not None:
            row_metrics.update(
                compute_answer_support_metrics(
                    answer_text, result.selected_chunk_ids, chunk_id_to_text
                )
            )
        metrics.append(row_metrics)

    return metrics


def store_eval_stats(results_df: pl.DataFrame) -> None:
    print('\n=== Evaluation Summary ===\n')

    summaries = []
    for k in sorted(results_df['k'].unique().to_list()):
        print(f'--- k = {k} ---')
        subset = results_df.filter(pl.col('k') == k)

        ans_aggs = []
        if 'answer_rouge1_recall' in results_df.columns:
            ans_aggs = [
                pl.col('answer_rouge1_recall').mean().alias('ans_rouge1_rec'),
                pl.col('answer_rouge1_precision').mean().alias('ans_rouge1_prec'),
                pl.col('answer_tfidf_cosine').mean().alias('ans_tfidf'),
                pl.col('answer_rouge1_f1').mean().alias('ans_rouge1_f1'),
            ]
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
                *ans_aggs,
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
    has_answer_metrics = 'answer_rouge1_recall' in results_df.columns
    metric_cols = ['AR', 'WAR', 'GP', 'GR']
    if has_answer_metrics:
        metric_cols = [
            *metric_cols,
            'ans_rouge1_rec',
            'ans_rouge1_prec',
            'ans_tfidf',
            'ans_rouge1_f1',
        ]

    ans_aggs = []
    if has_answer_metrics:
        ans_aggs = [
            pl.col('answer_rouge1_recall').mean().alias('ans_rouge1_rec'),
            pl.col('answer_rouge1_precision').mean().alias('ans_rouge1_prec'),
            pl.col('answer_tfidf_cosine').mean().alias('ans_tfidf'),
            pl.col('answer_rouge1_f1').mean().alias('ans_rouge1_f1'),
        ]
    summary = results_df.group_by('k', 'lam', 'strategy').agg(
        pl.col('aspect_recall').mean().alias('AR'),
        pl.col('weighted_aspect_recall').mean().alias('WAR'),
        pl.col('gold_precision').mean().alias('GP'),
        pl.col('gold_recall').mean().alias('GR'),
        *ans_aggs,
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
