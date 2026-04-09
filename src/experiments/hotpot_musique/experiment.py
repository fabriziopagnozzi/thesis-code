import json
import time
from pathlib import Path

import numpy as np
import polars as pl
from tqdm import tqdm

from helpers.embedder import Embedder
from helpers.metrics import compute_metrics, jaccard
from helpers.query_algorithms import select

from .config import ExperimentConfig
from .data_loaders import HotpotChunk, load_dataset
from .qa_processing import QARecord


def run_experiment(cfg: ExperimentConfig) -> pl.DataFrame:
    records = load_dataset(
        cfg.dataset,
        cfg.dataset_path,
        chunk_mode=cfg.chunk_mode,
        max_docs=cfg.max_docs,
        w=cfg.w,
        chunk_tokens=cfg.chunk_tokens,
        stride=cfg.stride,
    )
    if cfg.max_records is not None:
        records = records[: cfg.max_records]

    print(f'Loaded {len(records)} records from {cfg.dataset}')

    embedder = Embedder(
        model_name=cfg.embedding_model,
        device=cfg.device,
        batch_size=cfg.batch_size,
    )

    # Strategy x lambda grid
    strategy_configs = cfg.strategies_with_lambda()

    # Run
    rows: list[dict] = []
    t0 = time.perf_counter()

    for rec in tqdm(records, desc='Records'):
        if not rec.chunks:
            continue

        chunk_texts = [c.text for c in rec.chunks]
        n_chunks = len(chunk_texts)
        if cfg.max_cands is not None and n_chunks > cfg.max_cands:
            chunk_texts = chunk_texts[: cfg.max_cands]
            rec = QARecord(
                id=rec.id,
                question=rec.question,
                answer=rec.answer,
                question_type=rec.question_type,
                chunks=rec.chunks[: cfg.max_cands],
                gold_doc_titles=rec.gold_doc_titles,
                gold_facts=rec.gold_facts,
            )
            n_chunks = cfg.max_cands

        query_emb, chunk_embs = embedder.embed_qa_record(rec.question, chunk_texts)
        sim_to_query = chunk_embs @ query_emb  # (n,)
        sim_matrix = chunk_embs @ chunk_embs.T  # (n, n)

        # Collect selections per (strategy, lambda, k) for Jaccard
        selections: dict[tuple[str, float | None, int], np.ndarray] = {}

        for strategy, lam in strategy_configs:
            for k in cfg.k_values:
                if k > n_chunks:
                    continue

                effective_lam = lam if lam is not None else 0.5

                if cfg.t_max is not None:
                    # Budget mode: rank all candidates, then truncate
                    ranked = select(
                        strategy=strategy,
                        sim_to_query=sim_to_query,
                        k=n_chunks,
                        sim_matrix=sim_matrix,
                        embeddings=chunk_embs,
                        query_embedding=query_emb,
                        lam=effective_lam,
                        window=cfg.mmr_window,
                        theta=cfg.theta,
                    )
                    selected = _budget_truncate(ranked, rec.chunks, cfg.t_max)
                    effective_k = len(selected)
                else:
                    selected = select(
                        strategy=strategy,
                        sim_to_query=sim_to_query,
                        k=k,
                        sim_matrix=sim_matrix,
                        embeddings=chunk_embs,
                        query_embedding=query_emb,
                        lam=effective_lam,
                        window=cfg.mmr_window,
                        theta=cfg.theta,
                    )
                    effective_k = k

                m = compute_metrics(
                    selected_indices=selected,
                    chunks=rec.chunks,
                    answer=rec.answer,
                    gold_doc_titles=rec.gold_doc_titles,
                    gold_facts=rec.gold_facts,
                    sim_to_query=sim_to_query,
                    sim_matrix=sim_matrix,
                )

                key = (strategy, lam, k)
                selections[key] = selected

                rows.append(
                    {
                        'record_id': rec.id,
                        'question_type': rec.question_type,
                        'n_chunks': n_chunks,
                        'n_gold_docs': rec.n_gold_docs,
                        'n_gold_facts': rec.n_gold_facts,
                        'strategy': strategy,
                        'lambda': lam,
                        'k': effective_k,
                        't_max': cfg.t_max,
                        **m,
                    }
                )

        # Compute Jaccard between each strategy and top_k baseline
        for k in cfg.k_values:
            if k > n_chunks:
                continue
            baseline_key = ('top_k', None, k)
            baseline_sel = selections.get(baseline_key)
            if baseline_sel is None:
                continue
            for (strategy, lam, sk), sel in selections.items():
                if sk != k:
                    continue
                j = jaccard(baseline_sel, sel)
                # Find the matching row and add jaccard
                sel_k = len(sel)
                for row in reversed(rows):
                    if (
                        row['record_id'] == rec.id
                        and row['strategy'] == strategy
                        and row['lambda'] == lam
                        and row['k'] == sel_k
                    ):
                        row['jaccard_vs_topk'] = j
                        break

    elapsed = time.perf_counter() - t0
    print(f'Done in {elapsed:.1f}s - {len(rows)} result rows')

    df = pl.DataFrame(rows)
    save_results(df, cfg)
    return df


def save_results(df: pl.DataFrame, cfg: ExperimentConfig) -> None:
    from experiments.hotpot_musique.result_analysis import save_report

    out_dir = Path(cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    prefix = f'{cfg.dataset}_{cfg.chunk_mode}'

    raw_path = out_dir / f'{prefix}_raw.json'
    _write_pretty_json(df, raw_path)
    print(f'Per-record results saved to {raw_path}')

    summary = summarize(df)
    summary_path = out_dir / f'{prefix}_summary.json'
    _write_pretty_json(summary, summary_path)
    print(f'Aggregated results saved to {summary_path}')

    report_path = out_dir / f'{prefix}_stats.json'
    save_report(df, report_path)

    cfg.save(out_dir / f'{prefix}_config.json')


def summarize(df: pl.DataFrame) -> pl.DataFrame:
    """Aggregate per-record results into per-(strategy, lambda, k) means."""
    group_cols = ['strategy', 'lambda', 'k']
    metric_cols = [
        'doc_rec',
        'fact_rec',
        'hit_rate',
        'fac_cov_score',
        'avg_cos',
        'jaccard_vs_topk',
    ]
    agg_exprs = [pl.col(c).mean().alias(c) for c in metric_cols if c in df.columns]
    agg_exprs.append(pl.col('record_id').count().alias('n_records'))

    return df.group_by(group_cols).agg(agg_exprs).sort(group_cols)


def _write_pretty_json(df: pl.DataFrame, path: Path) -> None:
    with open(path, 'w') as f:
        json.dump(df.to_dicts(), f, indent=2)


def _word_count(text: str) -> int:
    return len(text.split())


def _budget_truncate(
    ordered_indices: np.ndarray,
    chunks: list[HotpotChunk],
    t_max: int,
) -> np.ndarray:
    selected: list[int] = []
    tokens_used = 0
    for idx in ordered_indices:
        cost = _word_count(chunks[idx].text)
        if tokens_used + cost > t_max and selected:
            break
        selected.append(int(idx))
        tokens_used += cost
    return np.array(selected, dtype=np.intp)
