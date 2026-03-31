import json
from pathlib import Path

import polars as pl

METRICS = ['doc_rec', 'fact_rec', 'hit_rate', 'fac_cov_score', 'avg_cos']


def deltas_vs_baseline(
    df: pl.DataFrame,
    baseline: str = 'top_k',
) -> pl.DataFrame:
    """Compute Δ = strategy - baseline for each metric, per (strategy, λ, k)."""
    base_means = (
        df.filter(pl.col('strategy') == baseline)
        .group_by('k')
        .agg(
            *[pl.col(metric).mean().alias(f'{metric}_base') for metric in METRICS],
            pl.col('record_id').count().alias('n_records'),
        )
    )

    strategy_aggs = [pl.col(metric).mean() for metric in METRICS]
    if 'jaccard_vs_topk' in df.columns:
        strategy_aggs.append(pl.col('jaccard_vs_topk').mean())

    strategy_means = (
        df.filter(pl.col('strategy') != baseline)
        .group_by('strategy', 'lambda', 'k')
        .agg(strategy_aggs)
    )

    delta_columns = [
        (pl.col(metric) - pl.col(f'{metric}_base')).round(6).alias(f'Δ{metric}')
        for metric in METRICS
    ]

    return (
        strategy_means.join(base_means, on='k')
        .with_columns(delta_columns)
        .sort('strategy', 'lambda', 'k')
    )


def breakdown_by_question_type(df: pl.DataFrame) -> pl.DataFrame:
    group_cols = ['question_type', 'strategy', 'lambda', 'k']
    return (
        df.group_by(group_cols)
        .agg(
            *[pl.col(metric).mean() for metric in METRICS],
            pl.col('record_id').count().alias('n'),
        )
        .sort(group_cols)
    )


def _format_lambda(value: float | None) -> str:
    return f'{value:.1f}' if value is not None else '   -'


def _format_jaccard(row: dict) -> str:
    if 'jaccard_vs_topk' not in row:
        return '     -'
    return f'{row["jaccard_vs_topk"]:.3f}'


def _format_deltas_table(rows: list[dict]) -> str:
    lines = [
        '=' * 80,
        'DELTAS VS BASELINE (top_k)',
        '=' * 80,
        '',
        (
            f'{"strategy":<22s} {"λ":>4s} {"k":>3s}  '
            f'{"Δdoc_rec":>9s} {"Δfact_rec":>10s} '
            f'{"Δhit_rate":>10s} {"Δfac_cov":>9s} '
            f'{"Δavg_cos":>9s} {"jaccard":>8s}'
        ),
        '-' * 80,
    ]

    for row in rows:
        lam = _format_lambda(row['lambda'])
        jac = _format_jaccard(row)
        lines.append(
            f'{row["strategy"]:<22s} '
            f'{lam:>4s} {row["k"]:3d}  '
            f'{row["Δdoc_rec"]:>+9.4f} '
            f'{row["Δfact_rec"]:>+10.4f} '
            f'{row["Δhit_rate"]:>+10.4f} '
            f'{row["Δfac_cov_score"]:>+9.4f} '
            f'{row["Δavg_cos"]:>+9.4f} {jac:>8s}'
        )

    return '\n'.join(lines)


def _format_breakdown_table(rows: list[dict]) -> str:
    lines = [
        '',
        '=' * 80,
        'BY QUESTION TYPE',
        '=' * 80,
        '',
    ]

    lines.append(
        f'{"type":<12s} {"strategy":<22s} '
        f'{"λ":>4s} {"k":>3s}  '
        f'{"doc_rec":>8s} {"fact_rec":>9s} '
        f'{"hit_rate":>9s} {"fac_cov":>8s} '
        f'{"avg_cos":>8s} {"n":>6s}'
    )
    lines.append('-' * 90)

    for row in rows:
        lines.append(
            f'{row["question_type"]:<12s} {row["strategy"]:<22s} '
            f'{_format_lambda(row["lambda"]):>4s} {row["k"]:3d}  '
            f'{row["doc_rec"]:>8.4f} {row["fact_rec"]:>9.4f} {row["hit_rate"]:>9.4f} '
            f'{row["fac_cov_score"]:>8.4f} {row["avg_cos"]:>8.4f} {row["n"]:>6d}'
        )

    return '\n'.join(lines)


def format_report(deltas: list[dict], by_type: list[dict]) -> str:
    return _format_deltas_table(deltas) + '\n' + _format_breakdown_table(by_type) + '\n'


def save_report(df: pl.DataFrame, path: str | Path) -> None:
    deltas = deltas_vs_baseline(df)
    by_type = breakdown_by_question_type(df)

    report = {
        'deltas_vs_baseline': deltas.to_dicts(),
        'by_question_type': by_type.to_dicts(),
    }

    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)

    out.write_text(json.dumps(report, indent=2))
    out.with_suffix('.txt').write_text(
        format_report(report['deltas_vs_baseline'], report['by_question_type'])
    )

    print(f'Reports saved to {out} and {out.with_suffix(".txt")}')
