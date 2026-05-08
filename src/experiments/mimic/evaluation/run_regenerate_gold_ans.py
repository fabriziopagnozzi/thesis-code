import json
from typing import cast

from experiments.mimic.evaluation.schemas_evaluation import ExtractedFact, GoldAnnotationCfg
from experiments.mimic.global_configs import get_table_path, global_cfg, setup_logging
from experiments.mimic.queries.schemas_queries import QueryModifier, QueryModifierLabelId, QueryRow
from experiments.mimic.utils.utils import load_filtered_queries

from .run_gold_annotation import (
    GoldAnnotationPersistenceHandler,
    reduce_answer,
)

if __name__ == '__main__':
    setup_logging()
    from experiments.mimic.global_configs import load_config_from_main

    raw = load_config_from_main(key='evaluation')
    gold_annotation_cfg = GoldAnnotationCfg(**raw['gold_annotation'])
    persistence = GoldAnnotationPersistenceHandler(gold_annotation_cfg)

    # load cached MAP facts
    map_jsonl = get_table_path('gold_annotations', ext='jsonl')
    if not map_jsonl.exists():
        raise FileNotFoundError(f'No MAP cache at {map_jsonl}')

    facts_by_query: dict[int, dict[QueryModifierLabelId, list[ExtractedFact]]] = {}
    with map_jsonl.open() as f:
        for line in f:
            e = json.loads(line.strip())
            query_id = e['query_id']
            if query_id not in facts_by_query:
                facts_by_query[query_id] = {}
            label = e['facet_label']
            if label not in facts_by_query[query_id]:
                facts_by_query[query_id][label] = []
            decisions = e.get('decisions', [])
            if decisions != '<LLM_ERROR>':
                facts_by_query[query_id][label].extend(persistence.dedupe_facts(decisions))

    print(f'MAP cache: {len(facts_by_query)} query_ids with facts')

    # load already-generated answers

    answers_jsonl = get_table_path('gold_answers', ext='jsonl')
    done_ids: set[int] = set()
    if answers_jsonl.exists():
        with answers_jsonl.open() as f:
            for line in f:
                e = json.loads(line.strip())
                if 'query_id' in e:
                    done_ids.add(e['query_id'])

    pending = sorted(set(facts_by_query) - done_ids)
    print(f'Already done: {len(done_ids)}  |  Pending: {len(pending)}  →  {pending}')

    if not pending:
        print('Nothing to do.')
        raise SystemExit(0)

    # load queries to reconstruct aspects
    queries_df = load_filtered_queries(global_cfg.embedding_model)
    query_meta: dict[int, QueryRow] = {}
    for row in queries_df.iter_rows(named=True):
        query_row = cast(QueryRow, row)
        query_meta[query_row['query_id']] = query_row

    # run reduce for each pending query
    for query_id in pending:
        meta = query_meta.get(query_id)
        if meta is None:
            print(f'[WARN] query_id={query_id} not found in queries_df, skipping')
            continue

        query_text = meta['query_text']
        condition_name = meta.get('condition_name', '')
        modifiers = QueryModifier.parse_list(meta.get('modifiers_json', '') or '')
        facts_per_facet = facts_by_query[query_id]

        n_facts = sum(len(v) for v in facts_per_facet.values())
        print(f'\n[query_id={query_id}] {query_text[:80]}...')
        print(f'  facts: { {k: len(v) for k, v in facts_per_facet.items()} }  total={n_facts}')

        answer_text = reduce_answer(
            query_text=query_text,
            condition_name=condition_name,
            modifiers=modifiers,
            facts_per_facet=facts_per_facet,
            query_id=query_id,
            persistence=persistence,
        )
        if answer_text is None:
            continue

        with answers_jsonl.open('a') as f:
            f.write(
                json.dumps({
                    'query_id': query_id,
                    'query_text': query_text,
                    'answer_text': answer_text,
                })
                + '\n'
            )

        print(f'  → saved ({len(answer_text)} chars)')

    print(f'\nDone. {len(pending)} answers written to {answers_jsonl}')
