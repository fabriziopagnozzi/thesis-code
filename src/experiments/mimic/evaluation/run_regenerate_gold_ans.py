import json

from experiments.mimic.global_configs import get_result_dir, get_table_path, global_cfg
from experiments.mimic.queries.schemas_queries import QueryModifier
from experiments.mimic.utils.utils import load_filtered_queries

from .run_gold_annotation import reduce_answer

if __name__ == '__main__':
    # load cached MAP facts
    map_jsonl = get_table_path('gold_annotations', ext='jsonl')
    if not map_jsonl.exists():
        raise FileNotFoundError(f'No MAP cache at {map_jsonl}')

    facts_by_query: dict[int, dict[str, list[dict]]] = {}
    with map_jsonl.open() as f:
        for line in f:
            e = json.loads(line.strip())
            qidx = e['query_idx']
            if qidx not in facts_by_query:
                facts_by_query[qidx] = {}
            label = e['facet_label']
            if label not in facts_by_query[qidx]:
                facts_by_query[qidx][label] = []
            facts_by_query[qidx][label].extend(e.get('decisions', []))

    print(f'MAP cache: {len(facts_by_query)} query_idxs with facts')

    # load already-generated answers

    answers_jsonl = get_result_dir('gold_annotations') / 'gold_answers.jsonl'
    done_idxs: set[int] = set()
    if answers_jsonl.exists():
        with answers_jsonl.open() as f:
            for line in f:
                e = json.loads(line.strip())
                done_idxs.add(e['query_idx'])

    pending = sorted(set(facts_by_query) - done_idxs)
    print(f'Already done: {len(done_idxs)}  |  Pending: {len(pending)}  →  {pending}')

    if not pending:
        print('Nothing to do.')
        raise SystemExit(0)

    # load queries to reconstruct aspects

    queries_df = load_filtered_queries(global_cfg.embedding_model)
    query_meta: dict[int, dict] = {
        i: dict(row) for i, row in enumerate(queries_df.iter_rows(named=True))
    }

    # run reduce for each pending query

    for qidx in pending:
        meta = query_meta.get(qidx)
        if meta is None:
            print(f'[WARN] query_idx={qidx} not found in queries_df, skipping')
            continue

        query_text = meta['query_text']
        condition_name = meta.get('condition_name', '')
        modifiers = QueryModifier.parse_list(meta.get('modifiers_json', '') or '')
        facts_per_facet = facts_by_query[qidx]

        n_facts = sum(len(v) for v in facts_per_facet.values())
        print(f'\n[q{qidx}] {query_text[:80]}...')
        print(f'  facts: { {k: len(v) for k, v in facts_per_facet.items()} }  total={n_facts}')

        answer_text = reduce_answer(
            query_text=query_text,
            condition_name=condition_name,
            modifiers=modifiers,
            facts_per_facet=facts_per_facet,
            prompt_dump_dir=None,
            query_idx=qidx,
        )

        with answers_jsonl.open('a') as f:
            f.write(
                json.dumps({
                    'query_idx': qidx,
                    'query_text': query_text,
                    'answer_text': answer_text,
                })
                + '\n'
            )

        print(f'  → saved ({len(answer_text)} chars)')

    print(f'\nDone. {len(pending)} answers written to {answers_jsonl}')
