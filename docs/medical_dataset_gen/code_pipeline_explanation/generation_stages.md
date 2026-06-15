# Generation Stages

This note explains the implemented generation stages in the synthetic medical benchmark. It extends [deterministic_construction.md](/home/pagnozzi/thesis/docs/medical_dataset_gen/code_pipeline_explanation/deterministic_construction.md) by covering the full code path, including LLM chunk generation, cache behavior, query dropping, and qrel projection.

Source files:

- [query_plans.py](/home/pagnozzi/thesis/src/experiments/medical_dataset_gen/generation/query_plans.py)
- [facts.py](/home/pagnozzi/thesis/src/experiments/medical_dataset_gen/generation/facts.py)
- [chunks.py](/home/pagnozzi/thesis/src/experiments/medical_dataset_gen/generation/chunks.py)
- [chunk_rendering.py](/home/pagnozzi/thesis/src/experiments/medical_dataset_gen/generation/chunk_rendering.py)
- [chunk_grouped_llm.py](/home/pagnozzi/thesis/src/experiments/medical_dataset_gen/generation/chunk_grouped_llm.py)
- [chunk_cache.py](/home/pagnozzi/thesis/src/experiments/medical_dataset_gen/generation/chunk_cache.py)
- [text_templates.py](/home/pagnozzi/thesis/src/experiments/medical_dataset_gen/generation/text_templates.py)
- [queries_answers.py](/home/pagnozzi/thesis/src/experiments/medical_dataset_gen/generation/queries_answers.py)
- [qrels.py](/home/pagnozzi/thesis/src/experiments/medical_dataset_gen/generation/qrels.py)
- [schemas.py](/home/pagnozzi/thesis/src/experiments/medical_dataset_gen/generation/schemas.py)

## Stage 1: Query Plans

`run_make_query_plans()` creates `query_plans.parquet`. Its job is to define the hidden benchmark geometry before any text exists.

The relevant config fields are:

- `global.seed`
- `global.n_queries`
- `global.conditions`
- `generation.ontology_path`
- `generation.query_types`
- `generation.gold_chunks_dominant`
- `generation.gold_chunks_complementary`
- `generation.distractors_per_query`

The stage loads the ontology, takes the first `global.conditions` condition entries in YAML order, builds every unique unordered subgroup pair, and validates that the clinical axes are exactly `treatment_duration` and `rehab_outcome`.

The query scheduler is condition-balanced. For each selected condition, it builds a list of `QueryPlanSpec` objects over every configured `query_type` and every subgroup pair. It then emits plans round-robin over conditions until `global.n_queries` rows have been produced or the spec lists are exhausted. This prevents the first condition from occupying a large prefix of the dataset.

Each plan has four facets:

1. subgroup A + treatment duration
2. subgroup A + rehabilitation outcome
3. subgroup B + treatment duration
4. subgroup B + rehabilitation outcome

The dominant facet is selected by:

```python
dominant_slot = (plan_idx - 1) % 4
```

That rotating dominant slot avoids always making the same subgroup-axis combination dominant.

The plan row stores both scalar columns and JSON columns. `facets_json` contains serialized `QueryPlanFacet` objects. `logical_form_json` contains a `QueryLogicalForm` with the query type, condition id, subgroup ids, axis ids, facet ids, and dominant facet id. This is intentionally ordinary JSON, not a Prolog or symbolic solver layer.

The split assignment is deterministic:

```python
bucket = plan_idx % 20
test       if bucket in {0, 1, 2}
validation if bucket in {3, 4, 5}
train      otherwise
```

## Stage 2: Clinical Facts

`run_make_facts()` creates `clinical_facts.parquet`. It expands every query plan into hidden atomic facts. This is where the planned facet geometry becomes concrete evidence rows.

The relevant config fields are:

- `generation.gold_chunks_dominant`
- `generation.gold_chunks_complementary`
- `generation.distractors_per_query`

For each plan, the stage initializes `rng = Random(plan.plan_seed)`. It then generates:

- `target_gold_chunks` gold facts for each facet.
- `generation.distractors_per_query` hard distractor facts.

The gold count is already stored on each facet by the plan stage. The dominant facet receives `generation.gold_chunks_dominant`; every complementary facet receives `generation.gold_chunks_complementary`.

### Gold Facts

`make_gold_fact()` calls `make_base_fact()` with the facet's own condition, subgroup, axis, cluster id, and cluster role. A gold fact has:

- `is_gold = True`
- `facet_id = <actual supporting facet>`
- `target_facet_id = <same facet>`
- `cluster_role = dominant_gold` or `complementary_gold`

The `facet_id` is what later allows facet coverage metrics to know which answer facet a selected chunk supports.

### Distractor Facts

`make_distractor_facts()` cycles through three distractor types:

- `same_condition_wrong_subgroup`
- `same_subgroup_wrong_condition`
- `same_axis_wrong_condition`

Every distractor is anchored to a target facet, but does not support it. That is why distractors have:

- `is_gold = False`
- `facet_id = None`
- `target_facet_id = <the facet they are meant to distract from>`
- `cluster_role = hard_distractor`

The distractor types are close to the query in different ways. Same-condition wrong-subgroup distractors are clinically on-topic but fail the subgroup constraint. Same-subgroup wrong-condition distractors match the subgroup but fail the condition constraint. Same-axis wrong-condition distractors preserve the answer axis but fail both the condition and subgroup selection.

### Axis Values

`make_base_fact()` chooses an axis-specific `value_bin` with `_axis_value_bin()`. For gold facts, the facet's planned value bin is valid and used directly. For distractors, the target facet bin is reused when it exists for the distractor condition; otherwise the code cycles through that condition's available bins.

`_axis_values()` then converts the bin into concrete clinical values:

- `treatment_duration`: sample `duration_days` inside the ontology's range and choose a treatment from `duration_treatments` or `treatments`.
- `rehab_outcome`: choose a phrase from the ontology's `rehab_outcomes[value_bin]`.

`ClinicalFact` validates that these fields are mutually consistent. A duration fact must have `duration_days` and `treatment`, and must not have `rehab_outcome`. A rehab fact must have `rehab_outcome`, and must not have `duration_days` or `treatment`.

### Reuse Keys And Surface Seeds

The fact stage builds `chunk_reuse_key` from:

- schema version
- condition id
- subgroup id
- axis
- value bin
- local index

It does not include `query_id`. This is fundamental for controlled redundancy. Equivalent structural facts across queries can share a generated or rewritten surface form. The surface RNG is seeded from the SHA-256 hash of the reuse key, so patient age, sex, note style, treatment choices, and phrase choices are stable for that structural fact.

## Stage 3: Chunk Rendering

`run_make_chunks()` creates `chunks.parquet` and `generation_rejects.parquet`. This stage turns `ClinicalFact` rows into `ChunkRow` rows with actual note text.

The relevant config fields are:

- `generation.chunk_min_words`
- `generation.chunk_max_words`
- `generation.chunk_word_tolerance`
- `generation.llm_name`
- `generation.llm_workers`
- `generation.use_llm_chunk_generation`
- `generation.use_llm_chunk_rewriting`
- `generation.llm_chunk_max_attempts`
- `generation.llm_temperature`
- `generation.llm_num_ctx`

There are three modes:

1. Direct LLM chunk generation: `use_llm_chunk_generation = true`.
2. Deterministic template chunks: both LLM flags false.
3. Deterministic template chunks rewritten by LLM: `use_llm_chunk_generation = false` and `use_llm_chunk_rewriting = true`.

If both direct LLM generation and rewrite are enabled, direct LLM generation wins and rewrite is ignored.

### Deterministic Template Mode

When direct LLM generation and rewriting are both disabled, the code uses `_render_chunks_deterministic_parallel()`. This streams query-sized fact batches through a `ProcessPoolExecutor`. Each worker receives serialized config and ontology objects, reconstructs Pydantic models, and renders every fact in a query batch.

For each fact:

1. `render_chunk_text_template()` chooses the duration or rehab branch.
2. `validate_chunk_text()` checks the text against the hidden fact.
3. `finalize_chunk_row()` enforces hard validation and word-count bounds.
4. `ChunkRow.from_fact()` materializes the row with inherited labels and validation metadata.

If a deterministic validation failure occurs for any fact in a query batch, the whole query is marked failed and its rows are not written. This preserves the invariant that a kept query has complete local evidence.

### Template Rendering

`render_duration_chunk()` builds duration evidence from:

- patient descriptor
- condition presentation
- condition status phrase
- duration phrase template
- response verb
- condition-specific closing sentence
- duration chunk template

`render_rehab_chunk()` builds rehab evidence from:

- patient descriptor
- condition presentation
- condition- and bin-specific functional status phrase
- rehab transition phrase
- rehab outcome verb
- bin-specific closing sentence
- rehab chunk template

The phrase inventories are loaded from [text_templates_utils.yaml](/home/pagnozzi/thesis/src/experiments/medical_dataset_gen/generation/text_templates_utils.yaml) into the `ChunkTemplateUtils` schema.

### Chunk Validation

`validate_chunk_text()` returns hard errors and soft warnings.

Hard errors include:

- hidden benchmark terms appear in the text
- text starts with a banned note-section header
- missing condition evidence
- missing subgroup evidence
- duration fact missing exact duration days
- duration fact missing treatment
- rehab fact missing rehabilitation bin evidence
- unsupported axis

Soft warnings include:

- duration text includes extra condition treatments
- duration text contains rehabilitation-outcome language
- rehab text contains explicit duration days
- rehab text has bin-level evidence but not the exact chosen rehab phrase

Hard errors reject the chunk. Soft warnings are stored on the row as `validation_soft_warnings_json` and counted, but the chunk can remain in the benchmark.

### Direct LLM Generation

Direct LLM generation uses `generate_llm_chunk()` in [chunk_rendering.py](/home/pagnozzi/thesis/src/experiments/medical_dataset_gen/generation/chunk_rendering.py). Each attempt builds a prompt with `MedicalDatasetGenDefaultPrompts.chunk_generation_prompt()`, calls `helpers.ollama_client.generate()`, cleans the output, validates it, and checks word count.

If validation fails, the errors are converted into revision feedback and the next attempt asks the model to fix them. After `llm_chunk_max_attempts`, the function returns the last text plus errors.

In direct LLM mode, failed generation for one chunk drops the whole query. This is stricter than rewrite mode because direct LLM generation has no guaranteed deterministic fallback for that fact.

### LLM Rewrite Mode

Rewrite mode first renders deterministic template text, then asks the LLM to rewrite it with `rewrite_llm_chunk()`. The prompt explicitly supplies:

- the draft text
- the patient descriptor
- required facts
- forbidden facts
- the facet focus
- the word-count range

If rewriting fails validation, the deterministic draft is kept. The resulting chunk records:

- `text_generation_source = fallback`
- `llm_attempted = True`
- `llm_rejected = True`

This behavior is different from direct generation. Rewrite failure does not by itself drop the query, because the deterministic draft still has a valid evidence-preserving path.

### Grouped LLM Calls

`chunk_grouped_llm.py` deduplicates equivalent LLM work. It groups facts by cache key, generates one text per unique key, and materializes the same accepted text back onto all facts in the group after validation.

For direct generation, `render_chunks_grouped_llm()`:

1. Checks exact `fact_id` cache hits.
2. Checks reusable cache-key hits.
3. Groups remaining facts by `chunk_generation_cache_key()`.
4. Runs unique jobs sequentially or with a `ThreadPoolExecutor`.
5. Validates and materializes each group.
6. Writes accepted cache entries to local and shared caches.

For rewrite, `render_chunks_grouped_rewrite()` processes one query group at a time, renders deterministic drafts, checks rewrite cache hits, groups missing rewrite jobs by `chunk_rewrite_cache_key()`, and runs the missing jobs in a thread pool.

The grouping matters because the benchmark intentionally reuses structured facts. Without grouping and caching, LLM generation would waste work on duplicated or near-duplicated chunks.

## Stage 4: Queries And Gold Answers

`run_make_queries_answers()` creates `queries.parquet` and `gold_answers.parquet`.

The relevant config fields are:

- `generation.use_llm_query_paraphrase`
- `generation.llm_name`
- `generation.llm_temperature`
- `generation.llm_num_ctx`

Before creating queries, the stage reads `generation_rejects.parquet`. Any query id with a generation reject is skipped. This is why query dropping in the chunk stage propagates cleanly: downstream tables only include queries with complete accepted chunks.

`render_query()` supports two query types:

- `subgroup_comparison`: asks how treatment duration and rehabilitation outcome differ between the two subgroups.
- `outcome_synthesis`: asks to compare therapy-course length and discharge rehabilitation status for subgroup A versus subgroup B.

If query paraphrasing is enabled, `maybe_paraphrase_query()` calls the LLM and accepts the paraphrase only if it still contains the condition display label and both subgroup labels. Otherwise it falls back to the deterministic query text.

### Canonical Answer Construction

The gold answer is not LLM-authored. `_facet_summaries()` groups gold fact rows by facet id.

For `treatment_duration`, it computes:

- modal value bin
- average duration in days
- most common treatment

For `rehab_outcome`, it computes:

- modal value bin
- most common rehab outcome phrase

`canonical_answer()` then writes a compact four-part answer: subgroup A duration, subgroup A rehab, subgroup B duration, subgroup B rehab. The answer row also stores JSON for facet summaries, answer fact objects, supporting fact ids, and supporting facet ids.

## Stage 5: Qrels

`run_make_qrels()` creates `qrels.parquet` from `chunks.parquet`.

The rule is intentionally simple:

```text
is_gold = true  -> relevance_grade = 1, support_type = positive
is_gold = false -> relevance_grade = 0, support_type = hard_negative
```

The qrels row preserves:

- `query_id`
- `chunk_id`
- `fact_id`
- `facet_id`
- `target_facet_id`
- `cluster_id`
- `cluster_role`
- `is_gold`
- `distractor_type`
- `relevance_grade`
- `support_type`

This stage is fundamental because it keeps evaluation tied to the hidden construction. The qrels do not depend on lexical overlap, embedding similarity, or human annotation. They are a direct projection of the planned fact-to-facet mapping.
