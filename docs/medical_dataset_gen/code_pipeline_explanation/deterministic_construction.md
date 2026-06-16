# Deterministic Construction of Synthetic Medical Queries and Chunks

This note explains how the synthetic medical benchmark is built from `ontology.yaml` and the template code, with the deterministic path as the main focus. The key idea is simple: the benchmark geometry is decided first, in structured fields, and only then rendered into natural language. That is what makes the dataset reproducible and what gives the retrieval methods a real coverage problem to solve.

Source files used here:

- [ontology.yaml](/home/pagnozzi/thesis/src/experiments/medical_dataset_gen/ontology.yaml)
- [ontology.py](/home/pagnozzi/thesis/src/experiments/medical_dataset_gen/generation/ontology.py)
- [query_plans.py](/home/pagnozzi/thesis/src/experiments/medical_dataset_gen/generation/query_plans.py)
- [facts.py](/home/pagnozzi/thesis/src/experiments/medical_dataset_gen/generation/facts.py)
- [text_templates.py](/home/pagnozzi/thesis/src/experiments/medical_dataset_gen/generation/text_templates.py)
- [chunks.py](/home/pagnozzi/thesis/src/experiments/medical_dataset_gen/generation/chunks.py)
- [queries_answers.py](/home/pagnozzi/thesis/src/experiments/medical_dataset_gen/generation/queries_answers.py)
- [qrels.py](/home/pagnozzi/thesis/src/experiments/medical_dataset_gen/generation/qrels.py)
- [schemas.py](/home/pagnozzi/thesis/src/experiments/medical_dataset_gen/generation/schemas.py)

The deterministic walkthrough below assumes `use_llm_chunk_generation: false` and `use_llm_query_paraphrase: false`, which is the cleanest way to understand the template-based build.

## What Part Of The Ontology The MVP Uses

The config file in this repo sets `conditions: 4`, which means `selected_conditions()` takes the first four condition entries from `ontology.yaml` in file order. In the current ontology, those are:

- `encephalitis_myelitis` -> `encephalitis or myelitis`
- `pneumonia` -> `pneumonia`
- `ischemic_stroke` -> `ischemic stroke`
- `heart_failure` -> `acute decompensated heart failure`

The benchmark also assumes exactly two clinical axes:

- `treatment_duration`
- `rehab_outcome`

`run_make_query_plans()` enforces that set and raises if the ontology does not match. That check is important because the rest of the pipeline is written around a four-facet query shape: two subgroups times two axes.

For the examples in this document, the first subgroup pair is enough to show the mechanism:

- `age_over_75` -> `patients older than 75`
- `age_under_50` -> `patients younger than 50`

The full ontology has many more subgroups, but the first pair is enough to understand the deterministic ordering and the template structure.

## The Build Order

The pipeline stage order is:

1. Build hidden query plans.
2. Expand each plan into hidden clinical facts.
3. Render each fact into a chunk of text.
4. Render the natural-language query and canonical answer.
5. Derive qrels directly from the hidden gold/distractor labels.

That order matters. The benchmark does not start from free text and then try to infer structure. It starts from structure and uses text as a rendering layer.

## Step 1: Query Plans Are Built First

`run_make_query_plans()` is the first real construction step. It does three deterministic things:

1. It loads the ontology.
2. It selects the first `n` conditions.
3. It enumerates every unique subgroup pair and every query type, then emits plans in a round-robin order across conditions.

In pseudocode, the core shape is:

```python
conditions = ontology.conditions[:cfg.global.conditions]
pairs = all_unique_subgroup_pairs_in_ontology_order()
for condition in round_robin(conditions):
    for pair in pairs:
        for query_type in cfg.generation.query_types:
            emit_query_plan(condition, pair, query_type)
```

The round-robin emission is important. It prevents the first condition from monopolizing the first block of queries and keeps the dataset balanced across the chosen conditions.

### What A Plan Contains

Each plan stores the following hidden structure:

- `query_id`
- `plan_seed`
- `split`
- `query_type`
- `condition_id` and `condition_display`
- `subgroup_a_*` and `subgroup_b_*`
- `dominant_facet_id`
- `facets`
- `logical_form`

The `logical_form` is stored as JSON in `query_plans.parquet`. That is deliberate: the solver is ordinary structured filtering, not a symbolic reasoner.

### Facets Are Built Before Text

Every plan is expanded into exactly four facets:

- subgroup A + treatment duration
- subgroup A + rehabilitation outcome
- subgroup B + treatment duration
- subgroup B + rehabilitation outcome

The facet order is fixed. In rotating-dominance mode, what changes from query to query is which slot is dominant.

`dominant_slot = (plan_idx - 1) % 4`

That means the dominant facet position cycles through `f1`, `f2`, `f3`, `f4`, then repeats. This avoids a fixed position bias and makes the benchmark less brittle. In embedding-calibrated mode, this rotating slot is only the initial placeholder; `calibrate_plans` later selects the dominant facet from neutral probe embeddings.

The facet value bins are assigned by a separate deterministic value-pattern rotation:

- duration patterns cycle through `short/prolonged`, `prolonged/short`, `standard/prolonged`, and `prolonged/standard`
- rehabilitation patterns cycle independently through home, inpatient, and persistent-deficit pairings

That design keeps clinical values balanced without making value choice depend on the hidden dominant facet.

### Real Early Query Ordering

The first emitted queries are not random. They follow the round-robin order over conditions, and the dominant facet rotates with the plan index.

| emitted plan | query id | condition | query type | dominant facet |
| --- | --- | --- | --- | --- |
| 1 | `q00001` | `encephalitis or myelitis` | `subgroup_comparison` | `q00001_f1` |
| 2 | `q00002` | `pneumonia` | `subgroup_comparison` | `q00002_f2` |
| 3 | `q00003` | `ischemic stroke` | `subgroup_comparison` | `q00003_f3` |
| 4 | `q00004` | `acute decompensated heart failure` | `subgroup_comparison` | `q00004_f4` |
| 5 | `q00005` | `encephalitis or myelitis` | `outcome_synthesis` | `q00005_f1` |

That table is the clearest proof that the dataset is deterministic. The same config and ontology always produce the same order.

## Step 2: Hidden Facts Are Generated From The Plan

`run_make_facts()` takes each query plan and materializes hidden evidence rows. It does this per facet:

- gold rows for the facet itself
- hard distractor rows that are close to the query but wrong in one key way

### Gold Rows

Each facet gets a target number of gold chunks:

- dominant facet: `gold_chunks_dominant` = 18
- complementary facets: `gold_chunks_complementary` = 8

So each query gets `18 + 8 + 8 + 8 = 42` gold facts.

The facts are still hidden structure at this point. They are not yet natural-language chunks.

### Distractor Rows

Each query also gets `distractors_per_query` hard negatives. The current config uses 30.

The distractor types are:

- same condition, wrong subgroup
- same subgroup, wrong condition
- same axis, wrong condition

These are close enough to look plausible, but they break one of the hidden constraints. That is exactly what makes them useful retrieval negatives.

### Why `chunk_reuse_key` Exists

`chunk_reuse_key` is built from:

- condition
- subgroup
- axis
- value bin
- local index

It intentionally does not include `query_id`.

That means structurally equivalent facts can share the same surface realization. This is useful for two reasons:

- it makes the benchmark more redundant, which is what a coverage method should exploit
- it allows cache reuse when the same structural fact is rendered again

This redundancy is not a bug. It is part of the benchmark geometry.

## Step 3: The Fact Fields Are Axis-Specific

`ClinicalFact` enforces axis consistency in its validator:

- `treatment_duration` facts must have `duration_days` and `treatment`, and must not have `rehab_outcome`
- `rehab_outcome` facts must have `rehab_outcome`, and must not have `duration_days` or `treatment`

That axis-specific validation is what keeps the hidden labels clean.

`_axis_values()` then fills in concrete values:

- duration facts sample a duration inside the condition’s allowed range and choose a treatment from the condition’s duration-treatment list
- rehab facts choose one rehabilitation phrase from the condition’s rehab-outcome list

Because the values are sampled from condition-specific lists, the text still looks medically coherent while staying aligned with the hidden label.

## Step 4: Chunk Text Is Rendered From Templates

When LLM chunk generation is disabled, `render_chunk_text()` is the only renderer. It has two branches:

- `_render_duration_chunk()`
- `_render_rehab_chunk()`

Both templates use seeded randomness, so the output is deterministic for the same hidden fact.

### Duration Template

The duration template follows this pattern:

1. patient descriptor
2. condition mention
3. condition presentation phrase
4. explicit treatment duration sentence
5. short closing sentence about follow-up or stability

Example from the first query plan:

```text
The 80-year-old woman was admitted with encephalitis or myelitis, with headache, altered mental status, and lower-extremity weakness. The active treatment course used corticosteroids for 5 days, and the record described resolution of fever and stable neurologic examination before discharge. The discharge medication list matched the completed neurologic treatment plan and outpatient neurology follow-up.
```

Why this passes validation:

- it names the condition
- it identifies the subgroup through age and patient descriptor
- it includes the duration and treatment
- it avoids hidden benchmark terms such as `gold` or `qrel`
- it does not use a banned note-section header

### Rehab Template

The rehab template follows a different pattern:

1. patient descriptor
2. condition mention
3. condition presentation phrase
4. functional status sentence
5. explicit rehabilitation outcome
6. short closing sentence

Example from the same query plan:

```text
The 90-year-old man was managed for encephalitis or myelitis, with headache, altered mental status, and lower-extremity weakness. At discharge, mental status returned near baseline with residual fatigue; the discharge record described home therapy with improving gait stability. The patient left with clear activity precautions and follow-up for functional recovery.
```

Why this works:

- the condition is explicit
- the subgroup is recoverable from the age and the hidden subgroup phrase
- the rehab-outcome bin is expressed through a phrase such as `home therapy with improving gait stability`
- the chunk remains short, realistic, and aligned with the hidden axis

### A Hard Negative Example

Here is one distractor fact and the chunk it renders:

```text
The 74-year-old man with diabetes without documented end-organ complications was admitted with encephalitis or myelitis, with headache, altered mental status, and lower-extremity weakness. The inpatient treatment course used acyclovir for 6 days, and the record described resolution of fever and stable neurologic examination before discharge. Repeat examination was stable, and the team documented no escalation beyond the completed inpatient course.
```

This is a hard negative because it is:

- the right condition
- the right axis
- a plausible treatment-duration note
- but the wrong subgroup for the query

That is the exact kind of negative that makes coverage-based selection more valuable than plain top-k.

## Step 5: Queries Are Rendered After The Hidden Plan Exists

`render_query()` turns the plan into a natural-language query.

There are two templates:

- `subgroup_comparison`
- `outcome_synthesis`

The query text is not generated from scratch. It is a direct template over the plan fields.

### Subgroup Comparison

For a `subgroup_comparison` plan, the query is:

```text
For patients diagnosed with <condition>, how do treatment duration and rehabilitation outcome differ between <subgroup A> and <subgroup B>?
```

For the first query plan, that becomes:

```text
For patients diagnosed with encephalitis or myelitis, how do treatment duration and rehabilitation outcome differ between patients older than 75 and patients younger than 50?
```

### Outcome Synthesis

For an `outcome_synthesis` plan, the query is:

```text
Among patients diagnosed with <condition>, compare therapy-course length and discharge rehabilitation status for <subgroup A> versus <subgroup B>.
```

The first outcome-synthesis plan in the actual emission order is `q00005`, not `q00002`, because the pipeline round-robins across conditions. Its query text is:

```text
Among patients diagnosed with encephalitis or myelitis, compare therapy-course length and discharge rehabilitation status for patients older than 75 versus patients younger than 50.
```

### Why The Query Template Works

The query template is stable because it names:

- the condition
- the two subgroup labels
- the two axis labels or their paraphrased equivalents

That makes the hidden structure recoverable from the query text while still leaving retrieval work to do.

## Step 6: Canonical Answers Are Built From The Hidden Facets

`run_make_queries_answers()` reads the gold facts back, groups them by facet, and summarizes them into a canonical answer.

The summarization rule is simple:

- for `treatment_duration`, it reports the modal duration bin, the average duration, and the most common treatment
- for `rehab_outcome`, it reports the modal bin and a representative rehab phrase

So the answer is a structured summary of the facet evidence, not a free-form opinion.

The code then stores:

- `queries.parquet` with the rendered query text
- `gold_answers.parquet` with the canonical answer plus the supporting fact IDs and facet summaries

That separation is useful. It keeps the query text and the answer text consistent without requiring a human annotation step.

## Step 7: Qrels Are Derived, Not Manually Written

`run_make_qrels()` projects the hidden chunk metadata into relevance labels.

The rule is:

- `is_gold = true` -> `relevance_grade = 1`, `support_type = positive`
- `is_gold = false` -> `relevance_grade = 0`, `support_type = hard_negative`

That is all qrels need here, because the facet geometry is already encoded in the hidden facts.

The important point is that the benchmark does not rely on manual relevance annotation. Relevance is a direct consequence of the ontology, the plan, and the hidden fact construction.

## Why This Construction Creates A Real Coverage Problem

This is the thesis-relevant part.

The benchmark is designed so that a query is not satisfied by finding one relevant chunk. It is satisfied by covering several distinct answer facets.

The setup creates that pressure in three ways:

- one facet is intentionally overrepresented, so top-k can drift into redundancy
- complementary facets are still present, so a diversity-aware selector can do better
- hard negatives are semantically close, so the task is not trivial lexical matching

That is why facility-location style coverage is the method the benchmark is meant to reward. It can select a smaller but more facet-complete set of chunks than a selection rule that mainly chases similarity or local dispersion.

## Minimal Mental Model

If you only keep one mental model from this document, it should be this:

1. The ontology defines the legal medical vocabulary and the allowed bins.
2. The query plan fixes the hidden facets before any text exists.
3. The fact generator creates redundant golds and close negatives from that plan.
4. The chunk renderer turns reusable facts into canonical note documents with seeded templates.
5. Query memberships link those documents back to facets, roles, and gold/distractor labels.
6. The query renderer turns the same hidden plan into a natural-language question.
7. The qrels are then a direct projection of the membership labels.

That is how the dataset stays deterministic while still producing a nontrivial retrieval benchmark.
