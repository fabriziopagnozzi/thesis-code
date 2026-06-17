# Deterministic Construction of Synthetic Medical Queries, Chunks, and Answers

This note explains the deterministic construction path for the curated medical RAG benchmark. The important design choice is that the benchmark geometry is created first in structured fields, and only then rendered into text. That gives the experiment explicit facet membership, cluster roles, qrels, and canonical answers without relying on noisy post-hoc annotation.

The concrete examples below come from this experiment directory:

```text
/home/fab/Projects/thesis/src/experiments/medical_dataset_gen/_results/03_bge_m3_pool_1_with_distractors_and_improved_chunks
```

For that run, deterministic text rendering was enabled:

- `use_llm_chunk_generation: false`
- `use_llm_chunk_rewriting: false`
- `use_llm_query_paraphrase: false`

So the queries, chunks, and answers shown here are all produced by code and YAML templates, not by live LLM calls.

Source files used in this construction:

- [ontology.yaml](/home/fab/Projects/thesis/src/experiments/medical_dataset_gen/ontology.yaml)
- [ontology.py](/home/fab/Projects/thesis/src/experiments/medical_dataset_gen/generation/ontology.py)
- [query_plans.py](/home/fab/Projects/thesis/src/experiments/medical_dataset_gen/generation/query_plans.py)
- [calibrate_plans.py](/home/fab/Projects/thesis/src/experiments/medical_dataset_gen/generation/calibrate_plans.py)
- [facts.py](/home/fab/Projects/thesis/src/experiments/medical_dataset_gen/generation/facts.py)
- [chunk_templates.py](/home/fab/Projects/thesis/src/experiments/medical_dataset_gen/generation/chunk_templates.py)
- [query_templates.py](/home/fab/Projects/thesis/src/experiments/medical_dataset_gen/generation/query_templates.py)
- [templates_data](/home/fab/Projects/thesis/src/experiments/medical_dataset_gen/generation/templates_data)
- [chunks.py](/home/fab/Projects/thesis/src/experiments/medical_dataset_gen/generation/chunks.py)
- [queries_answers.py](/home/fab/Projects/thesis/src/experiments/medical_dataset_gen/generation/queries_answers.py)
- [qrels.py](/home/fab/Projects/thesis/src/experiments/medical_dataset_gen/generation/qrels.py)
- [schemas.py](/home/fab/Projects/thesis/src/experiments/medical_dataset_gen/generation/schemas.py)

## Template Data Boundaries

The deterministic renderers use typed YAML data under [templates_data](/home/fab/Projects/thesis/src/experiments/medical_dataset_gen/generation/templates_data). Chunk text is loaded into `ChunkTemplateUtils`; query text is loaded into `QueryTemplateData`.

| file | responsibility |
| --- | --- |
| [condition_context.yaml](/home/fab/Projects/thesis/src/experiments/medical_dataset_gen/generation/templates_data/condition_context.yaml) | condition presentation phrases and condition response/status phrases shared by duration and rehab chunks |
| [duration_templates.yaml](/home/fab/Projects/thesis/src/experiments/medical_dataset_gen/generation/templates_data/duration_templates.yaml) | treatment-duration closing sentences, duration phrase templates, response verbs, and full duration chunk templates |
| [rehab_templates.yaml](/home/fab/Projects/thesis/src/experiments/medical_dataset_gen/generation/templates_data/rehab_templates.yaml) | functional-status phrases, rehab closing sentences, rehab transition phrases, rehab outcome verbs, and full rehab chunk templates |
| [validation_terms.yaml](/home/fab/Projects/thesis/src/experiments/medical_dataset_gen/generation/templates_data/validation_terms.yaml) | hidden benchmark terms, subgroup lexical evidence, rehab bin terms, persistent-deficit terms, and token stopwords used during validation |
| [query_answer_templates.yaml](/home/fab/Projects/thesis/src/experiments/medical_dataset_gen/generation/templates_data/query_answer_templates.yaml) | query templates with stable variant ids plus canonical answer templates used by deterministic query and answer rendering |

`chunk_templates.py` loads the chunk-related files in an explicit order:

```python
_TEMPLATE_DATA_FILES = (
    'condition_context.yaml',
    'duration_templates.yaml',
    'rehab_templates.yaml',
    'validation_terms.yaml',
)
```

The chunk loader merges the top-level YAML keys and validates the result with `ChunkTemplateUtils`. It raises if two files define the same top-level key. Query and answer templates are loaded separately by `query_templates.py` from `query_answer_templates.yaml` and validated with `QueryTemplateData`. This keeps chunk wording, query wording, answer wording, and ontology facts separate while preserving strongly typed runtime objects.

## Experiment Shape

The example experiment uses:

- `conditions: 17`
- 14 ontology subgroups
- 91 unique subgroup pairs
- two query types: `subgroup_comparison` and `outcome_synthesis`
- `dominance_mode: embedding_calibrated`
- `gold_chunks_dominant: 26`
- `gold_chunks_complementary: 14`
- `distractors_per_query: 25`
- one background outlier cluster of size 8 per query

The config cap is `n_queries: 6000`, but the ontology only yields `17 * 91 * 2 = 3094` unique condition/subgroup-pair/query-type plans. The run therefore contains 3094 query plans and 3094 final queries.

Each final query has four intended answer facets:

| facet slot | subgroup | axis |
| --- | --- | --- |
| `f1` | subgroup A | `treatment_duration` |
| `f2` | subgroup A | `rehab_outcome` |
| `f3` | subgroup B | `treatment_duration` |
| `f4` | subgroup B | `rehab_outcome` |

For a query with one dominant facet, the planned positive evidence count is:

```text
26 dominant gold chunks + 14 + 14 + 14 complementary gold chunks = 68 gold chunks
```

The non-gold local pool adds:

```text
25 near-miss hard distractors + 8 background outlier chunks = 33 non-gold chunks
```

So a normal query-local pool has 101 memberships: 68 positives and 33 negatives.

## Stage 1: Query Plans

`run_make_query_plans()` builds hidden plans before any free text exists. It loads the ontology, selects the configured conditions, enumerates subgroup pairs and query types, and emits plans round-robin over conditions.

The initial plan uses a rotating dominant slot:

```python
dominant_slot = (plan_idx - 1) % 4
```

In this experiment, `dominance_mode: embedding_calibrated` then runs `calibrate_plans.py`. Calibration renders neutral probe chunks for each facet, embeds those probes against the natural query, and selects the facet that is most naturally close to the query in embedding space. The query plan is rewritten with that selected dominant facet before facts are generated.

For `q00001`, calibration changed the dominant facet from the initial `q00001_f1` to `q00001_f4`.

### Query Template Variants Are Assigned Here

Query wording diversity is also decided at the plan stage. For each `query_type`, `query_plans.py` reads the available template variant ids from `query_answer_templates.yaml` and assigns them in round-robin order as plans are emitted. The chosen variant is stored in `plan.template_id`.

That means:

- `query_type` still controls the semantic shape of the benchmark item
- `template_id` now records the exact surface wording variant used later by the renderer
- the renderer does not randomly choose a query template at render time

## Stage 2: Hidden Facts

`run_make_facts()` turns each query plan into hidden `ClinicalFact` rows.

For gold facts, each facet produces its target number of rows:

- the dominant facet produces 26 rows
- each complementary facet produces 14 rows

For non-gold facts, the generator adds:

- `same_condition_wrong_subgroup`
- `same_subgroup_wrong_condition`
- `same_axis_wrong_condition`
- `background_clinical_cluster`

These negatives are intentionally close. They often share the condition, subgroup, or axis with the query, but break at least one hidden constraint.

Every fact stores both the natural clinical fields and the benchmark fields:

- condition, subgroup, axis, and value bin
- `facet_id` for gold support
- `target_facet_id` for near-miss negatives
- `cluster_role`
- `is_gold`
- treatment duration or rehab outcome fields, depending on axis

## Stage 3: Deterministic Chunk Rendering

`render_chunk_text_template()` in `chunk_templates.py` has two branches:

- `render_duration_chunk()`
- `render_rehab_chunk()`

Both use seeded randomness. The seed comes from the fact's `chunk_reuse_key`, so structurally equivalent facts can reuse the same rendered chunk across queries. That is why the project has both:

- `chunk_documents.parquet`: unique rendered chunk documents
- `chunk_memberships.parquet`: query-local membership rows that connect chunks to queries, facets, and roles

### Duration Chunks

A duration chunk includes:

1. patient descriptor
2. condition mention
3. condition presentation phrase
4. explicit treatment-duration phrase
5. condition-specific status/closing language

The pieces come mainly from:

- `condition_context.yaml`
- `duration_templates.yaml`

### Rehab Chunks

A rehab chunk includes:

1. patient descriptor
2. condition mention
3. condition presentation phrase
4. functional status phrase
5. explicit rehabilitation outcome phrase
6. rehab planning/closing language

The pieces come mainly from:

- `condition_context.yaml`
- `rehab_templates.yaml`

### Validation

`validate_chunk_text()` checks the rendered text before accepting it. It verifies condition evidence, subgroup evidence, axis-specific evidence, and banned hidden benchmark terms. Its lexical support lists live in `validation_terms.yaml`.

## Stage 4: Queries and Answers

`render_query()` in `queries_answers.py` delegates to `render_query_template()` in [query_templates.py](/home/fab/Projects/thesis/src/experiments/medical_dataset_gen/generation/query_templates.py). That renderer looks up the exact template variant named by `plan.template_id`, then fills in the query-plan fields and ontology axis labels.

For `subgroup_comparison`, the YAML now stores explicit variant ids:

```yaml
query_templates:
  subgroup_comparison:
    - id: subgroup_comparison
      template: "For patients diagnosed with {condition}, how do {treatment_duration_label} and {rehab_outcome_label} differ between {subgroup_a} and {subgroup_b}?"
    - id: subgroup_comparison_compare
      template: "Among patients diagnosed with {condition}, compare {treatment_duration_label} and {rehab_outcome_label} between {subgroup_a} and {subgroup_b}."
```

If `plan.template_id == "subgroup_comparison"`, then with the current ontology `treatment_duration_label` becomes `treatment duration` and `rehab_outcome_label` becomes `rehabilitation outcome`, so that renders:

```text
For patients diagnosed with <condition>, how do treatment duration and rehabilitation outcome differ between <subgroup A> and <subgroup B>?
```

For `outcome_synthesis`, alternate surface variants live in the same list:

```yaml
query_templates:
  outcome_synthesis:
    - id: outcome_synthesis
      template: "Among patients diagnosed with {condition}, compare therapy-course length and discharge rehabilitation status for {subgroup_a} versus {subgroup_b}."
    - id: outcome_synthesis_summary
      template: "For patients with {condition}, summarize therapy-course length and discharge rehabilitation status for {subgroup_a} compared with {subgroup_b}."
```

That renders:

```text
Among patients diagnosed with <condition>, compare therapy-course length and discharge rehabilitation status for <subgroup A> versus <subgroup B>.
```

`canonical_answer()` then summarizes the hidden gold facts per facet:

- treatment facets report the modal duration bin, average duration, and most common treatment
- rehab facets report the modal rehab bin and a representative rehab phrase

The answer is therefore a deterministic summary of the facet evidence.

## Real Example: `q00001`

These rows come from:

- `queries.parquet`
- `query_plans.parquet`
- `gold_answers.parquet`
- `chunk_documents.parquet`
- `chunk_memberships.parquet`

### Query

```text
For patients diagnosed with encephalitis or myelitis, how do treatment duration and rehabilitation outcome differ between patients older than 75 and patients younger than 50?
```

### Hidden Facets

| facet | role | subgroup | axis | value bin | target gold chunks |
| --- | --- | --- | --- | --- | --- |
| `q00001_f1` | complementary | patients older than 75 | `treatment_duration` | `short` | 14 |
| `q00001_f2` | complementary | patients older than 75 | `rehab_outcome` | `home_rehab` | 14 |
| `q00001_f3` | complementary | patients younger than 50 | `treatment_duration` | `prolonged` | 14 |
| `q00001_f4` | dominant | patients younger than 50 | `rehab_outcome` | `inpatient_rehab` | 26 |

### Canonical Answer

```text
For patients older than 75, a short course, averaging 6.0 days, most often with corticosteroids for treatment duration and a home rehab pattern, commonly described as home therapy with improving gait stability for rehabilitation outcome. For patients younger than 50, a prolonged course, averaging 24.4 days, most often with corticosteroids for treatment duration and an inpatient rehab pattern, commonly described as required acute rehabilitation for mobility and cognitive deficits for rehabilitation outcome.
```

### Gold Chunk From `q00001_f1`

This chunk supports the older-than-75 treatment-duration facet.

```text
The 83-year-old woman older than 75 required hospitalization for encephalitis or myelitis after developing new neurologic deficits and inflammatory cerebrospinal fluid findings. For treatment duration, the team documented a 7-day course of acyclovir, and by discharge clinicians documented improving mentation and reduced headache. Neurology follow-up was arranged to monitor recovery after completion of the anti-inflammatory or antiviral course.
```

Structured fields:

| field | value |
| --- | --- |
| `chunk_id` | `chunk_0000001` |
| `facet_id` | `q00001_f1` |
| `cluster_role` | `complementary_gold` |
| `axis` | `treatment_duration` |
| `subgroup_label` | `patients older than 75` |
| `value_bin` | `short` |
| `duration_days` | `7` |
| `treatment` | `acyclovir` |

### Gold Chunk From `q00001_f2`

This chunk supports the older-than-75 rehabilitation-outcome facet.

```text
The 86-year-old man older than 75 received inpatient care for encephalitis or myelitis after headache, altered mental status, and lower-extremity weakness. Before leaving the hospital, orientation improved and gait was safe with supervised exercises, and the discharge summary described the rehabilitation outcome as home therapy with improving gait stability. The patient left with clear activity precautions and follow-up for functional recovery.
```

Structured fields:

| field | value |
| --- | --- |
| `chunk_id` | `chunk_0000015` |
| `facet_id` | `q00001_f2` |
| `cluster_role` | `complementary_gold` |
| `axis` | `rehab_outcome` |
| `subgroup_label` | `patients older than 75` |
| `value_bin` | `home_rehab` |
| `rehab_outcome` | `home therapy with improving gait stability` |

### Gold Chunk From `q00001_f3`

This chunk supports the younger-than-50 treatment-duration facet.

```text
The 28-year-old man younger than 50 required hospitalization for encephalitis or myelitis after developing fever, confusion, and gait change. For treatment duration, the team documented a 21-day course of corticosteroids, and by discharge clinicians documented resolution of fever and stable neurologic examination. Repeat examination was stable, and the team documented no escalation beyond the completed inpatient course.
```

Structured fields:

| field | value |
| --- | --- |
| `chunk_id` | `chunk_0000029` |
| `facet_id` | `q00001_f3` |
| `cluster_role` | `complementary_gold` |
| `axis` | `treatment_duration` |
| `subgroup_label` | `patients younger than 50` |
| `value_bin` | `prolonged` |
| `duration_days` | `21` |
| `treatment` | `corticosteroids` |

### Gold Chunk From `q00001_f4`

This chunk supports the younger-than-50 rehabilitation-outcome facet. This is also the dominant facet for `q00001`.

```text
The 26-year-old man younger than 50 was managed for encephalitis or myelitis, with new neurologic deficits and inflammatory cerebrospinal fluid findings. On the day of discharge, cognitive slowing and balance deficits still limited safe transfers; the discharge record described the rehabilitation outcome as transferred to inpatient rehabilitation for persistent weakness. The plan emphasized supervised strengthening, mobility training, and reassessment before return home.
```

Structured fields:

| field | value |
| --- | --- |
| `chunk_id` | `chunk_0000043` |
| `facet_id` | `q00001_f4` |
| `cluster_role` | `dominant_gold` |
| `axis` | `rehab_outcome` |
| `subgroup_label` | `patients younger than 50` |
| `value_bin` | `inpatient_rehab` |
| `rehab_outcome` | `transferred to inpatient rehabilitation for persistent weakness` |

## Real Near-Miss Chunks For `q00001`

Near-miss chunks are not random junk. They are clinically plausible, close to the query, and wrong in a controlled way.

### Same Condition, Wrong Subgroup

This chunk has the right condition and the right axis, but it is about uncomplicated diabetes rather than either queried age subgroup.

```text
The 74-year-old woman with diabetes without documented end-organ complications was treated inpatient for encephalitis or myelitis, with new neurologic deficits and inflammatory cerebrospinal fluid findings on presentation. For treatment duration, corticosteroids remained in place across a 7-day course; clinicians documented improved strength and stable neurologic checks before transition out of the hospital. Repeat examination was stable, and the team documented no escalation beyond the completed inpatient course.
```

Structured fields:

| field | value |
| --- | --- |
| `chunk_id` | `chunk_0000069` |
| `cluster_role` | `hard_distractor` |
| `target_facet_id` | `q00001_f1` |
| `distractor_type` | `same_condition_wrong_subgroup` |
| `condition_display` | `encephalitis or myelitis` |
| `subgroup_label` | `patients with uncomplicated diabetes` |
| `axis` | `treatment_duration` |
| `relevance_grade` | `0` |

### Same Subgroup, Wrong Condition

This chunk has the queried older-than-75 subgroup and a rehab outcome, but the condition is ischemic stroke rather than encephalitis or myelitis.

```text
The 78-year-old woman older than 75 remained hospitalized for ischemic stroke with acute dysarthria and unilateral weakness. At discharge, mild dysarthria persisted but transfers were independent; the discharge record described the rehabilitation outcome as discharged home with outpatient stroke therapy. The plan emphasized caregiver teaching, home safety, and close outpatient reassessment.
```

Structured fields:

| field | value |
| --- | --- |
| `chunk_id` | `chunk_0000070` |
| `cluster_role` | `hard_distractor` |
| `target_facet_id` | `q00001_f2` |
| `distractor_type` | `same_subgroup_wrong_condition` |
| `condition_display` | `ischemic stroke` |
| `subgroup_label` | `patients older than 75` |
| `axis` | `rehab_outcome` |
| `relevance_grade` | `0` |

### Background Clinical Cluster

This chunk belongs to a coherent background cluster. It is medically fluent, but outside the query's condition/subgroup/facet structure.

```text
The 66-year-old woman with chronic inflammatory autoimmune disease was treated inpatient for ulcerative colitis flare, with frequent stools, cramping, and worsening inflammatory bowel disease activity on presentation. For treatment duration, the hospital treatment interval used infliximab for 7 days; the team documented reduced stool frequency and improved abdominal pain before transition out of the hospital. The plan emphasized gastroenterology follow-up after flare-directed therapy.
```

Structured fields:

| field | value |
| --- | --- |
| `chunk_id` | `chunk_0000094` |
| `cluster_role` | `background_outlier` |
| `distractor_type` | `background_clinical_cluster` |
| `condition_display` | `ulcerative colitis flare` |
| `subgroup_label` | `patients with autoimmune disease` |
| `axis` | `treatment_duration` |
| `relevance_grade` | `0` |

## Second Real Query: `q01548`

The first `outcome_synthesis` query in this run is `q01548`.

```text
Among patients diagnosed with encephalitis or myelitis, compare therapy-course length and discharge rehabilitation status for patients older than 75 versus patients younger than 50.
```

Its canonical answer is:

```text
For patients older than 75, a standard course, averaging 11.3 days, most often with acyclovir for treatment duration and an inpatient rehab pattern, commonly described as transferred to inpatient rehabilitation for persistent weakness for rehabilitation outcome. For patients younger than 50, a prolonged course, averaging 24.4 days, most often with corticosteroids for treatment duration and a home rehab pattern, commonly described as discharged home with outpatient neurorehabilitation for rehabilitation outcome.
```

Two real gold chunks for that query:

```text
The 84-year-old woman older than 75 was treated inpatient for encephalitis or myelitis, with new neurologic deficits and inflammatory cerebrospinal fluid findings on presentation. For treatment duration, corticosteroids remained in place across a 11-day course; the chart reflected improving mentation and reduced headache before transition out of the hospital. Neurology follow-up was arranged to monitor recovery after completion of the anti-inflammatory or antiviral course.
```

```text
The 89-year-old man older than 75 received inpatient care for encephalitis or myelitis after headache, altered mental status, and lower-extremity weakness. In the final therapy assessment, orientation improved but gait instability still made independent mobility unsafe, and the final rehabilitation assessment documented the rehabilitation outcome as required acute rehabilitation for mobility and cognitive deficits. The plan emphasized supervised strengthening, mobility training, and reassessment before return home.
```

## Qrels Are A Projection Of Memberships

`run_make_qrels()` does not infer relevance from text. It projects the hidden membership labels:

| hidden membership | qrel output |
| --- | --- |
| `is_gold = true` | `relevance_grade = 1`, `support_type = positive` |
| `is_gold = false` and `cluster_role = background_outlier` | `relevance_grade = 0`, `support_type = background_outlier` |
| any other `is_gold = false` row | `relevance_grade = 0`, `support_type = hard_negative` |

For `q00001`, that means:

- `chunk_0000001`, `chunk_0000015`, `chunk_0000029`, and `chunk_0000043` are positive support chunks
- `chunk_0000069` and `chunk_0000070` are hard negatives
- `chunk_0000094` is a background outlier

The qrels remain deterministic because they come from structured fact generation, not from a model judging the rendered text.

## Why This Creates A Coverage Problem

The query is not answered by one relevant chunk. It needs evidence for all four facets:

- subgroup A treatment duration
- subgroup A rehabilitation outcome
- subgroup B treatment duration
- subgroup B rehabilitation outcome

The dominant facet is intentionally overrepresented, so similarity-only top-k can spend too many slots on redundant evidence from one cluster. The complementary facets are still present, so a coverage-oriented selector can improve facet coverage by choosing across clusters. The near-miss negatives keep the problem nontrivial because many non-gold chunks are clinically similar to the query.

That is the intended benchmark geometry: enough redundancy for top-k to become wasteful, enough explicit facet structure for facility-location coverage to help, and enough controlled negatives to prevent the task from collapsing into simple lexical matching.

## Minimal Mental Model

1. The ontology defines legal conditions, subgroups, axes, bins, treatments, and rehab outcomes.
2. Query plans define the four hidden facets before any text exists.
3. Calibration can choose the dominant facet from embedding-space probe evidence.
4. Fact generation creates redundant gold clusters and close non-gold clusters.
5. Template data renders those facts into deterministic clinical text.
6. Query rendering turns the same plan into a natural-language question.
7. Canonical answers summarize hidden gold facts per facet.
8. Qrels project membership labels directly into retrieval relevance.

This is why the dataset remains reproducible while still exposing a real multi-aspect retrieval problem.
