import re
from dataclasses import dataclass
from random import Random
from typing import Any

from experiments.medical_dataset_gen.generation.prompts_default import (
    MedicalDatasetGenDefaultPrompts,
)
from helpers.ollama_client import generate

_HIDDEN_BENCHMARK_TERMS = [
    'admission adm_',
    'benchmark',
    'chunk',
    'cohort descriptor',
    'distractor',
    'facet',
    'gold',
    'index terms',
    'qrel',
    'source query',
    'structured cohort',
    'synthetic',
    'target query',
]

_DURATION_RE = re.compile(r'\b\d+\s*[- ]?(?:day|days)\b', re.IGNORECASE)
_AGE_RE = re.compile(r'\b(\d{2,3})\s*[- ]year[- ]old\b', re.IGNORECASE)
_SECTION_HEADER_RE = re.compile(
    r'^\s*(?:brief hospital course|hospital course|discharge summary|'
    r'discharge diagnosis|clinical summary)\s*:\s*',
    re.IGNORECASE,
)


@dataclass
class ChunkValidation:
    hard_errors: list[str]
    soft_warnings: list[str]


def render_chunk_text(fact: dict[str, Any], ontology: dict[str, Any], rng: Random) -> str:
    """Deterministic clinical fallback used when LLM generation is disabled or rejected."""
    condition_display = fact['condition_display']
    axis = fact['axis']
    patient = _patient_descriptor(fact)
    patient_start = _sentence_start(patient)

    if axis == 'treatment_duration':
        status = rng.choice(_condition_status_phrases(fact['condition_id']))
        body = (
            f'{patient_start} was treated for {condition_display}. '
            f'Treatment with {fact["treatment"]} continued for {fact["duration_days"]} days, '
            f'with {status} before discharge.'
        )
    elif axis == 'rehab_outcome':
        transition = rng.choice([
            'At discharge',
            'By the time of discharge',
            'Near discharge',
        ])
        body = (
            f'{patient_start} was managed for {condition_display}. '
            f'{transition}, the record described {fact["rehab_outcome"]}.'
        )
    else:
        raise ValueError(f'Unsupported axis: {axis}')

    followup = rng.choice([
        'Follow-up with the treating service was arranged.',
        'Vital signs were stable at discharge, and follow-up with the treating service was arranged.',
        'The discharge plan focused on the acute illness and immediate recovery.',
    ])
    return _squash_ws(f'{body} {followup}')


def maybe_generate_chunk_text(
    fallback_text: str,
    fact: dict[str, Any],
    ontology: dict[str, Any],
    llm_name: str,
    use_llm: bool,
    temperature: float,
    num_ctx: int,
    chunk_min_words: int,
    chunk_max_words: int,
    revision_feedback: str | None = None,
) -> str:
    if not use_llm:
        return fallback_text

    prompt = MedicalDatasetGenDefaultPrompts.chunk_generation_prompt(
        fact=fact,
        ontology=ontology,
        patient_descriptor=_patient_descriptor(fact),
        forbidden_terms=_HIDDEN_BENCHMARK_TERMS,
        min_words=chunk_min_words,
        max_words=chunk_max_words,
        revision_feedback=revision_feedback,
    )
    generated = generate(
        prompt,
        model=llm_name,
        system=MedicalDatasetGenDefaultPrompts.chunk_generation_system,
        temperature=temperature,
        num_ctx=num_ctx,
    )
    cleaned = _cleanup_generated_text(generated)
    return cleaned


def render_query(plan: dict[str, Any], ontology: dict[str, Any]) -> str:
    condition = plan['condition_display']
    a = plan['subgroup_a_label']
    b = plan['subgroup_b_label']

    if plan['query_type'] == 'outcome_synthesis':
        return (
            f'Among patients diagnosed with {condition}, compare therapy-course length and '
            f'discharge rehabilitation status for {a} versus {b}.'
        )

    duration = ontology['clinical_axes']['treatment_duration']['label']
    rehab = ontology['clinical_axes']['rehab_outcome']['label']
    return (
        f'For patients diagnosed with {condition}, how do {duration} and {rehab} differ '
        f'between {a} and {b}?'
    )


def maybe_paraphrase_query(
    query_text: str,
    plan: dict[str, Any],
    llm_name: str,
    use_llm: bool,
    temperature: float,
    num_ctx: int,
) -> str:
    if not use_llm:
        return query_text

    prompt = MedicalDatasetGenDefaultPrompts.query_paraphrase_prompt(query_text, plan)
    paraphrase = generate(prompt, model=llm_name, temperature=temperature, num_ctx=num_ctx).strip()
    required = [plan['condition_display'], plan['subgroup_a_label'], plan['subgroup_b_label']]
    if paraphrase and all(label.lower() in paraphrase.lower() for label in required):
        return paraphrase
    return query_text


def canonical_answer(plan: dict[str, Any], facet_summaries: dict[str, str]) -> str:
    a = plan['subgroup_a_label']
    b = plan['subgroup_b_label']

    return (
        f'For {a}, the synthetic corpus shows {facet_summaries[facets_by(plan, a, "treatment_duration")]} '
        f'for treatment duration and {facet_summaries[facets_by(plan, a, "rehab_outcome")]} for rehabilitation outcome. '
        f'For {b}, it shows {facet_summaries[facets_by(plan, b, "treatment_duration")]} for treatment duration '
        f'and {facet_summaries[facets_by(plan, b, "rehab_outcome")]} for rehabilitation outcome.'
    )


def facets_by(plan: dict[str, Any], subgroup_label: str, axis: str) -> str:
    for facet in plan['facets']:
        if facet['subgroup_label'] == subgroup_label and facet['axis'] == axis:
            return facet['facet_id']
    raise KeyError((subgroup_label, axis))


def validate_chunk_text(text: str, fact: dict[str, Any], ontology: dict[str, Any]) -> ChunkValidation:
    lower = text.lower()
    hard_errors: list[str] = []
    soft_warnings: list[str] = []

    for term in _HIDDEN_BENCHMARK_TERMS:
        if term in lower:
            hard_errors.append(f'contains hidden benchmark term: {term}')
    if _SECTION_HEADER_RE.match(text):
        hard_errors.append('contains leading note-section header')

    if not _contains_condition(text, fact, ontology):
        hard_errors.append(f'missing condition evidence: {fact["condition_display"]}')
    if not _contains_subgroup_evidence(text, fact, ontology):
        hard_errors.append(f'missing subgroup evidence: {fact["subgroup_label"]}')

    if fact['axis'] == 'treatment_duration':
        duration = str(fact['duration_days'])
        treatment = str(fact['treatment']).lower()
        if duration not in lower:
            hard_errors.append(f'missing treatment duration days: {duration}')
        if treatment and treatment not in lower:
            hard_errors.append(f'missing treatment: {fact["treatment"]}')
        extra_treatments = _extra_condition_treatments(text, fact, ontology)
        if extra_treatments:
            soft_warnings.append(f'contains extra treatment(s): {", ".join(extra_treatments)}')
        if _contains_rehab_language(text):
            soft_warnings.append('duration chunk contains rehabilitation-outcome language')
    elif fact['axis'] == 'rehab_outcome':
        has_exact_rehab = _contains_exact_rehab_outcome(text, fact)
        has_bin_rehab = _contains_rehab_bin_evidence(text, fact, ontology)
        if not has_bin_rehab:
            hard_errors.append(f'missing rehabilitation outcome evidence: {fact["rehab_outcome"]}')
        elif not has_exact_rehab:
            soft_warnings.append(f'missing exact rehabilitation phrase: {fact["rehab_outcome"]}')
        if _DURATION_RE.search(text):
            soft_warnings.append('rehab chunk contains explicit duration days')
    else:
        hard_errors.append(f'unsupported axis: {fact["axis"]}')

    return ChunkValidation(hard_errors=hard_errors, soft_warnings=soft_warnings)


def _cleanup_generated_text(text: str) -> str:
    text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL).strip()
    fence = re.search(r'```(?:text)?\s*(.*?)```', text, flags=re.DOTALL)
    if fence:
        text = fence.group(1).strip()
    text = text.strip().strip('"').strip("'").strip()
    lines = [line.strip('- ').strip() for line in text.splitlines() if line.strip()]
    if len(lines) > 1:
        text = ' '.join(lines)
    return _squash_ws(_SECTION_HEADER_RE.sub('', text))


def _patient_descriptor(fact: dict[str, Any]) -> str:
    age = int(fact['patient_age'])
    noun = 'woman' if fact['patient_sex'] == 'female' else 'man'
    phrase = fact['clinical_subgroup_phrase']
    if fact['subgroup_axis'] == 'demographic':
        return f'the {age}-year-old {noun}'
    return f'the {age}-year-old {noun} with {phrase}'


def _sentence_start(text: str) -> str:
    return text[:1].upper() + text[1:]


def _contains_condition(text: str, fact: dict[str, Any], ontology: dict[str, Any]) -> bool:
    lower = text.lower()
    if fact['condition_display'].lower() in lower:
        return True
    condition = ontology['conditions'][fact['condition_id']]
    return any(str(term).lower() in lower for term in condition['terms'][:3])


def _contains_subgroup_evidence(text: str, fact: dict[str, Any], ontology: dict[str, Any]) -> bool:
    lower = text.lower()
    subgroup_id = fact['subgroup_id']

    if subgroup_id == 'age_over_75':
        return _has_age_in_range(text, 76, 120) or 'older than 75' in lower or 'above age 75' in lower
    if subgroup_id == 'age_under_50':
        return _has_age_in_range(text, 18, 49) or 'younger than 50' in lower or 'below age 50' in lower

    phrase = str(fact['clinical_subgroup_phrase']).lower()
    if phrase and phrase in lower:
        return True

    subgroup = ontology['subgroups'].get(subgroup_id, {})
    aliases = [str(alias).lower() for alias in subgroup.get('aliases', [])]
    if any(alias in lower for alias in aliases):
        return True

    subgroup_terms = {
        'uncomplicated_diabetes': ['uncomplicated diabetes', 'type 2 diabetes', 'diabetes without'],
        'chronic_kidney_disease': ['chronic kidney disease', 'ckd'],
        'copd': ['copd', 'chronic obstructive pulmonary disease'],
        'immunosuppression': ['immunosuppression', 'immunosuppressed'],
    }
    return any(term in lower for term in subgroup_terms.get(subgroup_id, []))


def _has_age_in_range(text: str, low: int, high: int) -> bool:
    return any(low <= int(match.group(1)) <= high for match in _AGE_RE.finditer(text))


def _contains_rehab_language(text: str) -> bool:
    lower = text.lower()
    terms = [
        'acute rehabilitation',
        'cleared for home',
        'discharged home',
        'discharged to home',
        'home health',
        'home nursing',
        'home physical therapy',
        'home therapy',
        'home with',
        'inpatient rehabilitation',
        'outpatient neurorehabilitation',
        'outpatient therapy',
        'rehab',
        'rehabilitation',
        'therapy was arranged',
        'transferred to rehabilitation',
        'visiting nursing',
    ]
    return any(term in lower for term in terms)


def _contains_exact_rehab_outcome(text: str, fact: dict[str, Any]) -> bool:
    lower = text.lower()
    outcome = str(fact['rehab_outcome']).lower()
    return bool(outcome and outcome in lower)


def _contains_rehab_bin_evidence(
    text: str,
    fact: dict[str, Any],
    ontology: dict[str, Any],
) -> bool:
    lower = text.lower()
    condition = ontology['conditions'][fact['condition_id']]
    value_bin = fact['value_bin']

    for phrase in condition['rehab_outcomes'].get(value_bin, []):
        if str(phrase).lower() in lower:
            return True

    bin_terms = {
        'home_rehab': [
            'breathing exercises',
            'discharged home',
            'home physical therapy',
            'home therapy',
            'home with',
            'outpatient',
            'visiting nursing',
        ],
        'inpatient_rehab': [
            'acute rehabilitation',
            'inpatient rehabilitation',
            'required inpatient therapy',
            'transferred to rehabilitation',
        ],
        'persistent_deficit': [
            'continued exertional limitation',
            'impaired balance',
            'ongoing',
            'persistent',
            'reduced exercise tolerance',
            'residual',
        ],
    }
    return any(term in lower for term in bin_terms.get(value_bin, []))


def _extra_condition_treatments(
    text: str,
    fact: dict[str, Any],
    ontology: dict[str, Any],
) -> list[str]:
    lower = text.lower()
    expected = str(fact['treatment']).lower()
    extras = []
    for treatment in _duration_treatment_terms(fact, ontology):
        treatment_lower = str(treatment).lower()
        if treatment_lower != expected and treatment_lower in lower:
            extras.append(str(treatment))
    return extras


def _duration_treatment_terms(
    fact: dict[str, Any],
    ontology: dict[str, Any],
) -> list[str]:
    condition = ontology['conditions'][fact['condition_id']]
    treatments = condition.get('duration_treatments') or condition['treatments']
    return [str(treatment) for treatment in treatments]


def _condition_status_phrases(condition_id: str) -> list[str]:
    return {
        'encephalitis_myelitis': [
            'improving mentation and reduced headache',
            'improved strength and stable neurologic checks',
            'resolution of fever and stable neurologic examination',
        ],
        'pneumonia': [
            'improved oxygenation and reduced cough',
            'stable oxygen saturation on room air',
            'improving dyspnea and downtrending fever',
        ],
        'ischemic_stroke': [
            'stable neurologic examination',
            'improved speech clarity',
            'no new neurologic deficits',
        ],
        'heart_failure': [
            'improved volume status',
            'decreased edema and easier breathing',
            'stable renal function after diuresis',
        ],
    }.get(condition_id, ['clinical improvement'])


def _squash_ws(text: str) -> str:
    return re.sub(r'\s+', ' ', text).strip()
