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
    if fact['axis'] == 'treatment_duration':
        body = _render_duration_chunk(fact, rng)
    elif fact['axis'] == 'rehab_outcome':
        body = _render_rehab_chunk(fact, rng)
    else:
        raise ValueError(f'Unsupported axis: {fact["axis"]}')

    return _squash_ws(body)


def _render_duration_chunk(fact: dict[str, Any], rng: Random) -> str:
    patient = _sentence_start(_patient_descriptor(fact))
    condition = fact['condition_display']
    presentation = rng.choice(_condition_presentations(fact['condition_id']))
    response = rng.choice(_condition_status_phrases(fact['condition_id']))
    course_noun = rng.choice(
        ['active treatment course', 'inpatient treatment course', 'documented therapy course']
    )
    duration_phrase = rng.choice(
        [
            f'{fact["treatment"]} was continued for {fact["duration_days"]} days',
            f'the {course_noun} used {fact["treatment"]} for {fact["duration_days"]} days',
            f'clinicians completed {fact["duration_days"]} days of {fact["treatment"]}',
        ]
    )
    close = rng.choice(_duration_closing_sentences(fact['condition_id']))

    return (
        f'{patient} was admitted with {condition}, with {presentation}. '
        f'{_sentence_start(duration_phrase)}, and the record described {response} before discharge. '
        f'{close}'
    )


def _render_rehab_chunk(fact: dict[str, Any], rng: Random) -> str:
    patient = _sentence_start(_patient_descriptor(fact))
    condition = fact['condition_display']
    presentation = rng.choice(_condition_presentations(fact['condition_id']))
    functional_detail = rng.choice(
        _functional_status_phrases(fact['condition_id'], fact['value_bin'])
    )
    transition = rng.choice(
        [
            'By discharge',
            'At discharge',
            'Near the end of the hospitalization',
        ]
    )
    close = rng.choice(_rehab_closing_sentences(fact['condition_id'], fact['value_bin']))

    return (
        f'{patient} was managed for {condition}, with {presentation}. '
        f'{transition}, {functional_detail}; the discharge record described '
        f'{fact["rehab_outcome"]}. {close}'
    )


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


def validate_chunk_text(
    text: str, fact: dict[str, Any], ontology: dict[str, Any]
) -> ChunkValidation:
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
        return (
            _has_age_in_range(text, 76, 120) or 'older than 75' in lower or 'above age 75' in lower
        )
    if subgroup_id == 'age_under_50':
        return (
            _has_age_in_range(text, 18, 49) or 'younger than 50' in lower or 'below age 50' in lower
        )

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
        'obesity': ['obesity', 'obese', 'body mass index'],
        'malignancy': ['malignancy', 'active cancer', 'cancer treatment'],
        'atrial_fibrillation': ['atrial fibrillation', 'af'],
        'chronic_liver_disease': ['chronic liver disease', 'cirrhosis', 'hepatic disease'],
        'dementia': ['dementia', 'cognitive impairment', 'neurocognitive disorder'],
        'frailty': ['frailty', 'frail'],
        'peripheral_vascular_disease': [
            'peripheral vascular disease',
            'peripheral artery disease',
            'pad',
        ],
        'autoimmune_disease': [
            'autoimmune disease',
            'systemic autoimmune disease',
            'inflammatory autoimmune disease',
        ],
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


def _condition_presentations(condition_id: str) -> list[str]:
    return {
        'encephalitis_myelitis': [
            'fever, confusion, and gait change',
            'headache, altered mental status, and lower-extremity weakness',
            'new neurologic deficits and inflammatory cerebrospinal fluid findings',
        ],
        'pneumonia': [
            'hypoxemia, fever, and productive cough',
            'new oxygen requirement and bibasilar infiltrates',
            'dyspnea, leukocytosis, and radiographic consolidation',
        ],
        'ischemic_stroke': [
            'acute dysarthria and unilateral weakness',
            'new focal neurologic deficits on arrival',
            'hemiparesis with imaging consistent with acute infarct',
        ],
        'heart_failure': [
            'volume overload, edema, and exertional dyspnea',
            'pulmonary congestion and elevated filling pressures',
            'worsening orthopnea with lower-extremity edema',
        ],
    }.get(condition_id, ['acute symptoms requiring inpatient management'])


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


def _duration_closing_sentences(condition_id: str) -> list[str]:
    return {
        'encephalitis_myelitis': [
            'Neurology follow-up was arranged to monitor recovery after completion of the anti-inflammatory or antiviral course.',
            'The discharge medication list matched the completed neurologic treatment plan and outpatient neurology follow-up.',
            'Repeat examination was stable, and the team documented no escalation beyond the completed inpatient course.',
        ],
        'pneumonia': [
            'The discharge medication list matched the completed antimicrobial plan and outpatient pulmonary follow-up.',
            'Respiratory status was stable, and no additional inpatient antibiotic escalation was documented.',
            'The plan emphasized completion of antimicrobial management and reassessment by the primary clinician.',
        ],
        'ischemic_stroke': [
            'The discharge medication list emphasized vascular prevention and outpatient neurology follow-up.',
            'Repeat examination was stable, and the team documented no extension of the acute infarct.',
            'The plan focused on secondary stroke prevention and close neurologic reassessment.',
        ],
        'heart_failure': [
            'The discharge medication list reflected the completed decongestive plan and cardiology follow-up.',
            'Volume status was stable, and the team documented no further inpatient escalation.',
            'The plan emphasized weight monitoring, medication adjustment, and early outpatient reassessment.',
        ],
    }.get(condition_id, ['Follow-up with the treating service was arranged.'])


def _functional_status_phrases(condition_id: str, value_bin: str) -> list[str]:
    phrases = {
        'encephalitis_myelitis': {
            'home_rehab': [
                'orientation improved and gait was safe with supervised exercises',
                'headache resolved and mild balance deficits could be managed with supervision',
                'mental status returned near baseline with residual fatigue',
            ],
            'inpatient_rehab': [
                'orientation improved but gait instability still made independent mobility unsafe',
                'weakness persisted despite improved fever and stable neurologic checks',
                'cognitive slowing and balance deficits still limited safe transfers',
            ],
            'persistent_deficit': [
                'focal weakness and impaired balance remained prominent',
                'confusion improved only partially and mobility stayed limited',
                'residual neurologic deficits were still documented on the final examination',
            ],
        },
        'pneumonia': {
            'home_rehab': [
                'oxygen needs improved and endurance was adequate for supervised activity',
                'cough and fever improved, but conditioning remained below baseline',
                'breathing was stable on room air with mild residual weakness',
            ],
            'inpatient_rehab': [
                'respiratory status improved but deconditioning limited transfers',
                'oxygenation stabilized while weakness still prevented independent ambulation',
                'fatigue after respiratory illness made self-care unsafe',
            ],
            'persistent_deficit': [
                'exertional dyspnea persisted despite improvement in fever',
                'oxygen requirement remained a barrier to baseline activity',
                'exercise tolerance stayed reduced at the end of the stay',
            ],
        },
        'ischemic_stroke': {
            'home_rehab': [
                'speech improved and gait was safe with outpatient support',
                'mild dysarthria persisted but transfers were independent',
                'strength improved enough for home discharge planning',
            ],
            'inpatient_rehab': [
                'hemiparesis still limited transfers and gait safety',
                'aphasia improved but mobility deficits required supervised practice',
                'motor deficits remained too significant for independent self-care',
            ],
            'persistent_deficit': [
                'hemiparesis remained prominent on the final neurologic examination',
                'aphasia and mobility impairment persisted',
                'residual focal deficits continued to limit safe ambulation',
            ],
        },
        'heart_failure': {
            'home_rehab': [
                'dyspnea improved and walking tolerance was adequate for a supervised home plan',
                'edema decreased and activity tolerance was improving',
                'volume status stabilized with residual fatigue',
            ],
            'inpatient_rehab': [
                'deconditioning remained marked despite improved volume status',
                'fatigue and poor endurance still limited transfers',
                'breathing improved, but mobility remained unsafe without supervised strengthening',
            ],
            'persistent_deficit': [
                'exertional dyspnea continued to limit hallway ambulation',
                'fatigue and mobility limitation remained near discharge',
                'residual congestion symptoms kept exercise tolerance reduced',
            ],
        },
    }
    return phrases.get(condition_id, {}).get(
        value_bin, ['functional limitations remained documented']
    )


def _rehab_closing_sentences(condition_id: str, value_bin: str) -> list[str]:
    if value_bin == 'home_rehab':
        return [
            'The plan emphasized caregiver teaching, home safety, and close outpatient reassessment.',
            'The patient left with clear activity precautions and follow-up for functional recovery.',
            'The team documented a stable medical condition with continued recovery outside the hospital.',
        ]
    if value_bin == 'inpatient_rehab':
        return [
            'The plan emphasized supervised strengthening, mobility training, and reassessment before return home.',
            'The team documented a need for daily therapy intensity after medical stabilization.',
            'Transfer planning focused on fall prevention, endurance, and recovery of daily activities.',
        ]
    return {
        'encephalitis_myelitis': [
            'Neurology follow-up was arranged because recovery remained incomplete.',
            'The plan emphasized monitoring residual neurologic deficits after discharge.',
        ],
        'pneumonia': [
            'Pulmonary follow-up was arranged because respiratory recovery remained incomplete.',
            'The plan emphasized monitoring oxygen needs and gradual return of endurance.',
        ],
        'ischemic_stroke': [
            'Neurology follow-up was arranged because focal deficits remained functionally important.',
            'The plan emphasized continued monitoring of neurologic recovery after discharge.',
        ],
        'heart_failure': [
            'Cardiology follow-up was arranged because functional recovery remained limited.',
            'The plan emphasized symptom monitoring and gradual activity advancement.',
        ],
    }.get(condition_id, ['Follow-up was arranged because recovery remained incomplete.'])


def _squash_ws(text: str) -> str:
    return re.sub(r'\s+', ' ', text).strip()
