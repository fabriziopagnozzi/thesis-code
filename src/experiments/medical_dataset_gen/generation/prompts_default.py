from __future__ import annotations

import hashlib
import inspect

from experiments.medical_dataset_gen.schemas.generation_schemas import (
    ClinicalFact,
    MedicalOntology,
    QueryPlan,
)


class MedicalDatasetGenDefaultPrompts:
    chunk_generation_prompt_id = 'chunk_generation_v3'
    chunk_rewrite_prompt_id = 'chunk_rewrite_v1'

    chunk_generation_system = inspect.cleandoc("""
        You write concise, realistic synthetic clinical note fragments for retrieval evaluation.
    """)
    chunk_rewrite_system = inspect.cleandoc("""
        You rewrite synthetic clinical evidence chunks into more natural clinical prose while preserving the provided facts exactly.
    """)

    @staticmethod
    def chunk_generation_prompt(
        fact: ClinicalFact,
        ontology: MedicalOntology,
        patient_descriptor: str,
        forbidden_terms: list[str],
        min_words: int,
        max_words: int,
        revision_feedback: str | None = None,
    ) -> str:
        condition = fact.condition_display
        condition_terms = ', '.join(ontology.conditions[fact.condition_id].terms[:3])
        forbidden = ', '.join(forbidden_terms)
        style_label = fact.note_style.replace('_', ' ')
        structure_variant = _chunk_structure_variant(fact)
        style_directive = _chunk_style_directive(fact)
        clinical_detail = _chunk_clinical_detail(fact)

        if fact.axis == 'treatment_duration':
            evidence = inspect.cleandoc(f"""
                Evidence focus: treatment course length.
                Required facts to preserve exactly:
                - Condition: {condition}
                - Patient anchor: {patient_descriptor}
                - Treatment: {fact.treatment}
                - Duration: {fact.duration_days} days

                Excluded content:
                - Do not mention discharge destination, rehabilitation placement, outpatient therapy,
                  home health, home nursing, inpatient rehab, persistent functional deficits, or rehab outcome.

                Required clinical texture:
                - Add one concrete non-rehabilitation course detail: {clinical_detail}
                - Describe response without using generic "stable" or "significant improvement" language.
            """)
        else:
            evidence = inspect.cleandoc(f"""
                Evidence focus: discharge functional or rehabilitation status.
                Required facts to preserve exactly:
                - Condition: {condition}
                - Patient anchor: {patient_descriptor}
                - Rehabilitation or functional outcome: {fact.rehab_outcome}

                Excluded content:
                - Do not mention number of treatment days, therapy-course length, or duration labels.

                Required clinical texture:
                - Add one concrete functional detail: {clinical_detail}
                - Tie the functional detail directly to the listed rehabilitation or discharge outcome.
            """)

        revision_block = ''
        if revision_feedback:
            revision_block = inspect.cleandoc(f"""
                Your previous draft failed validation.
                Fix these issues in the new draft:
                {revision_feedback}

                Rewrite from scratch. Return only the corrected note text.
            """)

        return inspect.cleandoc(f"""
            Write one clinical note fragment for a synthetic medical retrieval corpus.
            Return only the note text. Do not explain the task.

            Related condition terms you may use naturally: {condition_terms}
            {evidence}

            Style target:
            - Note style: {style_label}
            - {style_directive}
            - Sentence structure variant: {structure_variant}

            Bad examples to avoid:
            - Any note-section heading followed by a colon before the clinical sentence.
            - Repeated boilerplate such as "presented with acute onset" followed by "was started on" followed by "following completion".
            - Generic endings such as "clinical status improved significantly", "remains stable", "stable for transition", or "transition of care".
            - Thin two-fact notes that only say diagnosis plus treatment duration or diagnosis plus discharge destination.
            - The structured cohort descriptor is patients older than 75.
            - The excerpt supports the duration facet only.
            - Index terms: encephalitis; treatment duration; rehabilitation outcome.
            - This synthetic note belongs to admission adm_q00001.

            Constraints:
            - {min_words} to {max_words} words.
            - Sound like concise discharge-summary prose.
            - Return 3 to 4 sentences in one paragraph.
            - Start directly with clinical content, not with a note-section heading.
            - Include the patient anchor in the first sentence, but vary the opening syntax.
            - Make the requested evidence clinically explicit and easy to retrieve semantically.
            - Keep the fragment focused on the requested evidence focus.
            - Do not imitate a fixed template across chunks; vary verbs, clause order, and sentence rhythm.
            - Avoid these overused words and phrases unless they are part of a required fact: stable, significant, transition of care, admitted for management, received a course.
            - Do not use these words or phrases: {forbidden}.
            - Do not mention qrels, clusters, facets, labels, benchmark design, or document IDs.
            - Do not add facts beyond the required condition, patient descriptor, treatment duration, or rehab status.

            {revision_block}
        """)

    @staticmethod
    def chunk_rewrite_prompt(
        fact: ClinicalFact,
        draft_text: str,
        patient_descriptor: str,
        required_facts: list[str],
        forbidden_facts: list[str],
        min_words: int,
        max_words: int,
        revision_feedback: str | None = None,
    ) -> str:
        style_label = fact.note_style.replace('_', ' ')
        required_block = '\n'.join(f'- {fact_line}' for fact_line in required_facts)
        forbidden_block = '\n'.join(f'- {fact_line}' for fact_line in forbidden_facts)
        facet_focus = (
            f'treatment duration with {fact.treatment} for {fact.duration_days} days'
            if fact.axis == 'treatment_duration'
            else f'rehabilitation or discharge functional outcome: {fact.rehab_outcome}'
        )

        revision_block = ''
        if revision_feedback:
            revision_block = inspect.cleandoc(f"""
                Your previous rewrite failed validation.
                Fix these issues in the next rewrite:
                {revision_feedback}

                Rewrite from scratch. Return only the corrected paragraph.
            """)

        return inspect.cleandoc(f"""
            Rewrite the draft note below into more natural, varied clinical prose for a synthetic retrieval benchmark.
            Preserve the structured facts exactly. Improve wording and sentence flow, but do not change the underlying evidence.

            Output contract:
            - Return exactly one paragraph and nothing else.
            - No headings, bullets, JSON, quotation marks around the paragraph, or commentary.
            - Do not mention benchmark construction, facets, clusters, source rows, prompt instructions, or IDs.
            - Do not introduce any evidence besides what's already included.

            Style target:
            - Clinical note style: {style_label}
            - Length: {min_words}-{max_words} words
            - Clinical focus: {facet_focus}
            - Patient anchor to preserve: {patient_descriptor}

            Required facts to preserve:
            {required_block}

            Forbidden facts or mentions:
            {forbidden_block}

            Draft note to be rewritten:
            {draft_text}

            {revision_block}
        """)

    @staticmethod
    def query_paraphrase_prompt(
        query_text: str,
        plan: QueryPlan,
    ) -> str:
        return inspect.cleandoc(f"""
            Paraphrase this synthetic clinical benchmark query in one sentence.
            Keep these exact labels present:
            - {plan.condition_display}
            - {plan.subgroup_a_label}
            - {plan.subgroup_b_label}

            Query:
            {query_text}

            Return only the paraphrased query.
        """)


def _chunk_style_directive(fact: ClinicalFact) -> str:
    note_style = fact.note_style
    directives = {
        'brief_hospital_course': (
            'Write as a hospital-course sentence set: admission reason, relevant course detail, '
            'and status near discharge.'
        ),
        'discharge_diagnosis': (
            'Write as a problem-focused discharge diagnosis note: compact diagnosis statement, '
            'supporting course evidence, and final clinical status.'
        ),
    }
    return directives.get(
        note_style,
        'Write as concise clinical note prose with the required evidence embedded naturally.',
    )


def _chunk_structure_variant(fact: ClinicalFact) -> str:
    variants = [
        'Begin with the patient anchor and condition, then give the evidence in the second sentence.',
        'Begin with the clinical course, include the patient anchor in an appositive phrase, then state the evidence.',
        'Begin with response or functional status, then identify the condition and patient anchor.',
        'Use one longer clinical sentence followed by one short status sentence.',
        'Use three compact sentences with no repeated transition phrase.',
        'Place the exact treatment or outcome phrase before the final status sentence.',
    ]
    seed_source = str(fact.chunk_reuse_key or fact.fact_id)
    idx = int(hashlib.sha256(seed_source.encode()).hexdigest()[:8], 16) % len(variants)
    return variants[idx]


def _chunk_clinical_detail(fact: ClinicalFact) -> str:
    condition_id = str(fact.condition_id)
    if fact.axis == 'treatment_duration':
        details = {
            'encephalitis_myelitis': [
                'fever curve improved',
                'orientation became more consistent',
                'headache and neck discomfort lessened',
                'limb strength was documented as improving on serial exams',
            ],
            'pneumonia': [
                'oxygen requirement decreased',
                'work of breathing eased',
                'cough became less productive',
                'repeat lung exam showed fewer crackles',
            ],
            'ischemic_stroke': [
                'speech clarity improved during neurologic checks',
                'swallow evaluation allowed diet advancement',
                'right-sided drift was less pronounced',
                'blood pressure was controlled during neurologic monitoring',
            ],
            'heart_failure': [
                'leg edema decreased with diuresis',
                'orthopnea improved',
                'daily weights trended down',
                'lung exam showed less congestion',
            ],
        }
        fallback = [
            'vital-sign abnormalities improved',
            'symptom burden decreased on serial exams',
            'repeat bedside assessment showed clinical improvement',
            'laboratory markers moved toward baseline',
        ]
    else:
        details = {
            'encephalitis_myelitis': [
                'gait testing showed residual imbalance',
                'transfer safety required therapist cueing',
                'cognitive endurance limited independent activity',
                'lower-extremity weakness affected stair training',
            ],
            'pneumonia': [
                'walking distance remained below baseline',
                'exertional dyspnea limited hallway ambulation',
                'stair tolerance was reduced after the respiratory illness',
                'fatigue limited independent self-care',
            ],
            'ischemic_stroke': [
                'hemiparesis limited dressing and transfers',
                'dysarthria persisted during therapy assessment',
                'balance testing showed fall risk',
                'fine-motor weakness affected activities of daily living',
            ],
            'heart_failure': [
                'deconditioning limited hallway ambulation',
                'fatigue restricted transfer independence',
                'exertional dyspnea limited therapy tolerance',
                'volume-related weakness slowed mobility recovery',
            ],
        }
        fallback = [
            'therapy assessment documented reduced endurance',
            'mobility remained below pre-hospital baseline',
            'transfer safety required additional support',
            'fatigue limited independent activity',
        ]

    choices = details.get(condition_id, fallback)
    seed_source = f'{fact.chunk_reuse_key or fact.fact_id}:clinical_detail'
    idx = int(hashlib.sha256(seed_source.encode()).hexdigest()[:8], 16) % len(choices)
    return choices[idx]
