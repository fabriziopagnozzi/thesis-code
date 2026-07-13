from __future__ import annotations

import hashlib
import inspect

from experiments.medical_dataset_gen.schemas.generation_schemas import (
    ClinicalFact,
    MedicalOntology,
    QueryPlan,
    parse_axis_payload,
)


class MedicalDatasetGenDefaultPrompts:
    chunk_generation_prompt_id = 'chunk_generation_v4_axis_payload'
    chunk_rewrite_prompt_id = 'chunk_rewrite_v3_lexical_artifact_control'

    chunk_generation_system = inspect.cleandoc("""
        You write concise, realistic synthetic clinical note fragments for retrieval evaluation.
    """)
    chunk_rewrite_system = inspect.cleandoc("""
        You rewrite synthetic clinical evidence chunks into natural clinical prose while preserving the provided facts exactly and avoiding benchmark-style wording.
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

        payload = parse_axis_payload(fact.axis_payload_json)
        required_lines = '\n'.join(f'- {item}' for item in fact.must_mention)
        evidence = inspect.cleandoc(f"""
            Evidence focus: {fact.axis.replace('_', ' ')}.
            Required facts to preserve exactly:
            - Condition: {condition}
            - Patient anchor: {patient_descriptor}
            {required_lines}
            - Typed axis payload: {payload.model_dump_json()}

            Excluded content:
            - Do not add evidence from any other clinical axis.

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
            - Do not add facts beyond the required condition, patient descriptor, target-axis category, and typed target-axis payload.

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
            Preserve the required clinical facts exactly in meaning. Improve wording and sentence flow, but do not change the underlying evidence.

            Output contract:
            - Return exactly one paragraph and nothing else.
            - No headings, bullets, JSON, quotation marks around the paragraph, or commentary.
            - Do not mention benchmark construction, facets, clusters, axes, labels, source rows, prompt instructions, or IDs.
            - Do not introduce any evidence besides what's already included.

            Style target:
            - Clinical note style: {style_label}
            - Length: {min_words}-{max_words} words
            - Patient anchor to preserve: {patient_descriptor}

            Lexical artifact control:
            - Do not copy the draft wording except for required clinical entities, cohort descriptors, and clinically necessary values.
            - Avoid abstract category labels such as treatment duration, rehabilitation outcome, complication burden, resource utilization, mortality risk, or follow-up intensity unless the wording is clinically unavoidable.
            - Express the evidence through concrete clinical observations, care decisions, course details, outcomes, or follow-up plans rather than naming the evidence category.
            - Vary sentence openings, verbs, clause order, and temporal framing across rewrites.

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
        ontology: MedicalOntology,
    ) -> str:
        return inspect.cleandoc(f"""
            Paraphrase this synthetic clinical benchmark query in one sentence.
            Keep these exact labels present:
            - {plan.condition_display}
            - {plan.subgroup_a_label}
            - {plan.subgroup_b_label}
            - Emphasized topic: {ontology.clinical_axes[plan.primary_axis].label}
            - Context topic: {ontology.clinical_axes[plan.secondary_axis].label}

            Preserve the hierarchy: discuss the emphasized topic first and in more detail.
            Keep the context topic explicit but subordinate. Do not reverse or equalize them.
            Scope the condition before the cohort contrast so it clearly applies to both groups.
            Avoid attachment ambiguities such as "compare A and B with C".
            Do not use the phrases "primary endpoint", "secondary endpoint", or "primary comparison".

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
        'progress_note': (
            'Write as a concise inpatient progress note: active problem, salient interval course, '
            'and the requested evidence.'
        ),
        'discharge_summary': (
            'Write as a problem-focused discharge summary: compact diagnosis statement, '
            'supporting course evidence, and discharge-relevant status.'
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
