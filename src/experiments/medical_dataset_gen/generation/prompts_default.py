import inspect
from typing import Any


class MedicalDatasetGenDefaultPrompts:
    chunk_generation_prompt_id = 'chunk_generation_v1'
    chunk_rewrite_prompt_id = 'chunk_rewrite_v1'

    chunk_generation_system = inspect.cleandoc("""
        You write concise, realistic synthetic clinical note fragments for retrieval evaluation.
    """)
    chunk_rewrite_system = inspect.cleandoc("""
        You rewrite synthetic clinical evidence chunks into more natural clinical prose while preserving the provided facts exactly.
    """)

    @staticmethod
    def chunk_generation_prompt(
        fact: dict[str, Any],
        ontology: dict[str, Any],
        patient_descriptor: str,
        forbidden_terms: list[str],
        min_words: int,
        max_words: int,
        revision_feedback: str | None = None,
    ) -> str:
        condition = fact['condition_display']
        condition_terms = ', '.join(ontology['conditions'][fact['condition_id']]['terms'][:3])
        forbidden = ', '.join(forbidden_terms)

        if fact['axis'] == 'treatment_duration':
            evidence = inspect.cleandoc(f"""
                Evidence focus: treatment course length.
                Must include: {condition}; {patient_descriptor}; {fact['treatment']}; {fact['duration_days']} days.
                Must not include: discharge destination, rehabilitation placement, outpatient therapy,
                home health, home nursing, inpatient rehab, persistent functional deficits, or rehab outcome.
            """)
            examples = inspect.cleandoc("""
                Good examples:
                - The 82-year-old woman was treated for encephalitis with acyclovir for 14 days. Fever resolved and orientation improved before discharge, and neurology follow-up was arranged.
                - The 79-year-old man with uncomplicated type 2 diabetes was managed for myelitis with intravenous methylprednisolone for 5 days, followed by an oral taper. Lower-extremity strength improved modestly before discharge.
                - The 43-year-old woman with immunosuppression was treated for encephalitis with acyclovir for 21 days. Headache lessened and mental status normalized by discharge.
            """)
        else:
            evidence = inspect.cleandoc(f"""
                Evidence focus: discharge functional or rehabilitation status.
                Must include: {condition}; {patient_descriptor}; {fact['rehab_outcome']}.
                Must not include: number of treatment days, therapy-course length, duration labels.
            """)
            examples = inspect.cleandoc("""
                Good examples:
                - The 84-year-old woman treated for encephalitis was discharged to inpatient rehabilitation after persistent gait instability and impaired balance limited safe ambulation.
                - The 67-year-old man with uncomplicated type 2 diabetes was hospitalized for myelitis. At discharge, he required home physical therapy because residual leg weakness continued to limit transfers.
                - The 48-year-old woman with chronic kidney disease was managed for encephalitis and was cleared for discharge home with visiting nursing after cognition returned to baseline and mobility remained intact.
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

            {examples}

            Bad examples to avoid:
            - Any note-section heading followed by a colon before the clinical sentence.
            - The structured cohort descriptor is patients older than 75.
            - The excerpt supports the duration facet only.
            - Index terms: encephalitis; treatment duration; rehabilitation outcome.
            - This synthetic note belongs to admission adm_q00001.

            Constraints:
            - {min_words} to {max_words} words.
            - Sound like concise discharge-summary prose.
            - Start directly with the patient and clinical content, not with a note-section heading.
            - Mention the subgroup-defining evidence early in the fragment.
            - Make the requested evidence clinically explicit and easy to retrieve semantically.
            - Keep the fragment focused on the requested evidence focus.
            - Do not use these words or phrases: {forbidden}.
            - Do not mention qrels, clusters, facets, labels, benchmark design, or document IDs.
            - Do not add facts beyond the required condition, patient descriptor, treatment duration, or rehab status.

            {revision_block}
        """)

    @staticmethod
    def chunk_rewrite_prompt(
        fact: dict[str, Any],
        draft_text: str,
        patient_descriptor: str,
        required_facts: list[str],
        forbidden_facts: list[str],
        min_words: int,
        max_words: int,
        revision_feedback: str | None = None,
    ) -> str:
        style_label = fact['note_style'].replace('_', ' ')
        required_block = '\n'.join(f'- {fact_line}' for fact_line in required_facts)
        forbidden_block = '\n'.join(f'- {fact_line}' for fact_line in forbidden_facts)
        facet_focus = (
            f'treatment duration with {fact["treatment"]} for {fact["duration_days"]} days'
            if fact['axis'] == 'treatment_duration'
            else f'rehabilitation or discharge functional outcome: {fact["rehab_outcome"]}'
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
            - Keep the note deidentified: no names, dates, addresses, phone numbers, medical record numbers, or real identifiers.
            - Keep the evidence explicit enough that dense retrieval should still match the note to the query.
            - Do not mention benchmark construction, facets, clusters, source rows, prompt instructions, or IDs.
            - Do not introduce any diagnosis, treatment, duration, rehabilitation outcome, demographic detail, comorbidity, lab result, imaging result, medication, or disposition detail that is not listed in the required facts.

            Style target:
            - Clinical note style: {style_label}
            - Length: {min_words}-{max_words} words
            - Clinical focus: {facet_focus}
            - Patient anchor to preserve: {patient_descriptor}

            Required facts to preserve:
            {required_block}

            Forbidden facts or mentions:
            {forbidden_block}

            Draft note:
            {draft_text}

            {revision_block}
        """)

    @staticmethod
    def query_paraphrase_prompt(query_text: str, plan: dict[str, Any]) -> str:
        return inspect.cleandoc(f"""
            Paraphrase this synthetic clinical benchmark query in one sentence.
            Keep these exact labels present:
            - {plan['condition_display']}
            - {plan['subgroup_a_label']}
            - {plan['subgroup_b_label']}

            Query:
            {query_text}

            Return only the paraphrased query.
        """)
