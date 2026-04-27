import inspect


class MimicDefaultPrompts:
    llm_conditions_cleaning_system = inspect.cleandoc("""
		You are a medical coding expert helping build a clinical QA benchmark.

		Given a list of ICD-10 sub-code descriptions sharing the same 3-character prefix, you must:
			1. Produce a single concise clinical label (up to 20 words) representing the group in a semantically meaningful way, focusing on the main condition shared by all sub-labels.
			2. Decide whether this code group is USEFUL for a multi-document clinical QA benchmark.
		Set "keep": false for codes that are administrative, procedural, or too vague to anchor a
		meaningful clinical question requiring multi-source synthesis.

		Examples of codes to EXCLUDE ("keep": false):
		- Intraoperative, postprocedural, and iatrogenic complications (e.g., T-codes, and specific complication blocks like L76, G97, N99)
		- Do-not-resuscitate orders (Z66) and Encounter/screening codes (Z00-Z13)
		- External cause codes (V/W/X/Y)
		- Morphology/histology codes
		- Unspecified injury catch-alls
		- Codes whose group contains only generic 'unspecified' or 'other' entries with no clear clinical entity.
		- Standalone symptom/sign codes (R-codes), UNLESS they represent a life-threatening acute state (e.g., Coma, Shock) or a complex syndrome.
		- Drug poisoning/adverse effect groups (T-codes).

		Prefer the most specific / prevalent clinical entity over generic catch-alls when naming.
	""")

    query_gen_template = inspect.cleandoc("""
		Below are excerpts from real discharge summaries of patients admitted for {condition}.

		<START DISCHARGE SUMMARIES EXAMPLES>
		{chunks_block}
		<END DISCHARGE SUMMARIES EXAMPLES>

		The examples above are provided ONLY so you understand what kinds of clinical information the dataset contains.
		Do NOT generate a question about these specific patients or these specific notes.
		Using the above as background context only, generate ONE generic question about patients with {condition}
		that asks for clinical patterns across multiple aspects of care.
		The question must be answerable by reading across many different patients in the dataset, not just the examples above.
		The question must address EXACTLY the following aspects:
		{modifier_list}

		Good examples:
		- "For patients admitted with sepsis, how does antibiotic management differ between those with chronic kidney disease and elderly patients over 75?"
		- "For patients with acute myocardial infarction, how do discharge medications and follow-up plans differ between those with chronic kidney disease and elderly patients?"
		- "In patients hospitalized with pneumonia, how does length of stay and treatment intensity compare between those with heart failure and elderly patients?"

		Requirements:
		- Asks about observable patterns in the data, not what "should" be done
		- Can be answered by reading across many discharge notes, not from a textbook

		Keep the question to 1-2 sentences. Return ONLY the question, nothing else.
	""")

    query_gen_template_alternative = inspect.cleandoc("""
		Below are excerpts from real discharge summaries of patients admitted for {condition}.
		<START DISCHARGE SUMMARIES EXAMPLES>
		{chunks_block}
		<END DISCHARGE SUMMARIES EXAMPLES>
		Scan the examples above to identify specific clinical artifacts that are actually documented:
		medication names, drug classes, procedures, lab values, timelines, complications, or discharge plans.
		Do NOT generate a question about these specific patients.

		Using these specifics as inspiration, generate ONE clinical question about {condition} that:
		1. Addresses ALL of the following aspects: {modifier_list}
		2. Can be answered by reading ACROSS many different patient discharge notes - not from a textbook

		The question MUST be anchored to 1-2 concrete clinical dimensions. Choose from:
		- specific medication classes, agents, or dose adjustments
		- procedure selection, timing, or sequencing
		- lab monitoring targets or physiologic thresholds
		- complication patterns and their temporal development
		- discharge disposition, LOS, or post-discharge care needs
		- care escalation decisions (ICU admission, pressor use, invasive monitoring)

		Requirements:
		- Each aspect listed above must be independently answerable from notes of patients with that characteristic - the two aspects should require DIFFERENT chunks to answer
		- Do NOT use vague terms as the main subject: avoid "management strategies", "clinical approaches", "overall outcomes", or "treatment plans" without specifying WHAT kind
		- Ask about observable, documented patterns - not what guidelines recommend

		Vary the question structure. Valid forms include:
		- "What [specific drug class / procedure / lab finding] characterizes {condition} patients with [aspect A] compared to [aspect B]?"
		- "Among {condition} patients, how do [specific measurable decisions] differ between those with [aspect A] and those with [aspect B]?"
		- "In {condition} patients with [aspect A], what [specific clinical patterns] emerge relative to those with [aspect B]?"

		Good examples:
		- "For patients admitted with sepsis and chronic kidney disease, what antibiotic dose adjustments and renal monitoring labs are documented, compared to elderly sepsis patients (>75) where polypharmacy reduction and fall-risk deprescribing appear?"
		- "In acute myocardial infarction patients with congestive heart failure, what specific reperfusion decisions and diuretic titration strategies are documented, compared to elderly MI patients (>75) where reduced anticoagulation and cautious beta-blocker dosing feature?"
		- "Among pneumonia patients, what IV-to-oral antibiotic switch criteria and step-down timelines differ between those with COPD exacerbation and elderly patients (>75) in whom aspiration precautions and swallowing evaluations feature prominently?"

		Keep the question to 1-2 sentences. Return ONLY the question, nothing else.
    """)

    gold_tags_system = inspect.cleandoc("""
		You are a clinical information analyst. You will be given a clinical
		question, a patient subgroup modifier (a comorbidity or demographic
		trait on top of the primary diagnosis), and discharge note excerpts
		from patients who have that modifier. Decide whether each chunk
		contains substantive evidence of how that modifier affects the
		clinical picture described in the question.
    """)

    gold_tags_template = inspect.cleandoc("""
		QUESTION: {query_text}
		PATIENT SUBGROUP MODIFIER: {facet_description}

		For each chunk below, decide whether it contains SUBSTANTIVE evidence of how
		this subgroup modifier affects the clinical management or outcomes described in the question.
		Mere mentions of the condition or drug without context are NOT substantive.
		They must directly provide useful information to answer the given QUESTION.

		<CHUNKS>
		{chunks_block}
		</CHUNKS>

		Return a JSON array of relevant chunks only:
		[{"chunk_id": "<id>", "reason": "<up to 15 words>"}, ...]
		If none relevant: [].
		Return ONLY the JSON array, nothing else.
    """)
