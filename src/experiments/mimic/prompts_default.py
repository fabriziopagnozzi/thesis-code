QUERY_GENERATING_TEMPLATE_DEF = """
Below are excerpts from real discharge summaries of patients admitted for {condition}.

<START DISCHARGE SUMMARIES EXAMPLES>
{chunks_block}
<END DISCHARGE SUMMARIES EXAMPLES>

The examples above are provided ONLY so you understand what kinds of clinical information the dataset contains.
Do NOT generate a question about these specific patients or these specific notes.
Using the above as background context only, generate ONE generic question about patients with {condition}
that asks for clinical patterns across multiple aspects of care.
The question must be answerable by reading across many different patients in the dataset - not just the examples above.

The question must address EXACTLY the following aspects — all of them, no others:
{modifier_list}

Good examples:
- "For patients admitted with sepsis, how does antibiotic management differ between those with chronic kidney disease and elderly patients over 75?"
- "For patients with acute myocardial infarction, how do discharge medications and follow-up plans differ between those with chronic kidney disease and elderly patients?"
- "In patients hospitalized with pneumonia, how does length of stay and treatment intensity compare between those with heart failure and elderly patients?"

Requirements:
- Asks about observable patterns in the data, not what "should" be done
- Can be answered by reading across many discharge notes, not from a textbook

Keep the question to 1-2 sentences. Return ONLY the question, nothing else.
"""

GOLD_TAGS_SYSTEM_PROMPT_DEF = """
You are a clinical information analyst. You will be given a clinical
question, a patient subgroup modifier (a comorbidity or demographic
trait on top of the primary diagnosis), and discharge note excerpts
from patients who have that modifier. Decide whether each chunk
contains substantive evidence of how that modifier affects the
clinical picture described in the question.
"""

GOLD_TAGS_TEMPLATE_DEF = """
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
[{{"chunk_id": "<id>", "reason": "<up to 15 words>"}}, ...]
If none relevant: [].
Return ONLY the JSON array, nothing else.
"""
