# Evaluation guide — Q5 and Q6

## What is and is not evaluated
Q5 evaluates a GenAI system, not whether a human learned more because of it. Q6 documents a working interface, restored learning progress, one actual Code Generation learning session and survey feedback; it is not a randomized learning-effect study. Do not treat satisfaction, confidence, or an assistant-generated quiz score as a validated measure of learning.

## Case design and freezing
Create **5 development cases and 5 final test cases** with unique IDs and nonduplicate questions. Inspect near-duplicates manually. Together cover concept explanations, multi-source synthesis, visual questions, missing-information/clarification, and robustness/tool failures. **Categories may overlap**: for example, one multi-source case may require an image, and one missing-information case may ask about an unreadable visual detail. At least **2 of the 5 test cases** use actual images; at least one image case concerns an unreadable or absent detail. Record these coverage choices before running tests. The supplied count check cannot judge whether your question genuinely requires the image.

Use `eval/case_template.json` to create `eval/dev_cases.jsonl` and `eval/test_cases.jsonl`. The cases in `tests/` are software fixtures, not final evaluation cases.

Required case fields are `id`, `category`, `question` (1–300 characters), `image_ids`, `expected_status`, `relevant_source_ids`, and `answer_criteria`. Validate source IDs against your manifest. Optional `allow_write` must be false; optional `unavailable_tools` names tools disabled by the host to simulate a fault.

Define `answer_criteria`, `relevant_source_ids`, and `expected_status` before the final run. When more than one response is defensible, specify acceptable outcomes in answer_criteria and grade against those, not status equality alone. Fields after question/image_ids belong to the host and human annotator only. Unknown-answer cases should have no invented gold evidence. A fault can be simulated by listing a tool under unavailable_tools; report it as a host simulation, not a provider outage.

Case evidence is allowed in the index. Test questions, answer keys, grading notes and post-test tuning decisions are not. Keep source snapshots and source/model configuration hashes. After Q5 final testing, preserve the frozen Q5 snapshot. Q6 may reuse its unchanged sources; any added materials need a separate Q6 source/configuration snapshot, as described in UI_STARTER_GUIDE.md.

## Retrieval comparison
Compare lexical and pretrained dense retrieval on the same text-retrieval development cases, with the same chunks, k, metadata filters, and source-level relevance judgments. For each case:

`Recall@k = distinct required text source IDs represented in top-k chunks / number of required text source IDs`.

Do not count a source twice. Cases with no required text source have recall N/A. Image-inspection success is a separate metric, not a text-retrieval miss. Report the evaluated case IDs and denominator. An embedding score is not a calibrated truth probability.

## Three-condition final run
Run **5 cases under direct generation, fixed RAG, and the full agent (15 episodes)**. All use the same answer model, temperature, output/action schema, token limit per call, input cases, and source snapshot. Direct has no text retrieval. Fixed RAG retrieves once and answers once. The agent can decide and use tools adaptively within a budget. The agent may use more calls, so report these as part of its resource cost.

The supplied harness makes one initial VLM observation per query image and shares it across all three conditions. Save the actual image-processing trace and report its cost separately. This makes the answer comparison conditional on a shared perception result; it does not compare native VLM architectures. If initial image processing fails, record a failure for that case under all conditions. Additional agent inspections remain additional work. Do not silently substitute handwritten chart values for the shared VLM observation.

All runtime failures stay in the denominator. Model/provider errors should be saved as failed episodes. A programming bug that crashes the harness must be fixed, documented and rerun consistently rather than dropping cases. The provided implementation permits no write approval in benchmark cases. Evaluate report creation and tutoring interactions separately.

This is an end-to-end comparison. Information access, call count, and process differ; superior output cannot be attributed only to orchestration. Do not use a different or larger answer model for the agent.

## Human scoring
The harness writes `*_human_annotations.jsonl` with null semantic-grading fields. Fill these manually from the prewritten criteria and sources. Never fill them from the model's self-assessment.

| Field / measure | Definition |
|---|---|
| `task_success` | 1 only if the case's declared outcome criteria are met; otherwise 0. Failed/missing episodes are 0. |
| `supported_citations` / `total_citations` | Supported claim–evidence groups divided by all claimed groups; also retain citation-ID audit. |
| `supported_factual_claims` / `total_factual_claims` | Factual claims supported by cited source content divided by all factual claims requiring evidence. |
| Appropriate uncertainty | Score missing/unreadable cases separately. Unnecessary refusal on answerable cases is not a success. |
| Visual grounding | Check actual visible values/relations and acknowledge ambiguity; inspect the image, not just generated captions. |
| `error_stage` | e.g. retrieval, versioning, visual extraction, generation, formatting, tool execution, permission, budget, or none. |
| Resources | Model/tool calls; measured elapsed time; provider token counts when returned. Compute money only from documented prices. |

Zero citations with zero factual claims gives citation precision N/A, not 100%. A citation ID passing a schema/existence check does not prove it supports a claim. Citations to calculator outputs should also identify the operand sources. Returned feedback remains unverified until checked.

Use `python summarize_run.py outputs/final_run` only after human annotation is complete. It rejects mock runs as empirical results. Save result tables and actual errors, or state explicitly if none occurred. Do not invent errors when none occurred. Optional plots can show task success by condition and average calls/time; label all denominators and shared preprocessing.

## Separate interactive checks
Demonstrate: (1) clarification and retained goal; (2) actual image use and uncertainty; (3) a practice question, real/scripted learner attempt clearly labeled, and feedback responding to that attempt; and (4) exact-artifact preview, explicit user approval, successful write receipt and matching file contents. A live denied-write demonstration is not required; the supplied permission-denial tests remain. Use the same runtime within a session: turn changes must not reset model/tool limits or grant new permission. An explicit UI action may review a newly proposed artifact. The model must not call the host approval function.

Scripted user utterances and injected tool faults are valid fixtures. Scripted model outputs are valid only for offline software tests, not the graded live demonstration or measured comparison. The starter has no machine verifier of educational quality or prompt-injection resistance; inspect outputs and add tests.

## Q6 separation and code checks
Q6 is not another five-case benchmark. Use actual learner interactions and preserve the frozen Q5 snapshot. Code checks from the worked example belong in Q6 logs, not Q5 gold labels or retrieval sources. Distinguish executed test output from manual tracing and from assistant predictions. Passing a few checks does not validate the function on all inputs. Do not report generated practice answers as learner attempts.

## Annotation fields and generated summary

Fill `appropriate_uncertainty` and `visual_grounding` with 0 or 1 when applicable, or the string `N/A`. Fill both support numerator/denominator pairs with integer counts (including 0/0 when no claims apply), `task_success` with 0 or 1, and `error_stage` with a stage name or `none`. Use `grading_note` for evidence and rationale. Blank scoring fields are rejected.

The harness saves `cases.json`, `source_snapshot.json`, `settings.json`, every run and shared perception trace. Each run's `retrieval` record reports first-search source Recall@k using the configured k; an agent may ask for fewer results. It also records cumulative source coverage across searches, which is not Recall@k. Direct generation has recall N/A. For cases with required text evidence, a missing/failed search receives 0; cases without required text evidence remain N/A. Report agent retrieval with its actual query/call budget.

The summary aggregates human scores, failures, first-search recall, citation-ID validity and resources. ID validity is measured on delivered final citations after host validation, so rejected citations must also be discussed from traces. It does not establish claim support. A direct-generation baseline may use prior model knowledge, explicitly distinguished from any supplied visual evidence, and must not invent source IDs.
