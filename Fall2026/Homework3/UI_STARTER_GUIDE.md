# Optional Q6 interface starter

Canvas defines the assignment requirements. A local web app or desktop UI is acceptable. The supplied **Streamlit shell is optional** and intentionally incomplete. A plain terminal chat or rerunning a notebook cell for each message does not meet Q6A.

## Install and launch

First complete Q5 and its tests. From the student package directory:

```bash
python -m pip install -r requirements.txt
python -m pip install -r requirements_ui.txt
# For a pretrained dense retriever, also install requirements_embeddings.txt.
python -m streamlit run ui_app.py --server.address 127.0.0.1
```

This binds the app to your own computer. Open the local address shown in the terminal. There is no public deployment requirement. Credentials belong in environment variables or a locally ignored `.env`, never in the UI, logs, screenshots, or committed JSON. This is a single-user educational application, not a hardened multi-user service.

Copy `config_q6.template.json` to `config_q6.json`. Set exact models and review data permissions and cost. Set its `manifest` path as follows:

- If you reuse only unchanged Q5 materials, point it at the frozen Q5 manifest.
- If you add materials, create `data/q6/manifest.json` from `data/project/manifest.template.json` and place the Q6 text and image files below `data/q6/`. Copy any reused authorized files into this new corpus, preserving their source IDs, provenance, and hashes. The loader rejects paths outside the manifest's directory. Do not modify the frozen Q5 files.

The template defaults to `data/q6/manifest.json`; create that file or change the path before connecting. Use [Source data in README.md](README.md#source-data) for the manifest fields. Keep learner attempts, logs, survey responses, and evaluation questions/answers out of the manifest. `enable_live` remains false until deliberately enabled. Embedding loading/downloading requires separate permission. Opening the shell alone does not download or call a model.

## Four student implementation points

Complete `ui_student.py`; retain the public interfaces and tests.

1. **Connect:** Build the configured retriever and create a `LearningSession` around your completed Q5 agent. Pass the learner's actual topic, two goals, and prior knowledge. Connecting is not itself a model turn. Do not silently use a different model/retriever.
2. **Summarize progress:** Read actual session/UI records to show turns, attempts, pending/unresolved questions, unverified feedback, learner-confirmed completed activities, self-reported confidence, and remaining execution counters. Do not assign mastery from fluent generated text or a progress bar.
3. **Save:** Make a JSON-only snapshot using the documented schema. Include profile/state, history, real turns, UI records, pending report, counters, source/model binding, and durations. Use the provided atomic JSON writer. Do not store credentials, Python objects, or restorable permission grants.
4. **Restore:** Validate the full record before mutating a new compatible session. Restore learner progress, pending questions, observed evidence and resource counters without replaying tool actions. Revoke report-write permissions and archive prior approval events. Restoring is not a way to obtain a new budget. Successful calculator receipts contribute derived IDs; use `calculation_evidence_ids(saved_trace)` when rebuilding the runtime known-ID set. Return the restored UI record.

`ui_support.py` supplies JSON/path safety, schema checks, model/source binding, typed UI records, and exact-artifact approval callbacks. It does not implement those four student functions. A different GUI may use these same contracts.

## State contract

`LearningSession.state` contains the topic, joined goals, stated prior knowledge, pending practice question, actual attempts and feedback, unresolved questions, and observed evidence IDs. The separate UI record contains:

```text
goals: [{id, text}, ...]
completed_activities: [{label, turn, origin="learner_confirmed", recorded_at}, ...]
confidence_ratings: [{goal_id, rating, origin="learner_reported", recorded_at}, ...]
```

Completing an activity records what was done, not what was mastered. An unanswered question or an explicit request for the solution must not become an independent attempt. AI feedback remains `feedback_verified=false` unless you separately implement and document a genuine verification process; the provided tests use the unverified setting.

Each saved snapshot is bound to the source hashes, chunk inventory, selected models/retriever, and configured limits. A changed corpus or model requires an explicitly separate session rather than silently relabeling old evidence. Local JSON is not authenticated: load only your own files. Never use pickle/eval or import a saved `system` message/permission as authority.

## Interaction and approval

Use `session.respond(message, kind="message")` for normal conversation, `kind="answer"` for the learner's actual attempt, and `kind="request_solution"` only when the learner asks to see an answer. The same runtime persists across turns. Rendering the UI or clicking Restore must not call `respond`.

A **Save progress** click is an explicit host bookkeeping action to checkpoint the learning session. It is distinct from an agent-proposed `save_report` tool call. The latter still needs an exact report preview and actual user approval. The supplied callback verifies the current filename/content digest before recording consent. Loading progress does not re-grant old write permission; the user must review the current artifact again.

## Restart demonstration

1. Run a real session with your model: record topic/goals, a cited explanation, actual attempts/feedback, and unresolved questions.
2. Save the snapshot and actual log. Note counters, pending question, attempts, activity labels, and confidence entries.
3. Stop the server with Ctrl+C; launch it again. Do not just switch tabs or rely on an in-memory object.
4. Select the same config and Restore progress. Confirm the recorded fields match and counters were not reset.
5. Continue an ordinary learning interaction. If a report is proposed, verify fresh approval is required. Do not automatically repeat prior tool calls.
6. Take the third screenshot after this restart/restore. Use captions and before/after records to explain the check. An empty shell screenshot is not evidence of a working assistant.

## Tests

```bash
# Supplied functions; should pass before student completion.
python -m unittest tests.test_provided tests.test_approval tests.test_learning_session tests.test_ui_support

# Q5 TODO contracts; expected to fail until Q5 is completed.
python -m unittest tests.test_student

# Q6 persistence/progress contracts; expected to fail until UI TODOs are completed.
python -m unittest tests.test_ui_student
```

Tests use labeled scripted fixtures, not real LLM outputs or evidence of learning. Passing them is not a substitute for the Q6 screenshots and live learning session. Extend the tests for your chosen interface. Stopping/restarting the entire application remains a manual acceptance check.

## Official implementation references

These are implementation aids, not additional assignment requirements.

- Streamlit session state: https://docs.streamlit.io/develop/api-reference/caching-and-state/st.session_state
- Streamlit chat elements: https://docs.streamlit.io/develop/api-reference/chat
- Streamlit forms: https://docs.streamlit.io/develop/api-reference/execution-flow/st.form

Checked 2026-10-23. Streamlit's in-memory Session State is tied to the browser connection; it is not durable storage across an application restart. That is why Q6 requires an explicit local saved record.


## Session settings
The Q6 topic is **Code Generation**; the UI opens with that topic as an editable default. Learners must enter their own two goals and prior knowledge. Use `CODE_GENERATION_SESSION_GUIDE.md` to connect specification, small-code evaluation, visual interpretation and three practice attempts. Q6A still requires three screenshots and an application restart.

Generated code is displayed as source text, not executed by this UI. Do not use `eval`, `exec`, shell commands or an unrestricted execution tool to run model replies. A learner may manually review and separately test a small pure function in a local environment. Record actual outcomes separately from model predictions.

`config_q6.template.json` permits 24 learner turns to accommodate the session. Review budgets before the session; preserve counters during restore. Do not increase a budget by editing the saved state. A deliberately new run with new limits must be identified as a separate session.

