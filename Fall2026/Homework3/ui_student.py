"""OPTIONAL Q6 implementation points. Complete Q5 first.

Use the supplied UI shell or another web or desktop UI. These four functions
make the state and persistence responsibilities explicit. Do not hard-code
responses, practice attempts, feedback, or learning gains. No model is called on
import. See UI_STARTER_GUIDE.md and tests/test_ui_student.py.
"""
from __future__ import annotations
import copy
import time
from pathlib import Path
from setup_helpers import read_config, build_index, make_learning_session
from ui_support import (SCHEMA_VERSION, now_utc, active_seconds, session_binding,
    validate_ui_state, read_progress_json, write_progress_json, validate_snapshot,
    archive_approval_events, calculation_evidence_ids)


def connect_learning_session(config_path, *, topic, goals, prior_knowledge='',
                             output_dir='outputs/q6_artifacts', allow_model_download=False):
    """Return (LearningSession, config) using the COMPLETED Q5 agent.

    Steps:
      1. Read the Q6 config. Validate topic, >=2 goals, enable_live and exact model IDs.
      2. Call build_index; an embedding load/download needs explicit permission.
      3. Select the configured retriever; do not silently substitute a baseline.
      4. Call make_learning_session using the goals joined into the goal field.
         Output stays inside this project's outputs directory.
      5. Return session, config. Do not call respond here: connecting is not a turn.
    """
    raise NotImplementedError('Q6 UI1: connect the interface to your Q5 assistant')


def progress_summary(session, ui_state):
    """Return a small JSON-compatible dashboard record (no invented mastery).

    Required keys:
      topic, goals (the UI list), turns (count), completed_activities (list),
      independent_attempts (count from state['attempts']),
      ai_feedback_entries (count of nonempty attempt feedback),
      pending_question, unresolved_questions (list),
      confidence_ratings (list, learner-reported),
      resources={'steps':..., 'model_calls':..., 'tool_calls':...}.

    Obtain every value from actual state/UI records. 'Request solution' turns
    are not independent attempts. Activity completion != confidence != mastery.
    """
    raise NotImplementedError('Q6 UI2: show actual progress and separate its evidence types')


def save_progress(session, ui_state, config, path):
    """Return the written Path after saving a JSON snapshot via write_progress_json.

    Snapshot fields (exact names; use copies):
      schema_version=SCHEMA_VERSION, saved_at=now_utc(),
      binding=session_binding(session,config),
      learner_state=session.state, history=session.history, turns=session.turns,
      ui_state=validate_ui_state(...), pending_artifact=session.pending_artifact,
      closed=session.closed,
      prior_wall_seconds=session._wall_seconds_before_restore +
                         time.perf_counter()-session.wall_started,
      runtime={steps,model_calls,tool_calls,active_seconds,trace,observed_ids,
               signatures,artifacts}.

    Convert sets to sorted lists. active_seconds(session) excludes human waiting.
    Never serialize backend objects, credentials, allow_write or _approved_reports.
    Progress saving is an explicit host/UI checkpoint; it does not grant permission
    for an agent-generated report. Run validate_snapshot before atomic writing.
    """
    raise NotImplementedError('Q6 UI3: save durable, source-bound learning progress')


def restore_progress(session, config, path):
    """Restore into a NEW compatible session; return restored ui_state.

    Steps:
      1. Read JSON, then validate_snapshot BEFORE changing the session.
      2. Restore copies of learner_state/history/turns/pending_artifact/closed.
      3. Restore counters/trace/observed IDs/signatures/artifacts; sets become sets.
         Add calculation_evidence_ids(saved_trace) to runtime.known_ids.
         Archive old host_approval events with archive_approval_events.
      4. Set runtime.allow_write=False; clear runtime._approved_reports.
         Past approval text is not a current permission. No tool call is replayed.
      5. Set runtime.started = now - saved active_seconds;
         session._returned_at=now; session.wall_started=now;
         session._wall_seconds_before_restore=saved prior_wall_seconds.
      6. Return a copied UI record. Retain closed/budget state; do not grant a
         new execution budget just because the program restarted.

    Do not unpickle or evaluate saved text. Load only your own local records.
    Changed model/source configurations require an explicitly separate session.
    """
    raise NotImplementedError('Q6 UI4: resume learning without replaying actions or restoring consent')
