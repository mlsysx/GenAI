"""Provided host/UI session state for Q5. Also used by the optional Q6 interface.

No persistent learner profiles are loaded automatically. Model-generated feedback
is recorded as feedback, not as verified mastery. This is an in-process teaching
boundary, not a production sandbox for untrusted Python code.
"""
from __future__ import annotations
import copy
from pathlib import Path
import time
from hw3_utils import save_json


class LearningSession:
    """Keep one student's REAL turns, budgets and approval records across calls.

    agent must implement run(case, initial_observations=(), user_history=()).
    The agent owns the LLM/tool loop; this host wrapper owns transcript and state.
    """
    def __init__(self, agent, *, topic='', goal='', prior_knowledge='', max_turns=12):
        if type(max_turns) is not int or not 1 <= max_turns <= 30:
            raise ValueError('max_turns must be an integer from 1 to 30')
        self.agent, self.runtime = agent, agent.runtime
        self.max_turns = max_turns
        self.history, self.turns = [], []
        self.state = {'topic': '', 'goal': '', 'prior_knowledge': '',
                      'pending_question': None, 'attempts': [],
                      'unresolved_questions': [], 'evidence_ids': []}
        self.set_profile(topic=topic, goal=goal, prior_knowledge=prior_knowledge)
        self.pending_artifact = None
        self._returned_at = time.perf_counter()  # Exclude user waiting before the first turn too.
        self.wall_started = time.perf_counter()
        self._wall_seconds_before_restore = 0.0
        self.closed = False

    def set_profile(self, *, topic=None, goal=None, prior_knowledge=None):
        """HOST/UI: record only information actually supplied by the learner."""
        for key, value in [('topic', topic), ('goal', goal), ('prior_knowledge', prior_knowledge)]:
            if value is not None:
                if not isinstance(value, str) or len(value) > 800:
                    raise ValueError('Profile fields must be strings of at most 800 characters')
                self.state[key] = value

    def note_unresolved(self, question):
        """HOST/UI: record a learner's stated uncertainty; do not infer a diagnosis."""
        if not isinstance(question, str) or not question.strip() or len(question) > 800:
            raise ValueError('Supply an actual short learner question')
        self.state['unresolved_questions'].append(question)

    def _past_observations(self):
        # Retain exact recent evidence rather than hallucinating a summary of it.
        observations = []
        for e in self.runtime.trace:
            if e.get('event') == 'tool' and e.get('observation', {}).get('ok'):
                observations.append(e['observation'])
        return observations[-8:]

    def respond(self, message, *, kind='message', image_ids=()):
        """One actual user turn. kind: message, answer, or request_solution.

        'answer' records the actual practice attempt; 'request_solution' is not
        counted as an independent attempt. Other follow-ups remain normal turns.
        The same runtime is reused, so model/tool budgets do not reset per turn.
        Human waiting time is excluded from the active runtime time budget.
        """
        import json
        if self.closed or len(self.turns) >= self.max_turns:
            raise RuntimeError('Session is closed or has reached its turn limit')
        if kind not in {'message', 'answer', 'request_solution'}:
            raise ValueError('Unknown user-turn kind')
        if not isinstance(message, str) or not message.strip() or len(message) > 4000:
            raise ValueError('Use a nonempty learner message of at most 4000 characters')
        for sid in image_ids:
            if sid not in self.runtime.sources or self.runtime.sources[sid]['kind'] != 'image':
                raise ValueError('Use an image ID from the current manifest')
        if kind in {'answer', 'request_solution'} and self.state['pending_question'] is None:
            raise ValueError('No pending practice question')
        state_before = copy.deepcopy(self.state)
        now = time.perf_counter()
        if self._returned_at is not None:
            self.runtime.started += now - self._returned_at  # exclude human waiting, not model work
        attempted = None
        if kind == 'answer':
            if self.state['pending_question'] is None:
                raise ValueError('No pending practice question to answer')
            attempted = {'question': self.state['pending_question'], 'learner_reply': message,
                         'feedback': None, 'feedback_verified': False}
            self.state['attempts'].append(attempted)
            self.state['pending_question'] = None
        elif kind == 'request_solution':
            if self.state['pending_question'] is None:
                raise ValueError('No pending practice question')
            self.state['pending_question'] = None
        # This is a host-authored record, not a claim of a new trusted system role.
        payload = {'learner_message': message, 'turn_kind': kind,
                   'learner_state': copy.deepcopy(self.state),
                   'recent_untrusted_tool_observations': self._past_observations(),
                   'host_approval_receipts': [e for e in self.runtime.trace if e.get('event')=='host_approval']}
        case = {'id': f'learning_turn_{len(self.turns)+1:02d}',
                'question': json.dumps(payload, ensure_ascii=False), 'image_ids':list(image_ids)}
        before = {'model_calls': self.runtime.model_calls, 'tool_calls': self.runtime.tool_calls}
        try:
            result = self.agent.run(case, user_history=copy.deepcopy(self.history))
        finally:
            self._returned_at = time.perf_counter()
        final = result['final']
        # History records real user messages, not generated fake learner continuations.
        self.history.append({'role':'user','content':message})
        self.history.append({'role':'assistant','content':json.dumps(final, ensure_ascii=False)})
        if final['status']=='awaiting_answer':
            self.state['pending_question'] = final['question']
        if attempted is not None:
            attempted['feedback'] = final['answer']
        self.state['evidence_ids'] = sorted(self.runtime.observed_ids)
        self.pending_artifact = copy.deepcopy(final.get('artifact')) if final['status']=='needs_approval' else None
        if final['status']=='failed':
            self.closed = True
        record={'turn':len(self.turns)+1,'kind':kind,'learner_message':message,'image_ids':list(image_ids),
                'final':copy.deepcopy(final),'state_before':state_before,'state_after':copy.deepcopy(self.state),
                'delta_model_calls':self.runtime.model_calls-before['model_calls'],
                'delta_tool_calls':self.runtime.tool_calls-before['tool_calls'],
                'active_seconds':result['metrics']['elapsed_seconds'],
                'mock_used':result['mock_used']}
        self.turns.append(record)
        return result

    def review_and_approve(self):
        """HOST/UI ONLY. Display exact artifact; obtain actual keyboard input.

        A model-generated field or retrieved text never calls this function.
        The method grants approval but does not write the report. The subsequent
        save_report tool must still match the filename and content exactly.
        """
        if self.pending_artifact is None:
            raise ValueError('No pending artifact proposal')
        a = self.pending_artifact
        print('Proposed file:', self.runtime.output_dir / a['filename'])
        print('----- EXACT CONTENT -----\n'+a['content']+'\n----- END CONTENT -----')
        decision = input('Type APPROVE to authorize this exact file; anything else denies: ')
        approved = decision.strip()=='APPROVE'
        digest=self.runtime.approve_report(a['filename'],a['content'],confirmed=approved)
        self.history.append({'role':'user', 'content':
            f'Actual host review: {"approved" if approved else "denied"} {a["filename"]}; content SHA256 {digest}.'})
        return approved

    def save_log(self, path):
        """Explicit host export for homework submission, not a model-accessible tool."""
        save_json(path, {'profile': {k:self.state[k] for k in ('topic','goal','prior_knowledge')},
                        'turns':self.turns,'state':self.state,'trace':self.runtime.trace,
                        'artifacts':self.runtime.artifacts,
                        'wall_seconds':self._wall_seconds_before_restore + time.perf_counter()-self.wall_started,
                        'model_config':self.runtime.backend.describe(),
                        'vision_config':self.runtime.vision_backend.describe(),
                        'source_hashes':{k:v['sha256'] for k,v in self.runtime.sources.items()},
                        'mock_used':any(e.get('is_mock',False) for e in self.runtime.trace)})
        return Path(path)
