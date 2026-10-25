"""Optional Streamlit UI STARTER for Q6. Default: no connection, no model calls.

Launch from this folder:
  python -m streamlit run ui_app.py --server.address 127.0.0.1
Complete ui_student.py and the Q5 TODOs first. The empty shell is not a completed
bonus demonstration. No scripted assistant replies are provided by this UI.
"""
from pathlib import Path
import json
import os
import sys

ROOT=Path(__file__).resolve().parent
os.chdir(ROOT)
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
import streamlit as st
from dotenv import load_dotenv
load_dotenv()
import ui_student as student
from ui_support import (new_ui_state, read_progress_json, artifact_digest,
    record_artifact_review, record_completed_activity, record_confidence)

st.set_page_config(page_title='Learning Code Generation — Q6', layout='wide')
st.title('Code Generation Learning Assistant')
st.caption('CAI 5607 · Q6 interface starter · reuse your completed Q5 agent')
st.info('This shell makes no model calls until you explicitly connect and send a message. '
        'Complete the student TODOs. Do not enter credentials or sensitive learner information.')
for key,default in [('session',None),('config',None),('progress',None)]:
    if key not in st.session_state:st.session_state[key]=default

def show_error(exc):
    # Raw provider messages are sanitized by the supplied model adapter.
    st.error(f'{type(exc).__name__}: {exc}')

with st.sidebar:
    st.header('Configure')
    config_path=st.text_input('Q6 configuration file','config_q6.json')
    download_ok=st.checkbox('Allow loading/downloading the selected embedding model',value=False)
    connection_ok=st.checkbox('I completed the TODOs and reviewed model access, data permissions, and cost',value=False)
    st.caption('Use environment variables or notebook secrets for credentials, never a UI textbox.')
    with st.form('learner_setup'):
        topic=st.text_input('Topic', value='Code Generation')
        goal1=st.text_input('Learning goal 1')
        goal2=st.text_input('Learning goal 2')
        prior=st.text_area('Stated prior knowledge (optional)')
        connect=st.form_submit_button('Start a new learning session',disabled=not connection_ok)
    if connect:
        try:
            if st.session_state.session is not None:
                raise ValueError('Use a fresh browser session or restart the application before replacing an active session; save progress first.')
            ui_state=new_ui_state([goal1,goal2])
            s,cfg=student.connect_learning_session(config_path,topic=topic,goals=[goal1,goal2],
                prior_knowledge=prior,allow_model_download=download_ok)
            st.session_state.session=s;st.session_state.config=cfg;st.session_state.progress=ui_state
            st.success('Connected. No model response has been requested yet.')
        except Exception as exc:show_error(exc)
    st.divider()
    st.header('Save and resume')
    state_path=st.text_input('Local progress record','outputs/q6_progress/session.json')
    st.caption('Stop/restart the application, then use Restore. Restoring never executes a tool or re-grants report approval.')
    if st.button('Save progress',disabled=st.session_state.session is None):
        try:
            path=student.save_progress(st.session_state.session,st.session_state.progress,
                                      st.session_state.config,state_path)
            st.success(f'Saved: {path.name}')
        except Exception as exc:show_error(exc)
    if st.button('Restore progress',disabled=not connection_ok):
        try:
            if st.session_state.session is not None:
                raise ValueError('Restore into a fresh application session; save and restart before loading.')
            raw=read_progress_json(state_path)
            profile=raw.get('learner_state',{});goals=[g['text'] for g in raw.get('ui_state',{}).get('goals',[])]
            s,cfg=student.connect_learning_session(config_path,topic=profile.get('topic',''),goals=goals,
                 prior_knowledge=profile.get('prior_knowledge',''),allow_model_download=download_ok)
            progress=student.restore_progress(s,cfg,state_path)
            st.session_state.session=s;st.session_state.config=cfg;st.session_state.progress=progress
            st.success('Progress restored. Report writes require fresh approval.')
        except Exception as exc:show_error(exc)

chat_tab,progress_tab,sources_tab=st.tabs(['Conversation and practice','Learning progress','Sources and visual evidence'])
s=st.session_state.session
with chat_tab:
    st.subheader('Conversation and practice')
    if s is None:
        st.write('Set a topic and two goals, then connect your completed assistant. This empty shell is not a model-generated learning session.')
    else:
        for t in s.turns:
            with st.chat_message('user'):st.write(t['learner_message'])
            with st.chat_message('assistant'):
                f=t['final'];st.caption(f"Status: {f['status']}");st.write(f['answer'])
                if f.get('question'):st.write('Question: '+f['question'])
                if f.get('citations'):st.json(f['citations'])
        if s.state['pending_question']:
            st.warning('Pending practice question: '+s.state['pending_question'])
        if s.closed:st.warning('This session stopped or exhausted its budget. It is not silently restarted.')
        with st.form('message_form',clear_on_submit=True):
            kinds={'Message / follow-up':'message','Submit my practice attempt':'answer','Request the solution':'request_solution'}
            label=st.selectbox('Interaction type',list(kinds))
            image_ids=st.multiselect('Relevant visual evidence (optional)',[sid for sid,src in s.runtime.sources.items() if src['kind']=='image'])
            msg=st.text_area('Your actual message or practice attempt',max_chars=4000)
            send=st.form_submit_button('Send to my assistant',disabled=s.closed)
        if send:
            try:
                with st.spinner('Running the bounded assistant…'):
                    s.respond(msg,kind=kinds[label],image_ids=image_ids)
                st.rerun()
            except Exception as exc:show_error(exc)
        if s.pending_artifact:
            a=s.pending_artifact;digest=artifact_digest(a)
            st.subheader('Review before writing a report')
            st.write('Destination: '+str(s.runtime.output_dir / a['filename']));st.code(a['content'],language='markdown')
            st.caption('A review decision authorizes only these exact bytes and filename. It does not execute a save itself.')
            allow,deny=st.columns(2)
            decision=None
            if allow.button('Approve this exact report'):decision=True
            if deny.button('Deny this report'):decision=False
            if decision is not None:
                try:
                    record_artifact_review(s,expected_filename=a['filename'],expected_digest=digest,approved=decision)
                    st.success('Decision recorded by the host. Send a follow-up to continue; no write has been performed by this button.')
                except Exception as exc:show_error(exc)

with progress_tab:
    st.subheader('Learning progress')
    st.caption('Activity completion ≠ learner confidence ≠ AI feedback ≠ independently demonstrated mastery.')
    if s is None:st.write('No learning interactions have been recorded.')
    else:
        try:
            p=student.progress_summary(s,st.session_state.progress)
            st.write('Topic: '+p['topic'])
            st.table(p['goals'])
            c1,c2,c3=st.columns(3)
            c1.metric('Actual turns',p['turns']);c2.metric('Independent attempts',p['independent_attempts']);c3.metric('AI feedback entries (unverified)',p['ai_feedback_entries'])
            st.write('Pending question:',p['pending_question'] or 'None')
            st.write('Unresolved questions:',p['unresolved_questions'])
            st.write('Learner-confirmed completed activities');st.table(p['completed_activities'])
            st.write('Learner-reported confidence (1 = not confident; 5 = very confident)');st.table(p['confidence_ratings'])
            st.write('Execution counters');st.json(p['resources'])
        except Exception as exc:show_error(exc)
        with st.form('activity_form'):
            act=st.text_input('Activity you actually completed (not a mastery claim)')
            turn=st.number_input('Supporting interaction turn',min_value=1,value=1,step=1)
            mark=st.form_submit_button('Record my completed activity',disabled=not s.turns)
        if mark:
            try:record_completed_activity(st.session_state.progress,act,int(turn),turn_count=len(s.turns));st.rerun()
            except Exception as exc:show_error(exc)
        with st.form('confidence_form'):
            g=st.selectbox('Goal',[x['id'] for x in st.session_state.progress['goals']])
            confidence=st.slider('My self-reported confidence (not an objective score)',1,5,3)
            confirm=st.form_submit_button('Record my confidence rating')
        if confirm:
            try:record_confidence(st.session_state.progress,g,int(confidence));st.rerun()
            except Exception as exc:show_error(exc)
        with st.form('question_form'):
            question=st.text_input('My unresolved question or next step')
            note=st.form_submit_button('Record my question')
        if note:
            try:s.note_unresolved(question);st.rerun()
            except Exception as exc:show_error(exc)
        if st.button('Export actual conversation and tool log'):
            try:s.save_log('outputs/q6_progress/session_log.json');st.success('Saved outputs/q6_progress/session_log.json')
            except Exception as exc:show_error(exc)

with sources_tab:
    st.subheader('Sources and visual evidence')
    if s is None:st.write('Sources will be listed after you connect the configured Q6 knowledge base.')
    else:
        for sid,src in sorted(s.runtime.sources.items()):
            with st.expander(f"{sid} — {src['title']}"):
                st.json({k:src[k] for k in ('source','locator','date','version','status','generated','content_origin') if k in src})
                if src['kind']=='image':st.image(str(src['resolved_path']),caption=f"{sid}: manifest image; origin: {src.get('content_origin', 'see source record')}")
                else:st.text(src['text'][:8000])
        st.subheader('Model-generated visual observations (unverified)')
        for event in s.runtime.trace:
            obs = event.get('observation', {}).get('evidence', {})
            if obs.get('provenance') == 'model_generated_image_observation':
                st.write(f"{obs['id']} — {obs['text']}")
        st.caption('Displaying a source is not verification of an assistant claim. Generated observations remain labeled as such.')
