"""Assignment constants and setup functions. Importing makes no network calls."""
from pathlib import Path
import json
import platform
import importlib.metadata
from hw3_utils import (STATUSES, TOOL_SPECS, load_corpus, validate_chunks, TfidfRetriever,
                       sentence_transformer_embedder, Runtime, Limits)
from llm_backends import ChatBackend
from learning_session import LearningSession
from student_agent import chunk_document, DenseRetriever, EvidenceAgent, build_system_prompt


# Shared quantities from the Canvas Q5/Q6 assignment.
MIN_TEXT_DOCUMENTS = 10
MIN_VISUAL_ITEMS = 5
DEVELOPMENT_CASES = 5
TEST_CASES = 5
MIN_IMAGE_TEST_CASES = 2
FINAL_EPISODES = 15
Q5_POINTS = 45
Q6_POINTS = {"interface": 10, "learning_session": 5, "survey": 5}
Q6_TOPIC = "Code Generation"


def read_config(path='config.json'):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def build_index(config, *, allow_model_download=False):
    sources = load_corpus(config['manifest'])
    chunks = [c for s in sources.values() if s['kind']=='text'
              for c in chunk_document(s, config['chunk_max_words'], config['chunk_overlap'])]
    validate_chunks(chunks, sources)
    lexical = TfidfRetriever(chunks)
    dense = None
    if allow_model_download:
        if not config.get('embedding_model'):
            raise ValueError('Set embedding_model explicitly before downloading/loading it')
        embed=sentence_transformer_embedder(config['embedding_model'],config.get('embedding_revision'))
        dense=DenseRetriever(chunks,embed)
    return sources, chunks, lexical, dense


def make_backends(config):
    if config.get('enable_live') is not True:
        raise ValueError('Set enable_live=true only after reviewing data, model access, and costs')
    answer=ChatBackend(**config['answer_backend'],enable_live=True)
    vision=ChatBackend(**config['vision_backend'],enable_live=True) if config.get('vision_backend') else answer
    return answer,vision


def make_learning_session(config, sources, chunks, retriever, *, topic='', goal='', prior_knowledge='', output_dir='outputs/session_artifacts'):
    answer,vision=make_backends(config)
    rt=Runtime(answer,retriever,chunks,sources,output_dir,vision_backend=vision,
               limits=Limits(**config['session_limits']))
    prompt=build_system_prompt(config['domain'])
    return LearningSession(EvidenceAgent(rt,prompt),topic=topic,goal=goal,prior_knowledge=prior_knowledge,
                           max_turns=config.get('max_session_turns',12))


def check_case_splits(dev, test):
    if len(dev)!=DEVELOPMENT_CASES or len(test)!=TEST_CASES:
        raise ValueError(f'Use exactly {DEVELOPMENT_CASES} development and {TEST_CASES} final test cases')
    for field in ('id','question'):
        a=[str(c[field]).strip() for c in dev];b=[str(c[field]).strip() for c in test]
        if len(set(a))!=len(a) or len(set(b))!=len(b) or set(a)&set(b):
            raise ValueError(f'Duplicate or overlapping {field} across evaluation cases')
    if sum(bool(c.get('image_ids')) for c in test)<MIN_IMAGE_TEST_CASES:
        raise ValueError(f'At least {MIN_IMAGE_TEST_CASES} final cases must use images')
    if any(c.get('allow_write',False) is not False for c in dev+test):
        raise ValueError('Do approval/write demonstrations separately, not in the common answer benchmark')
    for case in dev + test:
        validate_case(case)
    return True


def validate_case(case, sources=None):
    """Validate labels without ever forwarding them to the model."""
    import re
    if not isinstance(case, dict) or not re.fullmatch(r'[A-Za-z0-9_-]{1,80}', case.get('id', '')):
        raise ValueError('Use a plain case ID')
    if not isinstance(case.get('question'), str) or not 1 <= len(case['question'].strip()) <= 300:
        raise ValueError('Case question must contain 1–300 characters')
    if not isinstance(case.get('category'), str) or not case['category'].strip():
        raise ValueError('Declare a category')
    if case.get('expected_status') not in STATUSES:
        raise ValueError('Declare a valid expected status')
    for field in ('answer_criteria', 'relevant_source_ids', 'image_ids'):
        value = case.get(field)
        if not isinstance(value, list) or any(not isinstance(x, str) or not x.strip() for x in value):
            raise ValueError(f'{field} must be a list of strings')
        if len(set(value)) != len(value):
            raise ValueError(f'Duplicate entries in {field}')
    if not case['answer_criteria']:
        raise ValueError('Write reference criteria before testing')
    if case.get('allow_write', False) is not False:
        raise ValueError('Benchmark cases cannot grant write permission')
    faults = case.get('unavailable_tools', [])
    if not isinstance(faults, list) or any(t not in TOOL_SPECS for t in faults):
        raise ValueError('Unknown simulated tool fault')
    if sources is not None:
        if any(s not in sources for s in case['image_ids'] + case['relevant_source_ids']):
            raise ValueError('Unknown case source ID')
        if any(sources[s]['kind'] != 'image' for s in case['image_ids']):
            raise ValueError('image_ids must identify images')
    return True


def environment_record():
    result={'python':platform.python_version(),'platform':platform.platform(),'packages':{}}
    for p in ('numpy','scikit-learn','requests','jsonschema','sentence-transformers','Pillow','nbformat'):
        try:result['packages'][p]=importlib.metadata.version(p)
        except importlib.metadata.PackageNotFoundError:result['packages'][p]=None
    return result
