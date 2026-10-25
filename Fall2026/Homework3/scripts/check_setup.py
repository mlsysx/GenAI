"""Offline dependencies/config/corpus check. Does not implement student TODOs."""
from pathlib import Path
import argparse
import json
import os
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))

def main():
    os.chdir(ROOT)
    from setup_helpers import read_config,environment_record
    from hw3_utils import load_corpus,Limits
    from llm_backends import ChatBackend
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--config',default='config.json');args=ap.parse_args()
    c=read_config(args.config)
    if c.get('retriever') not in {'lexical','dense'} or type(c.get('top_k')) is not int or not 1<=c['top_k']<=6:raise ValueError('Set retriever and top_k (1–6)')
    if not 0<=c['chunk_overlap']<c['chunk_max_words']:raise ValueError('Invalid chunking settings')
    for key in ('limits','session_limits'):Limits(**c[key])
    for key in ('answer_backend','vision_backend'):
        b=c.get(key)
        if not b:continue
        if b.get('model'):ChatBackend(**b,enable_live=False)
        else:print(f'{key}: model is intentionally blank; configure before live work.')
    sources=load_corpus(c['manifest'])
    print(json.dumps(environment_record(),indent=2))
    print('Verified sources:',len(sources),'— text:',sum(s['kind']=='text' for s in sources.values()),'images:',sum(s['kind']=='image' for s in sources.values()))
    print('No live request, embedding load or student implementation was run.')

if __name__=='__main__':main()
