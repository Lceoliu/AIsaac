"""Rebuild an offline HTML viewer from an evaluation .jsonl.gz record."""
import argparse
import gzip
import json
from pathlib import Path


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('record',type=Path)
    parser.add_argument('--out',required=True,type=Path)
    args=parser.parse_args()
    with gzip.open(args.record,'rt',encoding='utf8') as f:
        metadata=json.loads(next(f))['metadata'];rows=[json.loads(line) for line in f]
    template=(Path(__file__).resolve().parents[2]/'bridge/python/isaac_bridge/replay.html').read_text(encoding='utf8')
    payload=json.dumps(dict(metadata=metadata,rows=rows),separators=(',',':')).replace('</','<\\/')
    args.out.write_text(template.replace('/*REPLAY_DATA*/',payload),encoding='utf8')
    print(json.dumps(dict(path=str(args.out.resolve()),frames=len(rows),seed=metadata['seed'])))


if __name__=='__main__':main()
