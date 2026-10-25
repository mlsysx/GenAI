"""Print the SHA256 of one local file; makes no changes."""
import argparse
import hashlib
from pathlib import Path

def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('path',type=Path);a=ap.parse_args()
    print(hashlib.sha256(a.path.read_bytes()).hexdigest())

if __name__=='__main__':main()
