"""`python -m tradex.research.loop [options]` is the same as `python -m tradex research loop [options]`."""
import sys

from tradex.cli import main

if __name__ == "__main__":
    sys.exit(main(["research", "loop", *sys.argv[1:]]))
