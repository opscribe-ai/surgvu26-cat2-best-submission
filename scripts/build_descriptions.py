"""Generate config/descriptions.yaml from the challenge label tables."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from surgvu.descriptions import build_corpus, write_corpus  # noqa: E402
from surgvu.labels import load_all_cases  # noqa: E402

if __name__ == "__main__":
    labels_root = Path(sys.argv[1])
    out = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("config/descriptions.yaml")
    out.parent.mkdir(parents=True, exist_ok=True)

    cases = load_all_cases(labels_root)
    corpus = build_corpus(cases)
    write_corpus(corpus, out)

    # NOTE: some description strings recur under more than one task class
    # (2 of the 21 strings are shared across 3 and 4 task classes
    # respectively), so sum(len(v) for v in corpus.values()) overcounts.
    # "Unique descriptions" means unique across the whole corpus, so we
    # dedupe with a set rather than summing per-task list lengths.
    all_descriptions = {desc for entries in corpus.values() for desc in entries}
    total = len(all_descriptions)
    print("cases: %d" % len(cases))
    print("tasks: %d" % len(corpus))
    print("unique descriptions: %d" % total)
    for task in sorted(corpus):
        print("  %-38s %d" % (task, len(corpus[task])))
    if total != 21:
        print("\nWARNING: expected 21 unique descriptions, got %d" % total)
