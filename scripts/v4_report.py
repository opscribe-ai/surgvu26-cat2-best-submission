"""Collect every v4 dump into one table, with the confounds spelled out.

WHY A SCRIPT AND NOT A HAND-WRITTEN TABLE. Ten arms finish across six hours,
each writing a .npz.json next to its dump, and the numbers inside them are not
interchangeable: an arm trained on the centre-only pool saw 1.07 s of a 30 s
window, an arm on the multi-burst pool saw four bursts spread across it, and
one arm was never trained at all. A table that lists them in one column
without saying which is which invites exactly the comparison that has already
produced two wrong conclusions tonight.

So every row carries its own provenance -- pool, mechanism, initialisation,
resolution, learning rate, whether BatchNorm was frozen -- and the reference
numbers are printed as rows rather than assumed.

THE REFERENCES, and what each one is for:

    2D ResNet-50, 16 frames over 30 s      0.7802   what the submission uses
    same weights, 8 frames of a 2 s burst  0.7283   the INPUT handicap alone
    + temporal shift, untrained            0.7108   the MECHANISM cost alone

An arm beats "temporal modelling helps" only if it clears 0.7802 on the same
windows and folds. An arm that clears 0.7283 has beaten the input it was given
but not the shipped model. Both facts are worth reporting, and conflating them
is how a null becomes a win in the retelling.
"""
import argparse
import json
import sys
from pathlib import Path

DUMPS = "/staging/n/nkalthoff/surgvu26/dumps"

#: Dumps produced by a run we now know was wrong. Listed with the reason
#: rather than deleted: the file is evidence of the bug, and a table that
#: silently drops rows is as misleading as one that silently keeps bad ones.
SUPERSEDED = {
    "temporal_tsm_control": "built with the shift ON despite fold_div=0 -- the "
                            "falsy-zero bug; superseded by temporal_tsm_control2",
    "temporal_tsm_init": "valid, but identical to the bugged control above "
                         "because TSM adds no parameters; kept for the record",
}
TWO_D = 0.7802
#: The task head is a different problem with a different reference. Scoring a
#: task arm's description accuracy against the TOOLS number produced
#: "+0.1846" for temporal_tasktsmmulti -- a row that looked like the best
#: result of the night and was a category error.
TWO_D_TASK_ACC = 0.9456
TWO_D_TASK_DESC = 0.9581
CONTROL_SAME_INPUT = 0.7283
UNTRAINED_SHIFT = 0.7108


def rows_from(directory):
    out = []
    for path in sorted(Path(directory).glob("*.npz.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception as error:                      # noqa: BLE001
            out.append({"name": path.name, "error": str(error)})
            continue
        arms = payload.get("arms") or {}
        if not arms:
            out.append({"name": path.name, "error": "no arms in the dump"})
            continue
        best = max(arms, key=lambda k: arms[k].get("honest", -1))
        meta = payload.get("meta") or {}
        out.append({
            "name": path.name.replace(".npz.json", ""),
            "arm": best,
            "honest": arms[best].get("honest"),
            "arms_detail": arms[best],
            "measurable": arms[best].get("honest_measurable"),
            "mechanism": meta.get("mechanism", "?"),
            # From the recorded shards path when present. Older dumps predate
            # that field, so they fall back to the filename -- flagged with a
            # trailing ? so a guessed label is never mistaken for a read one.
            "pool": (("multi" if "shards_multi" in payload["shards"]
                      else "dense") if payload.get("shards")
                     else ("multi?" if "multi" in path.name else "dense?")),
            "init": str(meta.get("initialised_from", "?")).split("/")[-1],
            "image_size": meta.get("image_size"),
            "lr": meta.get("lr"),
            "frozen_bn": meta.get("frozen_bn"),
            "epochs": meta.get("epochs"),
            "untrained": bool(meta.get("untrained")),
            "head": meta.get("head") or payload.get("head") or "tools",
        })
    return out


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dumps", default=DUMPS)
    parser.add_argument("--out")
    args = parser.parse_args(argv)

    rows = rows_from(args.dumps)
    scored = [r for r in rows if r.get("honest") is not None
              and r.get("head", "tools") == "tools"]
    scored.sort(key=lambda r: -r["honest"])
    task_rows = sorted((r for r in rows if r.get("head") == "task"
                        and r.get("honest") is not None),
                       key=lambda r: -r["honest"])

    print("REFERENCES (same windows, same folds, same protocol)")
    print("  %-38s %.4f" % ("2D ResNet-50, 16 frames over 30 s", TWO_D))
    print("  %-38s %.4f" % ("same weights, 8 frames of a burst",
                            CONTROL_SAME_INPUT))
    print("  %-38s %.4f" % ("+ temporal shift, untrained", UNTRAINED_SHIFT))

    print("\n%-26s %-6s %-5s %8s %8s %6s %5s %3s %s"
          % ("arm", "mech", "pool", "honest", "vs 2D", "px", "lr", "bn", "ep"))
    for row in scored:
        if row["name"] in SUPERSEDED:
            continue
        print("%-26s %-6s %-5s %8.4f %+8.4f %6s %5s %3s %s"
              % (row["name"][:26], row["mechanism"], row["pool"],
                 row["honest"], row["honest"] - TWO_D,
                 row["image_size"], row["lr"] or "-",
                 "Y" if row["frozen_bn"] else ("-" if row["frozen_bn"] is None
                                               else "N"),
                 "0" if row["untrained"] else row["epochs"]))

    if task_rows:
        print("\nTASK HEAD -- a different problem with its own reference "
              "(2D: accuracy %.4f, description %.4f)"
              % (TWO_D_TASK_ACC, TWO_D_TASK_DESC))
        print("%-28s %8s %8s %9s %9s"
              % ("arm", "acc", "desc", "vs acc", "vs desc"))
        for row in task_rows:
            arms = row.get("arms_detail") or {}
            acc = arms.get("accuracy")
            desc = row["honest"]
            print("%-28s %8s %8.4f %+9s %+9.4f"
                  % (row["name"][:28],
                     "%.4f" % acc if acc is not None else "-", desc,
                     "%.4f" % (acc - TWO_D_TASK_ACC) if acc is not None else "-",
                     desc - TWO_D_TASK_DESC))

    broken = [r for r in rows if "error" in r]
    if broken:
        print("\nUNREADABLE (%d): %s"
              % (len(broken), ", ".join(r["name"] for r in broken)))

    stale = [r for r in scored if r["name"] in SUPERSEDED]
    if stale:
        print("\nSUPERSEDED, shown for the record:")
        for row in stale:
            print("  %-26s %.4f  -- %s"
                  % (row["name"][:26], row["honest"], SUPERSEDED[row["name"]]))

    scored = [r for r in scored if r["name"] not in SUPERSEDED]
    if scored:
        best = scored[0]
        print("\nBEST ARM: %s at %.4f (%+.4f against the shipped 2D model)"
              % (best["name"], best["honest"], best["honest"] - TWO_D))
        if best["honest"] > TWO_D:
            print("It clears the shipped model on identical windows and folds. "
                  "Temporal modelling pays -- check the fusion test before "
                  "wiring it, because 'better alone' and 'adds something' are "
                  "different claims.")
        elif best["mechanism"] in ("tsm", "i3d"):
            # The same-input control is a property of the CONVERSION path --
            # our 2D weights, 8 frames, 384px. A Kinetics backbone at 112px
            # never had that input, so measuring it against 0.7283 would be
            # comparing two different handicaps and calling the difference a
            # mechanism. Only conversions get that row.
            if best["honest"] > CONTROL_SAME_INPUT:
                print("It clears the same-input control (%.4f) but NOT the "
                      "shipped model (%.4f): the temporal mechanism earned back "
                      "part of what the narrower input cost, and no more."
                      % (CONTROL_SAME_INPUT, TWO_D))
            else:
                print("It does not clear the same-input control (%.4f), so the "
                      "mechanism has not paid for itself even against the "
                      "handicapped baseline." % CONTROL_SAME_INPUT)
        else:
            print("This is a Kinetics arm, so the same-input control does not "
                  "apply to it -- that control is our 2D weights at 384px on 8 "
                  "frames, an input this arm never saw. Against the shipped "
                  "model it is %+.4f, with depth, pretraining and resolution "
                  "all still confounded." % (best["honest"] - TWO_D))
    else:
        print("\nNo arm has produced a dump yet.")

    if args.out:
        Path(args.out).write_text(json.dumps({
            "references": {"two_d": TWO_D,
                           "control_same_input": CONTROL_SAME_INPUT,
                           "untrained_shift": UNTRAINED_SHIFT},
            "rows": rows,
        }, indent=2), encoding="utf-8")
        print("\nwrote %s" % args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
