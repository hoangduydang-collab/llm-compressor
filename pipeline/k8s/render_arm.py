#!/usr/bin/env python3
"""Render one GLM-5.3 quality-arm pod spec from glm53-quality-arm.yaml.tmpl.

WHY THIS IS A FILE AND NOT AN INLINE SHELL LOOP. Rendering it inline from Git
Bash cost two failed launches and about nine hours:

  1. MSYS path conversion rewrote the model argv. `/mnt/cephfs/...` arrived as
     `C:/Program Files/Git/mnt/cephfs/...`, which the arm's own preflight caught
     -- but only after the pod was scheduled.
  2. Setting MSYS_NO_PATHCONV=1 to fix that broke the OUTPUT argv instead: the
     `/c/Users/...` destination stopped being translated, so Python resolved it
     against the current drive and wrote the corrected spec to a bogus `C:\\c\\...`
     tree. The render printed the right model path from memory while writing to a
     different file, and the STALE spec got applied -- so the second launch failed
     exactly like the first, and the printed output said it should have worked.

The lesson is not "remember the env var". It is that a renderer must validate its
own output instead of trusting the shell that invoked it, so this one:

  * refuses a model path that is not an absolute /mnt cluster path,
  * refuses any leftover @@PLACEHOLDER@@,
  * parses the result as YAML and re-reads the file it just wrote,
  * prints the values it actually PERSISTED, never the in-memory ones.
"""

from __future__ import annotations

import argparse
import pathlib
import re
import sys

_TMPL = pathlib.Path(__file__).with_name("glm53-quality-arm.yaml.tmpl")
_PLACEHOLDER = re.compile(r"@@[A-Z_]+@@")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--arm", required=True, choices=["ours", "phala"])
    ap.add_argument("--model", required=True, help="absolute /mnt path to the checkpoint")
    ap.add_argument("--run-tag", required=True)
    ap.add_argument("--ref", required=True, help="llm-compressor commit to pin")
    ap.add_argument("--out", required=True)
    ap.add_argument("--reasoning", default="", choices=["", "reasoning", "nonreasoning"])
    ap.add_argument("--limit", default="", help="items per general task; empty = full populations")
    ap.add_argument("--tasks", default="", help="GENERAL_TASKS override; empty = profile default")
    ap.add_argument("--aa-gpqa", default="", choices=["", "1"],
                    help="1 = run NVIDIA gpqa_diamond_aa_v3 after serve gates; empty = skip")
    ap.add_argument("--node", default="", help="pin to this node; empty = let the scheduler choose")
    a = ap.parse_args(argv)

    # A mangled path must never reach a pod spec, whatever shell called us.
    if not a.model.startswith("/mnt/"):
        print(f"REFUSING: --model {a.model!r} is not an absolute /mnt cluster path.\n"
              "  If this looks like C:/Program Files/Git/mnt/... then Git Bash "
              "rewrote it; invoke this from PowerShell, or pass the path through "
              "an environment variable rather than argv.", file=sys.stderr)
        return 2

    # An EMPTY selector must still be a valid mapping entry. Emitting a bare "{}"
    # here produced `  {}` on its own line, which is a YAML scanner error in a
    # mapping context -- caught by this module's own safe_load, which is the whole
    # reason that check exists. `nodeSelector: {}` means "any node".
    node_block = ("nodeSelector: {kubernetes.io/hostname: %s}" % a.node
                  if a.node else "nodeSelector: {}")
    text = _TMPL.read_text(encoding="utf-8")
    for key, value in {
        "@@ARM@@": a.arm,
        "@@MODEL@@": a.model,
        "@@RUN_TAG@@": a.run_tag,
        "@@REF@@": a.ref,
        "@@LIMIT@@": a.limit,
        "@@TASKS@@": a.tasks,
        "@@AA_GPQA@@": a.aa_gpqa,
        "@@REASONING@@": a.reasoning,
        "@@NODESELECTOR@@": node_block,
    }.items():
        text = text.replace(key, value)

    left = sorted(set(_PLACEHOLDER.findall(text)))
    if left:
        print(f"REFUSING: unsubstituted placeholders remain: {left}", file=sys.stderr)
        return 1

    out = pathlib.Path(a.out)
    # newline="\n": the pod spec is consumed by Linux tooling, and a CRLF spec has
    # already bitten us once via a shell profile whose values kept the CR.
    out.write_text(text, encoding="utf-8", newline="\n")

    # Re-READ what landed on disk. The whole point: an in-memory value that was
    # never persisted is what made the previous failure invisible.
    try:
        import yaml
    except ImportError:
        print("WARNING: pyyaml absent; wrote the spec without parsing it", file=sys.stderr)
        print(f"wrote {out}")
        return 0

    doc = yaml.safe_load(out.read_text(encoding="utf-8"))
    container = doc["spec"]["containers"][0]
    env = {e["name"]: e.get("value") for e in container["env"] if "value" in e}
    if env.get("MODEL_PATH") != a.model:
        print(f"REFUSING: persisted MODEL_PATH {env.get('MODEL_PATH')!r} does not "
              f"match the requested {a.model!r}", file=sys.stderr)
        return 3

    print(f"wrote {out}")
    print(f"  pod      {doc['metadata']['name']}")
    print(f"  node     {doc['spec'].get('nodeSelector') or '(scheduler chooses)'}")
    print(f"  gpus     {container['resources']['limits']['nvidia.com/gpu']}")
    print(f"  model    {env['MODEL_PATH']}")
    print(f"  ref      {env['REPO_REF']}")
    print(f"  profile  {env['PROFILE']}")
    print(f"  tasks    {env['GENERAL_TASKS'] or '(profile default)'}"
          f"  limit={env['LIMIT'] or '(full populations)'}"
          f"  reasoning={env['REASONING_MODE'] or '(none)'}"
          f"  aa_gpqa={env.get('AA_GPQA') or '(skip)'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
