# Design: Reader-first MiniMax-M3 spec-dec comparison note

## Reader and decision

The primary reader is a collaborator deciding whether synthetic versus real
prompt source explains a differing speculative-decoding result. They need the
answer and its evidentiary limit before setup details or reproduction commands.

## Information architecture

Replace the 272-line reference note with a decision memo of approximately 100
lines, in this order:

1. A title and short takeaway: acceptance is 2.450 versus 2.473 (+0.9%), which
   the primary study treats as noise; the comparison is cross-window and not a
   paired statistical test.
2. A compact `What was compared` table describing synthetic random-token
   prompts and first-turn ShareGPT text, including their input shapes and
   natural-stopping output policy.
3. A `Results` table with accepted length, per-position acceptance, and
   concurrency-1 per-user throughput; one sentence on the concurrency-10
   ShareGPT support point.
4. A `What this means for a differing result` section: prompt source alone is
   not supported as the explanation. List the more consequential controls to
   compare: output policy, temperature, draft depth, target/drafter pair, and
   topology.
5. A compact `Inspect or reproduce` section: private archive host,
   owner-provided access, window names, aggregators, and the Wave 1 missing AA
   summary limitation.

## Constraints

- Preserve the factual values and cross-window instrumentation caveat from
  `docs/m3-specdec-eagle3.md`.
- Keep the private archive host `138.252.188.36`, but do not disclose its
  owner-specific path.
- Do not add a second runbook or repeat serving-environment prerequisites.
- Link to the primary report and plan for readers who need full setup detail.
- Do not assert a collaborator configuration or explain the discrepancy
  without their data.

## Validation

Confirm the rewritten note:

- contains the specified result and caveat;
- identifies the exact two prompt constructions;
- directs a collaborator to compare the named control variables;
- retains a non-sensitive artifact-access path;
- is about 100 lines and has no placeholder language.
