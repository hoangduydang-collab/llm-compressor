# Aug 25 - Aug 29

## Duy

### What I worked on

**The DeepSeek-V4-Flash out-of-memory crash.** Our deployment works around it by
processing prompts in small pieces, which is slow, so the question was whether that
setting can be raised and what raising it costs. I checked what already existed before
building anything, then ran four experiments on 4×H100 to reproduce the crash and test
the best candidate fix.

**GLM-5.2 AWQ.** The first end-to-end run of our own quantization on a model family
other than MiniMax-M3.

### Key results — DeepSeek-V4-Flash

**Cause.** Sparse attention builds a temporary buffer sized chunk × conversation
length, and the engine's memory planning does not account for it at all. At the 1M
token limit that is 8 GB per GPU nobody budgeted for, which is what took production
down on 17 Aug. Worth being clear that this is a gap in one code path rather than
anything inherent to the model — an older implementation of the same feature, in the
same codebase, already carries the check that is missing here. That is most of the
reason the fix is low risk to take.

**The upstream fix.** Without it the server dies on an 800k-token prompt, and not just
that request: the whole engine goes down, from a single user. With it the same server
handles 1M tokens at 0.94–0.99× of previous speed, so there is no measurable cost, and
it held at 95% KV cache occupancy, which is the condition our incident happened under.

**Chunk size.** Raising it is not free. 2048 to 4096 costs no KV cache at all, but 8192
costs around 8%, because memory unrelated to the buffer also grows with chunk size.
With prefill CUDA graphs enabled the fix stops applying, since the buffer is fixed when
the graph is recorded, and 8192 in that configuration would cost roughly a third of the
cache. That combination is the one to avoid.

### Key results — GLM-5.2 AWQ

**Weight loading is the blocker, and it is storage rather than the pipeline.** The model
is 1,485 GB and our shared storage reads at about 31 MB/s, so it sat on the critical
path of every step: 86 minutes to walk layers we are not even quantizing, against
roughly 4 minutes to compress one. Holding 900 GB in RAM helped, though what actually
capped it was the container's shared-memory size rather than the memory budget itself.

**Current state.** The 8-GPU AWQ run is about 54% through loading after 45 minutes.
Several earlier AWQ and GPTQ runs failed and I am working through those.

### Plan for next week

GLM-5.2 AWQ through to a finished model, then the paired quality check against BF16. It
is the only item with a deadline I do not control, so it goes first.

Recommend adopting the DeepSeek-V4-Flash fix. Whether to raise the chunk size is a
separate call, and it now has numbers behind it.

Finish the serving measurements that the memory work displaced.
