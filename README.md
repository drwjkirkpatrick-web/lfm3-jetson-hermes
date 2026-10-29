# LFM3-1.7B on Jetson 8GB — Verified Profile

Liquid AI LFM3-1.7B (released 2026-10-27) tuned for the Jetson 8GB +
llama.cpp stack. Two quants, fully benchmarked: standard `Q4_K_M` and
Liquid's quantization-aware-distilled `QAD-Q4_0`.

**Status: suite + probes complete; BFCL v2 + 256K ceiling test running.**

## Models

| File | Size | Quant |
|---|---|---|
| `~/models/new-zoo/LFM3-1.7B-Q4_K_M.gguf` | 1.05 GB | standard Q4_K_M |
| `~/models/new-zoo/LFM3-1.7B-QAD-Q4_0.gguf` | 1001 MB | QAD — distilled to recover Q4_0 accuracy |

Architecture (`lfm3`, requires llama.cpp `2e0d86be` / v0.8.0-dev+):
24 blocks = 13 gated short-conv + 11 attention (5 SWA + 6 global),
4 KV heads × 128 head dim → **~22.5 KB/token f16 KV**.
**256K native context.** ChatML template with native `<tools>` injection
and structured tool-call emission — works with the llama.cpp autoparser
out of the box, no template fix needed.

## Verified launch profile

```bash
./launch_lfm3.sh q4km        # or: qad
```

- `-ngl 99 -c 65536 -ctk f16 -ctv f16` — 64K f16 context ≈ 1.5 GB KV
  (clears the Hermes agent 64K minimum with ~2 GB headroom)
- `GGML_CUDA_ENABLE_UNIFIED_MEMORY=1 GGML_CUDA_NO_PINNED=1` (both required)
- **Sampling: temp 0.1 / top-k 50 / repeat-penalty 1.1** (vendor rec).
  **Never serve at greedy temp 0.0** — the model loops inside its think
  block on open-ended prompts.
- `--chat-template-file templates/lfm3-nothink.jinja` — sets
  `enable_thinking` default false. The model re-opens its own think block
  anyway (training-baked); the value is dropping thinking from multi-turn
  history.

## Speed (12-prompt suite, 64K f16 ctx)

| Config | Generation |
|---|---|
| Q4_K_M, greedy temp 0.0 | 33.8 t/s |
| Q4_K_M, vendor sampling | 33.9 t/s |
| **QAD-Q4_0, vendor sampling** | **36.4 t/s** |

For scale: Ornith-1.0-9B IQ3_M runs 9.4 t/s; Gemma-4-E4B QAT 17-18 t/s;
Granite 4.0-3B 19-20 t/s. LFM3-1.7B is the fastest tool-calling model in
the zoo — at a quarter of the file size of the 9B.

## Key findings

1. **It's a planning thinker.** Complex one-shot tasks are fully reasoned
   in `reasoning_content` (~4.6K tokens of planning for a full HTML page)
   *before* any content is emitted — then the output arrives complete and
   valid in one pass (doctype → `</html>`). **Budget max_tokens ≥ 8000
   for one-shot generative work**; 4096 caps die mid-plan with empty
   content. Tool calls need only 512 (observed thinking: 60-428 tok).
2. **Thinking cannot be suppressed.** `enable_thinking=false` prefills a
   closed empty think block (template lines 273-280); the model re-opens
   its own. Thinking lands cleanly in `reasoning_content` — agent-stack
   compatible (bounded, terminates).
3. **Native structured tool calls work day-zero**: `finish_reason:
   tool_calls`, correct JSON args, correct tool selection on 4/6 suite
   prompts + correct direct-draft abstention on the no-tool prompt.
   Recurring near-miss: the SQLite prompt (wants `terminal` + a sqlite3
   query) got `web_search` (greedy) / `read_file` (Q4 vendor) / `terminal
   "ls -la clinic.db"` (QAD) — reads the right target, never writes the query.
4. **QAD-Q4_0 beats Q4_K_M on speed** (+7%) at equal behavior — the
   distilled checkpoint is the better pick pending BFCL accuracy results.

## Benchmarks

- 12-prompt suite results: `/tmp/test_results_lfm3-1.7b-{q4km,q4km-v2,qad}.json`
- BFCL v2 (52 tests, both quants): **running** —
  `~/projects/bfcl-jetson-test-v2/results/lfm3-*.json`
- 256K q4_0 KV ceiling test: **pending**

## Hermes agent link (granite4-jetson-hermes pattern)

```bash
hermes config set model.default lfm3-1.7b-qad
hermes config set model.provider custom
hermes config set model.base_url http://127.0.0.1:8090/v1
hermes config set model.api_key none
hermes config set model.context_length 65536   # match launched profile
# then: hermes gateway restart (from a separate shell — the gateway cannot
# restart itself, and a restart kills locally-launched llama-servers)
```

Caveats before switching the main agent: 1.7B-class reasoning — expect
BFCL-tier tool calling but well below GLM-5.3/Ornith on hard analysis.
Best fit: fast local triage, voice assistant, tool-routing workloads.