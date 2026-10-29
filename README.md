# LFM3-1.7B on Jetson 8GB — Verified Profile

Liquid AI LFM3-1.7B (released 2026-10-27) tuned for the Jetson 8GB +
llama.cpp stack. Two quants, fully benchmarked: standard `Q4_K_M` and
Liquid's quantization-aware-distilled `QAD-Q4_0`.

**Winner: QAD-Q4_0 — faster (36.4 vs 33.9 t/s) AND more accurate
(BFCL 48/52 vs 47/52). It is the default pick.**

## Models

| File (in `~/models/new-zoo/`) | Size | BFCL v2 | Speed |
|---|---|---|---|
| `LFM3-1.7B-QAD-Q4_0.gguf` **(default)** | 1001 MB | **48/52 (92.3%)** | 36.4 t/s |
| `LFM3-1.7B-Q4_K_M.gguf` | 1.05 GB | 47/52 (90.4%) | 33.9 t/s |

Architecture (`lfm3`, requires llama.cpp `2e0d86be` / v0.8.0-dev+):
24 blocks = 13 gated short-conv + 11 attention (5 SWA + 6 global),
4 KV heads × 128 head dim → **~22.5 KB/token f16 KV**.
**256K native context.** ChatML template with native `<tools>` injection
and structured tool-call emission — works with the llama.cpp autoparser
out of the box, no template fix needed.

## Verified launch profiles

```bash
./launch_lfm3.sh qad               # 64K f16 profile (default)
./launch_lfm3.sh qad 8090 256k     # full native 256K, q4_0 KV
```

| Profile | Flags | RSS | Use |
|---|---|---|---|
| 64k (default) | `-c 65536 -ctk f16 -ctv f16` | ~3.0 GB | best KV quality, clears the Hermes 64K agent minimum |
| 256k | `-c 262144 -ctk q4_0 -ctv q4_0` | ~3.3 GB | **full native context** — ~4 GB still free for co-resident services |

Both: `-ngl 99`, `GGML_CUDA_ENABLE_UNIFIED_MEMORY=1 GGML_CUDA_NO_PINNED=1`
(both required), `--jinja --fit off`, b/ub 512, `-fa on`, and
`--chat-template-file templates/lfm3-nothink.jinja`.

## BFCL v2 results (52 tests, 2026-10-28)

| Category | QAD-Q4_0 | Q4_K_M |
|---|---|---|
| simple | 28/30 (93.3%) | 28/30 (93.3%) |
| moderate_parallel | 11/12 (91.7%) | 10/12 (83.3%) |
| **moderate_chained** | **8/8 (100%)** | **8/8 (100%)** |
| coding | 1/2 | 1/2 |
| **Overall** | **48/52 (92.3%)** | **47/52 (90.4%)** |

Zoo context: ties Gemma-4-E4B-QAT and Granite 4.1-3B (92.3%); ahead of
Spark-X2.5-4B (90.4%); Granite 4.0-3B keeps the accuracy crown (96.2%) —
at 2x the file size, ~half the speed, and 1/4 the context. **Perfect 8/8
chained on both quants** — before LFM3 only E4B had ever done that.
Deriving step-2 tool calls from returned data is this model's superpower
at 1.7B.

## Key findings

1. **It's a planning thinker.** Complex one-shot tasks are fully reasoned
   in `reasoning_content` (~4.6K tokens of planning for a full HTML page)
   *before* any content — then the output arrives complete and valid in
   one pass (doctype → `</html>`). **Budget max_tokens ≥ 8000 for one-shot
   generative work**; a 4096 cap dies mid-plan with empty content. Tool
   calls need only 512 (observed thinking: 60-428 tok).
2. **Never serve at greedy temp 0.0** — the think block never closes on
   open-ended prompts (suite html/poetry/creative prompts burned their
   whole budget in reasoning at temp 0). Vendor sampling — **temp 0.1 /
   top-k 50 / repeat-penalty 1.1** — is required and verified.
3. **Thinking cannot be suppressed.** `enable_thinking=false` prefills a
   closed empty think block (template lines 273-280); the model re-opens
   its own. The nothink template override's real value is dropping
   thinking from multi-turn history (shorter prompts). Thinking is
   bounded and lands cleanly in `reasoning_content`.
4. **Native structured tool calls day-zero**: `finish_reason: tool_calls`,
   clean JSON args. Recurring near-miss: the SQLite prompt (wants
   `terminal` + a sqlite3 query) gets `read_file` / `ls -la clinic.db` —
   finds the right target, never writes the query.
5. **Long context is nearly free.** A 14.5K-token prompt prefilled at
   ~1,400 tok/s (hybrid conv layers); full 256K KV is only ~1.44 GB at
   q4_0. First zoo model serving its entire native window on 8 GB.
6. **Coding split**: html_profile 6/7 passes even at greedy; python_game
   fails at the harness's greedy sampling (think-budget starvation). At
   vendor sampling + max_tokens 8192 the same prompt produces complete
   valid HTML — the failure is the benchmark's greedy setting, not the
   model.

## Speed (12-prompt suite, 64K f16 ctx)

| Config | Generation |
|---|---|
| Q4_K_M, greedy temp 0.0 | 33.8 t/s |
| Q4_K_M, vendor sampling | 33.9 t/s |
| **QAD-Q4_0, vendor sampling** | **36.4 t/s** |

Scale: Ornith-1.0-9B IQ3_M 9.4 t/s · Gemma-4-E4B QAT 17-18 t/s ·
Granite 4.0-3B 19-20 t/s · **LFM3-1.7B 36.4 t/s** — fastest tool-caller
in the zoo at a quarter of the 9B's file size.

## Hermes agent link (granite4-jetson-hermes pattern)

```bash
hermes config set model.default lfm3-1.7b-qad
hermes config set model.provider custom
hermes config set model.base_url http://127.0.0.1:8090/v1
hermes config set model.api_key none
hermes config set model.context_length 65536   # or 262144 on the 256k profile
# then: hermes gateway restart (from a separate shell — the gateway cannot
# restart itself, and a restart kills locally-launched llama-servers)
```

Best fit: fast local triage, tool-routing, voice assistant, long-context
RAG summarization. Not a replacement for GLM-5.3/Ornith on hard analysis
— 1.7B-class reasoning, E4B-class tool calling.

## Artifacts

- Suite results: `/tmp/test_results_lfm3-1.7b-{q4km,q4km-v2,qad}.json`
- BFCL results: `~/projects/bfcl-jetson-test-v2/results/lfm3-1.7b-*.json`
  (archived to `~/projects/bfcl-jetson-test-v2-results/`)
- llama.cpp build: `2e0d86be` (v0.8.0-dev build 1597); rollback:
  `~/llama.cpp/bin-backup-1a24df16/`