# Groundline

A causal evaluation environment for hierarchical language-agent organizations. A deterministic operational world computes business truth. Persistent employees receive scoped evidence, take permitted actions, and decide what to report upward. The repository and Python package retain the working name `groundline`.

The narrow research claim is not "LLMs can simulate a company." It is that hierarchical reporting can be treated as a causal, replayable measurement problem: keep world randomness fixed, intervene on organizational conditions, and estimate how information changes as it climbs.

## Built with GPT-5.6 and Codex

Groundline was developed primarily with [Codex](https://developers.openai.com/codex/), from the initial research and architecture through the deterministic Python engine, statistical design, React control room, regression tests, browser validation, and final integration. [GPT-5.6](https://developers.openai.com/api/docs/guides/latest-model?model=gpt-5.6) was also used directly as the coding and reasoning model during a bounded compatibility and hardening phase.

There is one important implementation-history detail. Immediately after the GPT-5.6 rollout, early issues in the Codex client temporarily interrupted this project by consuming excessive amount of credits. For part of that hardening work, GPT-5.6 was accessed through Claude Code using CLIProxyAPI. Development then returned to Codex, where the final fixes, aggregate verification, documentation, and submission pass were completed. Claude Code and CLIProxyAPI are not runtime dependencies of Groundline.

GPT-5.6 is also supported at runtime as the live employee policy. The adapter sends only an employee's authorized context to an OpenAI-compatible chat-completions endpoint and requires a structured JSON decision. It never sends hidden world truth and never allows model output to become authoritative state. Live decisions can be recorded once, replayed without credentials, and rerun in network-disabled locked mode.

The key engineering decisions made with Codex were:

- operational truth belongs exclusively to the deterministic world engine;
- agents are never instructed to lie, conceal, or exaggerate;
- causal comparisons pair every treatment on the same seeds;
- fresh hosted output becomes reproducible only after it is recorded in the decision ledger;
- organizations are validated rooted trees, not a hardcoded pyramid;
- fixture-policy results verify the instrument but are not presented as evidence about GPT-5.6 behavior.

## Run the complete offline experiment

Prerequisites: Python 3.13, `uv`, Node 24, and npm.

```powershell
uv sync
npm --prefix frontend install
npm --prefix frontend run build
uv run groundline experiment --config configs/demo.yaml --artifacts artifacts
uv run groundline serve --artifacts artifacts
```

Open `http://127.0.0.1:8000`. The backend serves the artifact API and compiled interface. The operator controls launch fresh 2×2 paired intervention matrices; SQLite-backed job progress and expired-lease recovery are exposed through the same API.

A normal wheel build runs the frontend build and packages the compiled same-origin application:

```powershell
uv build --wheel
uv pip install dist/groundline-*.whl
```

The demo experiment is a general rooted tree with 13 persistent employees: nine contributors, three department directors, and one executive across Product, Engineering, and QA. The topology engine also accepts deeper, unbalanced trees.

## Run one company trajectory

```powershell
uv run groundline run --config configs/demo.yaml --seed 7 --policy fixture
```

Use GPT-5.6, or another language model exposed through an OpenAI-compatible chat-completions endpoint, as the employee policy. Configure the bare model name, base URL, and API key in a `.env` file (see `.env.example`) or export them as environment variables — real environment variables take precedence over the file:

```powershell
# .env
GROUNDLINE_MODEL=gpt-5.6
GROUNDLINE_API_BASE=https://api.openai.com/v1
GROUNDLINE_API_KEY=your-key

# Record mode captures unseen decisions from the provider:
uv run groundline run --config configs/demo.yaml --seed 7 --policy record --artifacts artifacts

# Any OpenAI-compatible endpoint works the same way — set GROUNDLINE_API_BASE
# to its URL and GROUNDLINE_MODEL to a bare model name. Providers without a
# native OpenAI-compatible surface (e.g. AWS Bedrock) must be reached through
# an OpenAI-compatible proxy.
```

Live decisions request JSON-object structured output, receive no hidden world state, and are cached by the hash of the complete authorized context. `record` captures unseen decisions. `locked` permits only already-captured decisions and fails before making a network call on a cache miss. Every finalized run can be reconstructed exactly without a network call:

```powershell
# After a record run, replay works with zero provider credentials:
uv run groundline replay artifacts/<run-id>

# Or re-run with locked mode — no network call, fails on unseen context:
uv run groundline run --config configs/demo.yaml --seed 7 --policy locked --model gpt-5.6 --artifacts artifacts
```

## Artifact contract

Each run contains:

- `request.json`: complete seed, organization, scenario, and treatment
- `events.jsonl`: truth, observations, verifications, reports, decisions, metrics, and consequences
- `decisions.jsonl`: context hashes and structured policy outputs
- `metrics.json`: tick-level distortion by agent, department, and hierarchy depth
- `manifest.json`: schema, policy, and engine fingerprints plus request, event, decision, and metrics hashes and counts

The manifest is published last. Shared verification rejects malformed, incomplete, tampered, identity-mismatched, or replay-inconsistent artifacts before replay, analysis, resume, or API reads.

Each experiment adds an atomic execution state, paired run index, typed JSON analysis, CSV and Parquet seed-level outcomes, and a Markdown report. Inference includes exact or seeded Monte Carlo sign-flip tests, deterministic BCa intervals with explicit percentile fallback, Holm adjustment within declared confirmatory families, preregistered sensitivities, within-seed 2×2 factorial contrasts, design-resolution diagnostics, and a report-level mixed-effects model with seed and reporting-agent random intercepts. Fixture-policy results are engineering verification, not evidence about language-agent behavior.

## Verification

```powershell
uv run pytest -q
uv run ruff check .
uv run ruff format --check .
uv run mypy
npm --prefix frontend run test
npm --prefix frontend run build
npm --prefix frontend run test:e2e
```

## Architecture and evidence

| Module | Responsibility |
|---|---|
| [`world/engine.py`](src/groundline/world/engine.py) | Deterministic authoritative business state |
| [`organization/models.py`](src/groundline/organization/models.py) | Arbitrary reporting-tree validation and derived topology |
| [`observation/engine.py`](src/groundline/observation/engine.py) | Scoped local evidence and manager verification |
| [`openai_compat_policy.py`](src/groundline/policy/openai_compat_policy.py) | GPT-5.6/OpenAI-compatible structured decisions, retries, and record/locked behavior |
| [`simulation/runner.py`](src/groundline/simulation/runner.py) | Tick, report, action, consequence, and metric orchestration |
| [`events/store.py`](src/groundline/events/store.py) | Canonical event ledger and finalized artifact manifest |
| [`replay/engine.py`](src/groundline/replay/engine.py) | Zero-network reconstruction and equivalence checks |
| [`experiments/runner.py`](src/groundline/experiments/runner.py) | Paired intervention execution, resume, recovery, and exports |
| [`statistics/inference.py`](src/groundline/statistics/inference.py) | Seed-level causal inference and sensitivity analysis |
| [`api/app.py`](src/groundline/api/app.py) | Artifact queries, experiment jobs, and compiled control-room hosting |

## LICENSE 
Apache 2.0
