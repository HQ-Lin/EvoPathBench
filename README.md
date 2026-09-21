# EvoPathBench

EvoPathBench is a longitudinal benchmark for evaluating artifact-level self-evolution in decision agents. It separates three questions that endpoint performance conflates:

1. **Learning generalization:** does experience improve performance on unseen
   near-distribution and transfer episodes?
2. **Capability retention:** does a learned capability survive subsequent
   updates on unrelated tasks?
3. **Rule adaptation:** can the agent revise an outdated rule after receiving
   counterevidence?

The released finance track uses a controlled, semi-synthetic market as the experimental carrier. 

## Release contents

```text
EvoPathBench/
├── data/finance/        # 4,800 episodes and 1,800 longitudinal streams
├── analysis/            # paired metrics and design-aware inference
├── results/main/        # aggregate, non-identifying paper results
├── scripts/             # offline validation and dataset summaries
├── tests/               # release integrity test
├── LICENSE
└── requirements.txt
```

Raw model prompts, responses, request identifiers, service endpoints, account
metadata, credentials, local paths, and provider-specific launch scripts are
intentionally excluded from this release.

## Dataset

The finance track contains six task families across two environment layers, with 4,800 episode specifications and 1,800 ordered task streams. Each stream provides five update opportunities and six frozen checkpoints. Episode rows are generated instances rather than independent semantic tasks, so statistical analysis should preserve stream and campaign clustering.

The dataset uses three evaluation templates:

- **Accumulation** provides consistent learning evidence and evaluates the
  resulting artifact on unseen near-distribution and transfer episodes.
- **Interference** evaluates the same retention anchor before and after updates
  from unrelated task families.
- **Reversal** introduces counterevidence that changes a previously useful
  relation and evaluates adaptation after successive updates.

The dataset files are organized as follows:

- `data/finance/manifest.json` records the schema, counts, generation settings,
  and file hashes.
- `data/finance/episodes.jsonl` contains parameterized episode specifications.
- `data/finance/streams.jsonl` contains ordered learning events and checkpoint
  probes.
- `data/finance/resources/` contains frozen calibration summaries and
  diagnostic resources.
- `data/finance/validation_report.json` records structural and simulator
  validation results.


## Quick start

Dataset validation uses only the Python standard library:

```bash
python scripts/validate_dataset.py --dataset data/finance
python scripts/summarize_dataset.py --dataset data/finance
python -m unittest discover -s tests -v
```

The statistical scripts additionally require NumPy and pandas:

```bash
python -m pip install -r requirements.txt
python analysis/evaluate_dimensions.py --help
```



## License

Code and generated benchmark artifacts are released under the MIT License. The public market statistics used for calibration remain subject to their original data-provider terms. This repository does not redistribute the raw market archives.

## Running model inference

The `evolens/` package contains the implementation used to execute benchmark episodes. It includes the simulator, model-facing prompts, persistent memory and skill updates, candidate validation, state-off evaluation, and generation of episode-level records. The implementation namespace is retained for compatibility with the frozen experiment code.

Install the package with the inference dependency:

```bash
python -m pip install -e ".[inference]"
```

The model adapter in `evolens/agent.py` calls an OpenAI-compatible Chat
Completions endpoint through HTTPX and does not depend on a provider-specific
SDK. The default URL reproduces the public endpoint used in our experiments.
Other providers and self-hosted models can be used by passing their HTTPS
`/chat/completions` URL with `--base-url` and `--allow-custom-base-url`. The
endpoint must accept the standard `model`, `messages`, `temperature`,
`max_tokens`, and JSON response-format fields.

Credentials are read only from an environment variable and are never stored in
the repository. The default variable is `EVOPATHBENCH_API_KEY`; use
`--api-key-env` to select a different variable. A bounded smoke test can be run
with:

```bash
export EVOPATHBENCH_API_KEY="<your-key>"
python -m evolens model-smoke \
  --model <model-name> \
  --condition episodic_memory
```

For another OpenAI-compatible provider:

```bash
export OTHER_PROVIDER_KEY="<your-key>"
python -m evolens model-smoke \
  --model <model-name> \
  --base-url https://provider.example/v1/chat/completions \
  --allow-custom-base-url \
  --api-key-env OTHER_PROVIDER_KEY \
  --condition episodic_memory
```

A small longitudinal evaluation can be inspected without sending requests by
adding `--dry-run`:

```bash
python -m evolens evaluate-model \
  --dataset data/finance \
  --output-root runs \
  --model <model-name> \
  --conditions baseline,skillboost \
  --templates accumulation \
  --max-streams 1 \
  --campaigns 1 \
  --repeats 1 \
  --max-calls 500 \
  --dry-run
```

Remove `--dry-run` only after checking the estimated call budget. The supported conditions are `baseline`, `reflection`, `context`, `episodic_memory`, `consolidated_memory`,`skillopt`, `skillboost`, `skillx`, `trace2skill`, and `skillgrad`. Raw model responses are disabled by default. Run directories may contain request metadata and locally resolved paths, so they are excluded from version control and should be reviewed separately before sharing.
