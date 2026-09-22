# Contributing guidelines

## Response Times

| Category | Target Response Time |
|----------|---------------------|
| Issues | 1 week |
| External pull requests | 2 weeks |
| Security vulnerabilities | 48 hours |
| RFCs | 2 weeks |

If you haven't received a response within these timeframes, feel free to ping the thread.

## AI Contribution Policy

- All pull requests must reference an actionable issue.
- AI-generated code is welcome, but the contributor is fully responsible for its correctness, testing, and compliance with project guidelines.

## Setup Development Environment

```sh
# Clone the repository
git clone <repository-url>

# Install prerequisites
# Installs necessary drivers and bazel (See BAZEL_BUILD.md)
./prerequisites.sh

# Create and activate virtual environment
uv venv
source .venv/bin/activate

#If permission are needed, execute below command
chmod +x ./tools/*

#Perform the build
./tools/build

## Install neuron compiler from neuron repo
uv pip install 'neuronx-cc==2.*' --extra-index-url https://pip.repos.neuron.amazonaws.com --prerelease=allow --index-strategy unsafe-best-match
uv pip install nki --extra-index-url https://pip.repos.neuron.amazonaws.com --prerelease=allow --index-strategy unsafe-best-match

# Install development dependencies including Ruff
uv pip install ruff

# Install pre-commit hooks
uv pip install pre-commit
pre-commit install
```

> [!NOTE]
> **Why `uv pip` instead of `pip`?** Virtual environments created by `uv` don't include `pip` by default. When your venv is activated, running `pip` will use the system pip (`/usr/bin/pip`) instead of the venv's. Always use `uv pip` for installing packages, or install pip into the venv with `uv pip install pip`.

> [!NOTE]
> **When to use `uv run`?** Use `uv run <command>` when your venv is **not activated** — it automatically finds the venv in current dir and runs the command inside it. If your venv is already activated (check with `echo $VIRTUAL_ENV`), you can run commands directly without `uv run`. Keep in mind that `uv run` looks for `.venv` in the current or parent directories, so it may fail if your venv is located elsewhere.

### Building and Testing

**Build only:**
```sh
./tools/build
```

**Build and generate wheel:**
```sh
./tools/build --wheel
```

**Build documentation:**
```sh
uv sync --group docs
./tools/build --docs
```
Documentation will be generated in `build/docs/html/`.

**Generate wheel only (after build):**
```sh
USE_BAZEL=1 python setup.py bdist_wheel release
```

**Python tests:**
```sh
# Run all tests in parallel
./tools/build-and-test-parallel

# Run all tests sequentially
./tools/build-and-test

# Run specific test
./tools/run-test device/test_device.py

# Pass pytest options
./tools/run-test device/test_device.py -v
```

**C++ tests:**
```sh
# Run all C++ unit tests
./tools/run-cpp-test

# Run specific test
./tools/run-cpp-test  //tests/csrc:KernelExecutionTest

# Verbose output
./tools/run-cpp-test -v

# Rebuild and test
./tools/run-cpp-test -r

# To run the profiler test suite.
./tools/run-cpp-test //:ProfilerTests
```

### Test Reports and Coverage

> [!NOTE]
> Currently all sync tests are running with the flag `NEURON_LAUNCH_BLOCKING=1`

```sh
./tools/run-test-parallel                              # HTML reports (default)
./tools/run-test-parallel --coverage                   # With coverage analysis
./tools/run-test-parallel --skip-distributed           # Skip distributed tests
./tools/run-test-parallel --coverage --skip-distributed
```

**Reports:** `test-reports/report.html`, `test-reports/coverage/index.html`

### Huggingface Tests

Tests top 50 Causal-LM models. Excluded by default (must specify path explicitly):

```sh
./tools/run-test tests/huggingface                    # All HF tests
./tools/run-test tests/huggingface/test_hf_models.py  # Specific test
./tools/run-test-parallel tests/huggingface           # Parallel
```

### Linting and Formatting

```sh
uv run ruff check .                            # Check linting
uv run ruff check . --fix                       # Auto-fix
uv run ruff format .                           # Format Python
clang-format -i torch_neuronx/csrc/**/*.cpp     # Format C++
uv run pre-commit run --all-files               # Run all checks
```

## CI Expectations

Every pull request triggers two CI workflows that must pass before merge:

1. **Lint** — runs pre-commit checks (Ruff linting/formatting, clang-format) on `ubuntu-latest`.
2. **PR Checks** — runs the full test suite on self-hosted trn2 hardware:
   - ATen unit tests (with coverage)
   - Distributed unit tests
   - Block performance tests (MFU thresholds enforced)
   - Torch Compile PyTorch unit tests
   - C++ unit tests
   - Concatenation unit tests
   - Torch-MLIR LIT tests

All jobs must pass. If a failure looks unrelated to your change, comment on the PR and a maintainer will investigate.

## Design Review (RFCs)

Some changes need a design review **before** the implementation is written. It's easier to change
direction while a feature is still an idea than once a large PR exists.

Open an RFC if your change touches shared runtime machinery (allocator, streams, dispatcher,
collectives), adds or redefines public API or environment variables, is hard to reverse, or makes a
deliberate performance/memory tradeoff. This also applies to cross-cutting changes where one
components's requirement lands in another component — recommended to have a design before a PR diff

Bug fixes with one obvious correct form, new op coverage, tests, docs, and behaviour-preserving
refactors generally do **not** need an RFC. Send those as a normal PR. If you're unsure, ask in a
regular issue, or just open the RFC — it's cheaper than a rejected PR.

### How

1. **Open an issue** with the **RFC (Design Proposal)** template, titled `[RFC] ...` and labelled
   `rfc`. The template is a guideline: mark sections that don't apply `n/a`, and add any it doesn't
   cover. If the design is long, link out to a doc, but keep the summary, motivation and proposal in
   the issue so the discussion has something to attach to.
2. **Say whether it's ready.** Use `rfc:draft` while you're still writing and `rfc:commenting` when
   you want review, so nobody has to guess.
3. **Bring in the right reviewers.** They're the owners of the code you're changing — check
   [.github/CODEOWNERS](.github/CODEOWNERS) and @-mention those owner groups. Response times are under
   [Response Times](#response-times).
4. **Revise until the affected owners agree on a path forward.** Strikethrough the changed sentences
   and update the issue body as the design changes, so it tracks the proposal changes.
5. **Write down the outcome** when review settles: what was decided, who agreed, and what happens
   next. An RFC that simply stops getting comments still needs a decision recorded.
6. **Then implement**, linking the RFC from the PR. Writing an RFC doesn't oblige you to build it —
   say so on the issue if you'd rather someone else picked it up.

Not every RFC ends in yes or no. One can also be **deferred**: the idea is reasonable, but nobody
wants to evaluate or own it yet. Close it with a comment on why, and on what would change the answer.

### What reviewers look for

This varies with different RFCs, but generally:

- **Evidence proportional to the claim.** A performance or memory change needs measurements, ideally
  a same-binary A/B — and a long-running one if it touches allocation or lifetime, since a short run
  can't show the absence of a leak. A correctness fix needs a repro that fails without it. An API
  change needs usage examples.
- **Precedent.** Does this mechanism already exist elsewhere in this repo, in upstream PyTorch, or in
  another backend? If there's no precedent, say so and explain why the case is novel.
- **Alternatives, including the ones you rejected**, and why — with data where you measured them.
- **Limitations, named by you** — what the change deliberately doesn't cover.
- **Blast radius**: what could break, how you'd notice, and how to roll it back.
- **Naming and docs**, for anything user-visible. If the terminology is new or existing docs stop
  being accurate

## Merging Process

1. Open a PR against `main` with a linked issue.
2. CI runs automatically. Fix any failures.
3. A maintainer will review your PR within the timeframes listed in
   [Response Times](#response-times). If you don't receive a response within
   that window, leave a comment on the PR to nudge it back onto our radar.
4. Address review feedback and push updates.
5. Once approved and CI is green, a maintainer will merge the PR.

## Expand op coverage

There are two ways to find what ops to implement:

1. **Generate a HTML table of categorized aten ops**: This shows all aten ops with core ops highlighted and already implemented ops struck out.
   ```sh
   uv run python ./tools/aten-functions.py \
       --yaml-path ~/pytorch/aten/src/ATen/native/native_functions.yaml \
       --html-table \
       > ops.html
   ```

Once you pick the op to implement, follow [docs/new_op.md](docs/new_op.md) to implement it.

## How to enable, use and contribute to torch_neuronx statistics

Statistics/metrics are great to get tools to get information about the execution and helps the users debug difficult problems like OOMs, long compilation time, etc. To know more follow the [TorchNeuronX Statistics README](docs/torch_neuronx_stats_readme.md)
