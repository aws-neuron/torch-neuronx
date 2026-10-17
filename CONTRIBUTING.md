# Contributing guidelines

Thank you for your interest in torch-neuronx. We welcome bug reports, fixes, new op coverage,
documentation, and design proposals. Read this guide before you open an issue or a pull request.

## Reporting Bugs / Feature Requests

Use the [issue forms](https://github.com/aws-neuron/torch-neuronx/issues/new/choose). Before you file,
check the existing issues and the [Neuron release notes](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/release-notes/index.html).
The release notes list the known issues and limitations of each release.

Each pull request must reference an issue. For a large change, open the issue first, so we can agree on
the direction before you write code.
Some changes need a design review. See [Design Review (RFCs)](#design-review-rfcs).

## Contributing via Pull Requests

We publish the source of `torch-neuronx` with each Neuron SDK release. The published source builds and
runs with the released Neuron SDK, so you can test your change with packages that you can install.

| Branch or tag | What it holds | Accepts pull requests |
|---|---|---|
| `release-N.M` | The source of release line N.M, including its patch releases | No |
| `contribute-N.M` | Contributions for release line N.M | **Yes** |

To contribute:

1. Fork the repository.
2. Create your branch from the latest `contribute-N.M` branch.
3. Make one atomic change in each pull request. Do not reformat code that you do not change.
4. Open a pull request against the same `contribute-N.M` branch. Fill in the pull request template.
5. A maintainer reviews your change and starts CI. See [Continuous integration](#continuous-integration).
6. When the review is complete and CI passes, a maintainer merges your pull request.

When a new release line ships, we create a new `contribute-N.M` branch from that release. We do not
merge the new release into the old branch. Maintainers will lock the old branch. Please move your open
pull requests to the new branch.

## AI-assisted contributions

Read [AI_POLICY.md](AI_POLICY.md) before you use an AI tool to write code, a pull request, or a comment.
A maintainer reviews every contribution. Do not send very large pull requests, or many pull
requests in a short time. Expect review to happen at a human pace.

## Setup Development Environment

You need a Trainium instance. We recommend a Neuron DLAMI or Neuron Deep Learning Container, because it
already has the Neuron driver, runtime, tools, compiler (`neuronx-cc`), and NKI. For more details, see [the Neuron SDK setup docs for PyTorch](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/setup/pytorch/index.html#pytorch-setup).

### 1. Clone your fork

```sh
git clone https://github.com/<your-github-user>/torch-neuronx.git
cd torch-neuronx
git checkout contribute-N.M
```

### 2. Prepare the Neuron environment

Do the part that matches your instance.

**On a Neuron DLAMI or Deep Learning Container:** activate the Python environment that comes with it, and
install Bazel:

```sh
sudo ./install_bazelisk.sh
```

**On any other Linux instance:** install the Neuron packages and Bazel, create a virtual environment, and
install the compiler and NKI. Use the `neuronx-cc` and `nki` versions that the release notes list for your
Neuron SDK release.

```sh
./prerequisites.sh
uv venv
source .venv/bin/activate
uv pip install neuronx-cc nki --extra-index-url https://pip.repos.neuron.amazonaws.com
```

### 3. Install and build torch-neuronx

```sh
uv pip install -e ".[dev]"   # torch-neuronx and the development tools (pytest, Ruff)
./tools/build                # build the C++ extension
```

### 4. Install the pre-commit hooks

```sh
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

**Generate wheel only (after build):**
```sh
USE_BAZEL=1 python setup.py bdist_wheel release
```

**Python tests:** run them with pytest.
```sh
# Run one test file
python -m pytest tests/device/test_device.py -v

# Run one test
python -m pytest tests/device/test_device.py::TestNeuronDevice::test_device_count -v

# Run a directory in parallel
python -m pytest tests/operations -n 8
```

**Run the tests that CI runs:** CI runs the internal tests as nodes.
`tests/pytorch_tests_native/internal_test_distribution_config.json` defines the nodes. To run one node:

```sh
python tests/pytorch_tests_native/runner/run_upstream_tests.py --node aten-unit-tests --output-dir test-reports
```

The nodes are `aten-unit-tests`, `aten-tail-tests`, `dist-unit-tests`, and `dynamic-nki-unit-tests`.
A separate CI job runs the C++, MLIR LIT, and op accuracy tests.

**C++ tests:**
```sh
# Run all C++ unit tests
./tools/run-cpp-test

# Run specific test
./tools/run-cpp-test  //tests/csrc:KernelExecutionTest

# Verbose output
./tools/run-cpp-test -v

# To run the profiler test suite.
./tools/run-cpp-test //:ProfilerTests
```

### Test Reports and Coverage

The node runner writes `test-reports/<name>.html` and `test-reports/test_results_<name>.json`.

To measure coverage, run pytest with `pytest-cov`:

```sh
python -m pytest tests/operations --cov=torch_neuronx --cov-report=html
```

The coverage report is in `htmlcov/index.html`.

### Vertical Tests

The vertical tests check the coverage, accuracy, and performance of the target ATen ops, in eager mode
and with `torch.compile`. See [tests/vertical/README.md](tests/vertical/README.md) for all options.

```sh
python -m pytest tests/vertical/test_target_ops_coverage.py -n 32 --tb=no -q          # Eager coverage
python -m pytest tests/vertical/test_target_ops_compile_coverage.py -n 32 --tb=no -q  # Compile coverage
python -m pytest tests/vertical/test_target_ops_coverage.py -v -k "relu or gelu"      # Ops that match a pattern
```

### Linting and Formatting

The pre-commit hooks run Ruff and clang-format with the versions and exclusions that CI uses:

```sh
pre-commit run --all-files
```

## Continuous integration

CI runs on Trainium instances with the released Neuron SDK.

1. **CI does not start on its own.** When you open a pull request, it shows a pending check. A
   maintainer reviews the change first, and then starts CI.
2. **PR checks** run the ATen and internal unit test suites, and the block performance tests.
3. **The merge queue** runs the relevant upstream PyTorch test suite depending on what your PR changes, and then merges
   it. A PR merges only through the merge queue.

Run `pre-commit run --all-files` before you push (No Neuron hardware needed).

If a failure looks unrelated to your change, comment on the pull request. A maintainer will
investigate.

## Design Review (RFCs)

Some changes need a design review **before** the implementation. It is easier to change
direction while a feature is still an idea than when a large PR exists.

Open an RFC if your change touches shared runtime machinery (allocator, streams, dispatcher,
collectives), adds or redefines public API or environment variables, is hard to reverse, or makes a
deliberate performance/memory tradeoff. This also applies to changes where one components's requirements
lands in another component.

Bug fixes with an obvious correct form, new op coverage, tests, docs, and behaviour-preserving
refactors generally do **not** need an RFC. Send those as a normal PR. If you're unsure, ask in a
regular issue, or just open the RFC — it's cheaper than a rejected PR.

### How

1. **Open an issue** with the **RFC (Design Proposal)** template. If the design is long, you may link it as a doc,
   but keep the summary, motivation and proposal in the issue.
2. **Bring in the right reviewers.** Check [.github/CODEOWNERS](.github/CODEOWNERS) and tag the relevant owner groups.
3. **Revise until the affected owners agree on a path forward.** Strikethrough the changed sentences
   and update the issue body as the design changes.
4. **Write down the outcome** when review settles: what was decided, who agreed, and what is next.
5. Include links to the RFC in the PR you raise.

## Finding contributions to work on
Looking at the existing issues is a great way to find something to contribute on. As our projects,
by default, use the default GitHub issue labels (enhancement/bug/duplicate/help wanted/invalid/question/wontfix),
looking at any 'help wanted' issues is a great place to start.

## Code of Conduct

This project follows the [Amazon Open Source Code of Conduct](https://aws.github.io/code-of-conduct).
For more information, see the [Code of Conduct FAQ](https://aws.github.io/code-of-conduct-faq), or contact
<opensource-codeofconduct@amazon.com>.

## Security issues

Do not report a security issue in a public GitHub issue. Follow [SECURITY.md](SECURITY.md).

## Licensing

See the [LICENSE](LICENSE) file for the licensing of this project. We will ask you to confirm the
licensing of your contribution.
