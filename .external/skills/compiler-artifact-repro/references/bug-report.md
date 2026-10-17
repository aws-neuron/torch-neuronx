name: Bug Report
description: Report a bug in torch-neuronx
labels:
  - bug
body:
  - type: markdown
    attributes:
      value: |
        Check existing issues, release notes, and troubleshooting documentation first.
        Open a feature request instead when the behavior is a documented limitation.
  - type: textarea
    attributes:
      label: Describe the bug
      description: |
        Describe the bug clearly. Include a minimal code snippet and the full error or stack trace.
      placeholder: |
        Describe the bug.

        ```python
        # Minimal sample code
        ```

        ```text
        Full error or stack trace
        ```
    validations:
      required: true
  - type: textarea
    attributes:
      label: Reproduction
      description: |
        Provide the smallest script or unit test that reproduces the failure.
      placeholder: |
        ```python
        import torch
        import torch_neuronx

        def test_repro():
            x = torch.randn(4, 4, device="neuron")
            ...
        ```

        Command:

        ```text
        ```
    validations:
      required: true
  - type: textarea
    attributes:
      label: Expected Behavior
      description: Describe the expected behavior.
    validations:
      required: true
  - type: markdown
    attributes:
      value: |
        Review all text before submission. Remove account identifiers, access keys,
        private hosts and paths, personal data, and other sensitive information.
  - type: textarea
    attributes:
      label: Environment
      description: |
        Run this command and paste its output:

        ```sh
        python -m torch_neuronx.collect_env
        ```

        Review the output before submission.
      placeholder: |
        Paste the output of `python -m torch_neuronx.collect_env`.
    validations:
      required: true
