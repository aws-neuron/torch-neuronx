# Security Reporting Policy

## Reporting Security Issues

We take all security reports seriously. When we receive such reports, we will
investigate and subsequently address any potential vulnerabilities as quickly as
possible.

Please do **not** create a public GitHub issue in this project. If you discover a
potential security issue, notify AWS/Amazon Security via our
[vulnerability reporting page](https://aws.amazon.com/security/vulnerability-reporting/)
or directly via email to [AWS Security](mailto:aws-security@amazon.com). If you
prefer to encrypt your report, use the
[AWS PGP public key](https://aws.amazon.com/security/aws-pgp-public-key/).

Please do **not** include sensitive information — credentials, private keys,
customer data, or personally identifiable information — in your report.

We will keep you updated while we investigate, and we ask that you not disclose the issue publicly until a fix has
shipped.

Thank you for collaborating with us to help protect our customers.

## What is not a vulnerability

These are ordinary bugs. Please open a regular issue instead:

- A crash, hang, or wrong numerical result from a model or input you control.
- An unsupported `aten` operation, or a `torch.compile` failure.
- Anything an attacker can only reach with code execution as the same user, that
  gains them no privilege they did not already have. Privilege escalation and
  failures of isolation between users or tenants on a Neuron device **are** in
  scope — report those.
- A CVE in a dependency with no demonstrated path to exploit it through
  torch-neuronx. Report those upstream, and tell us if torch-neuronx is what
  makes them reachable.

torch-neuronx runs the model you give it. Loading an untrusted model or
checkpoint is equivalent to running untrusted code, and is not by itself a
vulnerability in torch-neuronx. See
[PyTorch's security policy](https://github.com/pytorch/pytorch/blob/main/SECURITY.md)
for what that means in practice.

If you are unsure whether an issue is in scope, report it anyway. We would rather
investigate a potential concern than miss a real one.
