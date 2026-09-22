# Copybara public pull request import

This directory contains the trusted Copybara configuration. The configuration
imports an open public pull request into a draft pull request in this repository.

The public pull request must target a `contribute-N.M` branch. It must have the
`ready-for-private-import` label. A maintainer must approve its current head
commit. A maintainer must also trigger the import.

## Repository configuration

Create a `contribute-N.M` branch in `aws-neuron/torch-neuronx` for each
supported release version.

Create a GitHub App with access to these repositories:

* `aws-neuron/torch-neuronx`
* `aws-neuron/private-torch-neuronx`

Give the app these repository permissions:

* Read and write `Contents`.
* Read and write `Pull requests`.
* Read `Metadata` in both repositories.

Create the `public-pr-import` GitHub environment in the private repository.
Restrict the environment to the default branch.
Require a maintainer approval. Prevent self-approval.

Add this environment configuration:

* Variable `COPYBARA_GITHUB_APP_ID`
* Variable `PUBLIC_PR_DISPATCHER_LOGIN`
* Secret `COPYBARA_GITHUB_APP_PRIVATE_KEY`

## Run the workflow

Open the `Import public pull request` workflow.

Select `validate` to check the Copybara configuration. This operation does not
use the GitHub App.

Select `dry-run` to test an import without updating the private main branch.
Enter the public pull request number.

Select `import` to create or update the private draft pull request.
Enter the public pull request number.

The public dispatcher can send an `import-public-pr` event. The event payload
must contain these fields:

```json
{
  "public_pr": 123,
  "source_repository": "aws-neuron/torch-neuronx",
  "base_branch": "contribute-3.0",
  "head_sha": "0123456789abcdef0123456789abcdef01234567",
  "trigger_actor": "maintainer-login"
}
```

The workflow reads the current base branch and head commit from GitHub. It
rejects a branch that does not match `contribute-N.M`. For an event, it also
rejects a base branch or head commit that changed after the dispatch.
The workflow also verifies the label, trigger actor, and current-head approval.
The trigger actor and approver must have `maintain` or `admin` access.
The event sender must match `PUBLIC_PR_DISPATCHER_LOGIN`.

The import operation runs only from the default branch. Pull requests can run
the validation job. The workflow does not run code from the public pull request.

The public repository must gate contributor code before it sends a dispatch.
The public gate must verify the maintainer trigger and current-head approval.
The private import does not change the existing private CI workflows.

## Security limit

Copybara `v20260629` uses one GitHub credential for each GitHub host. The proof
of concept uses one app for both repositories.

Before launch, test an app that is installed only in the private repository.
Confirm that Copybara can read the public repository with that token.

If this test fails, use a Copybara version that supports separate credentials.
Another option is a trusted wrapper that separates the read and write steps.
