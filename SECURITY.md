# Security and responsible use

## Reporting a vulnerability

Please report security issues privately through GitHub's "Report a vulnerability" (Security tab) rather than in a public issue. Examples: a crafted `.wspec` that executes code, escapes its output directory, or makes `verify` report OK for weights that differ. We aim to acknowledge reports within 72 hours.

## What WeightSpec does and does not protect against

* Loading a patch never executes code. Patches are JSON plus safetensors and are parsed as data. They are never unpickled.
* `verify` guarantees that a patch rebuilds the exact recorded tensors from the recorded base. It does **not** tell you whether an edit is safe or beneficial.
* `Policy` limits *what* may be edited. It is not a sandbox. Running a model after an edit has the same risks as running any modified model.

## Dual use

Directly editing weights can repair a model, and it can also remove safety behaviour. For example, refusal in many chat models is mediated by a single direction (Arditi et al., 2024). We ask that you:

* do not publish patches whose purpose is to remove safety behaviour from deployed models;
* run your own safety evaluations before deploying an edited model;
* disclose the edit (publish the patch) when you distribute edited weights, so others can audit it.

Pull requests that add such patches to this repository will not be accepted.
