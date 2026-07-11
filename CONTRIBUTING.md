# Contributing

## Workflow

- Work happens on milestone branches (`m0-*`, `m1-*`, …), one pull request per milestone.
- Each PR should build cleanly and, when it touches plugin runtime behavior, be exercised
  end-to-end (built, loaded in Natron, output inspected) before merge.
- Keep large binaries out of git: model weights (`*.onnx`, `*.safetensors`) and test media are
  gitignored and fetched by scripts / pulled from CI artifacts.

## Code style

- **C++17**, following the OpenFX Support library conventions (see vendored examples).
- **Python** tooling formatted with `ruff`/`black` defaults; keep scripts runnable standalone.

## Commits

- Conventional, imperative subject lines (e.g. `Add CoreML session options`).
- Reference the milestone (e.g. `[M3]`) where useful.

## Licensing

By contributing you agree your contributions are licensed under the project's **Apache-2.0**
license (see [`LICENSE`](LICENSE)).
