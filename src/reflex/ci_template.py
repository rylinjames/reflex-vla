"""GitHub Actions workflow template emitter for `reflex validate`.

Generates a CI workflow that runs `reflex validate` on a SmolVLA export under
GitHub-hosted `ubuntu-latest` (7GB RAM). pi0 and GR00T blocks are included as
commented-out templates that require a self-hosted runner with 16GB+ RAM.

No jinja2 dependency — uses `str.format` with a single `{reflex_version}`
placeholder.
"""

from __future__ import annotations

from pathlib import Path

__all__ = ["TEMPLATE", "emit_ci_template", "validate_emitted_yaml"]


TEMPLATE = """name: Reflex Validate

on:
  pull_request:
  push:
    branches: [main]

concurrency:
  group: ${{{{ github.workflow }}}}-${{{{ github.ref }}}}
  cancel-in-progress: true

jobs:
  validate-smolvla:
    runs-on: ubuntu-latest
    permissions:
      contents: read
    steps:
      - name: Checkout
        uses: actions/checkout@v4

      - name: Set up Python 3.11
        uses: actions/setup-python@v5
        with:
          python-version: "3.11"

      - name: Install reflex-vla
        run: |
          pip install 'reflex-vla[serve,onnx,dev] @ git+https://github.com/rylinjames/reflex-vla@v{reflex_version}'

      - name: Export SmolVLA
        run: |
          reflex export lerobot/smolvla_base --target desktop --output ./sv_export

      - name: Validate round-trip parity
        run: |
          reflex validate ./sv_export --threshold 1e-4 --num-cases 3 --output-json > validate_result.json

      - name: Upload validation report
        if: always()
        uses: actions/upload-artifact@v4
        with:
          name: smolvla-validate-result
          path: validate_result.json

  # Requires self-hosted runner with 16GB+ RAM — uncomment and update runs-on:
  # validate-pi0:
  #   runs-on: [self-hosted, linux, x64, ram-16gb]
  #   permissions:
  #     contents: read
  #   steps:
  #     - name: Checkout
  #       uses: actions/checkout@v4
  #     - name: Set up Python 3.11
  #       uses: actions/setup-python@v5
  #       with:
  #         python-version: "3.11"
  #     - name: Install reflex-vla
  #       run: pip install 'reflex-vla[serve,onnx,dev] @ git+https://github.com/rylinjames/reflex-vla@v{reflex_version}'
  #     - name: Export pi0
  #       run: reflex export lerobot/pi0_base --target desktop --output ./pi0_export
  #     - name: Validate round-trip parity
  #       run: reflex validate ./pi0_export --threshold 1e-4 --num-cases 3 --output-json > validate_result.json
  #     - name: Upload validation report
  #       if: always()
  #       uses: actions/upload-artifact@v4
  #       with:
  #         name: pi0-validate-result
  #         path: validate_result.json

  # Requires self-hosted runner with 16GB+ RAM — uncomment and update runs-on:
  # validate-gr00t:
  #   runs-on: [self-hosted, linux, x64, ram-16gb]
  #   permissions:
  #     contents: read
  #   steps:
  #     - name: Checkout
  #       uses: actions/checkout@v4
  #     - name: Set up Python 3.11
  #       uses: actions/setup-python@v5
  #       with:
  #         python-version: "3.11"
  #     - name: Install reflex-vla
  #       run: pip install 'reflex-vla[serve,onnx,dev] @ git+https://github.com/rylinjames/reflex-vla@v{reflex_version}'
  #     - name: Export GR00T
  #       run: reflex export nvidia/GR00T-N1-2B --target desktop --output ./gr00t_export
  #     - name: Validate round-trip parity
  #       run: reflex validate ./gr00t_export --threshold 1e-4 --num-cases 3 --output-json > validate_result.json
  #     - name: Upload validation report
  #       if: always()
  #       uses: actions/upload-artifact@v4
  #       with:
  #         name: gr00t-validate-result
  #         path: validate_result.json
"""


def emit_ci_template(
    output_path: Path,
    reflex_version: str | None = None,
    *,
    overwrite: bool = False,
) -> None:
    """Write the GitHub Actions workflow YAML to ``output_path``.

    Creates parent directories as needed. Refuses to overwrite an existing file
    unless ``overwrite=True`` is passed. ``reflex_version`` defaults to the
    current installed ``reflex.__version__`` when not explicitly provided.
    """
    output_path = Path(output_path)
    if output_path.exists() and not overwrite:
        raise FileExistsError(
            f"{output_path} exists — pass overwrite=True to replace"
        )
    if reflex_version is None:
        # Late import avoids circularity with reflex/__init__.py exports.
        from reflex import __version__ as reflex_version  # type: ignore[no-redef]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    rendered = TEMPLATE.format(reflex_version=reflex_version)
    output_path.write_text(rendered)


def validate_emitted_yaml(path: Path) -> bool:
    """Return True if the emitted file at ``path`` parses as YAML.

    Falls back to False if pyyaml is not installed or parsing fails.
    """
    try:
        import yaml  # type: ignore[import-untyped]
    except ImportError:
        return False
    try:
        with Path(path).open("r") as fh:
            yaml.safe_load(fh)
        return True
    except Exception:
        return False
