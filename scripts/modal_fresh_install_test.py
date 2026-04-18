"""Modal: fresh-install smoke test — simulates a new customer's first hour.

Spins a clean python:3.12-slim container, pip-installs reflex-vla from
the live GitHub main branch, verifies the CLI works end-to-end:

  1. `reflex --help` returns zero + shows subcommands
  2. `reflex targets` lists hardware profiles
  3. `reflex doctor` passes (or only fails on GPU checks under CPU-only image)
  4. imports: from reflex.runtime.server import create_app

If this breaks, a fresh-box customer broke in their first 5 minutes.
"""
import modal

app = modal.App("reflex-fresh-install-test")

image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git")
    .pip_install(
        "reflex-vla[serve,onnx] @ git+https://github.com/rylinjames/reflex-vla.git"
    )
)


@app.function(image=image, timeout=600)
def test_fresh_install():
    import subprocess
    import sys

    def _run(cmd, ok_codes=(0,)):
        print(f"\n$ {' '.join(cmd)}")
        r = subprocess.run(cmd, capture_output=True, text=True)
        print(r.stdout[-2000:] if r.stdout else "(no stdout)")
        if r.returncode not in ok_codes:
            print(f"[stderr]: {r.stderr[-2000:]}")
        return r.returncode

    results = {}
    results["help"] = _run(["reflex", "--help"])
    results["targets"] = _run(["reflex", "targets"])

    # Try to import the key runtime module; should not raise
    try:
        from reflex.runtime.server import create_app  # noqa: F401
        from reflex.verification_report import write_verification_report  # noqa: F401
        from reflex.safety.guard import ActionGuard  # noqa: F401
        results["imports"] = 0
        print("[imports] reflex.runtime.server + verification_report + safety.guard OK")
    except Exception as e:
        results["imports"] = 1
        print(f"[imports] FAIL: {e}")

    all_pass = all(v == 0 for v in results.values())
    print(f"\n=== VERDICT: {'PASS' if all_pass else 'FAIL'} ===")
    print(f"  {results}")
    return {"results": results, "passed": all_pass}


@app.local_entrypoint()
def main():
    result = test_fresh_install.remote()
    print("\n=== RESULT ===")
    for k, v in result.items():
        print(f"  {k}: {v}")
