"""Test CLI wrapper on a sample reaction."""

import os
import shutil
import subprocess
import tempfile


def test_cli_diels_alder():
    """Run the CLI on the Diels Alder example with the EMT calculator."""
    example_dir = os.path.abspath("examples/data/06_diels_alder")

    with tempfile.TemporaryDirectory() as tmpdir:
        # Copy the example into a temporary directory
        rxn_dir = os.path.join(tmpdir, "06_diels_alder")
        shutil.copytree(example_dir, rxn_dir)

        # Run the CLI using the EMT calculator
        result = subprocess.run(
            ["pixi", "run", "mlfsm", rxn_dir, "--calculator", "emt", "--suffix", "cli_test"],
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )

        print("STDOUT:", result.stdout)
        print("STDERR:", result.stderr)

        # Check success
        assert result.returncode == 0
        assert "Gradient calls:" in result.stdout
