"""Test FSM script on a sample reaction using EMT calculator."""

import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path


def test_fsm_script_diels_alder() -> None:
    """Run fsm_example.py on the Hexadiene example with CG and LST using the EMT calculator."""
    example_dir = Path("examples/data/07_hexadiene")
    script_path = Path("examples/fsm_example.py")

    with tempfile.TemporaryDirectory() as tmpdir:
        # Copy the example into a temporary directory
        rxn_dir = Path(tmpdir) / "07_hexadiene"
        shutil.copytree(example_dir, rxn_dir)

        env = os.environ.copy()
        env["PYTHONPATH"] = os.getcwd()
        # Run the FSM script

        result = subprocess.run(
            [
                sys.executable,
                str(script_path),
                str(rxn_dir),
                "--calculator",
                "emt",
                "--interp",
                "lst",
                "--method",
                "CG",
                "--suffix",
                "test_fsm_script",
            ],
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            env=env,
        )

        print("STDOUT:", result.stdout)
        print("STDERR:", result.stderr)

        # Check that the script completed without error
        assert result.returncode == 0
        assert "Gradient calls:" in result.stdout
