"""Test FSM script on a sample reaction using EMT calculator."""

import os
import shutil
import subprocess
import tempfile


def test_fsm_script_diels_alder() -> None:
    """Run fsm_example.py on the Diels-Alder example with the EMT calculator."""
    example_dir = os.path.abspath("examples/data/06_diels_alder")
    script_path = os.path.abspath("examples/fsm_example.py")

    with tempfile.TemporaryDirectory() as tmpdir:
        # Copy the example into a temporary directory
        rxn_dir = os.path.join(tmpdir, "06_diels_alder")
        shutil.copytree(example_dir, rxn_dir)
        env = os.environ.copy()
        env["PYTHONPATH"] = os.getcwd()
        # Run the FSM script
        result = subprocess.run(
            ["pixi", "run", "python", script_path, rxn_dir, "--calculator", "emt", "--suffix", "test_fsm_script"],
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
