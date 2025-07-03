"""
Example script for running the Freezing String Method (FSM).

Users must install their desired quantum chemistry backend separately from the
mlfsm package. Currently supported calculators include:

    - QChem
    - xTB (GFN2-xTB)
    - FAIR UMA
    - AIMNet2
    - MACEOFF
    - SchNet
    - EMT

Only the selected calculator needs to be installed in the Python environment.
"""


import os
import shutil
import argparse

from mlfsm.cos import FreezingString
from mlfsm.opt import CartesianOptimizer, InternalsOptimizer
from mlfsm.utils import load_xyz


def run_fsm(
    reaction_dir,
    optcoords="cart",
    interp="lst",
    method="L-BFGS-B",
    maxls=3,
    maxiter=1,
    dmax=0.3,
    nnodes_min=10,
    ninterp=100,
    suffix=None,
    calculator="qchem",
    chg=0,
    mult=1,
    nt=1,
    verbose=False,
    ckpt="schnet_fine_tuned.ckpt",
    interpolate=False,
    **kwargs,
):
    """Run the Freezing String Method on a given reaction with user specified parameters."""
    if suffix:
        outdir = os.path.join(
            reaction_dir,
            f"fsm_interp_{interp}_method_{method}_maxls_{maxls}_maxiter_{maxiter}_nnodesmin_{nnodes_min}_{calculator}_{suffix}",
        )
    else:
        outdir = os.path.join(
            reaction_dir,
            f"fsm_interp_{interp}_method_{method}_maxls_{maxls}_maxiter_{maxiter}_nnodesmin_{nnodes_min}_{calculator}",
        )
    if interpolate:
        outdir = os.path.join(reaction_dir, f"interp_{interp}")
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    else:
        shutil.rmtree(outdir)
        os.makedirs(outdir)

    # Load structures
    reactant, product = load_xyz(reaction_dir)

    # Load calculator
    if calculator == "qchem":
        from ase.calculators.qchem import QChem

        calc = QChem(
            label="fsm",
            method="wb97x-v",
            basis="def2-tzvp",
            charge=chg,
            multiplicity=mult,
            sym_ignore="true",
            symmetry="false",
            scf_algorithm="diis_gdm",
            scf_max_cycles="500",
            nt=nt,
        )
    elif calculator == "xtb":
        from xtb.ase.calculator import XTB

        calc = XTB(method="GFN2-xTB")
    elif calculator == "uma":
        import torch
        from fairchem.core import FAIRChemCalculator, pretrained_mlip

        dev = "cuda" if torch.cuda.is_available() else "cpu"
        predictor = pretrained_mlip.get_predict_unit("uma-s-1", device=dev)
        calc = FAIRChemCalculator(predictor, task_name="omol")
    elif calculator == "torchmd":
        from torchmd_calc import TMDCalculator

        calc = TMDCalculator()
    elif calculator == "aimnet2":
        from aimnet2calc import AIMNet2ASE

        calc = AIMNet2ASE("aimnet2", charge=chg, mult=mult)
    elif calculator == "emt":
        from ase.calculators.emt import EMT

        calc = EMT()
    elif calculator == "mace":
        import torch
        from mace.calculators import mace_off

        dev = "cuda" if torch.cuda.is_available() else "cpu"
        calc = mace_off(model="large", device=dev)
    elif calculator == "schnet":
        from schnet_ase_calculator import SchNetCalculator

        calc = SchNetCalculator(checkpoint=ckpt)
    else:
        raise Exception(f"Unknown calculator {calculator}")

    # Initialize FSM string
    string = FreezingString(reactant, product, nnodes_min, interp, ninterp)
    if interpolate:
        string.interpolate(outdir)
        return

    # Choose optimizer
    if optcoords == "cart":
        optimizer = CartesianOptimizer(calc, method, maxiter, maxls, dmax)
    elif optcoords == "ric":
        optimizer = InternalsOptimizer(calc, method, maxiter, maxls, dmax)
    else:
        raise Exception("Check optimizer coordinates")

    # Run FSM
    while string.growing:
        string.grow()
        string.optimize(optimizer)
        string.write(outdir)

    print("Gradient calls:", string.ngrad)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("reaction_dir", type=str, help="absolute path to reaction")
    parser.add_argument("--optcoords", type=str, default="cart", choices=["cart", "ric"])
    parser.add_argument("--interp", type=str, default="ric", choices=["cart", "lst", "ric"])
    parser.add_argument("--nnodes_min", type=int, default=18)
    parser.add_argument("--ninterp", type=int, default=50)
    parser.add_argument("--suffix", type=str, default=None)
    parser.add_argument("--method", type=str, default="L-BFGS-B", choices=["L-BFGS-B", "CG"])
    parser.add_argument("--maxls", type=int, default=3)
    parser.add_argument("--maxiter", type=int, default=2)
    parser.add_argument("--dmax", type=float, default=0.05)
    parser.add_argument(
        "--calculator",
        type=str,
        default="qchem",
        choices=["qchem", "xtb", "schnet", "torchmd", "uma", "aimnet2", "emt"],
    )
    parser.add_argument("--ckpt", type=str, default="gnns/schnet_fine_tuned.ckpt")
    parser.add_argument("--chg", type=int, default=0)
    parser.add_argument("--mult", type=int, default=1)
    parser.add_argument("--nt", type=int, default=1)
    parser.add_argument("--verbose", action="store_true", default=False)
    parser.add_argument("--interpolate", action="store_true", default=False)

    args = parser.parse_args()
    run_fsm(**vars(args))
