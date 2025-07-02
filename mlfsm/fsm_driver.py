"""
Core driver for executing the Freezing String Method (FSM) reaction path search.

This module contains the main `run_fsm()` function which performs string initialization,
path interpolation, and iterative optimization using a specified energy/force calculator.
It supports both Cartesian and internal coordinate optimization, and integrates with a
variety of quantum chemistry and ML potential backends.

Functions
---------
run_fsm(reaction_dir, ...)
    Executes a full FSM calculation from user inputs and writes outputs to disk.

Raises
------
Exception
    If an unknown calculator or invalid coordinate system is specified.
"""

import os
import shutil
from .cos import FreezingString
from .opt import CartesianOptimizer, InternalsOptimizer
from .utils import load_xyz


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

    # load initial states
    reactant, product = load_xyz(reaction_dir)

    # set calculator
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
            nt=8,
        )
    elif calculator == "xtb":
        from xtb.ase.calculator import XTB

        calc = XTB(method="GFN2-xTB")
    elif calculator == "uma":
        from fairchem.core import pretrained_mlip, FAIRChemCalculator

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
        from mace.calculators import mace_off

        calc = mace_off(model="large", device=dev)
    elif calculator == "schnet":
        from schnet_ase_calculator import SchNetCalculator

        calc = SchNetCalculator(checkpoint=ckpt)
    else:
        raise Exception(f"Unknown calculator {calculator}")

    # string class
    string = FreezingString(reactant, product, nnodes_min, interp, ninterp)
    if interpolate:
        string.interpolate(outdir)
        return

    # optimizer class
    if optcoords == "cart":
        optimizer = CartesianOptimizer(calc, method, maxiter, maxls, dmax)
    elif optcoords == "ric":
        optimizer = InternalsOptimizer(calc, method, maxiter, maxls, dmax)
    else:
        raise Exception("Check optimizer coordinates")

    # run
    while string.growing:
        string.grow()
        string.optimize(optimizer)
        string.write(outdir)

    print("Gradient calls:", string.ngrad)
