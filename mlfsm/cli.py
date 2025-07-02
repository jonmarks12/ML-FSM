"""
Command-line interface (CLI) for the ML-FSM package.

This module defines the CLI entry point for running the Freezing String Method (FSM)
via command-line arguments. It wraps the core logic in `fsm_driver.py` and parses
user-specified options such as the reaction directory, calculator, optimizer settings,
and interpolation strategy.

Example
-------
Run FSM with default settings on a given reaction directory:

    $ pixi run mlfsm data/my_reaction/

Attributes
----------
main() : function
    Parses arguments and calls `run_fsm()` from `fsm_driver`.
"""

import argparse

from .fsm_driver import run_fsm


def main():
    """Run the FSM command-line interface."""
    parser = argparse.ArgumentParser()
    parser.add_argument("reaction_dir", type=str, help="absolute path to reaction")
    parser.add_argument(
        "--optcoords", type=str, help="optimization coordinate system", default="cart", choices=["cart", "ric"]
    )
    parser.add_argument(
        "--interp", type=str, help="interpolation method", default="ric", choices=["cart", "lst", "ric"]
    )
    parser.add_argument(
        "--nnodes_min", type=int, help="minimum number of nodes, stepsize=initial_dist/nnodes_min", default=18
    )
    parser.add_argument("--ninterp", type=int, help="number of interpolated images", default=50)
    parser.add_argument(
        "--suffix", type=str, help="string to add to end of filename for name customization", default=None
    )
    parser.add_argument(
        "--method", type=str, help="optimization method", default="L-BFGS-B", choices=["L-BFGS-B", "CG"]
    )
    parser.add_argument("--maxls", type=int, help="maximum number of line search steps", default=3)
    parser.add_argument("--maxiter", type=int, help="number of iterations", default=2)
    parser.add_argument("--dmax", type=float, help="max step size", default=0.05)
    parser.add_argument(
        "--calculator",
        type=str,
        help="energy/force method",
        default="qchem",
        choices=["qchem", "xtb", "schnet", "torchmd", "uma", "aimnet2", "emt"],
    )
    parser.add_argument(
        "--ckpt", type=str, help="ckpt file for pre-trained SchNet model", default="gnns/schnet_fine_tuned.ckpt"
    )
    parser.add_argument("--chg", type=int, help="total molecule charge", default=0)
    parser.add_argument("--mult", type=int, help="spin multiplicity", default=1)
    parser.add_argument("--nt", type=int, help="omp threads", default=1)
    parser.add_argument("--verbose", action="store_true", default=False, help="internal coords printing")
    parser.add_argument("--interpolate", action="store_true", default=False, help="interpolate only")
    args = parser.parse_args()
    run_fsm(**vars(args))
