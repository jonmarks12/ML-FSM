import os
import shutil
import argparse
from cos import FreezingString
from opt import CartesianOptimizer, InternalsOptimizer
from utils import load_xyz

def fsm(reaction_dir, optcoords='cart', interp='lst', 
        method="L-BFGS-B", maxls=3, maxiter=1, dmax=0.3, nnodes_min=10, ninterp=100,
        calculator="qchem", chg=0, mult=1, nt=1, verbose=False, ckpt='schnet_fine_tuned.ckpt', interpolate=False, **kwargs):

    outdir = os.path.join(reaction_dir, f"fsm_interp_{interp}_method_{method}_maxls_{maxls}_maxiter_{maxiter}_nnodesmin_{nnodes_min}_{calculator}")
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
    if args.calculator == "qchem":
        from ase.calculators.qchem import QChem
        calc = QChem(label="fsm", method='wb97x-v', basis='def2-tzvp', charge=chg, multiplicity=mult,
                                     sym_ignore='true',symmetry='false', scf_algorithm='diis_gdm',
                                     scf_max_cycles='500', nt=8)
    elif args.calculator == "xtb":
        from xtb.ase.calculator import XTB
        calc = XTB(method='GFN2-xTB')
    elif args.calculator == "uma":
        from fairchem.core import pretrained_mlip, FAIRChemCalculator
        predictor = pretrained_mlip.get_predict_unit("uma-s-1", device=dev)
        calc = FAIRChemCalculator(predictor, task_name="omol")   
    elif args.calculator == "torchmd":
        from torchmd_calc import TMDCalculator
        calc=TMDCalculator()
    elif args.calculator == "aimnet2":
        from aimnet2calc import AIMNet2ASE
        calc = AIMNet2ASE("aimnet2",charge=chg,mult=mult)
    elif args.calculator == "emt":
        from ase.calculators.emt import EMT
        calc = EMT()
    elif args.calculator == "mace":
        from mace.calculators import mace_off
        calc = mace_off(model='large',device=dev)
    elif calculator == "schnet":
        from schnet_ase_calculator import SchNetCalculator
        calc = SchNetCalculator(checkpoint = ckpt)
    else:
        raise Exception(f"Unknown calculator {calculator}")

    # string class
    string = FreezingString(reactant, product, nnodes_min, interp, ninterp)
    if interpolate:
        string.interpolate(outdir)
        return

    # optimizer class
    if optcoords == 'cart':
        optimizer = CartesianOptimizer(calc, method, maxiter, maxls, dmax)
    elif optcoords == 'ric':
        optimizer = InternalsOptimizer(calc, method, maxiter, maxls, dmax)
    else:
        raise Exception("Check optimizer coordinates")

    # run
    while (string.growing):
        string.grow()
        string.optimize(optimizer)
        string.write(outdir)

    print("Gradient calls:", string.ngrad)

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("reaction_dir",type=str,help="absolute path to reaction")
    parser.add_argument("--optcoords", type=str, help="optimization coordinate system", default='cart', choices=['cart', 'ric'])
    parser.add_argument("--interp", type=str, help="interpolation method", default='ric',
                          choices=['cart', 'lst', 'ric'])
    parser.add_argument("--nnodes_min",type=int,help="minimum number of nodes, stepsize=initial_dist/nnodes_min",default=18)
    parser.add_argument("--ninterp", type=int, help="number of interpolated images", default=50)
    parser.add_argument("--method", type=str, help="optimization method", default='L-BFGS-B', choices=['L-BFGS-B', 'CG'])
    parser.add_argument("--maxls",type=int,help="maximum number of line search steps", default=3)
    parser.add_argument("--maxiter", type=int, help="number of iterations", default=2)
    parser.add_argument("--dmax", type=float, help="max step size", default=0.05)
    parser.add_argument("--calculator", type=str, help="energy/force method", default='qchem', choices=['qchem', 'xtb','schnet','torchmd','uma','aimnet2','emt'])
    parser.add_argument("--ckpt", type=str, help='ckpt file for pre-trained SchNet model',default='gnns/schnet_fine_tuned.ckpt')
    parser.add_argument("--chg", type=int, help="total molecule charge", default=0)
    parser.add_argument("--mult", type=int, help="spin multiplicity", default=1)
    parser.add_argument("--nt", type=int, help="omp threads", default=1)
    parser.add_argument("--verbose", action='store_true', default=False, help='internal coords printing')
    parser.add_argument("--interpolate", action='store_true', default=False, help='interpolate only')
    args = parser.parse_args()

    fsm(**vars(args))
