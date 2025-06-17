"""
SchNet-based ASE calculator for Energy and Force Prediction

This script defines an ASE-compatible Calculator class using pretrained SchNet models wrapped in a PyTorch Lightning module. Supports both CPU/GPU and uses autograd to compute gradients. Energies are returned in eV and Forces in eV/Angstrom.

Author: Jonah Marks
Repository: https://github.com/jonmarks12/ML-FSM
"""
import os
import glob
import torch
import subprocess
import numpy as np
from torch.autograd import grad
from ase.units import Bohr,Hartree
from torch_geometric.nn import SchNet
from torchmetrics import MeanSquaredError
from pytorch_lightning import LightningModule
from pytorch_lightning.callbacks import ModelCheckpoint
from ase.calculators.calculator import Calculator, all_changes

class SchNetLightning(LightningModule):
    """
    PyTorch Lightning wrapper for the SchNet neural network model.
    Attributes:
        model (SchNet): The SchNet instance from torch_geometric.nn.
        train_mse, val_mse, test_mse (Metric): Track mean squared error during training.

    Args:
        hidden_channels (int): Number of hidden channels in the embedding layer.
        num_filters (int): Number of filters used in interaction blocks.
        num_interactions (int): Number of interaction blocks.
        num_gaussians (int): Number of Gaussian radial basis functions.
        cutoff (float): Cutoff radius for local environment.
    """
    def __init__(self, hidden_channels: int, num_filters: int,
                  num_interactions: int, num_gaussians: int,
                  cutoff: float):

        super().__init__()
        self.save_hyperparameters()

        self.model = SchNet(hidden_channels=hidden_channels,
                        num_filters=num_filters,
                        num_interactions=num_interactions,
                        num_gaussians=num_gaussians,
                        cutoff=cutoff)

        self.train_mse = MeanSquaredError()
        self.val_mse = MeanSquaredError()
        self.test_mse = MeanSquaredError()

    def forward(self, x: torch.tensor, pos: torch.tensor) -> torch.tensor:
        return self.model(x, pos)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=2e-4)

class SchNetCalculator(Calculator):
    """
    ASE-compatible calculator that wraps a pretrained SchNet model.

    This class implements the `calculate` method to compute single-point 
    energies and forces using a SchNet neural network model loaded from 
    a PyTorch Lightning checkpoint.

    Attributes:
        checkpoint (str): Path to the Lightning model checkpoint.
        model (SchNetLightning): Loaded model used for prediction.
        device (torch.device): Chosen device (CPU or CUDA).

    Args:
        checkpoint (str): Path to the .ckpt file containing the pretrained model.
    """
    implemented_properties = ['energy','forces']
    
    def __init__(self,checkpoint='gnns/schnet_fine_tuned.ckpt'):
        
        Calculator.__init__(self,checkpoint='gnns/schnet_fine_tuned.ckpt')
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.checkpoint = checkpoint
        self.model = SchNetLightning.load_from_checkpoint(self.checkpoint,strict=False).to(self.device)
        self.model.eval()
    
    def calculate(self, atoms=None, properties=None,system_changes=all_changes):
        """
        Compute energy and forces for the given ASE Atoms object.

        Args:
            atoms (ase.Atoms): The atomic structure to evaluate.
            properties (list of str): Properties to compute; defaults to ['energy', 'forces'].
            system_changes (list of str): Triggers recalculation if any listed property changes.

        Sets:
            self.results['energy']: Predicted total energy in eV.
            self.results['forces']: Force matrix with shape (N_atoms, 3) in eV/Å.
        """
        if properties is None:
            properties = ['energy','forces']
        Calculator.calculate(self,atoms,properties,system_changes)
        positions = torch.tensor(atoms.get_positions(),requires_grad=True).to(self.device).float()
        z = torch.tensor(atoms.get_atomic_numbers()).long().to(self.device)
        energy = self.model(z,positions)
        e_grad = grad(energy,positions)[0]
        energy = energy.squeeze().item()
        e_grad = e_grad.detach().cpu().numpy()
        self.results = {
            'energy': energy*Hartree,
            'forces': -1*e_grad*Hartree,
        }