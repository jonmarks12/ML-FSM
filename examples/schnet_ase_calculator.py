"""ASE calculator using a SchNet model for energy and force prediction.

This module defines a PyTorch Lightning wrapper around SchNet, and a corresponding
ASE-compatible calculator. Supports GPU acceleration and uses autograd to compute
energy gradients. Energies are returned in eV and forces in eV/Å.

Author: Jonah Marks
Repository: https://github.com/jonmarks12/ML-FSM
"""

from typing import Any, ClassVar

import torch
from ase import Atoms
from ase.calculators.calculator import Calculator, all_changes
from ase.units import Hartree
from pytorch_lightning import LightningModule
from torch import Tensor
from torch.autograd import grad
from torch_geometric.nn import SchNet  # type: ignore [import-untyped]
from torchmetrics import MeanSquaredError


class SchNetLightning(LightningModule):
    """PyTorch Lightning wrapper for the SchNet neural network model.

    Attributes
    ----------
        model (SchNet): The SchNet instance.
        train_mse (Metric): Tracks training mean squared error.
        val_mse (Metric): Tracks validation mean squared error.
        test_mse (Metric): Tracks test mean squared error.

    Args:
        hidden_channels (int): Number of hidden channels in embeddings.
        num_filters (int): Number of filters in interaction blocks.
        num_interactions (int): Number of interaction blocks.
        num_gaussians (int): Number of RBFs used in distance expansion.
        cutoff (float): Cutoff radius in Å.
    """

    def __init__(
        self,
        hidden_channels: int,
        num_filters: int,
        num_interactions: int,
        num_gaussians: int,
        cutoff: float,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()

        self.model = SchNet(
            hidden_channels=hidden_channels,
            num_filters=num_filters,
            num_interactions=num_interactions,
            num_gaussians=num_gaussians,
            cutoff=cutoff,
        )

        self.train_mse = MeanSquaredError()
        self.val_mse = MeanSquaredError()
        self.test_mse = MeanSquaredError()

    def forward(self, x: Tensor, pos: Tensor) -> Tensor:
        """Forward pass through SchNet."""
        return self.model(x, pos)

    def configure_optimizers(self) -> torch.optim.Optimizer:
        """Define optimizer for training."""
        return torch.optim.Adam(self.parameters(), lr=2e-4)


class SchNetCalculator(Calculator):
    """ASE-compatible calculator that wraps a pretrained SchNet model.

    Attributes
    ----------
        checkpoint (str): Path to the Lightning checkpoint.
        model (SchNetLightning): Loaded PyTorch Lightning model.
        device (torch.device): CPU or CUDA device for computation.

    Args:
        checkpoint (str): Path to the `.ckpt` file containing weights.
    """

    implemented_properties: ClassVar[list[str]] = ["energy", "forces"]  # type: ignore [misc]

    def __init__(self, checkpoint: str = "gnns/schnet_fine_tuned.ckpt") -> None:
        super().__init__()
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.checkpoint = checkpoint
        self.model = SchNetLightning.load_from_checkpoint(
            self.checkpoint,
            strict=False,
        ).to(self.device)
        self.model.eval()

    def calculate(  # type: ignore [override]
        self, atoms: Atoms, properties: list[Any] | None = None, system_changes: list[Any] = all_changes
    ) -> None:
        """Compute single-point energy and forces using SchNet.

        Args:
            atoms (ase.Atoms): ASE atoms object to evaluate.
            properties (list): Desired properties (default: ["energy", "forces"]).
            system_changes (list): Triggers that cause recalculation.

        Sets:
            self.results["energy"]: Total energy in eV.
            self.results["forces"]: Forces in eV/Å.
        """
        if properties is None:
            properties = ["energy", "forces"]

        super().calculate(atoms, properties, system_changes)

        positions = torch.tensor(atoms.get_positions(), dtype=torch.float32, requires_grad=True).to(self.device)
        z = torch.tensor(atoms.get_atomic_numbers(), dtype=torch.long).to(self.device)

        energy = self.model(z, positions)
        e_grad = grad(energy, positions)[0]

        self.results = {
            "energy": energy.item() * Hartree,
            "forces": -1.0 * e_grad.cpu().numpy() * Hartree,
        }
