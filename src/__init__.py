"""
MGA Cosmology — Numerical Simulation Framework
==============================================

A simulation framework for the Matrioshka-Genesis-Accretion cosmological model
with Einstein-Cartan torsion-induced quantum bounce and emergent dark energy.

Companion paper: DOI 10.5281/zenodo.21038370
Author: Nahuel Gomez (ORCID: 0009-0006-4420-2685)
Repository: https://github.com/NahuelGomez-Dev/MGA-Cosmology

Example usage:
    >>> from mga_cosmology import MGACosmology, ModelParameters
    >>> params = ModelParameters(Gamma=0.01)
    >>> sim = MGACosmology(params=params)
    >>> results = sim.run()
    >>> sim.plot(save_path="output/bounce.png")
    >>> sim.export_csv("output/data.csv")
"""

from .mga_cosmology import (
    MGACosmology,
    ModelParameters,
    InitialConditions,
    IntegrationOptions,
    SimulationResults,
)

__version__ = "1.0.0"
__author__ = "Nahuel Gomez"
__doi__ = "10.5281/zenodo.21038370"

__all__ = [
    "MGACosmology",
    "ModelParameters",
    "InitialConditions",
    "IntegrationOptions",
    "SimulationResults",
]
