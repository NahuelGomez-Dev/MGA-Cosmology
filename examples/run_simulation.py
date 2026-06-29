"""
MGA Cosmology — Basic Usage Example
====================================

Demonstrates how to run a single simulation with the MGA framework,
generate plots, and export data.

Repository: https://github.com/NahuelGomez-Dev/MGA-Cosmology
Companion paper DOI: 10.5281/zenodo.21041476
"""

import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.mga_cosmology import MGACosmology, ModelParameters, InitialConditions

# =============================================================================
# CONFIGURATION
# =============================================================================

# Physical parameters
params = ModelParameters(
    w_m=1/3,        # Radiation equation of state
    w_Phi=-1.0,     # Dilaton slow-roll (dark energy)
    w_T=+1.0,       # Torsion (fixed: stiff fluid)
    Gamma=0.01,     # Dilaton decay rate
)

# Initial conditions
initial = InitialConditions(
    a_0=10.0,              # Start with collapsing universe
    rho_m_0=1.0,           # Radiation density
    rho_Phi_0=0.5,         # Dilaton density
    a_bounce_target=1.0,   # Target bounce point
)

# =============================================================================
# RUN SIMULATION
# =============================================================================

sim = MGACosmology(params=params, initial=initial, verbose=True)
results = sim.run()

# =============================================================================
# OUTPUTS
# =============================================================================

# Print summary
print(sim.summary())

# Generate plot
sim.plot(save_path='output/basic_simulation.png', show=True)

# Export CSV data for reproducibility
sim.export_csv('output/basic_simulation_data.csv')

# Physical validation
checks = sim.validate_physical()
print("\n[VALIDATION] Physical consistency checks:")
all_passed = True
for name, passed in checks.items():
    status = "✓ PASS" if passed else "✗ FAIL"
    print(f"  {status}: {name}")
    if not passed:
        all_passed = False

if all_passed:
    print("\n✓ All physical checks passed!")
else:
    print("\n✗ Some checks failed — review results carefully.")
