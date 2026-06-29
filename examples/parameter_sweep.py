"""
MGA Cosmology — Parameter Sweep Example
========================================

Demonstrates how to use the OOP architecture to explore the parameter
space of the MGA model. Specifically, we study how the dilaton decay
rate Γ (which depends on the parent black hole mass M_•) affects
the late-time dark energy dominance.

From Eq. 6.8 of the companion paper:
    Γ = 2α_WH ħc⁴ / (G² M_•³)

So different values of Γ correspond to different parent BH masses.

Companion paper: DOI 10.5281/zenodo.21038370
"""

import sys
import os
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.mga_cosmology import MGACosmology, ModelParameters

# =============================================================================
# PARAMETER SWEEP: Different Γ values (different parent BH masses)
# =============================================================================

Gamma_values = [0.001, 0.005, 0.01, 0.05, 0.1]
colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(Gamma_values)))

print("=" * 70)
print(" MGA COSMOLOGY — PARAMETER SWEEP: Γ DEPENDENCE")
print(" Exploring different parent black hole masses M_•")
print(" Γ ∝ M_•⁻³ (smaller BH → larger Γ → faster dilaton decay)")
print("=" * 70)

# Storage for results
results_dict = {}
rho_Phi_final = []
rho_m_final = []
a_min_values = []

for i, Gamma in enumerate(Gamma_values):
    print(f"\n--- Running Γ = {Gamma} ---")

    params = ModelParameters(Gamma=Gamma)
    sim = MGACosmology(params=params, verbose=False)
    results = sim.run()
    results_dict[Gamma] = results

    rho_Phi_final.append(results.rho_Phi[-1])
    rho_m_final.append(results.rho_m[-1])
    a_min_values.append(results.a_min)

    ratio = results.rho_Phi[-1] / max(results.rho_m[-1], 1e-30)
    print(f"  a_min = {results.a_min:.4f}, ä_bounce = {results.a_ddot_bounce:.4f}")
    print(f"  ρ_Φ/ρ_m (final) = {ratio:.2e}")

# =============================================================================
# PLOTTING: Comparative analysis
# =============================================================================

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Panel 1: Scale factor for all Γ values
ax1 = axes[0, 0]
for i, Gamma in enumerate(Gamma_values):
    r = results_dict[Gamma]
    ax1.plot(r.t, r.a, color=colors[i], linewidth=2, label=f'Γ = {Gamma}')
ax1.set_xlabel('Time $t$', fontsize=12)
ax1.set_ylabel('Scale factor $a(t)$', fontsize=12)
ax1.set_title('Scale Factor Evolution for Different Γ', fontsize=13, fontweight='bold')
ax1.legend(fontsize=10)
ax1.grid(True, alpha=0.3)

# Panel 2: Dilaton density evolution
ax2 = axes[0, 1]
for i, Gamma in enumerate(Gamma_values):
    r = results_dict[Gamma]
    rho_Phi_plot = np.where(r.rho_Phi > 1e-20, r.rho_Phi, np.nan)
    ax2.semilogy(r.t, rho_Phi_plot, color=colors[i], linewidth=2, label=f'Γ = {Gamma}')
ax2.set_xlabel('Time $t$', fontsize=12)
ax2.set_ylabel(r'$\rho_\Phi$ (log scale)', fontsize=12)
ax2.set_title('Dilaton Density Decay', fontsize=13, fontweight='bold')
ax2.legend(fontsize=10)
ax2.grid(True, alpha=0.3, which='both')

# Panel 3: Dark energy dominance ratio
ax3 = axes[1, 0]
ratios = [phi / max(m, 1e-30) for phi, m in zip(rho_Phi_final, rho_m_final)]
ax3.semilogy(Gamma_values, ratios, 'ko-', linewidth=2, markersize=8)
ax3.axhline(1.0, color='r', linestyle='--', linewidth=1.5, label='DE dominance threshold')
ax3.set_xlabel('Decay rate Γ', fontsize=12)
ax3.set_ylabel(r'$\rho_\Phi / \rho_m$ (final)', fontsize=12)
ax3.set_title('Late-Time Dark Energy Dominance', fontsize=13, fontweight='bold')
ax3.legend(fontsize=10)
ax3.grid(True, alpha=0.3, which='both')

# Panel 4: Minimum scale factor (bounce)
ax4 = axes[1, 1]
ax4.plot(Gamma_values, a_min_values, 'bs-', linewidth=2, markersize=8)
ax4.set_xlabel('Decay rate Γ', fontsize=12)
ax4.set_ylabel(r'$a_{\min}$ (bounce point)', fontsize=12)
ax4.set_title('Bounce Point vs Γ', fontsize=13, fontweight='bold')
ax4.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('output/parameter_sweep_Gamma.png', dpi=150, bbox_inches='tight')
plt.show()

# =============================================================================
# SUMMARY TABLE
# =============================================================================

print("\n" + "=" * 70)
print(" PARAMETER SWEEP SUMMARY")
print("=" * 70)
print(f"{'Γ':<10} {'a_min':<12} {'ä_bounce':<12} {'ρ_Φ/ρ_m (final)':<20}")
print("-" * 70)
for i, Gamma in enumerate(Gamma_values):
    r = results_dict[Gamma]
    ratio = rho_Phi_final[i] / max(rho_m_final[i], 1e-30)
    print(f"{Gamma:<10.4f} {r.a_min:<12.4f} {r.a_ddot_bounce:<12.4f} {ratio:<20.2e}")
print("=" * 70)
print("\nInterpretation:")
print("  • Larger Γ (smaller parent BH) → faster dilaton decay → weaker DE")
print("  • Smaller Γ (larger parent BH) → slower decay → stronger DE dominance")
print("  • The bounce point a_min is robust to changes in Γ")
print("  • This is the basis for the M_• – δw correlation prediction (Sec. 8.2)")
