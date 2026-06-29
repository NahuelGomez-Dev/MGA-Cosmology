"""
MGA Cosmology — Physical Validation Test Suite
===============================================

Comprehensive tests to validate the physical consistency of the
MGA cosmological simulation framework.

Run with: pytest tests/ -v

Repository: https://github.com/NahuelGomez-Dev/MGA-Cosmology
Companion paper DOI: 10.5281/zenodo.21041476
"""

import sys
import os
import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.mga_cosmology import (
    MGACosmology,
    ModelParameters,
    InitialConditions,
    IntegrationOptions,
)


# =============================================================================
# FIXTURES
# =============================================================================

@pytest.fixture
def default_sim():
    """Create a default simulation instance."""
    return MGACosmology(verbose=False)


@pytest.fixture
def default_results(default_sim):
    """Run a default simulation and return results."""
    return default_sim.run()


# =============================================================================
# TEST 1: DEFAULT SIMULATION RUNS
# =============================================================================

class TestDefaultSimulation:
    """Tests for the default simulation configuration."""

    def test_simulation_completes(self, default_sim):
        """Test that the default simulation runs to completion."""
        results = default_sim.run()
        assert results.success, "Integration did not complete successfully"

    def test_results_have_correct_length(self, default_results):
        """Test that all output arrays have the same length."""
        r = default_results
        n = len(r.t)
        assert len(r.a) == n
        assert len(r.a_dot) == n
        assert len(r.rho_m) == n
        assert len(r.rho_Phi) == n
        assert len(r.rho_T) == n
        assert len(r.H) == n
        assert len(r.a_ddot) == n


# =============================================================================
# TEST 2: BOUNCE OCCURS
# =============================================================================

class TestBounce:
    """Tests validating the quantum bounce mechanism."""

    def test_bounce_occurs(self, default_results):
        """Test that the scale factor never reaches zero (singularity avoided)."""
        r = default_results
        assert r.a_min > 0.1, f"Bounce failed: a_min = {r.a_min} (too close to zero)"

    def test_bounce_validated(self, default_results):
        """Test that curvature inverts at bounce (ä > 0)."""
        r = default_results
        assert r.a_ddot_bounce > 0, (
            f"Bounce validation failed: ä = {r.a_ddot_bounce} (should be > 0)"
        )

    def test_a_dot_changes_sign(self, default_results):
        """Test that ȧ changes from negative (collapse) to positive (expansion)."""
        r = default_results
        # First points should be collapsing
        assert r.a_dot[0] < 0, "Initial state should be collapsing (ȧ < 0)"
        # Last points should be expanding
        assert r.a_dot[-1] > 0, "Final state should be expanding (ȧ > 0)"


# =============================================================================
# TEST 3: PHYSICAL CONSISTENCY
# =============================================================================

class TestPhysicalConsistency:
    """Tests for physical consistency of the evolution."""

    def test_rho_m_nonnegative(self, default_results):
        """Test that matter density stays non-negative."""
        r = default_results
        assert np.all(r.rho_m >= -1e-10), (
            f"Matter density went negative: min(ρ_m) = {r.rho_m.min()}"
        )

    def test_rho_Phi_nonnegative(self, default_results):
        """Test that dilaton density stays non-negative."""
        r = default_results
        assert np.all(r.rho_Phi >= -1e-10), (
            f"Dilaton density went negative: min(ρ_Φ) = {r.rho_Phi.min()}"
        )

    def test_torsion_scaling(self, default_results):
        """Test that torsion density scales as a^(-6)."""
        r = default_results
        # Pick two points away from the bounce for numerical stability
        i = len(r.a) // 4
        j = 3 * len(r.a) // 4

        a_ratio = r.a[i] / r.a[j]
        rho_T_ratio = r.rho_T[j] / r.rho_T[i]
        expected_ratio = a_ratio ** 6

        relative_error = abs(rho_T_ratio - expected_ratio) / expected_ratio
        assert relative_error < 0.01, (
            f"Torsion scaling violated: expected ratio {expected_ratio:.4f}, "
            f"got {rho_T_ratio:.4f} (error: {relative_error:.2%})"
        )


# =============================================================================
# TEST 4: DARK ENERGY EMERGENCE
# =============================================================================

class TestDarkEnergy:
    """Tests for the emergent dark energy phase."""

    def test_dark_energy_dominance(self, default_results):
        """Test that dilaton dominates over matter at late times."""
        r = default_results
        assert r.rho_Phi[-1] > r.rho_m[-1], (
            f"Dark energy did not emerge: ρ_Φ(final) = {r.rho_Phi[-1]:.2e} "
            f"< ρ_m(final) = {r.rho_m[-1]:.2e}"
        )

    def test_hubble_positive_final(self, default_results):
        """Test that the universe is expanding at the end."""
        r = default_results
        assert r.H[-1] > 0, f"Final Hubble should be positive, got {r.H[-1]}"


# =============================================================================
# TEST 5: PARAMETER VALIDATION
# =============================================================================

class TestParameterValidation:
    """Tests for parameter validation."""

    def test_invalid_w_m_rejected(self):
        """Test that invalid w_m values are rejected."""
        with pytest.raises(ValueError):
            ModelParameters(w_m=2.0)

    def test_invalid_w_Phi_rejected(self):
        """Test that invalid w_Phi values are rejected."""
        with pytest.raises(ValueError):
            ModelParameters(w_Phi=0.5)

    def test_invalid_w_T_rejected(self):
        """Test that w_T must be exactly +1."""
        with pytest.raises(ValueError):
            ModelParameters(w_T=0.5)

    def test_negative_Gamma_rejected(self):
        """Test that negative Gamma is rejected."""
        with pytest.raises(ValueError):
            ModelParameters(Gamma=-0.01)


# =============================================================================
# TEST 6: GAMMA DEPENDENCE
# =============================================================================

class TestGammaDependence:
    """Tests exploring the effect of different Gamma values."""

    def test_different_Gamma_gives_different_results(self):
        """Test that changing Gamma produces different dynamics."""
        sim1 = MGACosmology(params=ModelParameters(Gamma=0.001), verbose=False)
        sim2 = MGACosmology(params=ModelParameters(Gamma=0.1), verbose=False)

        r1 = sim1.run()
        r2 = sim2.run()

        # Final dilaton densities should differ significantly
        ratio = r1.rho_Phi[-1] / r2.rho_Phi[-1]
        assert ratio > 2 or ratio < 0.5, (
            f"Different Gamma should give different results, "
            f"but ratio = {ratio:.2f}"
        )

    def test_larger_Gamma_faster_decay(self):
        """Test that larger Gamma leads to faster dilaton decay."""
        sim_small = MGACosmology(params=ModelParameters(Gamma=0.001), verbose=False)
        sim_large = MGACosmology(params=ModelParameters(Gamma=0.1), verbose=False)

        r_small = sim_small.run()
        r_large = sim_large.run()

        # Larger Gamma should give smaller final dilaton density
        assert r_large.rho_Phi[-1] < r_small.rho_Phi[-1], (
            "Larger Gamma should lead to faster dilaton decay"
        )


# =============================================================================
# TEST 7: OUTPUT GENERATION
# =============================================================================

class TestOutputs:
    """Tests for output generation (plots, CSV)."""

    def test_csv_export(self, default_sim, default_results, tmp_path):
        """Test that CSV export works correctly."""
        csv_path = tmp_path / "test_data.csv"
        default_sim.export_csv(str(csv_path))

        assert csv_path.exists(), "CSV file was not created"

        # Check that file has content
        content = csv_path.read_text()
        assert len(content) > 1000, "CSV file seems too short"
        assert "MGA Cosmology" in content, "CSV missing header"
        assert "github.com/NahuelGomez-Dev/MGA-Cosmology" in content, "CSV missing repo URL"
        assert "10.5281/zenodo.21041476" in content, "CSV missing paper DOI"

    def test_summary_generation(self, default_sim, default_results):
        """Test that summary string is generated."""
        summary = default_sim.summary()
        assert isinstance(summary, str)
        assert "BOUNCE" in summary
        assert "github.com/NahuelGomez-Dev/MGA-Cosmology" in summary


# =============================================================================
# RUN ALL TESTS
# =============================================================================

if __name__ == '__main__':
    pytest.main([__file__, '-v'])
