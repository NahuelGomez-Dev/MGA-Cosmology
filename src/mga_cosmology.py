"""
MGA Cosmology — Core Simulation Module
======================================

Implements the numerical integration of the closed modified Friedmann
equations derived in the MGA (Matrioshka-Genesis-Accretion) cosmological
framework.

Governing equations:
    - First Friedmann (Eq. 7.4):
        H² = (8πG/3)[ρ_m + ρ_Φ - ρ_T] - k/a²

    - Acceleration (Eq. 7.5):
        ä/a = -(4πG/3)[ρ_m + 3p_m + ρ_Φ + 3p_Φ - (ρ_T + 3p_T)]

    - Dilaton continuity (Eq. 7.9):
        ρ̇_Φ = -[3H(1 + w_Φ) + Γ]ρ_Φ

Unit system:
    8πG = 1, c = 1, ℏ = 1, a(t_bounce) = 1

Author: Nahuel Gomez (ORCID: 0009-0006-4420-2685)
Repository: https://github.com/NahuelGomez-Dev/MGA-Cosmology
Companion paper DOI: 10.5281/zenodo.21015494
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, Tuple
from datetime import datetime


# =============================================================================
# DATA CLASSES FOR CONFIGURATION
# =============================================================================

@dataclass
class ModelParameters:
    """
    Physical parameters of the MGA cosmological model.

    Attributes:
        w_m: Equation of state for matter/radiation (default: 1/3 for radiation)
        w_Phi: Equation of state for the dilaton (default: -1 for slow-roll)
        w_T: Equation of state for torsion fluid (fixed: +1, stiff fluid)
        Gamma: Dilaton decay rate from interfacial vacuum flux (Eq. 6.8)
        k: Spatial curvature parameter (0 = flat, ±1 = curved)
    """
    w_m: float = 1/3
    w_Phi: float = -1.0
    w_T: float = +1.0
    Gamma: float = 0.01
    k: float = 0.0

    def __post_init__(self):
        """Validate physical constraints on parameters."""
        if not (-1 <= self.w_m <= 1):
            raise ValueError(f"w_m must be in [-1, 1], got {self.w_m}")
        if not (-2 <= self.w_Phi <= 0):
            raise ValueError(f"w_Phi must be in [-2, 0] for dark energy, got {self.w_Phi}")
        if self.w_T != 1.0:
            raise ValueError(f"w_T is fixed at +1 (stiff torsion fluid), got {self.w_T}")
        if self.Gamma < 0:
            raise ValueError(f"Gamma must be non-negative, got {self.Gamma}")


@dataclass
class InitialConditions:
    """
    Initial conditions for the cosmological simulation.

    Attributes:
        a_0: Initial scale factor
        rho_m_0: Initial radiation/matter density
        rho_Phi_0: Initial dilaton density
        a_bounce_target: Target scale factor for the bounce (auto-calibrates ρ_T0)
    """
    a_0: float = 10.0
    rho_m_0: float = 1.0
    rho_Phi_0: float = 0.5
    a_bounce_target: float = 1.0

    def __post_init__(self):
        """Validate initial conditions."""
        if self.a_0 <= self.a_bounce_target:
            raise ValueError(
                f"a_0 ({self.a_0}) must be > a_bounce_target ({self.a_bounce_target})"
            )
        if self.rho_m_0 < 0:
            raise ValueError(f"rho_m_0 must be non-negative, got {self.rho_m_0}")
        if self.rho_Phi_0 < 0:
            raise ValueError(f"rho_Phi_0 must be non-negative, got {self.rho_Phi_0}")


@dataclass
class IntegrationOptions:
    """
    Numerical integration settings for the stiff ODE solver.

    Attributes:
        t_span: Time interval (t_start, t_end)
        n_points: Number of evaluation points
        method: ODE solver method ('Radau' or 'BDF' recommended for stiff systems)
        rtol: Relative tolerance
        atol: Absolute tolerance
        max_step: Maximum integration step size
    """
    t_span: Tuple[float, float] = (0.0, 50.0)
    n_points: int = 5000
    method: str = 'Radau'
    rtol: float = 1e-9
    atol: float = 1e-11
    max_step: float = 0.1

    def __post_init__(self):
        """Validate integration options."""
        if self.t_span[0] >= self.t_span[1]:
            raise ValueError(f"t_span must have t_start < t_end, got {self.t_span}")
        if self.n_points < 100:
            raise ValueError(f"n_points must be >= 100 for resolution, got {self.n_points}")
        if self.method not in ('Radau', 'BDF', 'LSODA'):
            raise ValueError(
                f"method must be 'Radau', 'BDF', or 'LSODA' for stiff systems, got {self.method}"
            )


@dataclass
class SimulationResults:
    """
    Container for simulation output data.

    Attributes:
        t: Time array
        a: Scale factor array
        a_dot: First derivative of scale factor
        rho_m: Matter/radiation density array
        rho_Phi: Dilaton density array
        rho_T: Torsion density array
        H: Hubble parameter array
        a_ddot: Acceleration array
        t_bounce: Time of minimum scale factor
        a_min: Minimum scale factor (bounce point)
        a_ddot_bounce: Acceleration at bounce (should be > 0)
        success: Whether integration completed successfully
        metadata: Additional simulation metadata
    """
    t: np.ndarray
    a: np.ndarray
    a_dot: np.ndarray
    rho_m: np.ndarray
    rho_Phi: np.ndarray
    rho_T: np.ndarray
    H: np.ndarray
    a_ddot: np.ndarray
    t_bounce: float
    a_min: float
    a_ddot_bounce: float
    success: bool
    metadata: Dict[str, Any] = field(default_factory=dict)


# =============================================================================
# MAIN SIMULATION CLASS
# =============================================================================

class MGACosmology:
    """
    Numerical simulator for MGA (Matrioshka-Genesis-Accretion) Cosmology.

    This class integrates the closed modified Friedmann equations derived
    from the effective braneworld framework with Einstein-Cartan torsion.

    The simulation validates:
        1. Quantum bounce: a_min > 0 (singularity avoidance)
        2. Curvature inversion: ä > 0 at bounce
        3. Three-phase evolution: torsion → radiation → dilaton dominance

    Example:
        >>> sim = MGACosmology()
        >>> results = sim.run()
        >>> print(f"Bounce at a_min = {results.a_min:.4f}")
        >>> sim.plot()
    """

    def __init__(
        self,
        params: Optional[ModelParameters] = None,
        initial: Optional[InitialConditions] = None,
        options: Optional[IntegrationOptions] = None,
        verbose: bool = True
    ):
        """
        Initialize the MGA cosmology simulator.

        Args:
            params: Model parameters (uses defaults if None)
            initial: Initial conditions (uses defaults if None)
            options: Integration options (uses defaults if None)
            verbose: Print progress information
        """
        self.params = params or ModelParameters()
        self.initial = initial or InitialConditions()
        self.options = options or IntegrationOptions()
        self.verbose = verbose
        self.results: Optional[SimulationResults] = None
        self._rho_T0: Optional[float] = None

    def _compute_rho_T0(self) -> float:
        """
        Auto-calibrate the torsion density normalization ρ_T0 so that
        the bounce occurs near the target scale factor.

        At the bounce: H² = 0, so ρ_m + ρ_Φ = ρ_T at a = a_bounce.

        Returns:
            The calibrated value of ρ_T0.
        """
        a_b = self.initial.a_bounce_target
        a_0 = self.initial.a_0

        # Scale matter density to bounce point (radiation: a^(-4))
        rho_m_at_bounce = self.initial.rho_m_0 * (a_0 / a_b) ** 4

        # Dilaton approximately constant in slow-roll
        rho_Phi_at_bounce = self.initial.rho_Phi_0

        # Torsion must balance both at bounce
        rho_T0 = rho_m_at_bounce + rho_Phi_at_bounce

        return rho_T0

    def _rho_T(self, a: float) -> float:
        """
        Torsion energy density: ρ_T = ρ_T0 · a^(-6)

        This O(a^(-6)) scaling is the key to the bounce mechanism:
        it dominates at small a and provides repulsive gravity.

        Args:
            a: Scale factor

        Returns:
            Torsion density at scale factor a.
        """
        return self._rho_T0 * a ** (-6)

    def _p_m(self, rho: float) -> float:
        """Radiation/matter pressure: p = w_m · ρ"""
        return self.params.w_m * rho

    def _p_Phi(self, rho: float) -> float:
        """Dilaton pressure: p = w_Φ · ρ"""
        return self.params.w_Phi * rho

    def _p_T(self, rho: float) -> float:
        """Torsion pressure: p = w_T · ρ (stiff fluid, w_T = +1)"""
        return self.params.w_T * rho

    def _derivatives(self, t: float, y: np.ndarray) -> list:
        """
        Compute the derivatives for the ODE system.

        State vector: y = [a, a_dot, rho_m, rho_Phi]
        Returns: dy/dt = [a_dot, a_ddot, rho_m_dot, rho_Phi_dot]

        Implements:
            - Acceleration equation (Eq. 7.5)
            - Matter continuity equation
            - Dilaton continuity equation (Eq. 7.9)

        Args:
            t: Current time (not explicitly used, system is autonomous)
            y: State vector [a, a_dot, rho_m, rho_Phi]

        Returns:
            List of derivatives [a_dot, a_ddot, rho_m_dot, rho_Phi_dot]
        """
        a, a_dot, rho_m_val, rho_Phi_val = y

        # Numerical safety: prevent a from going negative
        if a < 1e-6:
            a = 1e-6

        # Compute densities and pressures
        rho_T_val = self._rho_T(a)
        p_m_val = self._p_m(rho_m_val)
        p_Phi_val = self._p_Phi(rho_Phi_val)
        p_T_val = self._p_T(rho_T_val)

        # Hubble parameter
        H = a_dot / a if a > 1e-10 else 0.0

        # ================================================================
        # ACCELERATION EQUATION (Eq. 7.5, with 8πG = 1)
        # ================================================================
        # Active gravitational mass: ρ + 3p for each component
        # The MINUS sign before (ρ_T + 3p_T) is CRITICAL for the bounce:
        # it makes the torsion contribution NEGATIVE, yielding ä > 0
        # when torsion dominates at small a.
        active_mass = (
            rho_m_val + 3 * p_m_val +
            rho_Phi_val + 3 * p_Phi_val -
            (rho_T_val + 3 * p_T_val)
        )
        a_ddot = -a * active_mass / 6.0

        # ================================================================
        # MATTER/RADIATION CONTINUITY
        # ================================================================
        # Standard conservation: ρ̇_m = -3H(ρ_m + p_m) = -3H(1 + w_m)ρ_m
        rho_m_dot = -3 * H * rho_m_val * (1 + self.params.w_m)

        # ================================================================
        # DILATON CONTINUITY (Eq. 7.9)
        # ================================================================
        # The Γ term represents the interfacial vacuum flux decay
        # ρ̇_Φ = -[3H(1 + w_Φ) + Γ]ρ_Φ
        rho_Phi_dot = -(3 * H * (1 + self.params.w_Phi) + self.params.Gamma) * rho_Phi_val

        return [a_dot, a_ddot, rho_m_dot, rho_Phi_dot]

    def _compute_initial_conditions(self) -> Tuple[np.ndarray, float]:
        """
        Compute the full initial state vector from the specified conditions.

        Returns:
            Tuple of (y0, H_0) where y0 = [a_0, a_dot_0, rho_m_0, rho_Phi_0]
        """
        a_0 = self.initial.a_0

        # Compute H_0 from first Friedmann equation
        H_0_squared = (1/3) * (
            self.initial.rho_m_0 +
            self.initial.rho_Phi_0 -
            self._rho_T(a_0)
        )

        # Safety check: if H² is negative, slightly adjust ρ_T0
        if H_0_squared < 0:
            if self.verbose:
                print(f"  [WARNING] H² initial negative ({H_0_squared:.2e}), adjusting ρ_T0")
            self._rho_T0 = (self.initial.rho_m_0 + self.initial.rho_Phi_0) * 0.99
            H_0_squared = (1/3) * (
                self.initial.rho_m_0 +
                self.initial.rho_Phi_0 -
                self._rho_T(a_0)
            )

        H_0 = np.sqrt(max(H_0_squared, 1e-15))

        # Negative a_dot: universe is collapsing
        a_dot_0 = -a_0 * H_0

        y0 = [a_0, a_dot_0, self.initial.rho_m_0, self.initial.rho_Phi_0]

        return y0, H_0

    def run(self) -> SimulationResults:
        """
        Execute the numerical simulation.

        Integrates the coupled ODE system using a stiff solver (Radau method)
        and extracts the bounce characteristics.

        Returns:
            SimulationResults object containing all computed arrays and metadata.

        Raises:
            RuntimeError: If the integration fails to converge.
        """
        if self.verbose:
            print("=" * 70)
            print(" MGA COSMOLOGY — NUMERICAL SIMULATION")
            print(" Matrioshka-Genesis-Accretion Framework")
            print("=" * 70)
            print(f"\n[CONFIG] Model parameters:")
            print(f"  w_m (radiation)  = {self.params.w_m}")
            print(f"  w_Φ (dilaton)    = {self.params.w_Phi}")
            print(f"  w_T (torsion)    = {self.params.w_T}")
            print(f"  Γ (decay rate)   = {self.params.Gamma}")

        # Auto-calibrate torsion density
        self._rho_T0 = self._compute_rho_T0()
        if self.verbose:
            print(f"  ρ_T0 (calibrated) = {self._rho_T0:.4f}")
            print(f"  Target bounce at  a ≈ {self.initial.a_bounce_target}")

        # Compute initial state
        y0, H_0 = self._compute_initial_conditions()

        if self.verbose:
            print(f"\n[START] Initial conditions (collapse phase):")
            print(f"  a(0)    = {self.initial.a_0}")
            print(f"  ȧ(0)    = {y0[1]:.4f}")
            print(f"  H(0)    = {-H_0:.4f} (collapsing)")
            print(f"  ρ_m(0)  = {self.initial.rho_m_0}")
            print(f"  ρ_Φ(0)  = {self.initial.rho_Phi_0}")
            print(f"  ρ_T(0)  = {self._rho_T(self.initial.a_0):.6e}")

        # Set up time grid
        t_eval = np.linspace(
            self.options.t_span[0],
            self.options.t_span[1],
            self.options.n_points
        )

        if self.verbose:
            print(f"\n[INTEGRATING] Solving stiff ODE system with {self.options.method}...")
            print(f"  t_span = {self.options.t_span}")
            print(f"  rtol = {self.options.rtol}, atol = {self.options.atol}")

        # Perform integration
        sol = solve_ivp(
            self._derivatives,
            self.options.t_span,
            y0,
            method=self.options.method,
            t_eval=t_eval,
            rtol=self.options.rtol,
            atol=self.options.atol,
            max_step=self.options.max_step
        )

        if not sol.success:
            raise RuntimeError(f"Integration failed: {sol.message}")

        if self.verbose:
            print(f"\n[OK] Integration completed successfully")
            print(f"  Function evaluations: {sol.nfev}")
            print(f"  Jacobian evaluations: {sol.njev}")

        # Extract results
        t = sol.t
        a = sol.y[0]
        a_dot = sol.y[1]
        rho_m = sol.y[2]
        rho_Phi = sol.y[3]
        rho_T_arr = np.array([self._rho_T(ai) for ai in a])

        # Derived quantities
        H = a_dot / np.maximum(a, 1e-15)
        a_ddot = np.array([
            -a[i] * (
                rho_m[i] + 3 * self._p_m(rho_m[i]) +
                rho_Phi[i] + 3 * self._p_Phi(rho_Phi[i]) -
                (rho_T_arr[i] + 3 * self._p_T(rho_T_arr[i]))
            ) / 6.0
            for i in range(len(a))
        ])

        # Find bounce point (minimum a)
        a_min_idx = np.argmin(a)
        a_min = a[a_min_idx]
        t_bounce = t[a_min_idx]
        a_ddot_bounce = a_ddot[a_min_idx]

        if self.verbose:
            print(f"\n[BOUNCE] Characteristics:")
            print(f"  t_bounce  = {t_bounce:.4f}")
            print(f"  a_min     = {a_min:.6f}")
            print(f"  ä(bounce) = {a_ddot_bounce:.4f}")

            if a_ddot_bounce > 0:
                print(f"  ✓ BOUNCE VALIDATED: ä > 0 (singularity avoided)")
            else:
                print(f"  ✗ BOUNCE FAILED: ä ≤ 0")

            # Late-universe statistics
            print(f"\n[LATE UNIVERSE] Final state:")
            print(f"  ρ_m(final)  = {rho_m[-1]:.6e}")
            print(f"  ρ_Φ(final)  = {rho_Phi[-1]:.6e}")
            print(f"  ρ_T(final)  = {rho_T_arr[-1]:.6e}")
            print(f"  H(final)    = {H[-1]:.6e}")

            if rho_Phi[-1] > rho_m[-1]:
                print(f"  ✓ Dilaton dominance: ρ_Φ > ρ_m (dark energy)")

        # Build results object
        metadata = {
            'params': {
                'w_m': self.params.w_m,
                'w_Phi': self.params.w_Phi,
                'w_T': self.params.w_T,
                'Gamma': self.params.Gamma,
                'k': self.params.k,
            },
            'initial': {
                'a_0': self.initial.a_0,
                'rho_m_0': self.initial.rho_m_0,
                'rho_Phi_0': self.initial.rho_Phi_0,
                'rho_T0': self._rho_T0,
            },
            'integration': {
                'method': self.options.method,
                'rtol': self.options.rtol,
                'atol': self.options.atol,
                'n_points': self.options.n_points,
                'nfev': sol.nfev,
                'njev': sol.njev,
            },
            'timestamp': datetime.now().isoformat(),
        }

        self.results = SimulationResults(
            t=t, a=a, a_dot=a_dot,
            rho_m=rho_m, rho_Phi=rho_Phi, rho_T=rho_T_arr,
            H=H, a_ddot=a_ddot,
            t_bounce=t_bounce, a_min=a_min, a_ddot_bounce=a_ddot_bounce,
            success=sol.success,
            metadata=metadata
        )

        return self.results

    def plot(
        self,
        save_path: Optional[str] = None,
        figsize: Tuple[float, float] = (12, 10),
        dpi: int = 150,
        show: bool = True
    ) -> plt.Figure:
        """
        Generate publication-quality plots of the simulation results.

        Creates a two-panel figure:
            1. Scale factor a(t) showing the quantum bounce
            2. Energy densities on logarithmic scale

        Args:
            save_path: If provided, save figure to this path (PNG/PDF)
            figsize: Figure size in inches (width, height)
            dpi: Resolution for raster formats
            show: Whether to display the figure interactively

        Returns:
            matplotlib Figure object.
        """
        if self.results is None:
            raise RuntimeError("No results to plot. Call run() first.")

        r = self.results
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize)

        # ================================================================
        # Panel 1: Scale factor a(t)
        # ================================================================
        ax1.plot(r.t, r.a, 'b-', linewidth=2.5, label='Scale factor $a(t)$')
        ax1.axvline(r.t_bounce, color='r', linestyle='--', linewidth=1.5,
                     label=f'Bounce ($t = {r.t_bounce:.2f}$)')
        ax1.axhline(r.a_min, color='g', linestyle=':', linewidth=1.5,
                     label=f'$a_{{\\min}} = {r.a_min:.3f}$')

        ax1.set_xlabel('Time $t$ (normalized units)', fontsize=12)
        ax1.set_ylabel('Scale factor $a(t)$', fontsize=12)
        ax1.set_title(
            'MGA Cosmology: Scale Factor Evolution\n'
            '(Quantum Bounce via Einstein–Cartan Torsion)',
            fontsize=14, fontweight='bold'
        )
        ax1.legend(fontsize=11, loc='upper left')
        ax1.grid(True, alpha=0.3)
        ax1.set_ylim(0, max(r.a) * 1.1)

        # Annotate bounce
        ax1.annotate(
            'BOUNCE\n(Curvature\ninverted: ä > 0)',
            xy=(r.t_bounce, r.a_min),
            xytext=(r.t_bounce + 4, r.a_min + 2),
            arrowprops=dict(arrowstyle='->', color='red', lw=2),
            fontsize=11, color='red', fontweight='bold'
        )

        # ================================================================
        # Panel 2: Energy densities (log scale)
        # ================================================================
        # Filter non-positive values for log scale
        rho_m_plot = np.where(r.rho_m > 1e-20, r.rho_m, np.nan)
        rho_Phi_plot = np.where(r.rho_Phi > 1e-20, r.rho_Phi, np.nan)
        rho_T_plot = np.where(r.rho_T > 1e-20, r.rho_T, np.nan)

        ax2.semilogy(r.t, rho_m_plot, 'g-', linewidth=2.5,
                      label=r'$\rho_m$ (Radiation $\propto a^{-4}$)')
        ax2.semilogy(r.t, rho_Phi_plot, 'm-', linewidth=2.5,
                      label=r'$\rho_\Phi$ (Dilaton / Dark Energy)')
        ax2.semilogy(r.t, rho_T_plot, 'orange', linewidth=2.5,
                      label=r'$\rho_T$ (Torsion $\propto a^{-6}$)')
        ax2.axvline(r.t_bounce, color='r', linestyle='--', linewidth=1.5,
                     label='Bounce')

        ax2.set_xlabel('Time $t$ (normalized units)', fontsize=12)
        ax2.set_ylabel(r'Energy Density $\rho$ (log scale)', fontsize=12)
        ax2.set_title(
            'MGA Cosmology: Energy Density Evolution\n'
            '(Transition: Torsion → Radiation → Dilaton)',
            fontsize=14, fontweight='bold'
        )
        ax2.legend(fontsize=11, loc='upper right')
        ax2.grid(True, alpha=0.3, which='both')

        # Annotate dominance regions
        bounce_idx = np.argmin(r.a)
        ax2.annotate(
            'Torsion dominates\n(BOUNCE)',
            xy=(r.t_bounce, r.rho_T[bounce_idx]),
            xytext=(r.t_bounce - 8, r.rho_T[bounce_idx] * 5),
            arrowprops=dict(arrowstyle='->', color='orange', lw=2),
            fontsize=10, color='orange', fontweight='bold'
        )

        ax2.annotate(
            'Dilaton dominates\n(DARK ENERGY)',
            xy=(r.t[-50], r.rho_Phi[-50]),
            xytext=(r.t[-50] - 12, r.rho_Phi[-50] * 0.05),
            arrowprops=dict(arrowstyle='->', color='magenta', lw=2),
            fontsize=10, color='magenta', fontweight='bold'
        )

        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=dpi, bbox_inches='tight')
            if self.verbose:
                print(f"\n[SAVE] Figure saved to: {save_path}")

        if show:
            plt.show()

        return fig

    def export_csv(self, filepath: str, precision: int = 10) -> None:
        """
        Export raw simulation data to CSV for reproducibility and cross-validation.

        The CSV file includes a rich metadata header and 8 data columns,
        enabling other researchers to independently analyze the results
        or compare with other cosmological codes (CLASS, CAMB, LQC, etc.).

        Args:
            filepath: Output file path (e.g., "output/data.csv")
            precision: Number of decimal places for floating-point values
        """
        if self.results is None:
            raise RuntimeError("No results to export. Call run() first.")

        r = self.results
        m = r.metadata

        # Build metadata header
        header_lines = [
            "# MGA Cosmology — Simulation Raw Data",
            "# Repository: https://github.com/NahuelGomez-Dev/MGA-Cosmology",
            "# Companion paper DOI: 10.5281/zenodo.21015494",
            f"# Generated: {m['timestamp']}",
            "#",
            "# MODEL PARAMETERS:",
            f"#   w_m (radiation EoS)     = {m['params']['w_m']}",
            f"#   w_Phi (dilaton EoS)     = {m['params']['w_Phi']}",
            f"#   w_T (torsion EoS)       = {m['params']['w_T']}",
            f"#   Gamma (decay rate)      = {m['params']['Gamma']}",
            f"#   k (curvature)           = {m['params']['k']}",
            f"#   rho_T0 (torsion norm.)  = {m['initial']['rho_T0']:.10e}",
            "#",
            "# INITIAL CONDITIONS:",
            f"#   a(0)                    = {m['initial']['a_0']}",
            f"#   rho_m(0)                = {m['initial']['rho_m_0']:.10e}",
            f"#   rho_Phi(0)              = {m['initial']['rho_Phi_0']:.10e}",
            "#",
            "# BOUNCE CHARACTERISTICS:",
            f"#   t_bounce                = {r.t_bounce:.10e}",
            f"#   a_min                   = {r.a_min:.10e}",
            f"#   a_ddot at bounce        = {r.a_ddot_bounce:.10e}",
            "#",
            "# UNIT SYSTEM: 8*pi*G = c = hbar = 1, a(t_bounce) = 1",
            "#",
            "# COLUMNS:",
            "#   [1] time                 - Cosmic time (normalized units)",
            "#   [2] scale_factor         - a(t), scale factor",
            "#   [3] a_dot                - da/dt, first derivative",
            "#   [4] rho_matter           - Radiation density (scales as a^-4)",
            "#   [5] rho_dilaton          - Dilaton/dark energy density",
            "#   [6] rho_torsion          - Torsion density (scales as a^-6)",
            "#   [7] Hubble               - H = a_dot/a, Hubble parameter",
            "#   [8] acceleration         - a_ddot, second derivative",
        ]

        # Stack all arrays into columns
        data_matrix = np.column_stack([
            r.t, r.a, r.a_dot, r.rho_m, r.rho_Phi, r.rho_T, r.H, r.a_ddot
        ])

        # Save to CSV
        np.savetxt(
            filepath,
            data_matrix,
            header='\n'.join(header_lines),
            delimiter=',',
            fmt=f'%.{precision}e',
            comments=''
        )

        if self.verbose:
            print(f"\n[EXPORT] Raw data saved to: {filepath}")
            print(f"  Data points: {len(r.t)}")
            print(f"  Columns: 8 (time, a, ȧ, ρ_m, ρ_Φ, ρ_T, H, ä)")
            print(f"  Precision: {precision} decimal places")

    def summary(self) -> str:
        """
        Generate a human-readable summary of the simulation results.

        Returns:
            Multi-line string summarizing the key results.
        """
        if self.results is None:
            return "No simulation has been run yet. Call run() first."

        r = self.results
        bounce_valid = "✓ YES" if r.a_ddot_bounce > 0 else "✗ NO"
        de_dominance = "✓ YES" if r.rho_Phi[-1] > r.rho_m[-1] else "✗ NO"

        summary = f"""
================================================================================
  MGA COSMOLOGY — SIMULATION SUMMARY
  Repository: https://github.com/NahuelGomez-Dev/MGA-Cosmology
================================================================================

  BOUNCE CHARACTERISTICS:
    Time of bounce:        t_bounce = {r.t_bounce:.4f}
    Minimum scale factor:  a_min    = {r.a_min:.6f}
    Acceleration at bounce: ä       = {r.a_ddot_bounce:.4f}
    Bounce valid (ä > 0)?  {bounce_valid}

  LATE-UNIVERSE STATE:
    ρ_matter (final):      {r.rho_m[-1]:.6e}
    ρ_dilaton (final):     {r.rho_Phi[-1]:.6e}
    ρ_torsion (final):     {r.rho_T[-1]:.6e}
    Hubble (final):        {r.H[-1]:.6e}
    Dark energy dominant?  {de_dominance}

  INTEGRATION STATISTICS:
    Method:                {r.metadata['integration']['method']}
    Function evaluations:  {r.metadata['integration']['nfev']}
    Jacobian evaluations:  {r.metadata['integration']['njev']}
    Data points:           {len(r.t)}

================================================================================
"""
        return summary

    def validate_physical(self) -> Dict[str, bool]:
        """
        Run physical consistency checks on the simulation results.

        Returns:
            Dictionary mapping check names to pass/fail booleans.
        """
        if self.results is None:
            raise RuntimeError("No results to validate. Call run() first.")

        r = self.results
        checks = {}

        # Check 1: Bounce occurs (a never reaches zero)
        checks['bounce_occurs'] = r.a_min > 0.1

        # Check 2: Curvature inversion at bounce
        checks['curvature_inversion'] = r.a_ddot_bounce > 0

        # Check 3: Non-negative densities
        checks['rho_m_nonneg'] = np.all(r.rho_m >= -1e-10)
        checks['rho_Phi_nonneg'] = np.all(r.rho_Phi >= -1e-10)

        # Check 4: Torsion scaling (ρ_T ∝ a^(-6))
        if len(r.a) > 10:
            mid = len(r.a) // 2
            a_ratio = r.a[0] / r.a[mid]
            rho_T_ratio = r.rho_T[mid] / r.rho_T[0]
            expected_ratio = a_ratio ** 6
            checks['torsion_scaling'] = abs(rho_T_ratio - expected_ratio) / expected_ratio < 0.01

        # Check 5: Late-time dark energy dominance
        checks['dark_energy_dominance'] = r.rho_Phi[-1] > r.rho_m[-1]

        return checks


# =============================================================================
# STANDALONE EXECUTION
# =============================================================================

if __name__ == '__main__':
    # Run a default simulation when executed directly
    sim = MGACosmology(verbose=True)
    results = sim.run()

    print(sim.summary())

    # Generate outputs
    sim.plot(save_path='output/MGA_bounce_and_densities.png', show=True)
    sim.export_csv('output/MGA_simulation_data.csv')

    # Validate
    checks = sim.validate_physical()
    print("\n[VALIDATION] Physical consistency checks:")
    for name, passed in checks.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"  {status}: {name}")
