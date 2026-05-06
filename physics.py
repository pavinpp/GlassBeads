import os
import sys
import numpy as np
import jax
import jax.numpy as jnp
from functools import partial
from pathlib import Path

# Add project root to path to allow importing from src
sys.path.append(str(Path(__file__).resolve().parent))

from src.models import BGKSim
from src.lattice import LatticeD3Q19
from src.boundary_conditions import EquilibriumBC, BounceBack

# =============================================================================
#  SCALING & UNIT CONVERSION
# =============================================================================
class ThermoScaling:
    def __init__(self, dx_microns=22.0, q_ml_hr=10.0, diameter_mm=1.2,
                 nu_phys=1.0e-6, d_phys=5.0e-10, alpha_fluid=1.43e-7, alpha_solid=5.0e-7,
                 g_phys=-9.81, tau=0.70,
                 delta_c_nuc_phys=5.0, delta_c_grow_phys=0.5,
                 k_nuc=0.0005, k_grow=0.025, phi_s_seed=0.01,
                 alpha_housing_flow=0.0001, alpha_housing_shutin=0.005):
        self.dx = dx_microns * 1e-6
        self.q_ml_hr = q_ml_hr
        self.diameter = diameter_mm * 1e-3
        # Split housing thermal resistance into dynamic phases
        self.alpha_housing_flow = alpha_housing_flow
        self.alpha_housing_shutin = alpha_housing_shutin

        # Fluid
        self.nu_phys = nu_phys
        self.g_phys = g_phys
        self.tau = tau
        self.nu_lu = (2 * self.tau - 1) / 6
        self.dt = self.nu_lu * (self.dx**2 / self.nu_phys)

        # Solute (CuSO4)
        self.d_phys = d_phys
        self.d_lu_target = self.d_phys * (self.dt / self.dx**2)
        self.cs2 = 1.0 / 3.0
        self.tau_c = max(0.55, self.d_lu_target / self.cs2 + 0.5)
        self.d_lu = (self.tau_c - 0.5) * self.cs2
        self.omega_c = 1.0 / self.tau_c

        # Thermodynamic reference values
        self.t_hot_phys = 75.0   # °C injection temperature
        self.t_amb_phys = 20.0   # °C ambient
        self.c_in_phys = 60.0    # g/100mL injection concentration
        self.c_ref_phys = self.get_solubility(self.t_hot_phys)
        self.c_in_norm = self.c_in_phys / self.c_ref_phys
        self.t_hot = 1.0         # normalised
        self.t_amb = 0.0

        # CNT Kinetics
        self.delta_c_nuc_norm = delta_c_nuc_phys / self.c_ref_phys
        self.delta_c_grow_norm = delta_c_grow_phys / self.c_ref_phys
        self.k_nuc = k_nuc
        self.k_grow = k_grow
        self.phi_s_seed = phi_s_seed
        assert delta_c_grow_phys < delta_c_nuc_phys, "Growth threshold must be below nucleation threshold (MZW physics)"

        # Thermal (CHT)
        self.alpha_f_phys = alpha_fluid
        self.alpha_s_phys = alpha_solid
        self.alpha_f_lu = self.alpha_f_phys * (self.dt / self.dx**2)
        self.alpha_s_lu = self.alpha_s_phys * (self.dt / self.dx**2)
        self.tau_t_f = max(0.52, self.alpha_f_lu / self.cs2 + 0.5)
        self.tau_t_s = max(0.52, self.alpha_s_lu / self.cs2 + 0.5)
        self.omega_t_f = 1.0 / self.tau_t_f
        self.omega_t_s = 1.0 / self.tau_t_s

        # Volume / mass conversion
        self.v_vox_ml  = (self.dx * 100)**3          # m → cm → mL
        self.v_vox_mm3 = (self.dx * 1000)**3
        self.mass_per_voxel_mg = (self.c_ref_phys / 100.0) * self.v_vox_ml * 1000.0

        # Volumetric scaling: convert normalized solute mass → solid volume fraction
        # CuSO4·5H2O solid density: 2.284 g/cm³
        # Conversion factor: (c_ref_phys g/100mL) / (100 mL/100mL × 2.284 g/cm³)
        #                  = c_ref_phys / 228.4 (dimensionless volume fraction per unit normalized mass)
        self.rho_solid_phys = 2.284  # g/cm³
        self.mass_to_volume_factor = self.c_ref_phys / (100.0 * self.rho_solid_phys)

        # Velocities
        self.u_darcy_phys = ((self.q_ml_hr * 1e-6) / 3600.0) / (np.pi * (self.diameter / 2.0)**2)
        self.u_in_lu = self.u_darcy_phys * (self.dt / self.dx)
        self.g_lu = self.g_phys * (self.dt**2 / self.dx)
        self.cs = 1.0 / jnp.sqrt(3.0)
        self.ma_target = self.u_in_lu / float(self.cs)

    def get_solubility(self, t_phys):
        return 0.0051 * (t_phys**2) + 0.384 * t_phys + 23.09

    def _compute_rigorous_precip(self):
        """
        Compute the equilibrium precipitation accounting for solution volume reduction.
        Solves: (c_in - x) / (1 - x/rho_solid/100*c_ref) = s(T_amb)
        where x = mass precipitated per 100 mL initial solution
        Returns x in g/100mL
        """
        c_in = self.c_in_phys
        s_eq = self.get_solubility(self.t_amb_phys)
        # Rearrange: (c_in - x) = s_eq * (1 - x/(rho*100/c_ref))
        # c_in - x = s_eq - s_eq*x*c_ref/(rho*100)
        # c_in - s_eq = x - s_eq*x*c_ref/(rho*100)
        # c_in - s_eq = x(1 - s_eq*c_ref/(rho*100))
        #denominator = 1.0 - s_eq * self.c_ref_phys / (self.rho_solid_phys * 100.0)
        denominator = 1.0 - s_eq / (self.rho_solid_phys * 100.0)
        return (c_in - s_eq) / denominator

    def ramp_steps_from_time(self, ramp_time_phys):
        return max(1, int(ramp_time_phys / self.dt))

    def print_summary(self, physical_time_s, ramp_time_phys=0.05):
        steps = int(physical_time_s / self.dt)
        ramp_steps = self.ramp_steps_from_time(ramp_time_phys)
        print("\n" + "="*70)
        print("  JAX-LaB DIGITAL TWIN: THERMO-SOLUTE SIMULATION")
        print("="*70)
        print(f"  Voxel Size (dx):      {self.dx*1e6:.2f} um")
        print(f"  Time Step (dt):       {self.dt:.4e} s")
        print(f"  Fluid Tau:            {self.tau:.4f}")
        print(f"  Solute Tau:           {self.tau_c:.4f}")
        print(f"  Nuc threshold:        {self.delta_c_nuc_norm * self.c_ref_phys:.2f} g/100mL ({self.delta_c_nuc_norm:.4f} norm)")
        print(f"  Grow threshold:       {self.delta_c_grow_norm * self.c_ref_phys:.2f} g/100mL ({self.delta_c_grow_norm:.4f} norm)")
        print(f"  Inlet Thermal BC:     Adiabatic during shut-in")
        print(f"  Housing Wall BC:      Resistive cooling toward {self.t_amb_phys}°C")
        print(f"                        Flow α={self.alpha_housing_flow:.4f} | Shut-in α={self.alpha_housing_shutin:.4f}")
        print(f"  k_nuc / k_grow:       {self.k_nuc:.4f} / {self.k_grow:.4f}")
        print(f"  Thermal Tau (fluid):  {self.tau_t_f:.4f}")
        print(f"  Injection:            {self.t_hot_phys}°C at {self.c_in_phys} g/100mL ({self.c_in_norm:.4f} norm, ref {self.c_ref_phys:.2f})")
        print(f"  Ambient:              {self.t_amb_phys}°C")
        print(f"  Target Mach:          {self.ma_target:.5f}")
        print(f"  Mass per Voxel:       {self.mass_per_voxel_mg:.3e} mg")
        print(f"  Mass→Volume factor:   {self.mass_to_volume_factor:.4f}  (ρ_solid={self.rho_solid_phys:.3f} g/cm³)")
        print(f"  Dilute approx Δm:     {(self.c_in_phys - self.get_solubility(self.t_amb_phys)):.2f} g/100mL")
        print(f"  Rigorous Δm:          {self._compute_rigorous_precip():.2f} g/100mL")
        print("-" * 70)
        print(f"  Simulation Goal:      {physical_time_s} s → {steps} steps")
        print(f"  Ramp:                 {ramp_time_phys} s → {ramp_steps} steps  ({100*ramp_steps/steps:.1f}% of run)")
        print("="*70 + "\n")
        return steps


# =============================================================================
#  SIMULATION ENGINE
# =============================================================================
class ThermoGravityFlowSim(BGKSim):
    def __init__(self, geometry_mask, housing_mask, inlet_idx, outlet_idx, scaling, **kwargs):
        self.geometry_mask = geometry_mask
        self.housing_mask  = housing_mask
        self.custom_inlet_idx  = inlet_idx
        self.custom_outlet_idx = outlet_idx
        self.scaling = scaling

        self.solute_BCs  = []
        self.thermal_BCs = []
        self.solid_idx   = np.argwhere(self.geometry_mask == 1)
        self.in_idx_tuple  = tuple(self.custom_inlet_idx.T)
        self.out_idx_tuple = tuple(self.custom_outlet_idx.T)

        nx_val, ny_val, nz_val = self.geometry_mask.shape
        self.x_coords = jnp.arange(nx_val).reshape(nx_val, 1, 1, 1)
        ys, zs = np.meshgrid(np.arange(ny_val), np.arange(nz_val), indexing='ij')
        mid_y, mid_z = ny_val // 2, nz_val // 2
        rad_vox = (scaling.diameter / 2) / scaling.dx
        circle_mask = ((ys - mid_y)**2 + (zs - mid_z)**2) <= rad_vox**2

        # Inlet porosity correction
        inlet_face_fluid = np.sum((self.geometry_mask[0] == 0) & circle_mask)
        inlet_face_total = np.sum(circle_mask)
        self.inlet_porosity = inlet_face_fluid / max(1.0, inlet_face_total)

        # Planar fluid masks
        self.inlet_plane_fluid_mask  = jnp.array((self.geometry_mask[1]          == 0) & circle_mask)
        self.outlet_plane_fluid_mask = jnp.array((self.geometry_mask[nx_val - 2] == 0) & circle_mask)

        # Periodic isolation
        solid_in = np.argwhere(self.geometry_mask[0] == 1)
        solid_in = np.insert(solid_in, 0, 0, axis=1)
        self.solid_in_idx  = tuple(solid_in.T)
        self.solid_in_adj  = (solid_in[:, 0] + 1, solid_in[:, 1], solid_in[:, 2])

        solid_out = np.argwhere(self.geometry_mask[-1] == 1)
        solid_out = np.insert(solid_out, 0, nx_val - 1, axis=1)
        self.solid_out_idx = tuple(solid_out.T)
        self.solid_out_adj = (solid_out[:, 0] - 1, solid_out[:, 1], solid_out[:, 2])

        self.omega_t_field = jnp.where(
            self.geometry_mask == 1,
            self.scaling.omega_t_s,
            self.scaling.omega_t_f,
        )[..., None]

        # Housing-wall thermal BC infrastructure
        housing_idx = np.argwhere(self.housing_mask == 1)
        self.housing_idx_tuple = tuple(housing_idx.T)
        self.n_housing_cells = housing_idx.shape[0]

        super().__init__(**kwargs)

    def initialize_passive_fields(self, init_val=0.0):
        """Initialize scalar populations to equilibrium with u=0 and given normalized value."""
        scalar_mac = jnp.full((self.nx, self.ny, self.nz, 1), init_val)
        u_init = jnp.zeros((self.nx, self.ny, self.nz, 3))
        # Passive scalars assume density=1 locally in feq
        eq_pops = self.equilibrium(scalar_mac, u_init, cast_output=False)
        return self.precisionPolicy.cast_to_output(eq_pops)

    def initialize_macroscopic_fields(self):
        """Explicitly initialize fluid fields."""
        rho = jnp.ones((self.nx, self.ny, self.nz, 1))
        u   = jnp.zeros((self.nx, self.ny, self.nz, 3))
        return self.precisionPolicy.cast_to_output(rho), self.precisionPolicy.cast_to_output(u)

    def set_boundary_conditions(self):
        u_zeros_in  = jnp.zeros((len(self.custom_inlet_idx),  3))
        u_zeros_out = jnp.zeros((len(self.custom_outlet_idx), 3))
        rho_one_in  = jnp.ones ((len(self.custom_inlet_idx),  1))
        rho_one_out = jnp.ones ((len(self.custom_outlet_idx), 1))

        self.inlet_bc  = EquilibriumBC(self.in_idx_tuple,  self.gridInfo, self.precisionPolicy, rho_one_in,  u_zeros_in)
        self.outlet_bc = EquilibriumBC(self.out_idx_tuple, self.gridInfo, self.precisionPolicy, rho_one_out, u_zeros_out)
        self.BCs.append(BounceBack(tuple(self.solid_idx.T), self.gridInfo, self.precisionPolicy))

        self.solute_inlet_bc = EquilibriumBC(self.in_idx_tuple, self.gridInfo, self.precisionPolicy,
                                             jnp.zeros((len(self.custom_inlet_idx), 1)), u_zeros_in)
        self.solute_BCs.append(BounceBack(tuple(self.solid_idx.T), self.gridInfo, self.precisionPolicy))

        housing_indices = np.argwhere(self.housing_mask == 1)
        self.thermal_BCs.append(BounceBack(tuple(housing_indices.T), self.gridInfo, self.precisionPolicy))

    def apply_passive_bc(self, gout, gin, t, implementation_step, bc_list):
        for bc in bc_list:
            gout = bc.prepare_populations(gout, gin, implementation_step)
            if bc.implementationStep == implementation_step:
                if bc.isDynamic:
                    gout = bc.apply(gout, gin, t)
                else:
                    gout = gout.at[bc.indices].set(bc.apply(gout, gin))
        return gout

    def get_force(self):
        return None

    @partial(jax.jit, static_argnums=(0,), donate_argnums=(1, 2, 3, 4))
    def apply_optimized_physics(self, f, g, h, s, t, flow_steps, shutin=False):
        f_comp = self.precisionPolicy.cast_to_compute(f)
        g_comp = self.precisionPolicy.cast_to_compute(g)
        h_comp = self.precisionPolicy.cast_to_compute(h)
        s_comp = self.precisionPolicy.cast_to_compute(s)

        # Calculate Dynamic Ramp Times
        flow_time_phys = flow_steps * self.scaling.dt
        v_ramp_time_phys = jnp.minimum(flow_time_phys * 0.04, 0.4)
        c_ramp_add_time_phys = jnp.minimum(flow_time_phys * 0.06, 0.6)
        c_ramp_time_phys = v_ramp_time_phys + c_ramp_add_time_phys

        v_ramp_steps = (v_ramp_time_phys / self.scaling.dt).astype(jnp.int32)
        c_ramp_steps = (c_ramp_time_phys / self.scaling.dt).astype(jnp.int32)

        v_ramp = jnp.minimum(1.0, t / jnp.maximum(1, v_ramp_steps))
        c_ramp = jnp.minimum(1.0, t / jnp.maximum(1, c_ramp_steps))

        effective_u_in = jnp.where(shutin, 0.0, self.scaling.u_in_lu)
        fluid_mask = (self.geometry_mask == 0)[..., None]
        
        # Shared reverse indices for D3Q19
        opp_idx = jnp.array(self.lattice.opp_indices, dtype=jnp.int32)

        # 1. FLUID (Coupled with Solid Fraction)
        rho, u = self.update_macroscopic(f_comp)
        
        # A) Calculate solid fraction (phi_s). 
        phi_s = jnp.clip(s_comp / 1.0, 0.0, 1.0)
        
        # CREATE HYDRODYNAMIC VALVE FOR SHUT-IN (Perfect Mass Conservation)
        valve_mask = jnp.zeros_like(phi_s)
        valve_mask = valve_mask.at[self.in_idx_tuple].set(1.0)
        valve_mask = valve_mask.at[self.out_idx_tuple].set(1.0)
        
        effective_phi_s = jnp.where(shutin, jnp.maximum(phi_s, valve_mask), phi_s)
        
        # B) DAMPEN VELOCITY: Kill momentum inside solid to prevent acoustic shocks.
        u_damped = jnp.where(fluid_mask, u * (1.0 - effective_phi_s), 0.0)
        
        # C) BGK Collision 
        feq  = self.equilibrium(rho, u_damped, cast_output=False)
        f_fluid = f_comp - self.omega * (f_comp - feq)
        
        # D) Bounce-Back Collision
        f_solid = f_comp[..., opp_idx]
        
        # E) Continuous Partial Bounce-Back (uses the effective valve mask)
        fout = (1.0 - effective_phi_s) * f_fluid + effective_phi_s * f_solid
        
        fout = self.apply_bc(self.precisionPolicy.cast_to_output(fout), f, t, "PostCollision")
        f_post = self.streaming(fout)

        # ---------------------------------------------------------
        # 2. SOLUTE (WITH STRICT MASS CONSERVATION)
        # ---------------------------------------------------------
        c_mac = jnp.sum(g_comp, axis=-1, keepdims=True)
        
        t_mac_local = jnp.sum(h_comp, axis=-1, keepdims=True)
        t_phys_local = self.scaling.t_amb_phys + t_mac_local * (self.scaling.t_hot_phys - self.scaling.t_amb_phys)
        
        s_t = self.scaling.get_solubility(t_phys_local)
        c_eq_norm = s_t / self.scaling.c_ref_phys

        # Volume-corrected equilibrium: account for solution volume reduction as solid forms
        # When solid occupies fraction phi_s of voxel, solution occupies (1 - phi_s)
        # Actual concentration in solution = c_mac / (1 - phi_s)
        # Precipitation stops when c_mac / (1 - phi_s) = c_eq
        # Therefore effective threshold: c_mac > c_eq * (1 - phi_s)
        c_eq_effective = c_eq_norm * (1.0 - phi_s)
        
        # A) Calculate CPBB blend FIRST using the damped velocity and valve mask
        geq = self.equilibrium(c_mac, u_damped, cast_output=False)
        g_fluid = g_comp - self.scaling.omega_c * (g_comp - geq)
        g_solid = g_comp[..., opp_idx]
        gout = (1.0 - effective_phi_s) * g_fluid + effective_phi_s * g_solid
        
        # B) CALCULATE TWO-PHASE KINETIC SINK
        # Phase 1: NUCLEATION — fires only when Δc exceeds the metastable zone
        #          width AND the cell has not yet nucleated (φ_s < seed)
        # Phase 2: GROWTH — fires on already-nucleated cells (φ_s ≥ seed)
        #          at a lower threshold; continues building existing crystal
        delta_c = c_mac - c_eq_effective
        
        is_seeded = phi_s >= self.scaling.phi_s_seed
        can_nucleate = jnp.logical_and(
            jnp.logical_not(is_seeded),
            delta_c > self.scaling.delta_c_nuc_norm
        )
        can_grow = jnp.logical_and(
            is_seeded,
            delta_c > self.scaling.delta_c_grow_norm
        )
        
        # Driving Δc above each respective threshold (excess over barrier)
        nuc_drive = jnp.maximum(0.0, delta_c - self.scaling.delta_c_nuc_norm)
        grow_drive = jnp.maximum(0.0, delta_c - self.scaling.delta_c_grow_norm)
        
        raw_precip = jnp.where(
            fluid_mask,
            jnp.where(can_nucleate, self.scaling.k_nuc * nuc_drive * (1.0 - phi_s), 0.0)
            + jnp.where(can_grow, self.scaling.k_grow * grow_drive * (1.0 - phi_s), 0.0),
            0.0
        )
        
        # C) Strict Thermodynamic Clamp: 
        # Never extract more than 90% of the available supersaturation barrier (delta_c) 
        # in a single step to prevent kinetic overshoot > 100% completion.
        available_excess = jnp.maximum(0.0, delta_c)
        precipitation_amount = jnp.minimum(raw_precip, available_excess * 0.9)

        # Secondary numerical clamp: prevent extracting more than 10% of total mass
        precipitation_amount = jnp.minimum(precipitation_amount, c_mac * 0.1)
        
        # D) Apply sink exactly to the blended output to guarantee perfect mass tracking
        W = jnp.array(self.lattice.w, dtype=g_comp.dtype)
        gout = gout - W * precipitation_amount
        
        gout = self.apply_passive_bc(self.precisionPolicy.cast_to_output(gout), g, t, "PostCollision", self.solute_BCs)
        g_post = self.streaming(gout)
        # ---------------------------------------------------------

        # 3. THERMAL
        temp_mac = jnp.sum(h_comp, axis=-1, keepdims=True)
        heq      = self.equilibrium(temp_mac, u, cast_output=False)
        hout     = h_comp - self.omega_t_field * (h_comp - heq)
        hout     = self.apply_passive_bc(self.precisionPolicy.cast_to_output(hout), h, t, "PostCollision", self.thermal_BCs)
        h_post   = self.streaming(hout)

        # 4. BCs
        u_in_target = (effective_u_in / self.inlet_porosity) * v_ramp
        v_in_ramp   = jnp.zeros((len(self.custom_inlet_idx),  3)).at[:, 0].set(u_in_target)
        v_out_ramp  = jnp.zeros((len(self.custom_outlet_idx), 3)).at[:, 0].set(u_in_target)

        # Fluid BCs
        f_final = self.apply_bc(f_post, fout, t, "PostStreaming")
        feq_in = self.inlet_bc.equilibrium(jnp.ones((len(self.custom_inlet_idx), 1)), v_in_ramp)
        f_final_in = jnp.where(shutin, f_post[self.in_idx_tuple], self.precisionPolicy.cast_to_output(feq_in))
        f_final = f_final.at[self.in_idx_tuple].set(f_final_in)
        
        feq_out = self.outlet_bc.equilibrium(jnp.ones((len(self.custom_outlet_idx), 1)), v_out_ramp)
        f_final_out = jnp.where(shutin, f_post[self.out_idx_tuple], self.precisionPolicy.cast_to_output(feq_out))
        f_final = f_final.at[self.out_idx_tuple].set(f_final_out)

        # Solute BCs
        g_final  = self.apply_passive_bc(g_post, gout, t, "PostStreaming", self.solute_BCs)
        c_in_injection = jnp.where(shutin, 0.0, self.scaling.c_in_norm * c_ramp)
        geq_in_sol = self.solute_inlet_bc.equilibrium(jnp.full((len(self.custom_inlet_idx), 1), c_in_injection), v_in_ramp)
        g_final_in = jnp.where(shutin, g_post[self.solute_inlet_bc.indices], self.precisionPolicy.cast_to_output(geq_in_sol))
        g_final = g_final.at[self.solute_inlet_bc.indices].set(g_final_in)
        
        c_mac_xm2   = jnp.sum(self.precisionPolicy.cast_to_compute(g_post[-2]), axis=-1)
        outlet_count = jnp.maximum(1.0, jnp.sum(self.outlet_plane_fluid_mask))
        c_avg_xm2   = jnp.sum(jnp.where(self.outlet_plane_fluid_mask, c_mac_xm2, 0.0)) / outlet_count
        geq_out_sol = self.outlet_bc.equilibrium(jnp.full((len(self.custom_outlet_idx), 1), c_avg_xm2), v_out_ramp)
        g_final_out = jnp.where(shutin, g_post[self.out_idx_tuple], self.precisionPolicy.cast_to_output(geq_out_sol))
        g_final = g_final.at[self.out_idx_tuple].set(g_final_out)

        h_final = self.apply_passive_bc(h_post, hout, t, "PostStreaming", self.thermal_BCs)
        t_in_injection = self.scaling.t_hot * v_ramp
        heq_in_therm = self.inlet_bc.equilibrium(jnp.full((len(self.custom_inlet_idx), 1), t_in_injection), v_in_ramp)
        h_final_in = jnp.where(shutin, h_post[self.inlet_bc.indices], self.precisionPolicy.cast_to_output(heq_in_therm))
        h_final = h_final.at[self.inlet_bc.indices].set(h_final_in)
        t_mac_xm2   = jnp.sum(self.precisionPolicy.cast_to_compute(h_post[-2]), axis=-1)
        t_avg_xm2   = jnp.sum(jnp.where(self.outlet_plane_fluid_mask, t_mac_xm2, 0.0)) / outlet_count
        heq_out     = self.outlet_bc.equilibrium(jnp.full((len(self.custom_outlet_idx), 1), t_avg_xm2), v_out_ramp)
        h_final = h_final.at[self.out_idx_tuple].set(self.precisionPolicy.cast_to_output(heq_out))

        # Housing-wall resistive cooling.
        # Real chip walls have thermal mass and resistance — they don't 
        # instantly clamp to ambient. Relax toward t_amb at rate alpha_housing.
        # Housing-wall resistive cooling.
        u_zero_housing = jnp.zeros((self.n_housing_cells, 3))
        t_amb_housing = jnp.full((self.n_housing_cells, 1), self.scaling.t_amb)
        heq_amb = self.precisionPolicy.cast_to_output(
            self.equilibrium(t_amb_housing, u_zero_housing, cast_output=False)
        )
        h_housing_current = h_final[self.housing_idx_tuple]

        # JAX-compiled dynamic thermal switch
        alpha = jnp.where(
            shutin,
            jnp.float32(self.scaling.alpha_housing_shutin), # Fast forward cooling
            jnp.float32(self.scaling.alpha_housing_flow)    # High insulation during injection
        )

        h_housing_new = (1.0 - alpha) * h_housing_current + alpha * heq_amb
        h_final = h_final.at[self.housing_idx_tuple].set(h_housing_new)
        f_final = f_final.at[self.solid_in_idx].set(f_final[self.solid_in_adj])
        g_final = g_final.at[self.solid_in_idx].set(g_final[self.solid_in_adj])
        h_final = h_final.at[self.solid_in_idx].set(h_final[self.solid_in_adj])
        f_final = f_final.at[self.solid_out_idx].set(f_final[self.solid_out_adj])
        g_final = g_final.at[self.solid_out_idx].set(g_final[self.solid_out_adj])
        h_final = h_final.at[self.solid_out_idx].set(h_final[self.solid_out_adj])

        return f_final, g_final, h_final, precipitation_amount

    def run_peak_performance(self, t_start, t_end, f_state, g_state, h_state, s_state,
                             cum_mass_in, cum_mass_out, cum_vol_in, cum_vol_out,
                             flow_steps, shutin=False):
        def cond_fun(state):
            t, _, _, _, _, _, stable, _, _, _, _ = state
            return jnp.logical_and(t < t_end, stable)

        def body_fun(state):
            t, f, g, h, s, _, _, c_in_acc, c_out_acc, v_in_acc, v_out_acc = state
            c_mac_before = jnp.sum(self.precisionPolicy.cast_to_compute(g), axis=-1)
            total_mass_before = jnp.sum(c_mac_before)
            
            f_next, g_next, h_next, precip_step = self.apply_optimized_physics(f, g, h, s, t, flow_steps=flow_steps, shutin=shutin)
            s_next = jnp.clip(s + precip_step * self.scaling.mass_to_volume_factor, 0.0, 1.0)
            
            rho_n, u_n = self.update_macroscopic(self.precisionPolicy.cast_to_compute(f_next))
            c_mac_after = jnp.sum(self.precisionPolicy.cast_to_compute(g_next), axis=-1)
            t_mac = jnp.sum(self.precisionPolicy.cast_to_compute(h_next), axis=-1)
            fluid_mask = (self.geometry_mask == 0)
            total_mass_after = jnp.sum(c_mac_after)
            
            # ACCOUNTING FIX: Add precipitated mass back into the domain delta
            true_delta_mass_domain = (total_mass_after + jnp.sum(precip_step)) - total_mass_before
            
            cx = jnp.array(self.lattice.c[0], dtype=g_next.dtype)
            flux_x_field = jnp.sum(self.precisionPolicy.cast_to_compute(g_next) * cx, axis=-1)
            
            # DIAGNOSTIC FIX: During shut-in, the physical valves are closed. Any measured 
            # flux at the boundary is just internal solute diffusing against the closed wall.
            vol_out_step = jnp.where(shutin, 0.0, jnp.sum(u_n[self.out_idx_tuple][..., 0]))
            mass_out_step = jnp.where(shutin, 0.0, jnp.sum(flux_x_field[self.out_idx_tuple]))
            
            # Calculate true inlet flux (also strictly zeroed during shut-in)
            mass_in_step = jnp.where(shutin, 0.0, true_delta_mass_domain + mass_out_step)
            vol_in_step = jnp.where(shutin, 0.0, jnp.sum(u_n[self.in_idx_tuple][..., 0]))
            c_in_next = c_in_acc + mass_in_step
            c_out_next = c_out_acc + mass_out_step
            v_in_next = v_in_acc + vol_in_step
            v_out_next = v_out_acc + vol_out_step
            valid_ma = jnp.where(fluid_mask, jnp.sqrt(jnp.sum(u_n**2, axis=-1)), 0.0)
            ma_n = jnp.max(valid_ma) / self.scaling.cs
            r_err = jnp.max(jnp.where(fluid_mask, jnp.abs(rho_n[..., 0] - 1.0), 0.0)) * 100
            t_phys = self.scaling.t_amb_phys + t_mac * (self.scaling.t_hot_phys - self.scaling.t_amb_phys)
            s_t = self.scaling.get_solubility(t_phys)
            c_phys = c_mac_after * self.scaling.c_ref_phys
            delta_c_field = c_phys - s_t
            c_max = jnp.max(c_mac_after)
            max_delta_c = jnp.max(jnp.where(fluid_mask, delta_c_field, -999.0))
            stable = jnp.logical_and(~jnp.isnan(ma_n), ma_n < 0.15)
            stable = jnp.logical_and(stable, r_err < 30.0)
            stable = jnp.logical_and(stable, c_max < 2.5)

            def print_diagnostics():
                avg_t_phys = jnp.mean(t_phys[fluid_mask])
                avg_c = jnp.mean(c_mac_after[fluid_mask])
                s_max = jnp.max(jnp.where(fluid_mask, s[..., 0], 0.0))

                # Sample density ONE cell inside the boundary,
                # because the boundary nodes themselves are forced to rho=1 by the BC.
                rho_in_plane  = rho_n[1,  ..., 0]   # x = 1
                rho_out_plane = rho_n[-2, ..., 0]   # x = nx-2

                in_count  = jnp.maximum(1.0, jnp.sum(self.inlet_plane_fluid_mask))
                out_count = jnp.maximum(1.0, jnp.sum(self.outlet_plane_fluid_mask))

                rho_in  = jnp.sum(jnp.where(self.inlet_plane_fluid_mask,  rho_in_plane,  0.0)) / in_count
                rho_out = jnp.sum(jnp.where(self.outlet_plane_fluid_mask, rho_out_plane, 0.0)) / out_count

                dp_pa = (rho_in - rho_out) * jnp.float32(1.0/3.0) * 1000.0 * (self.scaling.dx / self.scaling.dt)**2

                jax.debug.print("Step:{t} | Ma:{ma:.4f} | R_err:{re:.2f}% | C_max:{c:.3f} | C_avg:{ca:.3f} | ΔC_max:{dc:.1f} | T_f:{tf:.1f}C | dP:{dp:.2f}Pa | Mode:{m} | s_max:{s_m:.3f}",
                                t=t, ma=ma_n, re=r_err, c=c_max, ca=avg_c, dc=max_delta_c, tf=avg_t_phys, dp=dp_pa, m=jnp.where(shutin, 1, 0), s_m=s_max)
            jax.lax.cond(t % 2500 == 0, print_diagnostics, lambda: None)
            return (t + 1, f_next, g_next, h_next, s_next, ma_n, stable, c_in_next, c_out_next, v_in_next, v_out_next)

        return jax.lax.while_loop(cond_fun, body_fun, (t_start, f_state, g_state, h_state, s_state, 0.0, True,
                                                        cum_mass_in, cum_mass_out, cum_vol_in, cum_vol_out))