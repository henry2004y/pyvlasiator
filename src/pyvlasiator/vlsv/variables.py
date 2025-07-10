# Predefined physical constants

from math import pi

QE = -1.60217662e-19  # electron charge, [C]
ME = 9.10938356e-31  # electron mass, [kg]
QI = 1.60217662e-19  # proton mass, [C]
MI = 1.673557546e-27  # proton mass, [kg]
C = 299792458.0  # speed of light, [m/s]
MU0 = 4 * pi * 1e-7  # Vacuum permeability, [H/m]
EPSILON0 = 1 / (C**2 * MU0)  # Vacuum permittivity, [F/m]
KB = 1.38064852e-23  # Boltzmann constant, [m²kg/(s²K)]
RE = 6.371e6  # Earth radius, [m]
RMERCURY = 2.4397e6 # Mercury radius, [m]

speciesdict = {
    "avgs": "p",
    "proton": "p",
    "helium": "He",
    "oxygen": "O",
    "electron": "e",
}
speciesamu = {
    "avgs": 1,
    "proton": 1,
    "helium": 4,
    "oxygen": 16,
    "electron": 5.4461702e-4,
}
speciescharge = {
    "avgs": 1,
    "proton": 1,
    "helium": 2,
    "oxygen": 1,
    "electron": -1,
}

# Define units, LaTeX markup names, and LaTeX markup units for intrinsic values
units_predefined = {
    "Rhom": ("kg/m3", r"$\rho_m$", r"$\mathrm{kg}\,\mathrm{m}^{-3}$"),
    "rhoq": ("C/m3", r"$\rho_q$", r"$\mathrm{C}\,\mathrm{m}^{-3}$"),
    "rho": ("1/m3", r"$n_\mathrm{p}$", r"$\mathrm{m}^{-3}$"),
    "rho_v": ("1/m2s", r"$\Gamma_\mathrm{p}$", r"$\mathrm{m}^{-2}$s"),
    "v": ("m/s", r"$V$", r"$\mathrm{m}\,\mathrm{s}^{-1}$"),
    "b": ("T", r"$B$", "T"),
    "b_vor": ("T", r"$B_\mathrm{vol}$", "T"),
    "background_b": ("T", r"$B_\mathrm{bg}$", "T"),
    "perturbed_b": ("T", r"$B_\mathrm{pert}$", "T"),
    "bgb": ("T", r"$B_\mathrm{bg}$", "T"),
    "perb": ("T", r"B_\mathrm{pert}$", "T"),
    "perb_vor": ("T", r"B_\mathrm{vol,pert}$", "T"),
    "e": ("V/m", r"$E$", r"$\mathrm{V}\,\mathrm{m}^{-1}$"),
    "e_vor": ("V/m", r"$E_\mathrm{vol}$", r"$\mathrm{V}\,\mathrm{m}^{-1}$"),
    "exhall_000_100": (
        "V/m",
        r"$E_\mathrm{Hall,000,100}$",
        r"$\mathrm{V}\,\mathrm{m}^{-1}$",
    ),
    "exhall_001_101": (
        "V/m",
        r"$E_\mathrm{Hall,001,101}$",
        r"$\mathrm{V}\,\mathrm{m}^{-1}$",
    ),
    "exhall_010_110": (
        "V/m",
        r"$E_\mathrm{Hall,010,110}$",
        r"$\mathrm{V}\,\mathrm{m}^{-1}$",
    ),
    "exhall_011_111": (
        "V/m",
        r"$E_\mathrm{Hall,011,111}$",
        r"$\mathrm{V}\,\mathrm{m}^{-1}$",
    ),
    "eyhall_000_010": (
        "V/m",
        r"$E_\mathrm{Hall,000,010}$",
        r"$\mathrm{V}\,\mathrm{m}^{-1}$",
    ),
    "eyhall_001_011": (
        "V/m",
        r"$E_\mathrm{Hall,001,011}$",
        r"$\mathrm{V}\,\mathrm{m}^{-1}$",
    ),
    "eyhall_100_110": (
        "V/m",
        r"$E_\mathrm{Hall,100,110}$",
        r"$\mathrm{V}\,\mathrm{m}^{-1}$",
    ),
    "eyhall_101_111": (
        "V/m",
        r"$E_\mathrm{Hall,101,111}$",
        r"$\mathrm{V}\,\mathrm{m}^{-1}$",
    ),
    "ezhall_000_001": (
        "V/m",
        r"$E_\mathrm{Hall,000,001}$",
        r"$\mathrm{V}\,\mathrm{m}^{-1}$",
    ),
    "ezhall_010_011": (
        "V/m",
        r"$E_\mathrm{Hall,010,011}$",
        r"$\mathrm{V}\,\mathrm{m}^{-1}$",
    ),
    "ezhall_100_101": (
        "V/m",
        r"$E_\mathrm{Hall,100,101}$",
        r"$\mathrm{V}\,\mathrm{m}^{-1}$",
    ),
    "ezhall_110_111": (
        "V/m",
        r"$E_\mathrm{Hall,110,111}$",
        r"$\mathrm{V}\,\mathrm{m}^{-1}$",
    ),
    "pressure": ("Pa", r"$P$", "Pa"),
    "pressure_dt2": ("Pa", r"$P_{\mathrm{d}t/2}}$", "Pa"),
    "pressure_r": ("Pa", r"$P_r$", "Pa"),
    "pressure_v": ("Pa", r"$P_v$", "Pa"),
    "ptensordiagonar": ("Pa", r"$\mathcal{P}_\mathrm{diag}$", "Pa"),
    "ptensoroffdiagonar": ("Pa", r"$\mathcal{P}_\mathrm{off-diag}$", "Pa"),
    "minvalue": ("s3/m6", r"$f_\mathrm{Min}$", r"$\mathrm{m}^{-6}\,\mathrm{s}^{3}$"),
    "effectivesparsitythreshold": (
        "s3/m6",
        r"$f_\mathrm{Min}$",
        r"$\mathrm{m}^{-6}\,\mathrm{s}^{3}$",
    ),
    "rho_loss_adjust": (
        "1/m3",
        r"$\Delta_\mathrm{loss} n_\mathrm{p}$",
        r"$\mathrm{m}^{-3}$",
    ),
    "energydensity": (
        "eV/cm3",
        r"$\rho_{\mathrm{energy}}$",
        r"$\mathrm{eV}\,\mathrm{cm}^{-3}$",
    ),
    "precipitationdiffflux": (
        "1/(cm2 sr s eV)",
        r"$\Delta F_\mathrm{precipitation}$",
        r"$\mathrm{cm}^{-2} \,\mathrm{sr}^{-1}\,\mathrm{s}^{-1}\,\mathrm{eV}^{-1}$",
    ),
    "T": ("K", r"$T$", "K"),
    "Tpar": ("K", r"$T$", "K"),
    "Tperp": ("K", r"$T$", "K"),
    "Panisotropy": ("", r"$P_\perp / P_\parallel$", ""),
    "Tanisotropy": ("", r"$T_\perp / T_\parallel$", ""),
    "VS": ("m/s", r"$V_S$", r"$\mathrm{m}\,\mathrm{s}^{-1}$"),
    "VA": ("m/s", r"$V_A$", r"$\mathrm{m}\,\mathrm{s}^{-1}$"),
    "MS": ("", r"$M_S$", ""),
    "MA": ("", r"$M_A$", ""),
    "Ppar": ("Pa", r"$P_\parallel$", "Pa"),
    "Pperp": ("Pa", r"$P_\perp$", "Pa"),
    "Beta": ("", r"$\beta$", ""),
    "BetaStar": ("", r"$\beta^\ast$", ""),
    "Gyroperiod": ("s", r"$T_{gyro}$", "s"),
    "PlasmaPeriod": ("s", r"$T_{plasma}$", "s"),
    "Gyrofrequency": ("rad/s", r"$\omega_{g}$", r"\mathrm{rad}/\mathrm{s}"),
    "Omegap": ("rad/s", r"$\omega_{p}$", r"\mathrm{rad}/\mathrm{s}"),
}


def pass_op(variable):
    # do nothing
    return variable


def magnitude(variable):
    return np.linalg.norm(np.asarray(variable), axis=-1)


def sumv(variable):
    # Note: this is used to sum over multipops, thus the summing axis is zero
    if np.ndim(variable) > 3:
        print("Error: Number of dimensions is too large")
        return
    else:
        # First dimension: populations
        # Second dimension: cells
        # Third dimension: components
        return np.sum(np.array(variable), axis=0)


# Dict of operators. The user can apply these to any variable,
# including more general datareducers. Can only be used to reduce one
# variable at a time
data_operators = {}
data_operators["pass"] = pass_op
data_operators["magnitude"] = magnitude
data_operators["sum"] = sumv
