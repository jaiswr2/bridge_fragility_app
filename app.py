import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


# ============================================================
# Logistic function
# ============================================================
def logistic(z):
    return 1.0 / (1.0 + np.exp(-z))


# ============================================================
# Fragility equations (FSR1 = FSR2 = FSR)
# ============================================================
def z_minor(W, hs, G, FSR):
    return (
        1.14205
        - 0.166 * hs * G
        + 0.0442 * FSR * G
        + 0.00147 * W * G
        + 1.11923 * W * FSR
    )


def z_moderate(W, hs, Bf, Ag, Dc, G, FSR):
    return (
        -2.4355
        + 0.52 * W * FSR
        - 0.0573 * hs * G
        + 0.0562 * Bf**2
        + 0.362 * W * FSR
        - 1.2595 * Bf * FSR
        - 0.949 * Ag * Dc
        + 27.0408 * FSR**2
    )


def z_extensive(W, hs, Bf, Ag, Dc, hf, G, FSR):
    return (
        -3.52141
        + 38.31346 * FSR**2
        - 0.0432 * G
        + 0.000186 * G**2
        + 0.438 * W * FSR
        - 1.72978 * Ag * Dc
        - 1.60891 * hf * FSR
        - 0.000643 * W * G
        + 0.822 * hs * Bf
        - 0.0468 * FSR * G
    )


def z_complete(W, Dc, G, FSR):
    return (
        -5.857
        + 82.15 * FSR**2
        - 0.0102 * W * G
        - 13.996 * Dc * FSR
        + 0.525 * W * FSR
    )


# ============================================================
# AXIS LABEL SPLITTER (auto two-row)
# ============================================================
def split_label(text):
    """Automatically split long axis label into two lines."""
    if "(" in text:
        name, unit = text.split("(", 1)
        unit = "(" + unit
        return f"{name.strip()}\n{unit.strip()}"
    return text


# ============================================================
# STREAMLIT APP
# ============================================================
st.set_page_config(layout="wide")
st.title("Fragility Surface of Shallow Foundation Bridges")


# ============================================================
# SIDEBAR INPUTS — COMPLETE TABLE 5
# ============================================================

# ---- Superstructure ----
st.sidebar.header("Superstructure Parameters")
Ls = st.sidebar.number_input("Span Length Ls (m)", 10.0, 60.0, 30.0)
W = st.sidebar.number_input("Deck Width W (m)", 4.88, 35.0, 12.01)
hs = st.sidebar.number_input("Slab Thickness hs (m)", 0.10, 0.30, 0.225, step=0.005)
Ag = st.sidebar.number_input("Girder Area Ag (m²)", 0.15, 0.70, 0.40)

# ---- Substructure ----
st.sidebar.header("Substructure Parameters")
ncol = st.sidebar.number_input("Number of Columns per Bent ncol", 2, 6, 2)
Hc = st.sidebar.number_input("Column Height Hc (m)", 3.0, 14.0, 7.0)
Dc = st.sidebar.number_input("Column Diameter Dc (m)", 0.5, 2.5, 1.2)

# ---- Foundation ----
st.sidebar.header("Foundation Parameters")
Bf = st.sidebar.number_input("Footing Width Bf (m)", 1.5, 8.0, 5.2)
hf = st.sidebar.number_input("Footing Thickness hf (m)", 0.30, 3.00, 1.5)

# ---- Soil ----
st.sidebar.header("Soil")
G_input = st.sidebar.number_input("Soil Shear Modulus G (MPa)", 10.0, 300.0, 98.0)

# ---- Traffic ----
st.sidebar.header("Traffic Load (Included for completeness)")
Wt = st.sidebar.number_input("Truck Weight Wt (kN)", 0.0, 800.0, 300.0)
Tpx = st.sidebar.number_input("Truck Position Tpx (0–0.75)", 0.00, 0.75, 0.30)

# ---- Settings ----
st.sidebar.header("Fragility Surface Settings")

damage_state = st.sidebar.selectbox(
    "Damage State",
    ["Minor", "Moderate", "Extensive", "Complete"]
)

y_choice = st.sidebar.selectbox(
    "Select Y-axis Parameter",
    [
        "Soil Shear Modulus G (MPa)",
        "Deck Width W (m)",
        "Footing Width Bf (m)",
        "Column Diameter Dc (m)",
        "Footing Thickness hf (m)",
        "Girder Area Ag (m²)",
        "Truck Weight Wt (kN)",
        "Truck Position Tpx"
    ]
)

camera = st.sidebar.radio(
    "Camera View",
    ["Left View (Default)", "Front View", "Top-Down View"]
)

resolution = st.sidebar.slider("Grid Resolution", 20, 80, 40)


# ============================================================
# GRID SETUP
# ============================================================
FSR_vals = np.linspace(0, 0.5, resolution)

ranges = {
    "Soil Shear Modulus G (MPa)": (10, 300),
    "Deck Width W (m)": (4.88, 35),
    "Footing Width Bf (m)": (1.5, 8),
    "Column Diameter Dc (m)": (0.5, 2.5),
    "Footing Thickness hf (m)": (0.30, 3.0),
    "Girder Area Ag (m²)": (0.15, 0.70),
    "Truck Weight Wt (kN)": (0, 800),
    "Truck Position Tpx": (0, 0.75),
}

low, high = ranges[y_choice]
y_vals = np.linspace(low, high, resolution)

FSR_grid, Y_grid = np.meshgrid(FSR_vals, y_vals)


# ============================================================
# ASSIGN BASE PARAMETER GRIDS
# ============================================================
W_grid = np.full_like(FSR_grid, W)
hs_grid = np.full_like(FSR_grid, hs)
Bf_grid = np.full_like(FSR_grid, Bf)
hf_grid = np.full_like(FSR_grid, hf)
Dc_grid = np.full_like(FSR_grid, Dc)
Ag_grid = np.full_like(FSR_grid, Ag)
G_grid = np.full_like(FSR_grid, G_input)
Wt_grid = np.full_like(FSR_grid, Wt)
Tpx_grid = np.full_like(FSR_grid, Tpx)

# Override based on Y-selection
if y_choice == "Soil Shear Modulus G (MPa)": G_grid = Y_grid
if y_choice == "Deck Width W (m)": W_grid = Y_grid
if y_choice == "Footing Width Bf (m)": Bf_grid = Y_grid
if y_choice == "Column Diameter Dc (m)": Dc_grid = Y_grid
if y_choice == "Footing Thickness hf (m)": hf_grid = Y_grid
if y_choice == "Girder Area Ag (m²)": Ag_grid = Y_grid
if y_choice == "Truck Weight Wt (kN)": Wt_grid = Y_grid
if y_choice == "Truck Position Tpx": Tpx_grid = Y_grid


# ============================================================
# COMPUTE PROBABILITY OF EXCEEDANCE
# ============================================================
if damage_state == "Minor":
    Z = z_minor(W_grid, hs_grid, G_grid, FSR_grid)

elif damage_state == "Moderate":
    Z = z_moderate(W_grid, hs_grid, Bf_grid, Ag_grid, Dc_grid, G_grid, FSR_grid)

elif damage_state == "Extensive":
    Z = z_extensive(W_grid, hs_grid, Bf_grid, Ag_grid, Dc_grid, hf_grid, G_grid, FSR_grid)

elif damage_state == "Complete":
    Z = z_complete(W_grid, Dc_grid, G_grid, FSR_grid)

P = logistic(Z)


# ============================================================
# CAMERA SELECTION
# ============================================================
if camera == "Left View (Default)":
    elev, azim = 25, 270
elif camera == "Front View":
    elev, azim = 25, 180
elif camera == "Top-Down View":
    elev, azim = 90, 270


# ============================================================
# PLOT
# ============================================================
fig = plt.figure(figsize=(9, 6))
ax = fig.add_subplot(111, projection="3d")

ax.view_init(elev=elev, azim=azim)

surface = ax.plot_surface(
    FSR_grid,
    Y_grid,
    P,
    cmap="viridis",
    edgecolor="none",
    shade=True,
    alpha=0.95
)

# MULTI-LINE LABELS
ax.set_xlabel(split_label("Foundation Scour Ratio (FSR₁ = FSR₂)"), labelpad=25)
ax.set_ylabel(split_label(y_choice), labelpad=25)
ax.set_zlabel(split_label("Probability of Exceedance"), labelpad=25)

# TITLES
ax.set_title("Fragility Surface of Shallow Foundation Bridges", pad=20)
plt.suptitle(f"{damage_state} Damage State", y=0.97, fontsize=12)

# REMOVE COLORBAR
# fig.colorbar(surface, shrink=0.6, aspect=12)

st.pyplot(fig)
