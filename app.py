import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # needed for 3D


# ----------------------------
# Logistic function
# ----------------------------
def logistic(z):
    return 1.0 / (1.0 + np.exp(-z))


# ----------------------------
# System-level fragility Z-equations (FSR1 = FSR2 = FSR)
# ----------------------------
def z_minor(W, hs, G, FSR):
    return (
        1.14205
        - 0.166 * hs * G
        + 0.0442 * FSR * G
        + 0.00147 * W * G
        + 1.11923 * W * FSR
    )


def z_moderate(W, hs, Bf, Ag, Dc, G, FSR):
    # Original had FSR1^2, FSR2^2, FSR1*FSR2; with FSR1 = FSR2 = FSR:
    # 35.0672*FSR^2 + 16.4784*FSR^2 - 24.5048*FSR^2 = 27.0408*FSR^2
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


# ----------------------------
# Streamlit Page
# ----------------------------
st.set_page_config(layout="wide")
st.title("System-Level Bridge Fragility Surface")

# ----------------------------
# Sidebar Inputs (case-study defaults)
# ----------------------------
st.sidebar.header("Bridge Geometry (Case Study Defaults)")

# Table 5 ranges, but defaults = case-study bridge
W = st.sidebar.number_input("Deck width W (m)", min_value=4.88, max_value=35.0, value=12.01)
hs = st.sidebar.number_input("Slab thickness hₛ (m)", min_value=0.10, max_value=0.30, value=0.225, step=0.005)
Bf = st.sidebar.number_input("Footing width Bf (m)", min_value=1.5, max_value=8.0, value=5.2, step=0.1)
hf = st.sidebar.number_input("Footing thickness hf (m)", min_value=0.30, max_value=3.0, value=1.5, step=0.05)
Dc = st.sidebar.number_input("Column diameter Dc (m)", min_value=0.5, max_value=2.5, value=1.2, step=0.05)
Ag = st.sidebar.number_input("Girder area Ag (m²)", min_value=0.15, max_value=0.70, value=0.40, step=0.01)

st.sidebar.header("Soil")
G_input = st.sidebar.number_input("Soil shear modulus G (MPa)", min_value=10.0, max_value=300.0, value=98.0, step=1.0)

st.sidebar.header("Surface Options")

damage_state = st.sidebar.selectbox(
    "Damage State",
    ["Minor", "Moderate", "Extensive", "Complete"]
)

y_var = st.sidebar.selectbox(
    "Y-axis parameter",
    ["G", "W", "Bf", "Dc", "hf", "Ag"]
)

resolution = st.sidebar.slider("Resolution (grid points per axis)", 20, 80, 40)

# ----------------------------
# Build grids
# ----------------------------
FSR_vals = np.linspace(0.0, 0.5, resolution)

if y_var == "G":
    y_vals = np.linspace(10.0, 300.0, resolution)
elif y_var == "W":
    y_vals = np.linspace(4.88, 35.0, resolution)
elif y_var == "Bf":
    y_vals = np.linspace(1.5, 8.0, resolution)
elif y_var == "Dc":
    y_vals = np.linspace(0.5, 2.5, resolution)
elif y_var == "hf":
    y_vals = np.linspace(0.30, 3.0, resolution)
elif y_var == "Ag":
    y_vals = np.linspace(0.15, 0.70, resolution)

FSR_grid, Y_grid = np.meshgrid(FSR_vals, y_vals)

# Start from case-study defaults
W_grid = np.full_like(FSR_grid, W)
hs_grid = np.full_like(FSR_grid, hs)
Bf_grid = np.full_like(FSR_grid, Bf)
hf_grid = np.full_like(FSR_grid, hf)
Dc_grid = np.full_like(FSR_grid, Dc)
Ag_grid = np.full_like(FSR_grid, Ag)
G_grid = np.full_like(FSR_grid, G_input)

# Override chosen Y-axis parameter
if y_var == "G":
    G_grid = Y_grid
elif y_var == "W":
    W_grid = Y_grid
elif y_var == "Bf":
    Bf_grid = Y_grid
elif y_var == "Dc":
    Dc_grid = Y_grid
elif y_var == "hf":
    hf_grid = Y_grid
elif y_var == "Ag":
    Ag_grid = Y_grid

# ----------------------------
# Compute fragility
# ----------------------------
if damage_state == "Minor":
    Z = z_minor(W_grid, hs_grid, G_grid, FSR_grid)
elif damage_state == "Moderate":
    Z = z_moderate(W_grid, hs_grid, Bf_grid, Ag_grid, Dc_grid, G_grid, FSR_grid)
elif damage_state == "Extensive":
    Z = z_extensive(W_grid, hs_grid, Bf_grid, Ag_grid, Dc_grid, hf_grid, G_grid, FSR_grid)
elif damage_state == "Complete":
    Z = z_complete(W_grid, Dc_grid, G_grid, FSR_grid)
else:
    Z = z_minor(W_grid, hs_grid, G_grid, FSR_grid)

P = logistic(Z)

# ----------------------------
# Plot
# ----------------------------
fig = plt.figure(figsize=(9, 6))
ax = fig.add_subplot(111, projection="3d")

surf = ax.plot_surface(
    FSR_grid,
    Y_grid,
    P,
    cmap="viridis",
    edgecolor="none",
    antialiased=True,
    alpha=0.95,
)

ax.set_xlabel("FSR (FSR₁ = FSR₂)", labelpad=10)
ax.set_ylabel(y_var, labelpad=10)
ax.set_zlabel("P(DS)", labelpad=10)
ax.set_title(f"Fragility Surface – {damage_state}", pad=12)

fig.colorbar(surf, shrink=0.6, aspect=10, label="P(DS)")

st.pyplot(fig)
