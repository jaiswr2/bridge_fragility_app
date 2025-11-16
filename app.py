import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

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
    return (
        -2.4355
        + 0.52 * W * FSR
        - 0.0573 * hs * G
        + 0.0562 * Bf**2
        + 0.362 * W * FSR
        - 1.2595 * Bf * FSR
        - 0.949 * Ag * Dc
        + 35.0672 * FSR**2
        + 16.4784 * FSR**2
        - 24.5048 * FSR * FSR
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
        + 82.15 * FSR * FSR
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
# Sidebar Inputs
# ----------------------------
st.sidebar.header("Bridge Geometry")

W = st.sidebar.slider("Deck width W (m)", 4.88, 35.0, 12.0)
hs = st.sidebar.slider("Slab thickness hₛ (m)", 0.10, 0.30, 0.20)
Bf = st.sidebar.slider("Footing width Bf (m)", 1.5, 8.0, 4.0)
hf = st.sidebar.slider("Footing thickness hf (m)", 0.3, 3.0, 1.0)
Dc = st.sidebar.slider("Column diameter Dc (m)", 0.5, 2.5, 1.2)
Ag = st.sidebar.slider("Girder area Ag (m²)", 0.15, 0.70, 0.40)

st.sidebar.header("Soil")
G_input = st.sidebar.slider("Soil shear modulus G (MPa)", 10, 300, 80)

st.sidebar.header("Traffic Load")
Wt = st.sidebar.slider("Truck weight Wt (kN)", 0, 800, 300)
Tpx = st.sidebar.slider("Truck position Tpx (0–0.75)", 0.00, 0.75, 0.30)

st.sidebar.header("Surface Options")

damage_state = st.sidebar.selectbox(
    "Damage State",
    ["Minor", "Moderate", "Extensive", "Complete"]
)

y_var = st.sidebar.selectbox(
    "Y-axis parameter",
    ["G", "W", "Bf", "Dc", "hf", "Ag"]
)

resolution = st.sidebar.slider("Resolution", 20, 80, 40)

# ----------------------------
# Build grids
# ----------------------------
FSR_vals = np.linspace(0.0, 0.5, resolution)

if y_var == "G":
    y_vals = np.linspace(10, 300, resolution)
elif y_var == "W":
    y_vals = np.linspace(4.88, 35.0, resolution)
elif y_var == "Bf":
    y_vals = np.linspace(1.5, 8.0, resolution)
elif y_var == "Dc":
    y_vals = np.linspace(0.5, 2.5, resolution)
elif y_var == "hf":
    y_vals = np.linspace(0.3, 3.0, resolution)
elif y_var == "Ag":
    y_vals = np.linspace(0.15, 0.7, resolution)

FSR_grid, Y_grid = np.meshgrid(FSR_vals, y_vals)

# Assign variables
W_grid = np.full_like(FSR_grid, W)
hs_grid = np.full_like(FSR_grid, hs)
Bf_grid = np.full_like(FSR_grid, Bf)
hf_grid = np.full_like(FSR_grid, hf)
Dc_grid = np.full_like(FSR_grid, Dc)
Ag_grid = np.full_like(FSR_grid, Ag)
G_grid = np.full_like(FSR_grid, G_input)

# Y-axis overrides
if y_var == "G": G_grid = Y_grid
if y_var == "W": W_grid = Y_grid
if y_var == "Bf": Bf_grid = Y_grid
if y_var == "Dc": Dc_grid = Y_grid
if y_var == "hf": hf_grid = Y_grid
if y_var == "Ag": Ag_grid = Y_grid

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
    alpha=0.95
)

ax.set_xlabel("FSR (FSR₁ = FSR₂)", labelpad=10)
ax.set_ylabel(y_var, labelpad=10)
ax.set_zlabel("P(DS)", labelpad=10)
ax.set_title(f"Fragility Surface – {damage_state}")

fig.colorbar(surf, shrink=0.6, aspect=10)

st.pyplot(fig)
