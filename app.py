import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  (needed for 3D plots)

# ----------------------------------------------------
# Logistic helper
# ----------------------------------------------------
def logistic(z):
    return 1.0 / (1.0 + np.exp(-z))


# ----------------------------------------------------
# System-level fragility Z-equations (updated final)
# FSR1 = FSR2 = FSR in this GUI
# ----------------------------------------------------
def z_minor(W, hs, G, FSR):
    FSR1 = FSR2 = FSR
    return (
        1.14205
        - 0.166 * hs * G
        + 0.0442 * FSR1 * G
        + 0.00147 * W * G
        + 1.11923 * W * FSR2
    )


def z_moderate(W, hs, Bf, Ag, Dc, G, FSR):
    FSR1 = FSR2 = FSR
    return (
        -2.4355
        + 0.52 * W * FSR2
        - 0.0573 * hs * G
        + 0.0562 * (Bf ** 2)
        + 0.362 * W * FSR1
        - 1.2595 * Bf * FSR2
        - 0.949 * Ag * Dc
        + 35.0672 * (FSR2 ** 2)
        + 16.4784 * (FSR1 ** 2)
        - 24.5048 * FSR1 * FSR2
    )


def z_extensive(W, hs, Bf, Ag, Dc, hf, G, FSR):
    FSR1 = FSR2 = FSR
    return (
        -3.52141
        + 38.31346 * (FSR2 ** 2)
        - 0.0432 * G
        + 0.000186 * (G ** 2)
        + 0.438 * W * FSR2
        - 1.72978 * Ag * Dc
        - 1.60891 * hf * FSR1
        - 0.000643 * W * G
        + 0.822 * hs * Bf
        - 0.0468 * FSR2 * G
    )


def z_complete(W, Dc, G, FSR):
    FSR1 = FSR2 = FSR
    return (
        -5.857
        + 82.15 * FSR1 * FSR2
        - 0.0102 * W * G
        - 13.996 * Dc * FSR1
        + 0.525 * W * FSR2
    )


# ----------------------------------------------------
# Streamlit app
# ----------------------------------------------------
st.set_page_config(
    page_title="Bridge System Fragility Surface",
    layout="wide"
)

st.title("System-Level Fragility Surfaces for Shallow-Foundation Bridges")
st.markdown(
    """
This demo uses **closed-form system-level fragility functions** for a shallow-foundation bridge.
The x-axis is always the **Foundation Scour Ratio (FSR)**, with FSR₁ = FSR₂ = FSR.
You can choose a second parameter for the y-axis and a **damage state** to visualize the
probability of exceedance as a 3D surface.
"""
)

# ---------------- Sidebar: inputs --------------------
st.sidebar.header("Bridge Geometry & Material Inputs")

# Ranges from your Table 5 (Lf and n_gd excluded)
# Span length Ls is omitted from equations, so not needed here.

W = st.sidebar.slider("Deck width W [m]", 4.88, 35.0, 15.0)
hs = st.sidebar.slider("Slab thickness h_s [m]", 0.10, 0.30, 0.20)
Ag = st.sidebar.slider("Girder C/S area A_g [m²] (traffic parameter)", 0.15, 0.70, 0.40)
ncol = st.sidebar.slider("Number of columns per bent n_col [-]", 2, 6, 4)   # not used, for completeness
Hc = st.sidebar.slider("Column height H_c [m]", 3.0, 14.0, 7.0)           # not used in current equations
Dc = st.sidebar.slider("Column diameter D_c [m]", 0.5, 2.5, 1.5)
Bf = st.sidebar.slider("Footing width B_f [m]", 1.5, 8.0, 4.0)
hf = st.sidebar.slider("Footing thickness h_f [m]", 0.30, 3.0, 1.0)

st.sidebar.subheader("Soil & Vehicle Parameters")
G_input = st.sidebar.slider("Soil shear modulus G [MPa]", 10.0, 300.0, 80.0)
Wt = st.sidebar.slider("Truck weight W_t [kN] (not used in current equations)", 0.0, 800.0, 400.0)
Tpx = st.sidebar.slider("Truck position T_px [-] (not used in current equations)", 0.0, 0.75, 0.35)

st.sidebar.markdown(
    """
**Note:**  
Only parameters that appear in the fragility equations (W, h_s, B_f, h_f, D_c,
A_g, G, FSR) affect the probability numerically.  
Other inputs (n_col, H_c, W_t, T_px) are provided for completeness and
possible future extensions.
"""
)

st.sidebar.header("Surface Settings")

damage_state = st.sidebar.selectbox(
    "Select damage state",
    ["Minor", "Moderate", "Extensive", "Complete"]
)

y_var = st.sidebar.selectbox(
    "Select Y-axis parameter",
    [
        "Traffic parameter A_g",
        "Soil stiffness G",
        "Deck width W",
        "Footing width B_f",
        "Column diameter D_c",
        "Footing thickness h_f"
    ]
)

resolution = st.sidebar.slider(
    "Grid resolution (points per axis)",
    20, 80, 40
)

# ---------------- Build FSR and Y grids ----------------
FSR_vals = np.linspace(0.0, 0.5, resolution)

if y_var == "Traffic parameter A_g":
    y_vals = np.linspace(0.15, 0.70, resolution)
elif y_var == "Soil stiffness G":
    y_vals = np.linspace(10.0, 300.0, resolution)
elif y_var == "Deck width W":
    y_vals = np.linspace(4.88, 35.0, resolution)
elif y_var == "Footing width B_f":
    y_vals = np.linspace(1.5, 8.0, resolution)
elif y_var == "Column diameter D_c":
    y_vals = np.linspace(0.5, 2.5, resolution)
elif y_var == "Footing thickness h_f":
    y_vals = np.linspace(0.30, 3.0, resolution)
else:
    y_vals = np.linspace(0.0, 1.0, resolution)  # fallback

FSR_grid, Y_grid = np.meshgrid(FSR_vals, y_vals)

# ---------------- Map grids to actual parameters ----------------
# Start from fixed values
W_grid = np.full_like(FSR_grid, W)
hs_grid = np.full_like(FSR_grid, hs)
Bf_grid = np.full_like(FSR_grid, Bf)
hf_grid = np.full_like(FSR_grid, hf)
Dc_grid = np.full_like(FSR_grid, Dc)
Ag_grid = np.full_like(FSR_grid, Ag)
G_grid = np.full_like(FSR_grid, G_input)

if y_var == "Traffic parameter A_g":
    Ag_grid = Y_grid
elif y_var == "Soil stiffness G":
    G_grid = Y_grid
elif y_var == "Deck width W":
    W_grid = Y_grid
elif y_var == "Footing width B_f":
    Bf_grid = Y_grid
elif y_var == "Column diameter D_c":
    Dc_grid = Y_grid
elif y_var == "Footing thickness h_f":
    hf_grid = Y_grid

# ---------------- Compute fragility surface ----------------
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

# ---------------- Plot in main area ----------------
col1, col2 = st.columns([3, 1])

with col1:
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")

    ax.plot_surface(
        FSR_grid,
        Y_grid,
        P,
        rstride=1,
        cstride=1,
        linewidth=0,
        antialiased=True,
        alpha=0.9
    )

    ax.set_xlabel("FSR (FSR₁ = FSR₂)", labelpad=10)
    ax.set_ylabel(y_var, labelpad=10)
    ax.set_zlabel(f"P(exceedance | {damage_state})", labelpad=10)
    ax.set_title(f"System Fragility Surface – {damage_state} Damage")

    st.pyplot(fig)

with col2:
    st.subheader("Current Settings")
    st.markdown(f"**Damage state:** {damage_state}")
    st.markdown(f"**Y-axis variable:** {y_var}")
    st.markdown("**Fixed inputs:**")
    st.markdown(f"- W = {W:.2f} m")
    st.markdown(f"- h_s = {hs:.3f} m")
    st.markdown(f"- B_f = {Bf:.2f} m")
    st.markdown(f"- h_f = {hf:.2f} m")
    st.markdown(f"- D_c = {Dc:.2f} m")
    st.markdown(f"- A_g (if not varying) = {Ag:.3f} m²")
    st.markdown(f"- G (if not varying) = {G_input:.1f} MPa")
    st.info(
        "FSR varies from 0.0 to 0.5 along the x-axis. "
        "FSR₁ and FSR₂ are assumed equal in this visualization."
    )

st.markdown(
    """
---
**How to interpret:**  
Each point on the surface gives the **probability of exceeding the selected damage state** for a
combination of FSR and the chosen y-axis parameter, with all other bridge and soil parameters
held at the values selected in the sidebar.
"""
)
