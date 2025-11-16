import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


# ============================================================
# Logistic conversion
# ============================================================
def logistic(z):
    return 1.0 / (1.0 + np.exp(-z))


# ============================================================
# System-level fragility equations (FSR1 = FSR2 = FSR)
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
        + 27.0408 * FSR**2          # Combined FSR1² + FSR2² – 2*FSR1*FSR2
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
# STREAMLIT APP
# ============================================================
st.set_page_config(layout="wide")
st.title("System-Level Fragility Surface (Shallow Foundation Bridge)")


# ============================================================
# SIDEBAR INPUTS (ALL PARAMETERS)
# ============================================================

st.sidebar.header("Superstructure Parameters")

Ls = st.sidebar.number_input("Span Length Ls (m)", 10.0, 60.0, 30.0)
W = st.sidebar.number_input("Deck Width W (m)", 4.88, 35.0, 12.01)
hs = st.sidebar.number_input("Slab Thickness hs (m)", 0.10, 0.30, 0.225, step=0.005)
Ag = st.sidebar.number_input("Girder Cross-Sectional Area Ag (m²)", 0.15, 0.70, 0.40)

st.sidebar.header("Substructure Parameters")

ncol = st.sidebar.number_input("Number of Columns per Bent ncol", 2, 6, 2)
Hc = st.sidebar.number_input("Column Height Hc (m)", 3.0, 14.0, 7.0)
Dc = st.sidebar.number_input("Column Diameter Dc (m)", 0.5, 2.5, 1.2)

st.sidebar.header("Foundation Parameters")

Bf = st.sidebar.number_input("Footing Width Bf (m)", 1.5, 8.0, 5.2)
hf = st.sidebar.number_input("Footing Thickness hf (m)", 0.30, 3.00, 1.5)

st.sidebar.header("Soil Parameter")

G_input = st.sidebar.number_input("Soil Shear Modulus G (MPa)", 10.0, 300.0, 98.0)

st.sidebar.header("Traffic Load Parameters")

Wt = st.sidebar.number_input("Truck Weight Wt (kN)", 0.0, 800.0, 300.0)
Tpx = st.sidebar.number_input("Truck Position Tpx (0–0.75)", 0.00, 0.75, 0.30)

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
        "Girder Area Ag (m²)"
    ]
)

resolution = st.sidebar.slider("Grid Resolution", 20, 80, 40)


# ============================================================
# BUILD GRID
# ============================================================
FSR_vals = np.linspace(0.0, 0.5, resolution)

if y_choice == "Soil Shear Modulus G (MPa)":
    y_vals = np.linspace(10.0, 300.0, resolution)
elif y_choice == "Deck Width W (m)":
    y_vals = np.linspace(4.88, 35.0, resolution)
elif y_choice == "Footing Width Bf (m)":
    y_vals = np.linspace(1.5, 8.0, resolution)
elif y_choice == "Column Diameter Dc (m)":
    y_vals = np.linspace(0.5, 2.5, resolution)
elif y_choice == "Footing Thickness hf (m)":
    y_vals = np.linspace(0.30, 3.00, resolution)
elif y_choice == "Girder Area Ag (m²)":
    y_vals = np.linspace(0.15, 0.70, resolution)

FSR_grid, Y_grid = np.meshgrid(FSR_vals, y_vals)


# ============================================================
# ASSIGN GRID VALUES
# ============================================================
W_grid = np.full_like(FSR_grid, W)
hs_grid = np.full_like(FSR_grid, hs)
Bf_grid = np.full_like(FSR_grid, Bf)
hf_grid = np.full_like(FSR_grid, hf)
Dc_grid = np.full_like(FSR_grid, Dc)
Ag_grid = np.full_like(FSR_grid, Ag)
G_grid = np.full_like(FSR_grid, G_input)

# Override based on Y-axis
if y_choice == "Soil Shear Modulus G (MPa)":
    G_grid = Y_grid
elif y_choice == "Deck Width W (m)":
    W_grid = Y_grid
elif y_choice == "Footing Width Bf (m)":
    Bf_grid = Y_grid
elif y_choice == "Column Diameter Dc (m)":
    Dc_grid = Y_grid
elif y_choice == "Footing Thickness hf (m)":
    hf_grid = Y_grid
elif y_choice == "Girder Area Ag (m²)":
    Ag_grid = Y_grid


# ============================================================
# COMPUTE FRAGILITY
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
# PLOT
# ============================================================
fig = plt.figure(figsize=(9, 6))
ax = fig.add_subplot(111, projection="3d")

surface = ax.plot_surface(
    FSR_grid,
    Y_grid,
    P,
    cmap="viridis",
    edgecolor="none",
    antialiased=True,
    alpha=0.95
)

ax.set_xlabel("Foundation Scour Ratio FSR₁ = FSR₂", labelpad=20)
ax.set_ylabel(y_choice, labelpad=20)
ax.set_zlabel("Probability of Exceedance", labelpad=20)
ax.set_title(f"Fragility Surface – {damage_state}", pad=20)

fig.colorbar(surface, shrink=0.6, aspect=12, label="Probability of Exceedance")

st.pyplot(fig)
