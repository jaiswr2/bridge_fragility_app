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
# STREAMLIT APP
# ============================================================
st.set_page_config(layout="wide")
st.title("Fragility Surface of Shallow Foundation Bridges")


# ============================================================
# INPUT PANEL — TWO BALANCED COLUMNS
# ============================================================

col1, col2 = st.columns(2)

# Left column
with col1:
    st.subheader("Superstructure")
    Ls = st.number_input("Span Length Ls (m)", 10.0, 60.0, 30.0)
    W = st.number_input("Deck Width W (m)", 4.88, 35.0, 12.01)
    hs = st.number_input("Slab Thickness hs (m)", 0.10, 0.30, 0.225, step=0.005)
    Ag = st.number_input("Girder Area Ag (m²)", 0.15, 0.70, 0.40)

    st.subheader("Soil")
    G_input = st.number_input("Soil Shear Modulus G (MPa)", 10.0, 300.0, 98.0)

    st.subheader("Scour")
    FSR1 = st.number_input("Foundation Scour Ratio FSR1", 0.0, 0.50, 0.0)
    FSR2 = st.number_input("Foundation Scour Ratio FSR2", 0.0, 0.50, 0.0)

    st.subheader("Traffic")
    Wt = st.number_input("Truck Weight Wt (kN)", 0.0, 800.0, 300.0)
    Tpx = st.number_input("Truck Position Tpx", 0.00, 0.75, 0.30)

# Right column
with col2:
    st.subheader("Substructure")
    ncol = st.number_input("Number of Columns per Bent ncol", 2, 6, 2)
    Hc = st.number_input("Column Height Hc (m)", 3.0, 14.0, 7.0)
    Dc = st.number_input("Column Diameter Dc (m)", 0.5, 2.5, 1.2)
    Bf = st.number_input("Footing Width Bf (m)", 1.5, 8.0, 5.2)
    hf = st.number_input("Footing Thickness hf (m)", 0.30, 3.00, 1.5)

    st.subheader("Damage Model")
    damage_state = st.selectbox(
        "Damage State",
        ["Minor", "Moderate", "Extensive", "Complete"]
    )
    y_choice = st.selectbox(
        "Select Y-axis Parameter",
        [
            "Soil Shear Modulus G (MPa)",
            "Deck Width W (m)",
            "Footing Width Bf (m)",
            "Column Diameter Dc (m)",
            "Footing Thickness hf (m)",
            "Girder Area Ag (m²)",
            "Truck Weight Wt (kN)",
            "Truck Position Tpx",
        ]
    )

st.markdown("---")


# ============================================================
# GRID (constant 40 resolution)
# ============================================================
resolution = 40
FSR_vals = np.linspace(0, 0.5, resolution)

# Y-axis ranges
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

# Base grids
W_grid = np.full_like(FSR_grid, W)
hs_grid = np.full_like(FSR_grid, hs)
Bf_grid = np.full_like(FSR_grid, Bf)
hf_grid = np.full_like(FSR_grid, hf)
Dc_grid = np.full_like(FSR_grid, Dc)
Ag_grid = np.full_like(FSR_grid, Ag)
G_grid = np.full_like(FSR_grid, G_input)
Wt_grid = np.full_like(FSR_grid, Wt)
Tpx_grid = np.full_like(FSR_grid, Tpx)

# Apply Y override
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
FSR = (FSR1 + FSR2) / 2  # your earlier assumption (FSR1 = FSR2)

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
# 3D PLOT (unchanged)
# ============================================================
plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['axes.facecolor'] = 'white'

fig = plt.figure(figsize=(8, 6), dpi=150)
ax = fig.add_subplot(111, projection="3d")

ax.set_facecolor("white")
fig.patch.set_facecolor("white")

ax.view_init(elev=25, azim=235)

surf = ax.plot_surface(
    FSR_grid, Y_grid, P,
    cmap="viridis",
    linewidth=0.5,
    edgecolor='black',
    alpha=0.9
)

ax.set_xlabel("Foundation Scour Ratio (FSR)", fontsize=8, labelpad=10)
ax.set_ylabel(y_choice, fontsize=8, labelpad=10)
ax.set_zlabel("Probability of Exceedance", fontsize=8, labelpad=-2)

ax.tick_params(axis='both', which='major', labelsize=8)

plt.title(f"{damage_state} Damage State", fontsize=12, pad=20)
plt.tight_layout()

col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    st.pyplot(fig)
