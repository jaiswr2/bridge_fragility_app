import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401


# ============================================================
# Logistic function
# ============================================================
def logistic(z):
    return 1.0 / (1.0 + np.exp(-z))


# ============================================================
# Fragility equations (Bridge-level, explicit FSR1 & FSR2)
# From your latest table screenshot
# ============================================================
def z_minor(W, hs, G, FSR1, FSR2):
    # 1.14205 – 0.166·hs·G + 0.0442·FSR1·G + 0.00147·W·G + 1.11923·W·FSR2
    return (
        1.14205
        - 0.166 * hs * G
        + 0.0442 * FSR1 * G
        + 0.00147 * W * G
        + 1.11923 * W * FSR2
    )


def z_moderate(W, hs, Bf, Ag, Dc, G, FSR1, FSR2):
    # −2.4355 + 0.52·W·FSR2 − 0.0573·hs·G + 0.0562·Bf² + 0.362·W·FSR1
    # + 35.0672·FSR2² − 1.2595·Bf·FSR2 − 0.949·Ag·Dc + 16.4784·FSR1² − 24.5048·FSR1·FSR2
    return (
        -2.4355
        + 0.52 * W * FSR2
        - 0.0573 * hs * G
        + 0.0562 * (Bf ** 2)
        + 0.362 * W * FSR1
        + 35.0672 * (FSR2 ** 2)
        - 1.2595 * Bf * FSR2
        - 0.949 * Ag * Dc
        + 16.4784 * (FSR1 ** 2)
        - 24.5048 * FSR1 * FSR2
    )


def z_extensive(W, hs, Bf, Ag, Dc, hf, G, FSR1, FSR2):
    # −3.52141 + 38.31346·FSR2² − 0.0432·G + 0.000186·G²
    # + 0.438·W·FSR2 − 1.72978·Ag·Dc − 1.60891·hf·FSR1
    # − 0.000643·W·G + 0.822·hs·Bf − 0.0468·FSR2·G
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


def z_complete(W, Dc, G, FSR1, FSR2):
    # −5.857 + 82.15·FSR1·FSR2 − 0.0102·W·G − 13.996·Dc·FSR1 + 0.525·W·FSR2
    return (
        -5.857
        + 82.15 * FSR1 * FSR2
        - 0.0102 * W * G
        - 13.996 * Dc * FSR1
        + 0.525 * W * FSR2
    )


# ============================================================
# Axis mapping helpers
# ============================================================
AXIS_LABEL_TO_CODE = {
    "FSR (FSR1 = FSR2)": "FSR",
    "FSR1": "FSR1",
    "FSR2": "FSR2",
    "Span Length Ls (m)": "Ls",
    "Deck Width W (m)": "W",
    "Slab Thickness hs (m)": "hs",
    "Girder Area Ag (m²)": "Ag",
    "Number of Columns ncol": "ncol",
    "Column Height Hc (m)": "Hc",
    "Column Diameter Dc (m)": "Dc",
    "Footing Width Bf (m)": "Bf",
    "Footing Thickness hf (m)": "hf",
    "Soil Shear Modulus G (MPa)": "G",
    "Truck Weight Wt (kN)": "Wt",
    "Truck Position Tpx": "Tpx",
}

# ranges for possible axis variables
AXIS_RANGES = {
    "FSR": (0.0, 0.5),
    "FSR1": (0.0, 0.5),
    "FSR2": (0.0, 0.5),
    "Ls": (10.0, 60.0),
    "W": (4.88, 35.0),
    "hs": (0.10, 0.30),
    "Ag": (0.15, 0.70),
    "ncol": (2, 6),
    "Hc": (3.0, 14.0),
    "Dc": (0.5, 2.5),
    "Bf": (1.5, 8.0),
    "hf": (0.30, 3.0),
    "G": (10.0, 300.0),
    "Wt": (0.0, 800.0),
    "Tpx": (0.0, 0.75),
}


def var_used_on_axis(var_code, x_code, y_code):
    """Return True if this variable is controlled by X or Y axis."""
    # Direct match
    if var_code == x_code or var_code == y_code:
        return True
    # Special case: FSR axis controls BOTH FSR1 and FSR2
    if x_code == "FSR" and var_code in ["FSR1", "FSR2"]:
        return True
    if y_code == "FSR" and var_code in ["FSR1", "FSR2"]:
        return True
    return False


# ============================================================
# STREAMLIT APP
# ============================================================
st.set_page_config(layout="wide")
st.title("Fragility Surface of Shallow Foundation Bridges")

# ------------------------------------------------------------
# Damage model & axis selection (global controls)
# ------------------------------------------------------------
st.subheader("Damage Model and Axes")

damage_state = st.selectbox(
    "Damage State",
    ["Minor", "Moderate", "Extensive", "Complete"],
    index=0,
)

axis_labels = list(AXIS_LABEL_TO_CODE.keys())

col_x, col_y = st.columns(2)
with col_x:
    x_label = st.selectbox("X-axis parameter", axis_labels, index=0)
with col_y:
    y_label = st.selectbox("Y-axis parameter", axis_labels, index=5)

x_code = AXIS_LABEL_TO_CODE[x_label]
y_code = AXIS_LABEL_TO_CODE[y_label]

if x_label == y_label:
    st.error("X-axis and Y-axis cannot be the same parameter. Please choose different parameters.")
    st.stop()

st.markdown("---")

# ------------------------------------------------------------
# Inputs in three columns: bridge parameters + images
# ------------------------------------------------------------
col1, col2, col3 = st.columns([1, 1, 1])

# ---------- COLUMN 1: Superstructure + some substructure ----------
with col1:
    st.subheader("Superstructure")

    Ls_disabled = var_used_on_axis("Ls", x_code, y_code)
    W_disabled = var_used_on_axis("W", x_code, y_code)
    hs_disabled = var_used_on_axis("hs", x_code, y_code)
    Ag_disabled = var_used_on_axis("Ag", x_code, y_code)

    Ls = st.number_input(
        "Span Length Ls (m)",
        10.0, 60.0, 30.0,
        disabled=Ls_disabled,
    )
    W = st.number_input(
        "Deck Width W (m)",
        4.88, 35.0, 12.01,
        disabled=W_disabled,
    )
    hs = st.number_input(
        "Slab Thickness hs (m)",
        0.10, 0.30, 0.225,
        step=0.005,
        disabled=hs_disabled,
    )
    Ag = st.number_input(
        "Girder Area Ag (m²)",
        0.15, 0.70, 0.40,
        disabled=Ag_disabled,
    )

    st.subheader("Pier Layout")

    ncol_disabled = var_used_on_axis("ncol", x_code, y_code)
    Hc_disabled = var_used_on_axis("Hc", x_code, y_code)

    ncol = st.number_input(
        "Number of Columns per Bent ncol",
        2, 6, 2,
        step=1,
        disabled=ncol_disabled,
    )
    Hc = st.number_input(
        "Column Height Hc (m)",
        3.0, 14.0, 7.0,
        disabled=Hc_disabled,
    )

# ---------- COLUMN 2: Substructure, soil, traffic, scour ----------
with col2:
    st.subheader("Substructure & Foundation")

    Dc_disabled = var_used_on_axis("Dc", x_code, y_code)
    Bf_disabled = var_used_on_axis("Bf", x_code, y_code)
    hf_disabled = var_used_on_axis("hf", x_code, y_code)

    Dc = st.number_input(
        "Column Diameter Dc (m)",
        0.5, 2.5, 1.2,
        disabled=Dc_disabled,
    )
    Bf = st.number_input(
        "Footing Width Bf (m)",
        1.5, 8.0, 5.2,
        disabled=Bf_disabled,
    )
    hf = st.number_input(
        "Footing Thickness hf (m)",
        0.30, 3.00, 1.5,
        disabled=hf_disabled,
    )

    st.subheader("Soil")

    G_disabled = var_used_on_axis("G", x_code, y_code)
    G_input = st.number_input(
        "Soil Shear Modulus G (MPa)",
        10.0, 300.0, 98.0,
        disabled=G_disabled,
    )

    st.subheader("Traffic")

    Wt_disabled = var_used_on_axis("Wt", x_code, y_code)
    Tpx_disabled = var_used_on_axis("Tpx", x_code, y_code)

    Wt = st.number_input(
        "Truck Weight Wt (kN)",
        0.0, 800.0, 300.0,
        disabled=Wt_disabled,
    )
    Tpx = st.number_input(
        "Truck Position Tpx",
        0.00, 0.75, 0.30,
        disabled=Tpx_disabled,
    )

    st.subheader("Scour (Base Values)")

    FSR1_disabled = var_used_on_axis("FSR1", x_code, y_code)
    FSR2_disabled = var_used_on_axis("FSR2", x_code, y_code)

    FSR1_input = st.number_input(
        "Foundation Scour Ratio FSR1",
        0.0, 0.5, 0.10,
        disabled=FSR1_disabled,
    )
    FSR2_input = st.number_input(
        "Foundation Scour Ratio FSR2",
        0.0, 0.5, 0.10,
        disabled=FSR2_disabled,
    )

# ---------- COLUMN 3: Images / schematic ----------
with col3:
    st.subheader("Illustrations")

    st.image("1.png", caption="Bridge Elevation (placeholder)", use_container_width=True)
    st.image("2.png", caption="Scour Configuration (placeholder)", use_container_width=True)
    st.image("3.png", caption="Response / Fragility Concept (placeholder)", use_container_width=True)

st.markdown("---")

# ============================================================
# GRID for X and Y axes
# ============================================================
resolution = 40

# X range
x_min, x_max = AXIS_RANGES[AXIS_LABEL_TO_CODE[x_label]]
y_min, y_max = AXIS_RANGES[AXIS_LABEL_TO_CODE[y_label]]

x_vals = np.linspace(x_min, x_max, resolution)
y_vals = np.linspace(y_min, y_max, resolution)

X_grid, Y_grid = np.meshgrid(x_vals, y_vals)

# ============================================================
# Base parameter grids (start from input values)
# ============================================================
Ls_grid = np.full_like(X_grid, Ls)
W_grid = np.full_like(X_grid, W)
hs_grid = np.full_like(X_grid, hs)
Ag_grid = np.full_like(X_grid, Ag)
ncol_grid = np.full_like(X_grid, ncol)
Hc_grid = np.full_like(X_grid, Hc)
Dc_grid = np.full_like(X_grid, Dc)
Bf_grid = np.full_like(X_grid, Bf)
hf_grid = np.full_like(X_grid, hf)
G_grid = np.full_like(X_grid, G_input)
Wt_grid = np.full_like(X_grid, Wt)
Tpx_grid = np.full_like(X_grid, Tpx)
FSR1_grid = np.full_like(X_grid, FSR1_input)
FSR2_grid = np.full_like(X_grid, FSR2_input)

# ============================================================
# Apply X-axis overrides
# ============================================================
def apply_axis_override(code, grid, var_code):
    if code == var_code:
        return X_grid
    return grid


# FSR special handling for X
if x_code == "FSR":
    FSR1_grid = X_grid
    FSR2_grid = X_grid
else:
    FSR1_grid = apply_axis_override(x_code, FSR1_grid, "FSR1")
    FSR2_grid = apply_axis_override(x_code, FSR2_grid, "FSR2")

W_grid = apply_axis_override(x_code, W_grid, "W")
hs_grid = apply_axis_override(x_code, hs_grid, "hs")
Ag_grid = apply_axis_override(x_code, Ag_grid, "Ag")
ncol_grid = apply_axis_override(x_code, ncol_grid, "ncol")
Hc_grid = apply_axis_override(x_code, Hc_grid, "Hc")
Dc_grid = apply_axis_override(x_code, Dc_grid, "Dc")
Bf_grid = apply_axis_override(x_code, Bf_grid, "Bf")
hf_grid = apply_axis_override(x_code, hf_grid, "hf")
G_grid = apply_axis_override(x_code, G_grid, "G")
Wt_grid = apply_axis_override(x_code, Wt_grid, "Wt")
Tpx_grid = apply_axis_override(x_code, Tpx_grid, "Tpx")
Ls_grid = apply_axis_override(x_code, Ls_grid, "Ls")

# ============================================================
# Apply Y-axis overrides
# ============================================================
def apply_axis_override_y(code, grid, var_code):
    if code == var_code:
        return Y_grid
    return grid


if y_code == "FSR":
    FSR1_grid = Y_grid
    FSR2_grid = Y_grid
else:
    FSR1_grid = apply_axis_override_y(y_code, FSR1_grid, "FSR1")
    FSR2_grid = apply_axis_override_y(y_code, FSR2_grid, "FSR2")

W_grid = apply_axis_override_y(y_code, W_grid, "W")
hs_grid = apply_axis_override_y(y_code, hs_grid, "hs")
Ag_grid = apply_axis_override_y(y_code, Ag_grid, "Ag")
ncol_grid = apply_axis_override_y(y_code, ncol_grid, "ncol")
Hc_grid = apply_axis_override_y(y_code, Hc_grid, "Hc")
Dc_grid = apply_axis_override_y(y_code, Dc_grid, "Dc")
Bf_grid = apply_axis_override_y(y_code, Bf_grid, "Bf")
hf_grid = apply_axis_override_y(y_code, hf_grid, "hf")
G_grid = apply_axis_override_y(y_code, G_grid, "G")
Wt_grid = apply_axis_override_y(y_code, Wt_grid, "Wt")
Tpx_grid = apply_axis_override_y(y_code, Tpx_grid, "Tpx")
Ls_grid = apply_axis_override_y(y_code, Ls_grid, "Ls")

# ============================================================
# COMPUTE PROBABILITY OF EXCEEDANCE
# ============================================================
if damage_state == "Minor":
    Z = z_minor(W_grid, hs_grid, G_grid, FSR1_grid, FSR2_grid)
elif damage_state == "Moderate":
    Z = z_moderate(W_grid, hs_grid, Bf_grid, Ag_grid, Dc_grid, G_grid, FSR1_grid, FSR2_grid)
elif damage_state == "Extensive":
    Z = z_extensive(W_grid, hs_grid, Bf_grid, Ag_grid, Dc_grid, hf_grid, G_grid, FSR1_grid, FSR2_grid)
elif damage_state == "Complete":
    Z = z_complete(W_grid, Dc_grid, G_grid, FSR1_grid, FSR2_grid)
else:
    Z = np.zeros_like(W_grid)

P = logistic(Z)

# ============================================================
# 3D PLOT - clean white, thin edges, centered
# ============================================================
plt.rcParams["figure.facecolor"] = "white"
plt.rcParams["axes.facecolor"] = "white"

fig = plt.figure(figsize=(8, 6), dpi=150)
ax = fig.add_subplot(111, projection="3d")

ax.set_facecolor("white")
fig.patch.set_facecolor("white")

# Camera angle
ax.view_init(elev=25, azim=235)

# Surface with thin black edges
surf = ax.plot_surface(
    X_grid,
    Y_grid,
    P,
    cmap="viridis",
    linewidth=0.5,
    edgecolor="black",
    alpha=0.9,
)

# Axis labels
ax.set_xlabel(x_label, fontsize=8, labelpad=10)
ax.set_ylabel(y_label, fontsize=8, labelpad=10)
ax.set_zlabel("Probability of Exceedance", fontsize=8, labelpad=-2)

# Tick labels
ax.tick_params(axis="both", which="major", labelsize=8)

plt.title(f"{damage_state} Damage State", fontsize=12, pad=20)
plt.tight_layout()

# Center the plot in the page
c1, c2, c3 = st.columns([1, 2, 1])
with c2:
    st.pyplot(fig)


