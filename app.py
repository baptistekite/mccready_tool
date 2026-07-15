import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

# --- PAGE SETUP ---
st.set_page_config(page_title="MacCready Polar Tool", layout="wide")
st.title("M A C C R E A D Y   P O L A R   T O O L")

# --- 1. Define & Process Glider Polars (Cached for web speed) ---
@st.cache_data
def get_glider_data():
    GLIDER_DATA = {
        "Enzo 3": {
            "VX": [28.8, 29.0, 33.3, 42.70, 44.31, 46.1, 50.5, 56.34, 63.8, 64.8],
            "VY": [-1.19, -0.98, -0.87, -0.99, -1.04, -1.10, -1.28, -1.60, -2.14, -2.23]
        },
        "Photon": {
            "VX": [27.2, 27.8, 31.9, 41.0, 45.8, 50.5, 53.4, 57.7, 56.2, 56.9],
            "VY": [-1.22, -1.03, -0.91, -0.99, -1.14, -1.33, -1.49, -1.75, -1.65, -1.70]
        },
        "Delta 5": {
            "VX": [26.8, 27.4, 31.5, 40.45, 43.68, 47.8, 50.4, 51.54, 52.8, 53.4],
            "VY": [-1.24, -1.06, -0.94, -1.00, -1.09, -1.24, -1.36, -1.42, -1.48, -1.52]
        }
    }

    for name, data in GLIDER_DATA.items():
        vx, vy = np.array(data["VX"]), np.array(data["VY"])
        sort_idx = np.argsort(vx)
        vx, vy = vx[sort_idx], vy[sort_idx]
        
        idx_ms = np.argmax(vy)
        v_ms, s_ms = vx[idx_ms], vy[idx_ms]
        
        poly = np.polyfit(vx[idx_ms:], vy[idx_ms:], 2)
        poly[2] += (s_ms - np.polyval(poly, v_ms))
        
        data.update({'vx': vx, 'vy': vy, 'idx_ms': idx_ms, 'v_ms': v_ms, 'poly': poly})
    return GLIDER_DATA

GLIDER_DATA = get_glider_data()

def get_sink_rate(glider_name, v_array):
    data = GLIDER_DATA[glider_name]
    v_ms, vx, vy, idx_ms, poly = data['v_ms'], data['vx'], data['vy'], data['idx_ms'], data['poly']
    
    s_array = np.zeros_like(v_array)
    for i, v in enumerate(v_array):
        if v <= v_ms:
            s_array[i] = np.interp(v, vx[:idx_ms+1], vy[:idx_ms+1])
        else:
            s_array[i] = np.polyval(poly, v)
    return s_array

# --- 2. STREAMLIT UI SIDEBAR ---
st.sidebar.header("Flight Parameters")
glider_name = st.sidebar.radio("Select Glider", ("Enzo 3", "Photon", "Delta 5"), index=1)
hw = st.sidebar.slider("Headwind (+ Head, - Tail) [km/h]", -30.0, 30.0, 0.0, 1.0)
dZ = st.sidebar.slider("Glide Airmass (+ Lift, - Sink) [m/s]", -3.0, 4.0, 0.0, 0.1)
wc = st.sidebar.slider("Next Thermal (MacCready) [m/s]", 0.0, 4.0, 0.0, 0.1)

# --- 3. MATH & OPTIMIZATION ---
data = GLIDER_DATA[glider_name]
v_min, v_max = data['vx'][0], data['vx'][-1]
origin_x, origin_y = hw, wc - dZ

search_v = np.linspace(v_min, v_max, 1000)
search_s = get_sink_rate(glider_name, search_v)

with np.errstate(divide='ignore', invalid='ignore'):
    slopes = (search_s - origin_y) / (search_v - origin_x)

valid_idx = search_v > origin_x
if np.any(valid_idx):
    best_idx = np.argmax(slopes[valid_idx])
    opt_v = search_v[valid_idx][best_idx]
    opt_s = search_s[valid_idx][best_idx]
else:
    opt_v, opt_s = v_max, get_sink_rate(glider_name, [v_max])[0]

if opt_v < v_min:
    opt_v, opt_s = v_min, get_sink_rate(glider_name, [v_min])[0]

ground_speed = opt_v - hw

# --- 4. TOP METRICS DASHBOARD ---
col1, col2, col3 = st.columns(3)
col1.metric(label="Optimal Airspeed", value=f"{opt_v:.1f} km/h")
col2.metric(label="Ground Speed", value=f"{ground_speed:.1f} km/h")
col3.metric(label="Effective Vertical Shift", value=f"{origin_y:.1f} m/s")

# --- 5. MATPLOTLIB PLOTTING ---
plt.style.use('dark_background')
BG_COLOR, ACCENT_BLUE, ACCENT_PINK, ACCENT_GREEN, TEXT_COLOR = '#1e1e2e', '#00E5FF', '#FF007F', '#00FF66', '#cdd6f4'

fig, ax = plt.subplots(figsize=(10, 6))
fig.patch.set_facecolor(BG_COLOR)
ax.set_facecolor(BG_COLOR)

# Generate curves
v_range = np.linspace(v_min, v_max, 300)
s_range = get_sink_rate(glider_name, v_range)

# Plot elements
ax.plot(v_range, s_range, color=ACCENT_BLUE, linewidth=2.5, label='Glider Polar')
ax.plot(data['vx'], data['vy'], 'x', color='#6c7086', markersize=6, alpha=0.7, label='Raw Data')
ax.plot([origin_x, opt_v], [origin_y, opt_s], '--', color=ACCENT_PINK, linewidth=1.5, label='Tangent')
ax.plot([origin_x], [origin_y], 'o', color=ACCENT_GREEN, markersize=8, zorder=5, label='Virtual Origin')
ax.plot([opt_v], [opt_s], 'o', color=ACCENT_PINK, markersize=8, zorder=5, label='Optimal Speed')

# Optional Logos (will render if they exist in the GitHub repo)
try:
    ax_logo1 = fig.add_axes([0.02, 0.85, 0.12, 0.12], zorder=10)
    ax_logo1.imshow(plt.imread('Oz_logo.png'))
    ax_logo1.axis('off')
except: pass

try:
    ax_logo2 = fig.add_axes([0.86, 0.02, 0.12, 0.12], zorder=10)
    ax_logo2.imshow(plt.imread('Ozone_logo.png'))
    ax_logo2.axis('off')
except: pass

# Aesthetics
ax.set_xlabel("Horizontal air speed (km/h)", color=TEXT_COLOR)
ax.set_ylabel("Vertical speed (m/s)", color=TEXT_COLOR)
ax.grid(True, which='both', color='#45475a', linestyle=':', linewidth=1)
ax.axhline(0, color='#585b70', linewidth=1.5)
ax.axvline(0, color='#585b70', linewidth=1.5)
for spine in ['top', 'right']: ax.spines[spine].set_visible(False)
for spine in ['bottom', 'left']: ax.spines[spine].set_color('#585b70')
ax.tick_params(colors=TEXT_COLOR)
ax.legend(loc='upper right', facecolor=BG_COLOR, edgecolor='none', labelcolor=TEXT_COLOR)
ax.set_xlim(-5, 70)
ax.set_ylim(-3, 4)

# Render plot in Streamlit
st.pyplot(fig)