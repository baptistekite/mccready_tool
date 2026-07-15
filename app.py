import streamlit as st
import numpy as np
import plotly.graph_objects as go
from PIL import Image

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

# --- 5. PLOTLY CHARTING ---
v_range = np.linspace(v_min, v_max, 300)
s_range = get_sink_rate(glider_name, v_range)

fig = go.Figure()

# Glider Polar Curve
fig.add_trace(go.Scatter(
    x=v_range, y=s_range, mode='lines', name='Glider Polar',
    line=dict(color='#00E5FF', width=3), hoverinfo='x+y'
))

# Raw Data Points
fig.add_trace(go.Scatter(
    x=data['vx'], y=data['vy'], mode='markers', name='Raw Data',
    marker=dict(color='#6c7086', symbol='x', size=8), hoverinfo='skip'
))

# Tangent Line
fig.add_trace(go.Scatter(
    x=[origin_x, opt_v], y=[origin_y, opt_s], mode='lines', name='Speed to Fly Tangent',
    line=dict(color='#FF007F', width=2, dash='dash'), hoverinfo='skip'
))

# Virtual Origin Point
fig.add_trace(go.Scatter(
    x=[origin_x], y=[origin_y], mode='markers', name='Virtual Origin',
    marker=dict(color='#00FF66', size=12, line=dict(color='white', width=1)),
    hovertemplate="Origin<br>Wind: %{x} km/h<br>Shift: %{y} m/s<extra></extra>"
))

# Optimal Speed Point
fig.add_trace(go.Scatter(
    x=[opt_v], y=[opt_s], mode='markers', name='Optimal Speed',
    marker=dict(color='#FF007F', size=12, line=dict(color='white', width=1)),
    hovertemplate="Optimal Speed<br>Airspeed: %{x:.1f} km/h<br>Sink: %{y:.2f} m/s<extra></extra>"
))

# Configure Layout and Theme
fig.update_layout(
    plot_bgcolor='#1e1e2e',
    paper_bgcolor='#1e1e2e',
    font=dict(color='#cdd6f4'),
    xaxis=dict(
        title="Horizontal air speed (km/h)",
        range=[-5, 70],
        gridcolor='#313244',
        zeroline=True, zerolinecolor='#585b70', zerolinewidth=2
    ),
    yaxis=dict(
        title="Vertical speed (m/s)",
        range=[-3, 4],
        gridcolor='#313244',
        zeroline=True, zerolinecolor='#585b70', zerolinewidth=2
    ),
    legend=dict(
        yanchor="top", y=0.99, xanchor="right", x=0.99,
        bgcolor="rgba(30, 30, 46, 0.8)", bordercolor="#45475a", borderwidth=1
    ),
    height=600,
    margin=dict(l=40, r=40, t=40, b=40)
)

# Render Logos
try:
    logo1 = Image.open('Oz_logo.png')
    fig.add_layout_image(
        dict(source=logo1, xref="paper", yref="paper", x=0.01, y=0.99,
             sizex=0.15, sizey=0.15, xanchor="left", yanchor="top", opacity=0.9)
    )
except: pass

try:
    logo2 = Image.open('Ozone_logo.png')
    fig.add_layout_image(
        dict(source=logo2, xref="paper", yref="paper", x=0.99, y=0.01,
             sizex=0.15, sizey=0.15, xanchor="right", yanchor="bottom", opacity=0.9)
    )
except: pass

# Display Chart in Streamlit
st.plotly_chart(fig, use_container_width=True)