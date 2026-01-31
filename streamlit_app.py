import streamlit as st
import plotly.graph_objects as go
import time
import serial
import datetime
import pandas as pd
import numpy as np
from scipy.stats import linregress
from sklearn.ensemble import RandomForestClassifier

# ============================================================================
# CONFIGURATION
# ============================================================================

SENSOR_PORT = "/dev/tty.usbserial-0001"
BAUDRATE = 9600
READING_INTERVAL = 5
filename='/Users/haroldhoffmann/Desktop/Research FY26/REDO_ADC_GAS/TidyData_E2/E2_Features_ZEROED_PPB.xlsx'
train_df=pd.read_excel(filename)
outliers=['Healthy-16 D2', 'Inoculated-8 D2', 'Baseline-Blank T9 D4', 'Baseline-Blank T7 D3', 'Baseline-Blank T4 D3','Baseline-Blank T7 D1']
df=train_df[~train_df['Plant Id'].isin(outliers)]

X=train_df.drop(columns=['Label', 'Plant Id', 'Day'])
y=train_df['Label']
model = RandomForestClassifier(n_estimators=1000, random_state=42)
model.fit(X, y)




# ============================================================================
# SENSOR FUNCTIONS
# ============================================================================

def getSingleReading(ser):
    ser.flushInput()
    ser.write('\r'.encode('utf-8'))
    time.sleep(0.5)
    measurementdata = ser.read_all().decode('utf-8', errors='ignore').strip()
    return measurementdata if measurementdata else "[TIMEOUT/NO DATA]"
def extract_ml_features(df):
    # 1. Force Numeric Types to prevent ufunc 'divide' errors
    sensor_cols = ['ethylene_ppm', 'temperature', 'humidity']
    for col in sensor_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    # 2. Add Time Seconds Column and Cutoff at 610s
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df['Time Seconds'] = (df['timestamp'] - df['timestamp'].iloc[0]).dt.total_seconds()
    df = df[df['Time Seconds'] <= 610].copy()
    
    # 3. Shift PPB (Value at 0s = 0)
    first_val = df['ethylene_ppm'].iloc[0]
    df['PPB_Shifted'] = df['ethylene_ppm'] - first_val
    
    # 4. Stabilization Split Logic (Pivot)
    STABILIZATION_THRESHOLD = 0.5 
    CONSECUTIVE_POINTS = 3       
    
    df['dt'] = df['Time Seconds'].diff()
    df['dV'] = df['PPB_Shifted'].diff()
    df['Derivative'] = df['dV'] / df['dt'].replace(0, np.nan)
    
    # Identify stable period
    df['Is_Stable'] = np.abs(df['Derivative']) < STABILIZATION_THRESHOLD
    df['Stable_Count'] = df['Is_Stable'].rolling(window=CONSECUTIVE_POINTS).sum()
    
    sustained = df[df['Stable_Count'] == CONSECUTIVE_POINTS]
    if not sustained.empty:
        idx_loc = df.index.get_loc(sustained.index[0])
        split_time = df.iloc[max(0, idx_loc - (CONSECUTIVE_POINTS - 1))]['Time Seconds']
    else:
        split_time = df['Time Seconds'].median()

    # 5. Grouping (Matching List 2 Categories)
    # Note: Using exact spacing used in List 2 keys
    groups = {
        'Steady_ ': df[df['Time Seconds'] > split_time],
        'Total_ ': df,
        'Warm_ ': df[df['Time Seconds'] <= split_time]
    }
    
    features = {}
    for prefix, data in groups.items():
        if data.empty:
            # Fill with 0s if a phase is missing to keep column count consistent
            y, x, temp, hum, derivs = np.array([0]), np.array([0]), np.array([0]), np.array([0]), np.array([0])
        else:
            y = data['PPB_Shifted'].values
            x = data['Time Seconds'].values
            temp = data['temperature'].values
            hum = data['humidity'].values
            derivs = data['Derivative'].dropna().values
        
        # Mapping features to your List 2 names
        features[f'{prefix}Mean'] = float(y.mean())
        features[f'{prefix}Maximum'] = float(y.max())
        features[f'{prefix}Minimum'] = float(y.min())
        features[f'{prefix}Standard_Deviation'] = float(y.std())
        features[f'{prefix}AUC'] = float(np.trapz(y, x))
        features[f'{prefix}duration'] = float(x.max() - x.min())
        
        # Derivative/Slope
        features[f'{prefix}avg derivative'] = float(derivs.mean()) if len(derivs) > 0 else 0.0
        features[f'{prefix}max derivative'] = float(derivs.max()) if len(derivs) > 0 else 0.0
        features[f'{prefix}min derivative'] = float(derivs.min()) if len(derivs) > 0 else 0.0
        
        if len(x) > 1:
            slope, *_ = linregress(x, y)
            features[f'{prefix}slope'] = float(slope)
        else:
            features[f'{prefix}slope'] = 0.0
            
        # Environmental Features (Crucial for List 2)
        features[f'{prefix}Mean Humidity'] = float(hum.mean())
        features[f'{prefix}Max Humidity'] = float(hum.max())
        features[f'{prefix}Min Humidity'] = float(hum.min())
        features[f'{prefix}Mean Temperature'] = float(temp.mean())
        features[f'{prefix}Max Temperature'] = float(temp.max())
        features[f'{prefix}Min Temperature'] = float(temp.min())

    # 6. Final Reordering to match List 2 EXACTLY
    requested_order = [
        'Steady_ Mean', 'Steady_ Maximum', 'Steady_ Minimum', 'Steady_ Standard_Deviation', 'Steady_ AUC', 
        'Steady_ duration', 'Steady_ avg derivative', 'Steady_ max derivative', 'Steady_ min derivative', 
        'Steady_ slope', 'Steady_ Mean Humidity', 'Steady_ Max Humidity', 'Steady_ Min Humidity', 
        'Steady_ Mean Temperature', 'Steady_ Max Temperature', 'Steady_ Min Temperature', 
        'Total_ Mean', 'Total_ Maximum', 'Total_ Minimum', 'Total_ Standard_Deviation', 'Total_ AUC', 
        'Total_ duration', 'Total_ avg derivative', 'Total_ max derivative', 'Total_ min derivative', 
        'Total_ slope', 'Total_ Mean Humidity', 'Total_ Max Humidity', 'Total_ Min Humidity', 
        'Total_ Mean Temperature', 'Total_ Max Temperature', 'Total_ Min Temperature', 
        'Warm_ Mean', 'Warm_ Maximum', 'Warm_ Minimum', 'Warm_ Standard_Deviation', 'Warm_ AUC', 
        'Warm_ duration', 'Warm_ avg derivative', 'Warm_ max derivative', 'Warm_ min derivative', 
        'Warm_ slope', 'Warm_ Mean Humidity', 'Warm_ Max Humidity', 'Warm_ Min Humidity', 
        'Warm_ Mean Temperature', 'Warm_ Max Temperature', 'Warm_ Min Temperature'
    ]
    
    features_df = pd.DataFrame([features])
    
    # Return reordered dataframe (this fixes the Model Predict ValueError)
    return features_df[requested_order]

# ============================================================================
# UI COMPONENTS
# ============================================================================

def create_mini_gauge(label, value, max_value, color):
    """Compact gauge chart"""
    gauge_max = max(100, max_value * 1.2)
    
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=value,
        title={'text': label, 'font': {'size': 10}},
        number={'font': {'size': 14}},
        gauge={
            'axis': {'range': [0, gauge_max]},
            'bar': {'color': color},
            'bgcolor': "white",
            'borderwidth': 1,
            'bordercolor': "gray",
            'steps': [
                {'range': [0, gauge_max * 0.7], 'color': "#f0f2f6"},
                {'range': [gauge_max * 0.7, gauge_max], 'color': "#e1e4e8"}
            ],
        }
    ))
    fig.update_layout(height=120, margin=dict(l=10, r=10, t=20, b=0))
    return fig

# ============================================================================
# STREAMLIT APP SETUP
# ============================================================================

st.set_page_config(page_title="Crop Monitor", layout="wide")

# Initialize session state
if 'ethylene_value' not in st.session_state:
    st.session_state.ethylene_value = 0
if 'max_ethylene_value' not in st.session_state:
    st.session_state.max_ethylene_value = 0
if 'ultrasonic_value' not in st.session_state:
    st.session_state.ultrasonic_value = 0
if 'max_ultrasonic_value' not in st.session_state:
    st.session_state.max_ultrasonic_value = 0
if 'last_reading' not in st.session_state:
    st.session_state.last_reading = "No data"
if 'readings_history' not in st.session_state:
    st.session_state.readings_history = []
if 'monitoring' not in st.session_state:
    st.session_state.monitoring = False
if 'monitoring_start_time' not in st.session_state:
    st.session_state.monitoring_start_time = None
if 'last_update_time' not in st.session_state:
    st.session_state.last_update_time = None
if 'continuous_df' not in st.session_state:
    st.session_state.continuous_df = pd.DataFrame(columns=[
        'timestamp', 'ethylene_ppm', 'ultrasonic_khz', 'temperature', 'humidity', 'plant_type', 'state'
    ])

# ============================================================================
# COMPACT HEADER
# ============================================================================

col1, col2, col3, col4, col5 = st.columns([2, 1, 1, 1, 1.5])

with col1:
    if st.button("⏯ START/STOP", use_container_width=True, type="primary"):
        st.session_state.monitoring = not st.session_state.monitoring
        if st.session_state.monitoring:
            st.session_state.monitoring_start_time = datetime.datetime.now()
            st.toast("Started")
        else:
            st.session_state.monitoring_start_time = None
            st.toast("Stopped")

    # Tiny elapsed timer (updates on rerun)
    if st.session_state.monitoring and st.session_state.monitoring_start_time:
        elapsed = datetime.datetime.now() - st.session_state.monitoring_start_time
        elapsed_str = str(elapsed).split('.')[0]  # HH:MM:SS
        st.caption(f"⏱ {elapsed_str}")

with col2:
    st.markdown("**Plant**")
    plant_choice = st.selectbox("Plant", ["Watermelon", "Tomato"], label_visibility="collapsed")

with col3:
    eth_toggle = st.toggle("Ethylene", value=True)

with col4:
    ultra_toggle = st.toggle("Ultrasonic", value=False)

with col5:
    com_port = st.text_input("COM Port", value=SENSOR_PORT, placeholder="/dev/tty.usbserial-0001")

# ============================================================================
# CONTINUOUS SENSOR READING
# ============================================================================

if st.session_state.monitoring and eth_toggle:
    try:
        # Use the user-specified COM port
        ser = serial.Serial(com_port, BAUDRATE, timeout=1)
        reading_lst = getSingleReading(ser)
        
        if reading_lst != "[TIMEOUT/NO DATA]" and ',' in reading_lst:
            parts = reading_lst.split(',')
            eth_reading = parts[1] if len(parts) > 1 else "0"
            temp_reading = parts[2] if len(parts) > 2 else "0"
            hum_reading = parts[3] if len(parts) > 3 else "0"
            ultra_reading = parts[4] if len(parts) > 4 else "0"
            
            st.session_state.last_reading = eth_reading
            
            eth_val = float(eth_reading)
            ultra_val = float(ultra_reading) if ultra_toggle else 0
            
            st.session_state.ethylene_value = eth_val
            st.session_state.ultrasonic_value = ultra_val
            st.session_state.last_update_time = datetime.datetime.now()
            
            new_row = pd.DataFrame([{
                'timestamp': datetime.datetime.now(),
                'ethylene_ppm': eth_val,
                'ultrasonic_khz': ultra_val,
                'temperature': temp_reading,
                'humidity': hum_reading,
                'plant_type': plant_choice,
                'state': 'monitoring'
            }])

            st.session_state.continuous_df = pd.concat(
                [st.session_state.continuous_df, new_row], 
                ignore_index=True
            )
            
            if eth_val > st.session_state.max_ethylene_value:
                st.session_state.max_ethylene_value = eth_val
            
            if ultra_val > st.session_state.max_ultrasonic_value:
                st.session_state.max_ultrasonic_value = ultra_val
            
            st.session_state.readings_history.append({
                'timestamp': datetime.datetime.now(),
                'value': eth_val
            })
        
        ser.close()
        
    except Exception as e:
        st.error(f"Sensor error: {e}")

# ============================================================================
# COMPACT LIVE DATA + GAUGES (SINGLE ROW)
# ============================================================================

col_status, col_last, col_curr, col_max, col_g1, col_g2 = st.columns([0.8, 0.8, 0.8, 0.8, 1.5, 1.5])

with col_status:
    status = "🟢 Live" if st.session_state.monitoring else "⚫ Off"
    st.markdown(f"**{status}**")
    if st.session_state.last_update_time:
        time_since = (datetime.datetime.now() - st.session_state.last_update_time).total_seconds()
        st.caption(f"{time_since:.0f}s ago")

with col_last:
    st.markdown("**Last**")
    st.markdown(f"`{st.session_state.last_reading}`")

with col_curr:
    st.markdown("**Current**")
    st.markdown(f"`{st.session_state.ethylene_value:.1f}`")

with col_max:
    st.markdown("**Max**")
    st.markdown(f"`{st.session_state.max_ethylene_value:.1f}`")

with col_g1:
    st.plotly_chart(
        create_mini_gauge(
            "Ethylene", 
            st.session_state.ethylene_value,
            st.session_state.max_ethylene_value, 
            "#2E7D32"
        ), 
        use_container_width=True
    )

with col_g2:
    st.plotly_chart(
        create_mini_gauge(
            "Ultrasonic", 
            st.session_state.ultrasonic_value,
            max(100, st.session_state.max_ultrasonic_value),
            "#1565C0"
        ), 
        use_container_width=True
    )

st.divider()

# ============================================================================
# ML SECTION (COMPACT)
# ============================================================================

st.subheader("🤖 ML Diagnosis")

col_btn1, col_btn2, col_btn3 = st.columns([1, 1, 2])
# 1. Prediction Trigger
with col_btn1:
    if st.button("▶ ANALYZE CROP HEALTH", use_container_width=True, type="primary"):
        if len(st.session_state.continuous_df) > 0:
            # Keep ML processing exactly as it was
            features = extract_ml_features(st.session_state.continuous_df)
            st.session_state.prediction = model.predict(features)[0]
        else:
            st.warning("⚠️ No data collected yet.")

# 2. FULL-WIDTH PROFESSIONAL DISPLAY
# This part stays OUTSIDE of any columns to take up the full screen width
if 'prediction' in st.session_state and st.session_state.prediction:
    pred = st.session_state.prediction.upper()
    
    # Professional Color Palettes
    is_healthy = "HEALTHY" in pred
    bg_color = "#E8F5E9" if is_healthy else "#FFEBEE"
    border_color = "#2E7D32" if is_healthy else "#C62828"
    text_color = "#1B5E20" if is_healthy else "#B71C1C"

    st.markdown(f"""
        <div style="
            display:block;
            background-color: {bg_color}; 
            padding: 16px 18px; 
            border-radius: 10px; 
            border: 4px solid {border_color};
            text-align: center;
            width: 92%;
            max-width: 980px;
            margin: 12px auto;
            box-shadow: 0px 3px 12px rgba(0,0,0,0.06);">
            <p style="color: {text_color}; font-size: 18px; font-weight: 600; font-family: sans-serif; margin-bottom: 6px; letter-spacing: 1px;">
                 DIAGNOSIS REPORT
            </p>
            <h1 style="color: {text_color}; font-size: 60px; font-weight: 900; font-family: sans-serif; margin: 0; line-height: 1;">
                {pred}
            </h1>
            <div style="margin-top: 10px; height: 5px; width: 90px; background-color: {border_color}; margin-left: auto; margin-right: auto; border-radius: 5px;"></div>
        </div>
        """, unsafe_allow_html=True)

with col_btn2:
    if st.button("🗑 Clear", use_container_width=True):
        st.session_state.readings_history = []
        st.session_state.continuous_df = pd.DataFrame(columns=[
            'timestamp', 'ethylene_ppm', 'ultrasonic_khz', 'temperature', 'humidity', 'plant_type', 'state'
        ])
        st.session_state.max_ethylene_value = 0
        st.session_state.ethylene_value = 0
        st.session_state.max_ultrasonic_value = 0
        st.session_state.ultrasonic_value = 0
        st.toast("Cleared")

# Results Display
if len(st.session_state.continuous_df) > 0:
    df = st.session_state.continuous_df
    
    col_s1, col_s2, col_s3, col_s4, col_s5 = st.columns(5)
    
    with col_s1:
        st.metric("Current", f"{st.session_state.ethylene_value:.1f}", label_visibility="visible")
    with col_s2:
        st.metric("Avg", f"{df['ethylene_ppm'].mean():.1f}")
    with col_s3:
        st.metric("Max", f"{df['ethylene_ppm'].max():.1f}")
    with col_s4:
        st.metric("StdDev", f"{df['ethylene_ppm'].std():.1f}")
    with col_s5:
        st.metric("Count", len(df))
    
    # Status indicator
    avg_ethylene = df['ethylene_ppm'].mean()
    if avg_ethylene < 50:
        st.success("✓ Status: Healthy")
    elif avg_ethylene < 100:
        st.warning("⚠ Status: Monitor")
    else:
        st.error("⚠ Status: Action Needed")
    
else:
    st.info("Start monitoring to collect data")

# ============================================================================
# DATA TABLE (COLLAPSED BY DEFAULT)
# ============================================================================

with st.expander("📊 View Data History"):
    if len(st.session_state.continuous_df) > 0:
        df_display = st.session_state.continuous_df.sort_values('timestamp', ascending=False)
        st.dataframe(df_display, use_container_width=True, height=200)
        st.line_chart(st.session_state.continuous_df.set_index('timestamp')['ethylene_ppm'], height=200)
    else:
        st.info("No data yet")

# ============================================================================
# AUTO-REFRESH
# ============================================================================

if st.session_state.monitoring:
    time.sleep(READING_INTERVAL)
    st.rerun()