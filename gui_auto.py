# src/gui_auto.py - Multi-Model Detection Dashboard (7 Models + Last 50 Flows + WORKING FILTERS)
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np
from datetime import datetime
from streamlit_autorefresh import st_autorefresh

# ============================================================
# PAGE CONFIGURATION
# ============================================================
st.set_page_config(
    page_title="Machine Learning Driven Smart Intrusion Detection System",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("🛡️ Machine Learning Driven Smart Intrusion Detection System")
st.markdown("**7 Detection Engines: Rule-Based + 5 ML Models + Ensemble**")
st.markdown("---")

# ============================================================
# SESSION STATE
# ============================================================
if 'last_update' not in st.session_state:
    st.session_state.last_update = datetime.now()

# ============================================================
# MODEL CONFIGURATION
# ============================================================
MODEL_FILES = {
    'rule': {'name': 'Rule-Based', 'file': 'rule_predictions.csv', 'color': '#3498db', 'icon': '📏', 'order': 1},
    'rf': {'name': 'Random Forest', 'file': 'rf_predictions.csv', 'color': '#e74c3c', 'icon': '🌲', 'order': 2},
    'xgb': {'name': 'XGBoost', 'file': 'xgb_predictions.csv', 'color': '#2ecc71', 'icon': '⚡', 'order': 3},
    'et': {'name': 'Extra Trees', 'file': 'et_predictions.csv', 'color': '#9b59b6', 'icon': '🌳', 'order': 4},
    'dt': {'name': 'Decision Tree', 'file': 'dt_predictions.csv', 'color': '#e67e22', 'icon': '🌿', 'order': 5},
    'gb': {'name': 'Gradient Boosting', 'file': 'gb_predictions.csv', 'color': '#f1c40f', 'icon': '📈', 'order': 6},
    'ensemble': {'name': 'Ensemble', 'file': 'ensemble_predictions.csv', 'color': '#e74c3c', 'icon': '🎯', 'order': 7}
}

# ============================================================
# SIDEBAR - DASHBOARD SETTINGS
# ============================================================
st.sidebar.header("Dashboard Settings")

# Auto-refresh settings
refresh_options = {
    "5 seconds": 5,
    "10 seconds": 10,
    "15 seconds": 15,
    "30 seconds": 30,
    "35 seconds": 35,
    "40 seconds": 40,
    "45 seconds": 45,
    "50 seconds": 50,
    "55 seconds": 55,
    "60 seconds": 60,
    "Off": 0
}

refresh_choice = st.sidebar.selectbox(
    "Auto-refresh every:", 
    list(refresh_options.keys()), 
    index=4
)

if refresh_choice == "Off":
    refresh_interval = 0
else:
    refresh_interval = refresh_options[refresh_choice]

if st.sidebar.button("Refresh Now"):
    st.rerun()

if refresh_interval > 0:
    st_autorefresh(interval=refresh_interval * 1000, key="auto_refresh")
    st.sidebar.success(f"Auto-refresh: every {refresh_interval}s")
else:
    st.sidebar.warning("Auto-refresh OFF")

current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
st.sidebar.info(f"Last updated: {current_time}")

# ============================================================
# LOAD ALL MODEL PREDICTIONS
# ============================================================
st.sidebar.markdown("---")
st.sidebar.subheader("Model Status")

flows_dir = 'data/processed_flows'
flows_file = f'{flows_dir}/flows.csv'

# Check if flows exist
if not os.path.exists(flows_file):
    st.warning("Waiting for flow data... Dashboard will update automatically when data arrives.")
    st.stop()

# Load flows
df_flows = pd.read_csv(flows_file)

# Load all model predictions
model_data = {}
for model_id, config in MODEL_FILES.items():
    file_path = f'{flows_dir}/{config["file"]}'
    if os.path.exists(file_path):
        try:
            df = pd.read_csv(file_path)
            if 'prediction' in df.columns:
                model_data[model_id] = {
                    'df': df,
                    'name': config['name'],
                    'color': config['color'],
                    'icon': config['icon'],
                    'order': config['order'],
                    'available': True,
                    'anomalies': int(df['prediction'].sum()),
                    'total': len(df),
                    'anomaly_rate': (df['prediction'].sum() / len(df) * 100) if len(df) > 0 else 0,
                    'avg_confidence': df['anomaly_probability'].mean() * 100 if 'anomaly_probability' in df.columns else 0
                }
                st.sidebar.success(f"[OK] {config['name']}")
            else:
                model_data[model_id] = {'available': False, 'name': config['name'], 'icon': config['icon']}
                st.sidebar.warning(f"[MISSING] {config['name']} - no prediction column")
        except Exception as e:
            model_data[model_id] = {'available': False, 'name': config['name'], 'icon': config['icon'], 'error': str(e)}
            st.sidebar.error(f"[ERROR] {config['name']}")
    else:
        model_data[model_id] = {'available': False, 'name': config['name'], 'icon': config['icon']}
        st.sidebar.warning(f"[WAITING] {config['name']}")

# PCAP count
if os.path.exists('data/raw_pcaps'):
    pcap_count = len([f for f in os.listdir('data/raw_pcaps') if f.endswith('.pcap')])
else:
    pcap_count = 0

# ============================================================
# SIDEBAR FILTERS
# ============================================================
st.sidebar.markdown("---")
st.sidebar.header("Filters")

filter_type = st.sidebar.radio(
    "Show flows:", 
    ["All", "Normal Only", "Anomaly Only"],
    help="Filter flows by prediction result"
)

if 'protocol' in df_flows.columns:
    protocols = ['All'] + sorted(df_flows['protocol'].dropna().unique().tolist())
    filter_protocol = st.sidebar.selectbox("Filter by Protocol:", protocols)
else:
    filter_protocol = 'All'

# Show active filters in sidebar
st.sidebar.markdown("---")
st.sidebar.caption(f"**Active Filters:**")
st.sidebar.caption(f"  • Status: {filter_type}")
st.sidebar.caption(f"  • Protocol: {filter_protocol}")

# ============================================================
# HELPER FUNCTIONS FOR FILTERING
# ============================================================
def apply_flow_filters(df):
    """Apply protocol filter to flows dataframe"""
    if df is None:
        return None
    
    filtered = df.copy()
    
    if filter_protocol != 'All' and 'protocol' in filtered.columns:
        filtered = filtered[filtered['protocol'] == filter_protocol]
    
    return filtered

def apply_prediction_filters(df, pred_column='prediction'):
    """Apply both status and protocol filters to prediction dataframe"""
    if df is None:
        return None
    
    filtered = df.copy()
    
    # Filter by prediction type
    if filter_type == "Normal Only" and pred_column in filtered.columns:
        filtered = filtered[filtered[pred_column] == 0]
    elif filter_type == "Anomaly Only" and pred_column in filtered.columns:
        filtered = filtered[filtered[pred_column] == 1]
    
    # Filter by protocol
    if filter_protocol != 'All' and 'protocol' in filtered.columns:
        filtered = filtered[filtered['protocol'] == filter_protocol]
    
    return filtered

def create_model_table(model_id, model_info):
    """Create a table for a single model WITH FILTERS APPLIED"""
    if not model_info.get('available', False):
        st.markdown(f"### {model_info.get('icon', '❓')} {model_info.get('name', model_id)}")
        st.error("Model Not Working - Prediction file not available")
        if 'error' in model_info:
            st.caption(f"Error: {model_info['error']}")
        else:
            st.caption("Waiting for auto_processor to generate predictions...")
        return False
    
    # Apply filters to the model's dataframe
    df_filtered = apply_prediction_filters(model_info['df'])
    
    if df_filtered is None or len(df_filtered) == 0:
        st.info(f"No flows match the current filters for {model_info['name']}")
        return True
    
    anomalies = df_filtered[df_filtered['prediction'] == 1].copy()
    total_filtered = len(df_filtered)
    anomalies_count = len(anomalies)
    anomaly_percentage = (anomalies_count / total_filtered * 100) if total_filtered > 0 else 0
    
    # Show metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Flows (Filtered)", total_filtered)
    with col2:
        st.metric("Anomalies", anomalies_count, delta=f"{anomaly_percentage:.1f}%")
    with col3:
        st.metric("Normal", total_filtered - anomalies_count)
    with col4:
        st.metric("Anomalies %", f"{anomaly_percentage:.1f}%")
    
    if len(anomalies) > 0:
        anomalies = anomalies.sort_values('anomaly_probability', ascending=False)
        
        display_cols = ['src_ip', 'dst_ip', 'protocol', 'packets', 'bytes', 
                       'anomaly_probability', 'prediction_label']
        available_cols = [c for c in display_cols if c in anomalies.columns]
        
        if 'anomaly_probability' in anomalies.columns:
            anomalies['confidence'] = (anomalies['anomaly_probability'] * 100).round(1).astype(str) + '%'
            if 'anomaly_probability' in available_cols:
                available_cols.remove('anomaly_probability')
            available_cols.append('confidence')
        
        st.dataframe(anomalies[available_cols].head(20), use_container_width=True, height=400)
        
        csv = anomalies.to_csv(index=False)
        st.download_button(
            label=f"Download {model_info['name']} Anomalies (CSV)",
            data=csv,
            file_name=f"{model_id}_anomalies_filtered_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv"
        )
    else:
        st.info(f"No anomalies match the current filters for {model_info['name']}")
    
    return True

# ============================================================
# METRICS OVERVIEW (All Models - WITH FILTERS)
# ============================================================
st.subheader("Model Overview")

available_models = {k: v for k, v in model_data.items() if v.get('available', False)}

if available_models:
    # Calculate filtered metrics for each model
    filtered_metrics = {}
    for model_id, data in available_models.items():
        df_filtered = apply_prediction_filters(data['df'])
        if df_filtered is not None and len(df_filtered) > 0:
            anomalies = int(df_filtered['prediction'].sum())
            total = len(df_filtered)
        else:
            anomalies = 0
            total = 0
        anomaly_rate = (anomalies / total * 100) if total > 0 else 0
        filtered_metrics[model_id] = {'anomalies': anomalies, 'total': total, 'rate': anomaly_rate}
    
    cols = st.columns(min(len(available_models), 4))
    col_idx = 0
    for model_id, data in available_models.items():
        with cols[col_idx % 4]:
            metrics = filtered_metrics[model_id]
            st.metric(
                label=f"{data['icon']} {data['name']}",
                value=f"{metrics['anomalies']:,}",
                delta=f"{metrics['rate']:.1f}% anomalies"
            )
        col_idx += 1
else:
    st.warning("No model predictions available yet. Waiting for auto_processor...")

st.markdown("---")

# ============================================================
# COMPARISON CHART (WITH FILTERS)
# ============================================================
if available_models:
    st.subheader("Model Comparison Chart")
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    model_names = []
    anomaly_counts = []
    colors = []
    
    for model_id, data in available_models.items():
        df_filtered = apply_prediction_filters(data['df'])
        if df_filtered is not None:
            anomalies = int(df_filtered['prediction'].sum())
        else:
            anomalies = 0
        model_names.append(data['name'])
        anomaly_counts.append(anomalies)
        colors.append(data['color'])
    
    bars = ax.bar(model_names, anomaly_counts, color=colors, edgecolor='black')
    ax.set_ylabel('Number of Anomalies')
    ax.set_title(f'Anomalies Detected by Each Model ({filter_type}, {filter_protocol})')
    ax.set_xticklabels(model_names, rotation=45, ha='right')
    
    for bar, val in zip(bars, anomaly_counts):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(), str(val), 
               ha='center', va='bottom')
    
    plt.tight_layout()
    st.pyplot(fig)
    st.markdown("---")

# ============================================================
# INDIVIDUAL MODEL TABLES (7 Tables - WITH FILTERS)
# ============================================================
st.subheader("Detection Results by Model")

# Create tabs for each model in specified order
tabs = st.tabs([f"{model_data[mid]['icon']} {model_data[mid]['name']}" for mid in MODEL_FILES.keys()])

for tab, (model_id, config) in zip(tabs, MODEL_FILES.items()):
    with tab:
        model_info = model_data.get(model_id, {'available': False, 'name': config['name'], 'icon': config['icon']})
        create_model_table(model_id, model_info)

st.markdown("---")

# ============================================================
# ALL FLOWS (FILTERED, EXPANDABLE) - FIXED INDEX ISSUE
# ============================================================
st.subheader("All Flows (Filtered List)")

# Apply protocol filter to flows
filtered_flows = apply_flow_filters(df_flows)

if filtered_flows is not None and len(filtered_flows) > 0:
    # Further filter by prediction type if needed
    if filter_type != "All":
        # Need to check at least one model to filter by prediction
        reference_model = None
        for model_id, data in available_models.items():
            reference_model = data['df']
            break
        
        if reference_model is not None:
            # Get indices where prediction matches filter
            if filter_type == "Normal Only":
                valid_mask = reference_model['prediction'] == 0
            else:
                valid_mask = reference_model['prediction'] == 1
            
            # Get the filtered indices (based on position, not original index)
            valid_positions = valid_mask[valid_mask].index.tolist()
            
            # Filter flows by these positions
            filtered_flows = filtered_flows.iloc[valid_positions].copy()
    
    display_df = filtered_flows.copy()
    display_df = display_df.reset_index(drop=True)
    
    # Add predictions from each available model
    for model_id, data in model_data.items():
        if data.get('available', False):
            df_pred = data['df']
            # Align by position (reset index for both)
            pred_aligned = df_pred.reset_index(drop=True)
            if len(display_df) <= len(pred_aligned):
                display_df[f'{model_id}_label'] = pred_aligned['prediction_label'].iloc[:len(display_df)].values
            else:
                display_df[f'{model_id}_label'] = 'N/A'
    
    # Select columns
    base_cols = ['src_ip', 'dst_ip', 'protocol', 'packets', 'bytes']
    model_cols = [f'{model_id}_label' for model_id in MODEL_FILES.keys() if f'{model_id}_label' in display_df.columns]
    display_cols = base_cols + model_cols
    display_cols = [c for c in display_cols if c in display_df.columns]
    
    with st.expander(f"Click to expand and view all {len(display_df)} filtered flows", expanded=False):
        st.dataframe(display_df[display_cols].tail(100), use_container_width=True, height=500)
        
        csv_all = display_df.to_csv(index=False)
        st.download_button(
            label="Download Filtered Flow List (CSV)",
            data=csv_all,
            file_name=f"all_flows_filtered_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv"
        )
        
        # Show summary statistics
        st.markdown("**Summary Statistics (Filtered):**")
        col_stat1, col_stat2, col_stat3, col_stat4 = st.columns(4)
        with col_stat1:
            st.metric("Total Flows", f"{len(display_df):,}")
        with col_stat2:
            unique_src = display_df['src_ip'].nunique() if 'src_ip' in display_df.columns else 0
            st.metric("Unique Source IPs", f"{unique_src:,}")
        with col_stat3:
            unique_dst = display_df['dst_ip'].nunique() if 'dst_ip' in display_df.columns else 0
            st.metric("Unique Dest IPs", f"{unique_dst:,}")
        with col_stat4:
            protocols_count = display_df['protocol'].nunique() if 'protocol' in display_df.columns else 0
            st.metric("Protocols", f"{protocols_count}")
else:
    st.warning(f"No flows match the current filters (Status: {filter_type}, Protocol: {filter_protocol})")

st.markdown("---")

# ============================================================
# LAST 50 FLOWS TABLE (FILTERED) - ULTRA ROBUST VERSION
# ============================================================
st.subheader("Last 50 Flows (Filtered)")

# Apply protocol filter to flows
filtered_flows_for_last = apply_flow_filters(df_flows)

if filtered_flows_for_last is not None and len(filtered_flows_for_last) > 0:
    # Get last 50 flows
    last_50_df = filtered_flows_for_last.tail(50).copy()
    original_length = len(last_50_df)
    
    # RESET INDEX to avoid index mismatch with predictions
    last_50_df = last_50_df.reset_index(drop=True)
    
    # Add prediction labels from priority models
    priority_models = ['ensemble', 'rf', 'gb', 'dt', 'rule']
    for model_id in priority_models:
        if model_id in model_data and model_data[model_id].get('available', False):
            try:
                df_pred = model_data[model_id]['df']
                
                # Safely get predictions - handle case where predictions have fewer rows
                pred_count = len(df_pred)
                target_count = len(last_50_df)
                
                if pred_count >= target_count:
                    # Take last target_count predictions
                    pred_last_50 = df_pred.tail(target_count).reset_index(drop=True)
                    last_50_df[f'{model_id}_result'] = pred_last_50['prediction_label'].values
                else:
                    # If predictions have fewer rows, pad with N/A
                    pred_last_50 = df_pred.reset_index(drop=True)
                    last_50_df[f'{model_id}_result'] = 'N/A'
                    last_50_df.loc[:pred_count-1, f'{model_id}_result'] = pred_last_50['prediction_label'].values
                    
            except Exception as e:
                print(f"Error adding {model_id} to last 50: {e}")
                last_50_df[f'{model_id}_result'] = 'N/A'
    
    # Select columns for display
    display_last_50_cols = ['src_ip', 'dst_ip', 'protocol', 'packets', 'bytes', 'duration']
    for model_id in priority_models:
        if f'{model_id}_result' in last_50_df.columns:
            display_last_50_cols.append(f'{model_id}_result')
    
    display_last_50_cols = [c for c in display_last_50_cols if c in last_50_df.columns]
    
    st.info(f"Showing last {original_length} flows matching filters: Status={filter_type}, Protocol={filter_protocol}")
    st.dataframe(last_50_df[display_last_50_cols], use_container_width=True, height=450)
    
    csv_last_50 = last_50_df.to_csv(index=False)
    st.download_button(
        label="Download Filtered Last 50 Flows (CSV)",
        data=csv_last_50,
        file_name=f"last_50_flows_filtered_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
        mime="text/csv"
    )
else:
    st.warning(f"No flows match the current filters for Last 50 Flows")

st.markdown("---")

# ============================================================
# SYSTEM STATUS
# ============================================================
st.subheader("System Status")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.write("**Auto-Refresh:**")
    if refresh_interval > 0:
        st.success(f"Every {refresh_interval}s")
    else:
        st.warning("Disabled")

with col2:
    st.write("**Active Models:**")
    active_count = len([m for m in model_data.values() if m.get('available', False)])
    st.info(f"{active_count}/{len(MODEL_FILES)}")

with col3:
    st.write("**Total Flows (Filtered):**")
    filtered_count = len(filtered_flows) if filtered_flows is not None else 0
    st.info(f"{filtered_count:,} / {len(df_flows):,}")

with col4:
    st.write("**PCAP Files:**")
    st.info(f"{pcap_count}")

st.markdown("---")

# ============================================================
# MODEL AVAILABILITY LEGEND
# ============================================================
st.subheader("Model Availability")

model_items = list(model_data.items())
for i in range(0, len(model_items), 4):
    cols = st.columns(4)
    for j in range(4):
        idx = i + j
        if idx < len(model_items):
            model_id, data = model_items[idx]
            with cols[j]:
                if data.get('available', False):
                    st.success(f"{data['icon']} {data['name']}: Active")
                else:
                    st.error(f"{data['icon']} {data['name']}: Not Working")

st.markdown("---")
st.caption(f"""
Last Update: {current_time}
Auto-Refresh: {'Active' if refresh_interval > 0 else 'Inactive'}
Active Models: {active_count}/{len(MODEL_FILES)}
Total Flows: {len(df_flows):,} total, {filtered_count} filtered
PCAP Files: {pcap_count}
Active Filters: Status = {filter_type}, Protocol = {filter_protocol}
""")