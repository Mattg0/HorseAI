import streamlit as st
import pandas as pd
from datetime import datetime, timedelta
import yaml
import json
import os
from UI.UIhelper import PipelineHelper
import plotly.express as px
import plotly.graph_objects as go

# Page configuration
st.set_page_config(
    page_title="Horse Racing Pipeline",
    page_icon="🏇",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for color palette
st.markdown("""
<style>
    /* Color palette: #244855 / #E64833 / #874F41 / #90AEAD / #FBE9D0 */

    .stApp {
        background-color: #FBE9D0;
    }

    .stSidebar {
        background-color: #244855;
    }

    .stSidebar .stMarkdown,
    .stSidebar .stRadio label,
    .stSidebar h2 {
        color: #FBE9D0 !important;
    }

    .stSidebar .stRadio > div {
        background-color: transparent;
        padding: 1rem;
    }

    .stSidebar .stRadio > div > label {
        display: flex;
        align-items: center;
        cursor: pointer;
        margin: 0.25rem 0;
    }

    .stSidebar .stRadio > div > label > div:first-child {
        display: none !important; /* Hide radio button */
    }

    .stSidebar .stRadio > div > label > div:last-child {
        background: rgba(251, 233, 208, 0.1) !important;
        backdrop-filter: blur(10px);
        color: #FBE9D0 !important;
        font-weight: 500;
        padding: 0.75rem 1rem;
        border-radius: 8px;
        border: 1px solid rgba(251, 233, 208, 0.2);
        width: 100%;
        transition: all 0.3s ease;
    }

    .stSidebar .stRadio > div > label > div:last-child:hover {
        background: rgba(251, 233, 208, 0.2) !important;
        border: 1px solid rgba(230, 72, 51, 0.5);
        transform: translateY(-1px);
    }

    .main-header {
        background-color: #244855;
        color: #FBE9D0;
        padding: 2rem;
        border-radius: 10px;
        margin-bottom: 2rem;
        text-align: center;
    }

    .config-panel {
        background-color: #90AEAD;
        color: #244855;
        padding: 2rem;
        border-radius: 10px;
        margin-bottom: 1rem;
        border: 1px solid #874F41;
    }

    .output-panel {
        background-color: #874F41;
        color: #FBE9D0;
        padding: 2rem;
        border-radius: 10px;
        margin-bottom: 1rem;
        border: 1px solid #244855;
    }

    .stButton > button {
        background-color: #E64833;
        color: #FBE9D0;
        border: none;
        border-radius: 8px;
        padding: 0.75rem 1.5rem;
        font-weight: bold;
        transition: all 0.3s ease;
    }

    .stButton > button:hover {
        background-color: #E64833;
        backdrop-filter: blur(10px);
        box-shadow: 0 8px 32px rgba(230, 72, 51, 0.3);
        transform: translateY(-2px);
    }

    .success-message {
        background-color: #90AEAD;
        color: #244855;
        padding: 1rem;
        border-radius: 5px;
        margin: 1rem 0;
        border-left: 4px solid #244855;
    }

    .error-message {
        background-color: #E64833;
        color: #FBE9D0;
        padding: 1rem;
        border-radius: 5px;
        margin: 1rem 0;
        border-left: 4px solid #244855;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state and helper
if 'output_logs' not in st.session_state:
    st.session_state.output_logs = []
if 'helper' not in st.session_state:
    st.session_state.helper = PipelineHelper()
if 'config_loaded' not in st.session_state:
    st.session_state.config_loaded = False
if 'training_active' not in st.session_state:
    st.session_state.training_active = False


# Config management functions
def load_config():
    """Load config.yaml file"""
    try:
        config_path = 'config.yaml'
        if os.path.exists(config_path):
            with open(config_path, 'r') as file:
                config_data = yaml.safe_load(file)
                st.session_state.config_data = config_data
                st.session_state.config_loaded = True
                log_output("Configuration loaded successfully", "success")
                return config_data
        else:
            log_output("config.yaml not found", "error")
            return None
    except Exception as e:
        log_output(f"Error loading config: {str(e)}", "error")
        return None


def save_config(updated_config):
    """Save updated config to config.yaml"""
    try:
        config_path = 'config.yaml'
        with open(config_path, 'w') as file:
            yaml.dump(updated_config, file, default_flow_style=False, sort_keys=False)
        st.session_state.config_data = updated_config
        log_output("Configuration saved successfully", "success")
        return True
    except Exception as e:
        log_output(f"Error saving config: {str(e)}", "error")
        return False


def display_config_json():
    """Display current config as formatted JSON"""
    if st.session_state.config_data:
        return json.dumps(st.session_state.config_data, indent=2)
    return "No configuration loaded"


# Helper functions (placeholders for now)
def log_output(message, message_type="info"):
    """Add message to output logs"""
    timestamp = datetime.now().strftime("%H:%M:%S")
    st.session_state.output_logs.append({
        "timestamp": timestamp,
        "message": message,
        "type": message_type
    })


def clear_logs():
    """Clear output logs"""
    st.session_state.output_logs = []


def display_logs():
    """Display output logs"""
    if st.session_state.output_logs:
        st.markdown("### 📋 Output Logs")
        for log in st.session_state.output_logs[-10:]:  # Show last 10 logs
            if log["type"] == "success":
                st.markdown(f'<div class="success-message">[{log["timestamp"]}] ✅ {log["message"]}</div>',
                            unsafe_allow_html=True)
            elif log["type"] == "error":
                st.markdown(f'<div class="error-message">[{log["timestamp"]}] ❌ {log["message"]}</div>',
                            unsafe_allow_html=True)
            else:
                st.write(f'[{log["timestamp"]}] ℹ️ {log["message"]}')


# Placeholder functions for pipeline operations
def mysql_sqlite_sync(db):
    """Placeholder for MySQL to SQLite sync"""
    log_output("Starting MySQL to SQLite synchronization...", "info")
    from core.orchestrators.mysql_sqlite_sync import sync_data
    sync_data(db)
    log_output("MySQL to SQLite sync completed successfully!", "success")


def execute_full_training(progress_bar, status_text):
    """Execute full training using pipeline helper with async support"""

    # Check if training is already running
    if st.session_state.helper.is_training:
        # Training in progress - check for updates
        updates = st.session_state.helper.get_training_updates()

        for update in updates:
            if update['type'] == 'progress':
                progress_bar.progress(update['percent'])
                status_text.text(f"Progress: {update['percent']}% - {update['message']}")
                log_output(update['message'], "info")
            elif update['type'] == 'complete':
                if update['success']:
                    progress_bar.progress(100)
                    status_text.text("Training completed!")
                    log_output(update['message'], "success")
                else:
                    progress_bar.progress(0)
                    status_text.text("Training failed!")
                    log_output(update['message'], "error")

                # Clear training state
                st.session_state.training_active = False

        # Auto-refresh if still training
        if st.session_state.helper.is_training:
            st.session_state.training_active = True
            # This will cause the UI to refresh and check again
            return
    else:
        # Not training - start new training
        log_output("Starting background training...", "info")

        if st.session_state.helper.start_training_async():
            progress_bar.progress(0)
            status_text.text("Training started in background...")
            st.session_state.training_active = True
            log_output("Training started successfully in background", "success")
        else:
            log_output("Failed to start training", "error")


def execute_predictions(selected_races, progress_bar, status_text, force_reprediction=False):
    """Execute predictions using pipeline helper"""
    prediction_type = "force reprediction" if force_reprediction else "standard prediction"
    log_output(f"Starting race {prediction_type}...", "info")

    def progress_callback(percentage, message):
        """Callback function to update progress"""
        progress_bar.progress(percentage)
        status_text.text(f"Progress: {percentage}% - {message}")
        log_output(message, "info")

    try:
        result = st.session_state.helper.execute_predictions(
            races_to_predict=selected_races if selected_races else None,
            progress_callback=progress_callback,
            force_reprediction=force_reprediction
        )

        if result["success"]:
            progress_bar.progress(100)
            status_text.text("Predictions completed!")
            log_output(result["message"], "success")
        else:
            progress_bar.progress(0)
            status_text.text("Predictions failed!")
            log_output(result["message"], "error")

    except Exception as e:
        progress_bar.progress(0)
        status_text.text("Prediction error!")
        log_output(f"Prediction error: {str(e)}", "error")


def execute_comprehensive_evaluation(progress_bar, status_text):
    """Execute comprehensive evaluation using new PredictEvaluator"""
    log_output("Starting comprehensive evaluation...", "info")

    def progress_callback(percentage, message):
        """Callback function to update progress"""
        progress_bar.progress(percentage)
        status_text.text(f"Progress: {percentage}% - {message}")
        log_output(message, "info")

    try:
        result = st.session_state.helper.evaluate_all_predictions_comprehensive(progress_callback=progress_callback)

        if result["success"]:
            progress_bar.progress(100)
            status_text.text("Comprehensive evaluation completed!")
            log_output(result["message"], "success")

            # Store results in session state for display
            st.session_state.evaluation_results = result

        else:
            progress_bar.progress(0)
            status_text.text("Evaluation failed!")
            log_output(result["message"], "error")

    except Exception as e:
        progress_bar.progress(0)
        status_text.text("Evaluation error!")
        log_output(f"Evaluation error: {str(e)}", "error")


def display_evaluation_charts(chart_data):
    """Display evaluation charts using Plotly"""

    # Overall Performance Summary
    st.markdown("### 📊 Overall Performance")
    summary = chart_data['overall_summary']

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Races", summary['total_races'])
    with col2:
        st.metric("Winner Accuracy", f"{summary['winner_accuracy']:.1f}%")
    with col3:
        st.metric("Podium Accuracy", f"{summary['podium_accuracy']:.1f}%")
    with col4:
        st.metric("Total Winning Bets", summary['total_winning_bets'])

    # Bet Type Performance Chart
    if chart_data['bet_performance']:
        st.markdown("### 🎯 Bet Type Performance")

        bet_df = pd.DataFrame(chart_data['bet_performance'])

        # Create side-by-side charts
        col1, col2 = st.columns(2)

        with col1:
            # Win Rate Bar Chart
            fig_winrate = px.bar(
                bet_df,
                x='win_rate',
                y='bet_type',
                orientation='h',
                title="Win Rate by Bet Type (%)",
                color='win_rate',
                color_continuous_scale=['#E64833', '#90AEAD', '#244855'],
                text='win_rate'
            )
            fig_winrate.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
            fig_winrate.update_layout(
                showlegend=False,
                height=400,
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)'
            )
            st.plotly_chart(fig_winrate, use_container_width=True)

        with col2:
            # Wins vs Losses Stacked Bar
            fig_wins = go.Figure(data=[
                go.Bar(name='Wins', y=bet_df['bet_type'], x=bet_df['wins'],
                       orientation='h', marker_color='#90AEAD'),
                go.Bar(name='Losses', y=bet_df['bet_type'], x=bet_df['losses'],
                       orientation='h', marker_color='#E64833')
            ])
            fig_wins.update_layout(
                barmode='stack',
                title="Wins vs Losses by Bet Type",
                height=400,
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)'
            )
            st.plotly_chart(fig_wins, use_container_width=True)

        # Detailed table
        st.markdown("**Detailed Bet Performance**")
        display_df = bet_df[['bet_type', 'wins', 'total', 'win_rate']].copy()
        display_df.columns = ['Bet Type', 'Wins', 'Total Races', 'Win Rate (%)']
        display_df['Win Rate (%)'] = display_df['Win Rate (%)'].round(1)
        st.dataframe(display_df, hide_index=True, use_container_width=True)

    # Quinte Analysis
    if chart_data.get('quinte_summary'):
        st.markdown("### 🌟 Quinte Performance")

        quinte = chart_data['quinte_summary']
        left, center, right = st.columns(3)
        with left:
            st.metric("Quinte Races", quinte['total_quinte_races'])
        with center:
            st.metric("Quinte Winner Accuracy", f"{quinte['winner_accuracy']:.1f}%")
        with right:
            st.metric("Quinte Bet Win Rate", f"{quinte['quinte_bet_win_rate']:.1f}%")

        # Quinte Horse Strategy Analysis
        if chart_data['quinte_strategy']:
            st.markdown("### 🐎 Quinte Betting Strategy Analysis")

            strategy_df = pd.DataFrame(chart_data['quinte_strategy'])

            # Strategy comparison chart
            fig_strategy = px.bar(
                strategy_df,
                x='strategy',
                y='win_rate',
                title="Win Rate by Number of Horses",
                color='win_rate',
                color_continuous_scale=['#E64833', '#874F41', '#244855'],
                text='win_rate'
            )
            fig_strategy.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
            fig_strategy.update_layout(
                showlegend=False,
                height=400,
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                xaxis_title="Betting Strategy",
                yaxis_title="Win Rate (%)"
            )
            st.plotly_chart(fig_strategy, use_container_width=True)

            # Recommendations
            st.markdown("### 💡 Betting Recommendations")

            base_rate = strategy_df[strategy_df['strategy'] == '5 Horses']['win_rate'].iloc[0]
            rate_6 = strategy_df[strategy_df['strategy'] == '6 Horses']['win_rate'].iloc[0]
            rate_7 = strategy_df[strategy_df['strategy'] == '7 Horses']['win_rate'].iloc[0]

            improvement_6 = rate_6 - base_rate
            improvement_7 = rate_7 - base_rate

            subleft, subright = st.columns(2)

            with subleft:
                st.markdown("**6-Horse Strategy**")
                if improvement_6 > 5:  # 5% improvement threshold
                    st.success(f"✅ Significant improvement: +{improvement_6:.1f}%")
                    st.info("🎯 **Recommended:** Using 6 horses shows strong results")
                elif improvement_6 > 0:
                    st.warning(f"⚠️ Modest improvement: +{improvement_6:.1f}%")
                    st.info("💭 Consider cost vs benefit")
                else:
                    st.error(f"❌ No improvement: {improvement_6:+.1f}%")
                    st.info("🚫 **Not recommended:** Stick with 5 horses")

            with subright:
                st.markdown("**7-Horse Strategy**")
                if improvement_7 > 10:  # 10% improvement threshold for 7 horses
                    st.success(f"✅ Strong improvement: +{improvement_7:.1f}%")
                    st.info("🎯 **Recommended:** 7 horses worth the extra cost")
                elif improvement_7 > improvement_6:
                    st.warning(f"⚠️ Better than 6 horses: +{improvement_7:.1f}%")
                    st.info("💭 Compare with 6-horse strategy")
                else:
                    st.error(f"❌ Not recommended: {improvement_7:+.1f}%")
                    st.info("🚫 **Avoid:** Extra cost not justified")

            # Cost-benefit analysis
            st.markdown("### 💰 Cost-Benefit Analysis")

            cost_multiplier_6 = 6 / 5  # 20% more combinations
            cost_multiplier_7 = 7 / 5  # 40% more combinations

            if improvement_6 > 0:
                roi_6 = improvement_6 / ((cost_multiplier_6 - 1) * 100)
                st.info(f"**6-Horse ROI:** {roi_6:.1f}x return on extra investment")

            if improvement_7 > 0:
                roi_7 = improvement_7 / ((cost_multiplier_7 - 1) * 100)
                st.info(f"**7-Horse ROI:** {roi_7:.1f}x return on extra investment")

    else:
        st.info("No quinte races found for strategy analysis")


def execute_incremental_training(date_from, date_to, limit, update_model, create_enhanced,
                                archive_after, progress_bar, status_text):
    """Execute incremental training using regression enhancement pipeline"""
    log_output("Starting incremental training with regression enhancement...", "info")

    def progress_callback(percentage, message):
        """Callback function to update progress"""
        progress_bar.progress(percentage)
        status_text.text(f"Progress: {percentage}% - {message}")
        log_output(message, "info")

    try:
        result = st.session_state.helper.execute_incremental_training(
            date_from=date_from,
            date_to=date_to,
            limit=limit,
            update_model=update_model,
            create_enhanced=create_enhanced,
            archive_after=archive_after,
            progress_callback=progress_callback
        )

        if result["success"]:
            progress_bar.progress(100)
            status_text.text("Incremental training completed!")
            log_output(result["message"], "success")
            st.session_state.incremental_results = result
        else:
            progress_bar.progress(0)
            status_text.text("Incremental training failed!")
            log_output(result["message"], "error")

    except Exception as e:
        progress_bar.progress(0)
        status_text.text("Training error!")
        log_output(f"Incremental training error: {str(e)}", "error")



# Main application
def main():
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>🏇 Horse Racing Prediction Pipeline</h1>
        <p>Comprehensive data processing and model management interface</p>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("""
    <script>
    window.parent.document.querySelector('[data-testid="stSidebar"]').setAttribute('data-theme', 'light');
    </script>
    """, unsafe_allow_html=True)
    # Sidebar navigation
    st.sidebar.markdown("## 🚀 Pipeline Operations")

    operation = st.sidebar.radio(
        "Choose Operation:",
        [
            "🎲 Execute Prediction",
            "✨ AI Insight",
            "📈 Execute Evaluation",
            "🔄 Incremental Training",
            "🎯 Execute Full Training",
            "🔄 MySQL ↔ SQLite Sync",
            "⚙️ Update Config.yaml",
        ],
        index=0
    )

    # Main content area
    col1, col2 = st.columns([2, 1])

    with col1:
        # Configuration panel
        with st.container():
            if operation == "🔄 MySQL ↔ SQLite Sync":
                st.markdown('''
                <div class="config-panel">
                    <h3> 🔄 MySQL to SQLite Synchronization</h3>
                    <!-- Other content here -->
                </div>
                ''', unsafe_allow_html=True)

                mysql_db = st.selectbox(
                    "MySQL Database:",
                    ["pturf2024", "pturf2025", "custom"]
                )

                if mysql_db == "custom":
                    custom_db = st.text_input("Custom Database Name:")
                    mysql_db = custom_db if custom_db else "pturf2025"

                st.info("This will synchronize race data from MySQL to SQLite database.")

                if st.button("🚀 Start Sync", key="sync_btn"):
                    if mysql_db:
                        mysql_sqlite_sync(mysql_db)
                    else:
                        mysql_sqlite_sync(custom_db)

                st.markdown('</div>', unsafe_allow_html=True)
            elif operation == "⚙️ Update Config.yaml":
                st.markdown('''
                <div class="config-panel">
                    <h3> ⚙️ Configuration Management</h3>
                    <!-- Other content here -->
                </div>
                ''', unsafe_allow_html=True)

                # Load config on first visit or refresh
                if not st.session_state.config_loaded:
                    try:
                        st.session_state.helper.load_config()
                        st.session_state.config_loaded = True
                        log_output("Configuration loaded successfully", "success")
                    except Exception as e:
                        log_output(f"Error loading config: {str(e)}", "error")

                # Refresh button
                if st.button("🔄 Refresh Config", key="refresh_config"):
                    try:
                        st.session_state.helper.load_config()
                        log_output("Configuration refreshed", "success")
                        st.rerun()
                    except Exception as e:
                        log_output(f"Error refreshing config: {str(e)}", "error")

                if st.session_state.config_loaded:
                    # Create tabs for different config sections
                    tab1, tab2, tab3 = st.tabs(["📝 Edit Settings", "📋 View Full Config", "💾 Advanced"])

                    with tab1:
                        config_sections = st.session_state.helper.get_config_sections()

                        # Base Configuration
                        st.markdown("**Base Settings**")
                        col1_base, col2_base = st.columns(2)
                        with col1_base:
                            if config_sections['base']:
                                db_options = st.session_state.helper.get_sqlite_databases()
                                current_active = config_sections['base'].get('active_db', 'full')

                                if current_active in db_options:
                                    active_db_index = db_options.index(current_active)
                                else:
                                    active_db_index = 0

                                new_active_db = st.selectbox(
                                    "Active Database:",
                                    options=db_options,
                                    index=active_db_index,
                                    key="active_db_select"
                                )

                        with col2_base:
                            if config_sections['features']:
                                new_embedding_dim = st.number_input(
                                    "Embedding Dimension:",
                                    min_value=4, max_value=64,
                                    value=config_sections['features'].get('embedding_dim', 8),
                                    key="embedding_dim_input"
                                )

                        # LSTM Configuration
                        st.markdown("**LSTM Settings**")
                        col1_lstm, col2_lstm = st.columns(2)
                        with col1_lstm:
                            if config_sections['lstm']:
                                new_seq_length = st.number_input(
                                    "Sequence Length:",
                                    min_value=1, max_value=20,
                                    value=config_sections['lstm'].get('sequence_length', 5),
                                    key="seq_length_input"
                                )

                        with col2_lstm:
                            if config_sections['lstm']:
                                new_step_size = st.number_input(
                                    "Step Size:",
                                    min_value=1, max_value=5,
                                    value=config_sections['lstm'].get('step_size', 1),
                                    key="step_size_input"
                                )

                        # Dataset Configuration
                        st.markdown("**Dataset Settings**")
                        col1_dataset, col2_dataset = st.columns(2)
                        with col1_dataset:
                            if config_sections['dataset']:
                                new_test_size = st.slider(
                                    "Test Size:",
                                    min_value=0.1, max_value=0.5, step=0.05,
                                    value=config_sections['dataset'].get('test_size', 0.2),
                                    key="test_size_input"
                                )

                        with col2_dataset:
                            if config_sections['dataset']:
                                new_val_size = st.slider(
                                    "Validation Size:",
                                    min_value=0.05, max_value=0.3, step=0.05,
                                    value=config_sections['dataset'].get('val_size', 0.1),
                                    key="val_size_input"
                                )

                        # Cache Configuration
                        st.markdown("**Cache Settings**")
                        if config_sections['cache']:
                            new_use_cache = st.checkbox(
                                "Enable Cache",
                                value=config_sections['cache'].get('use_cache', True),
                                key="use_cache_input"
                            )

                        # Save button
                        if st.button("💾 Save Configuration", key="save_config_btn"):
                            try:
                                # Prepare updates
                                updates = {}
                                if 'base' in config_sections:
                                    updates['base'] = {'active_db': new_active_db}
                                if 'features' in config_sections:
                                    updates['features'] = {'embedding_dim': new_embedding_dim}
                                if 'lstm' in config_sections:
                                    updates['lstm'] = {'sequence_length': new_seq_length, 'step_size': new_step_size}
                                if 'dataset' in config_sections:
                                    updates['dataset'] = {'test_size': new_test_size, 'val_size': new_val_size}
                                if 'cache' in config_sections:
                                    updates['cache'] = {'use_cache': new_use_cache}

                                # Update each section
                                config_copy = st.session_state.helper._config_data.copy()
                                for section, section_updates in updates.items():
                                    if section in config_copy:
                                        config_copy[section].update(section_updates)

                                st.session_state.helper.save_config(config_copy)
                                log_output("Configuration saved successfully", "success")
                                st.rerun()
                            except Exception as e:
                                log_output(f"Error saving config: {str(e)}", "error")

                    with tab2:
                        st.subheader("Complete Configuration")
                        st.code(st.session_state.helper.get_config_json(), language="json")

                    with tab3:
                        st.subheader("Advanced Options")
                        st.warning("⚠️ Advanced configuration editing - use with caution")

                        # Text area for manual editing
                        config_text = st.text_area(
                            "Edit YAML directly:",
                            value=yaml.dump(st.session_state.helper._config_data, default_flow_style=False),
                            height=300,
                            key="manual_config_edit"
                        )

                        if st.button("💾 Save Manual Changes", key="save_manual_config"):
                            try:
                                manual_config = yaml.safe_load(config_text)
                                st.session_state.helper.save_config(manual_config)
                                log_output("Manual configuration saved successfully", "success")
                                st.rerun()
                            except yaml.YAMLError as e:
                                log_output(f"Invalid YAML format: {str(e)}", "error")
                            except Exception as e:
                                log_output(f"Error saving manual config: {str(e)}", "error")
                else:
                    st.error("Unable to load configuration file. Please ensure config.yaml exists.")
                    if st.button("🔄 Retry Loading Config", key="retry_config"):
                        try:
                            st.session_state.helper.load_config()
                            st.session_state.config_loaded = True
                            log_output("Configuration loaded successfully", "success")
                            st.rerun()
                        except Exception as e:
                            log_output(f"Error loading config: {str(e)}", "error")

                st.markdown('</div>', unsafe_allow_html=True)

            elif operation == "🎯 Execute Full Training":
                st.markdown('''
                <div class="config-panel">
                    <h3> 🎯 Full Model Training</h3>
                </div>
                ''', unsafe_allow_html=True)

                # Training Progress Section
                st.markdown("**Training Progress**")
                training_container = st.container()

                if st.button("🚀 Start Training", key="train_btn"):
                    with training_container:
                        # Create progress bar and status text
                        progress_bar = st.progress(0)
                        status_text = st.empty()
                        execute_full_training(progress_bar, status_text)

                st.markdown('</div>', unsafe_allow_html=True)

            elif operation == "🎲 Execute Prediction":

                st.markdown('''
                <div class="config-panel">
                    <h3>🎲 Race Prediction</h3>
                </div>
                ''', unsafe_allow_html=True)

                # Refresh races button
                date=st.date_input(label="Date to sync")
                if st.button("🔄 Refresh Races", key="refresh_races"):
                    st.session_state.helper.sync_daily_races(date)

                    # Get daily races
                daily_races = st.session_state.helper.get_daily_races()

                if daily_races:
                    st.markdown(f"{len(daily_races)}e races Today")
                # Create DataFrame for display
                races_df = []
                for race in daily_races:
                    # Handle Race name with quinte indicator
                    race_name = f"R{race.get('prix', 'N/A')} - {race.get('prixnom', 'Unknown')}"
                    if race.get('quinte', 0) == 1:
                        race_name = f"🌟 {race_name}"

                    status_indicators = []
                    if race.get('has_processed_data', 0):
                        status_indicators.append("✅ Processed")
                    if race.get('has_predictions', 0):
                        status_indicators.append("🔮 Predicted")
                    else:
                        race["prediction_results"]= json.dumps({"predicted_arriv": "N/A"})
                    if race.get('has_results', 0):
                        status_indicators.append("🏁 Results")
                    races_df.append({
                        "Date": race['jour'],
                        "Race ID": race['comp'],
                        "Track": race['hippo'],
                        "Race": race_name,
                        "Prediction": json.loads(race["prediction_results"]).get("predicted_arriv"),
                        "Type": race.get('typec', 'N/A'),
                        "Status": " | ".join(status_indicators) if status_indicators else "⏳ Pending"
                    })
                if races_df:
                    # Display races table with selection
                    edited_df = st.data_editor(
                        races_df,
                        column_config={
                            "Select": st.column_config.CheckboxColumn(
                                "Select for Prediction",
                                help="Select races to predict",
                                default=False,
                            )
                        },
                        disabled=["Date", "Race ID", "Track", "Race", "Prediction", "Type", "Status"],
                        hide_index=True,
                        key="races_editor"
                    )
                # Get races needing predictions
                races_needing_prediction = st.session_state.helper.get_races_needing_prediction()
                st.markdown(f"**{len(races_needing_prediction)} races need predictions**")
                if st.button("🔮 Predict All New Races", key="predict_all"):
                    if races_needing_prediction:
                        # Create progress bar and status text
                        progress_bar = st.progress(0)
                        status_text = st.empty()
                        execute_predictions(None, progress_bar, status_text, force_reprediction=False)
                        st.rerun()
                    else:
                        st.info("No races need predictions")
                if st.button(" 🔁Force Reprediction All"):
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    execute_predictions(None, progress_bar, status_text, force_reprediction=True)
                    st.rerun()

                st.markdown('</div>', unsafe_allow_html=True)
            elif operation == "📈 Execute Evaluation":
                st.markdown('''
                <div class="config-panel">
                    <h3>📈 Comprehensive Prediction Evaluation</h3>
                </div>
                ''', unsafe_allow_html=True)

                st.info("Evaluate all races with both predictions and results using advanced analytics")

                left, right = st.columns(2)
                with left:
                   include_charts = st.checkbox("Include Visualization Charts", value=True)
                with right:
                   include_recommendations = st.checkbox("Include Betting Recommendations", value=True)

                # Evaluation button
                if st.button("📈 Run Comprehensive Evaluation", key="eval_btn"):
                # Create progress bar and status text
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    execute_comprehensive_evaluation(progress_bar, status_text)
                    st.rerun()

            # Display results if available
                if 'evaluation_results' in st.session_state and st.session_state.evaluation_results:
                   results = st.session_state.evaluation_results
                   if results['success']:
                    st.success("✅ Evaluation completed successfully!")

                    # Display charts if requested
                    if include_charts and 'chart_data' in results:
                        display_evaluation_charts(results['chart_data'])

                    # Raw metrics summary
                    with st.expander("📊 Raw Metrics Summary"):
                        metrics = results['metrics']
                        st.json({
                        "total_races": metrics.total_races,
                        "races_evaluated": metrics.races_evaluated,
                        "overall_winner_accuracy": f"{metrics.overall_winner_accuracy:.2%}",
                        "overall_podium_accuracy": f"{metrics.overall_podium_accuracy:.2%}",
                        "total_winning_bets": metrics.total_winning_bets
                        })

                    # Detailed bet type wins
                    with st.expander("🎯 Detailed Bet Type Analysis"):
                        bet_wins = results['bet_type_wins']
                        for bet_type, races in bet_wins.items():
                            if bet_type:  # Only show bet types with wins
                                st.markdown(f"**{bet_type.replace('_', ' ').title()}:** {len(races)} winning races")
                                race_details = []
                                for race in races:  # Show first 5 races
                                    race_info = race.race_info
                                    race_details.append(
                                        f"- {race_info['comp']}:{race_info['jour']} {race_info['hippo']} R{race_info['prix']}"
                                    )
                                if race_details:
                                    st.markdown("\n".join(race_details))
                            else:
                                st.error(f"❌ Evaluation failed: {results['message']}")

                st.markdown('</div>', unsafe_allow_html=True)

            elif operation == "✨ AI Insight":
                st.markdown('''
                <div class="config-panel">
                    <h3>🤖 AI Betting Insights</h3>
                </div>
                ''', unsafe_allow_html=True)
                
                st.info("Get intelligent betting advice powered by AI analysis of your prediction results and market odds")
                
                # AI Insight tabs
                tab1, tab2, tab3 = st.tabs(["📊 Daily Betting Advice","🌟 Quinte Race Advice", "🏇 Race-Specific Advice"])
                
                with tab1:
                    st.markdown("### 📊 Daily Betting Performance Analysis")
                    st.markdown("Get comprehensive daily betting advice based on your model's recent performance")
                    
                    # Configuration options
                    col1, col2 = st.columns(2)
                    with col1:
                        lm_studio_url = st.text_input(
                            "LM Studio URL:", 
                            value="http://localhost:1234", 
                            help="Leave empty to use configuration default"
                        )
                    with col2:
                        verbose_ai = st.checkbox("Enable verbose AI output", value=False)
                    
                    # Generate daily advice button
                    if st.button("🧠 Generate Daily Betting Advice", key="daily_ai_advice"):
                        with st.spinner("Analyzing your prediction results and generating advice..."):
                            # Use empty string if default URL is used
                            url_param = lm_studio_url if lm_studio_url != "http://localhost:1234" else None
                            
                            result = st.session_state.helper.get_ai_betting_advice(
                                lm_studio_url=url_param,
                                verbose=verbose_ai
                            )
                            
                            if result["success"]:
                                st.success("✅ AI betting advice generated successfully!")
                                
                                # Display the AI advice
                                st.markdown("### 🎯 AI Betting Recommendations")
                                st.markdown(result["ai_advice"])
                                
                                # Show evaluation data used
                                with st.expander("📊 Evaluation Data Used"):
                                    eval_data = result["evaluation_data"]
                                    
                                    # Summary metrics
                                    if 'summary_metrics' in eval_data:
                                        st.markdown("**Performance Summary:**")
                                        metrics = eval_data['summary_metrics']
                                        
                                        col1, col2, col3 = st.columns(3)
                                        with col1:
                                            st.metric("Total Races", metrics.get('total_races', 0))
                                        with col2:
                                            st.metric("Winner Accuracy", f"{metrics.get('winner_accuracy', 0):.1%}")
                                        with col3:
                                            st.metric("Podium Accuracy", f"{metrics.get('podium_accuracy', 0):.1%}")
                                    
                                    # Bet performance
                                    if 'pmu_summary' in eval_data:
                                        st.markdown("**Bet Type Performance:**")
                                        pmu_summary = eval_data['pmu_summary']
                                        
                                        bet_df = []
                                        for bet_type, rate in pmu_summary.items():
                                            if rate > 0:
                                                bet_df.append({
                                                    'Bet Type': bet_type.replace('_rate', '').replace('_', ' ').title(),
                                                    'Win Rate': f"{rate:.1%}"
                                                })
                                        
                                        if bet_df:
                                            st.dataframe(pd.DataFrame(bet_df), hide_index=True)
                                
                            else:
                                st.error(f"❌ Failed to generate AI advice: {result['message']}")
                                if 'error' in result:
                                    st.error(f"Error details: {result['error']}")
                
                with tab2:
                    st.markdown("### 🌟 Quinté+ Specialized Betting Strategy")
                    st.markdown("Get **3 refined betting recommendations** specifically optimized for quinté races")
                    
                    # Info box about quinte focus
                    st.info("🎯 **Quinté+ Focus**: This analysis specifically targets quinté races and provides 3 structured betting recommendations: Conservative, Balanced, and Aggressive strategies based on historical quinte performance.")
                    
                    # Configuration options
                    col1, col2 = st.columns(2)
                    with col1:
                        quinte_lm_studio_url = st.text_input(
                            "LM Studio URL:", 
                            value="http://localhost:1234", 
                            help="Leave empty to use configuration default",
                            key="quinte_lm_url"
                        )
                    with col2:
                        quinte_verbose_ai = st.checkbox("Enable verbose AI output", value=False, key="quinte_verbose")
                    
                    # Generate quinte advice button
                    if st.button("🌟 Generate Quinté+ Betting Strategy", key="quinte_ai_advice"):
                        with st.spinner("Analyzing quinté performance and generating 3 refined betting recommendations..."):
                            # Use empty string if default URL is used
                            url_param = quinte_lm_studio_url if quinte_lm_studio_url != "http://localhost:1234" else None
                            
                            result = st.session_state.helper.get_ai_quinte_advice(
                                lm_studio_url=url_param,
                                verbose=quinte_verbose_ai
                            )
                            
                            if result["success"]:
                                st.success("✅ Quinté+ betting strategy generated successfully!")
                                
                                # Display the AI advice
                                st.markdown("### 🎯 3 Refined Quinté+ Betting Recommendations")
                                st.markdown(result["ai_advice"])
                                
                                # Show evaluation data used
                                with st.expander("📊 Quinté+ Performance Data Used"):
                                    eval_data = result["evaluation_data"]
                                    
                                    # Summary metrics
                                    if 'summary_metrics' in eval_data:
                                        st.markdown("**Overall Performance Summary:**")
                                        metrics = eval_data['summary_metrics']
                                        
                                        col1, col2, col3 = st.columns(3)
                                        with col1:
                                            st.metric("Total Races", metrics.get('total_races', 0))
                                        with col2:
                                            st.metric("Winner Accuracy", f"{metrics.get('winner_accuracy', 0):.1%}")
                                        with col3:
                                            st.metric("Podium Accuracy", f"{metrics.get('podium_accuracy', 0):.1%}")
                                    
                                    # Quinte specific analysis
                                    if 'quinte_analysis' in eval_data:
                                        st.markdown("**Quinté+ Specific Analysis:**")
                                        quinte_data = eval_data['quinte_analysis']
                                        
                                        # Display key quinte metrics
                                        if 'total_quinte_races' in quinte_data:
                                            st.metric("Total Quinté+ Races", quinte_data['total_quinte_races'])
                                        
                                        # Show betting scenarios performance
                                        if 'betting_scenarios' in quinte_data:
                                            st.markdown("**Horse Selection Strategies:**")
                                            scenarios = quinte_data['betting_scenarios']
                                            
                                            scenario_df = []
                                            for scenario, data in scenarios.items():
                                                scenario_df.append({
                                                    'Strategy': scenario.replace('_', ' ').title(),
                                                    'Wins': data.get('wins', 0),
                                                    'Total': data.get('total', 0),
                                                    'Win Rate': f"{data.get('win_rate', 0):.1%}"
                                                })
                                            
                                            if scenario_df:
                                                st.dataframe(pd.DataFrame(scenario_df), hide_index=True)
                                    
                                    # PMU summary focused on quinte
                                    if 'pmu_summary' in eval_data:
                                        st.markdown("**Quinté+ Bet Type Performance:**")
                                        pmu_summary = eval_data['pmu_summary']
                                        
                                        quinte_bet_df = []
                                        quinte_bets = [
                                            ('quinte_exact_rate', 'Quinté+ Exact'),
                                            ('quinte_desordre_rate', 'Quinté+ Désordre'),
                                            ('bonus4_rate', 'Bonus 4'),
                                            ('bonus3_rate', 'Bonus 3'),
                                            ('multi4_rate', 'Multi 4')
                                        ]
                                        
                                        for bet_key, bet_name in quinte_bets:
                                            if bet_key in pmu_summary and pmu_summary[bet_key] > 0:
                                                quinte_bet_df.append({
                                                    'Bet Type': bet_name,
                                                    'Win Rate': f"{pmu_summary[bet_key]:.1%}"
                                                })
                                        
                                        if quinte_bet_df:
                                            st.dataframe(pd.DataFrame(quinte_bet_df), hide_index=True)
                                        else:
                                            st.info("No quinté bet wins recorded in current data")
                                
                            else:
                                st.error(f"❌ Failed to generate quinté betting strategy: {result['message']}")
                                if 'error' in result:
                                    st.error(f"Error details: {result['error']}")
                
                with tab3:
                    st.markdown("### 🏇 Race-Specific AI Analysis")
                    st.markdown("Get detailed AI advice for specific races including odds analysis and betting recommendations")
                    
                    # Get available races
                    daily_races = st.session_state.helper.get_daily_races()
                    
                    if daily_races:
                        # Filter races with predictions
                        races_with_predictions = [race for race in daily_races if race.get('has_predictions', 0) == 1]
                        
                        if races_with_predictions:
                            # Race selection
                            race_options = []
                            for race in races_with_predictions:
                                race_name = f"{race['hippo']} - R{race.get('prix', 'N/A')} - {race.get('prixnom', 'Unknown')}"
                                if race.get('quinte', 0) == 1:
                                    race_name = f"🌟 {race_name}"
                                race_options.append(race_name)
                            
                            selected_race_idx = st.selectbox(
                                "Select a race for AI analysis:",
                                range(len(race_options)),
                                format_func=lambda x: race_options[x]
                            )
                            
                            selected_race = races_with_predictions[selected_race_idx]
                            
                            # Configuration for race analysis
                            col1, col2 = st.columns(2)
                            with col1:
                                race_lm_studio_url = st.text_input(
                                    "LM Studio URL:", 
                                    value="http://localhost:1234", 
                                    help="Leave empty to use configuration default",
                                    key="race_lm_url"
                                )
                            with col2:
                                race_verbose_ai = st.checkbox("Enable verbose AI output", value=False, key="race_verbose")
                            
                            # Generate race advice button
                            if st.button("🧠 Generate Race Analysis", key="race_ai_advice"):
                                with st.spinner(f"Analyzing race {selected_race['comp']} and generating advice..."):
                                    # Use empty string if default URL is used
                                    url_param = race_lm_studio_url if race_lm_studio_url != "http://localhost:1234" else None
                                    
                                    result = st.session_state.helper.get_ai_race_advice(
                                        race_comp=selected_race['comp'],
                                        lm_studio_url=url_param,
                                        verbose=race_verbose_ai
                                    )
                                    
                                    if result["success"]:
                                        st.success(f"✅ AI race analysis generated for {selected_race['comp']}!")
                                        
                                        # Display the AI advice
                                        st.markdown("### 🎯 AI Race Analysis & Recommendations")
                                        st.markdown(result["ai_advice"])
                                        
                                        # Show race data and predictions used
                                        with st.expander("📊 Race Data & Predictions Used"):
                                            race_data = result["race_data"]
                                            predictions = result["predictions"]
                                            
                                            # Race information
                                            st.markdown("**Race Information:**")
                                            col1, col2, col3 = st.columns(3)
                                            with col1:
                                                st.metric("Date", race_data.get('jour', 'N/A'))
                                            with col2:
                                                st.metric("Track", race_data.get('hippo', 'N/A'))
                                            with col3:
                                                st.metric("Race", f"R{race_data.get('prix', 'N/A')}")
                                            
                                            # Predictions
                                            if predictions and 'predictions' in predictions:
                                                st.markdown("**Top Predictions:**")
                                                pred_data = predictions['predictions']
                                                
                                                # Create DataFrame from predictions
                                                pred_df = []
                                                for pred in pred_data[:8]:  # Show top 8
                                                    pred_df.append({
                                                        'Horse': pred.get('numero', 'N/A'),
                                                        'Predicted Rank': pred.get('predicted_rank', pred.get('predicted_position', 'N/A')),
                                                        'Confidence': f"{pred.get('confidence', pred.get('predicted_prob', 0)):.2%}" if isinstance(pred.get('confidence', pred.get('predicted_prob', 0)), (int, float)) else 'N/A'
                                                    })
                                                
                                                if pred_df:
                                                    st.dataframe(pd.DataFrame(pred_df), hide_index=True)
                                    
                                    else:
                                        st.error(f"❌ Failed to generate race analysis: {result['message']}")
                                        if 'error' in result:
                                            st.error(f"Error details: {result['error']}")
                        else:
                            st.warning("No races with predictions found. Please run predictions first.")
                            st.info("Go to '🎲 Execute Prediction' to generate predictions for today's races.")
                    else:
                        st.warning("No races available for analysis.")
                        st.info("Go to '🎲 Execute Prediction' to sync and predict today's races.")
                
                st.markdown('</div>', unsafe_allow_html=True)

            elif operation == "🔄 Incremental Training":

                st.markdown('<div class="config-panel">', unsafe_allow_html=True)

                st.markdown("### 🔄 Incremental Training & Regression Enhancement")

                st.info("Process completed races with predictions and results to improve model performance")

                # Training parameters

                left, right = st.columns(2)

                with left:
                    date_from = st.date_input(

                        "Start Date:",

                        value=(datetime.now() - timedelta(days=30)).date(),

                        key="incr_date_from"

                    )

                    update_model = st.checkbox(

                        "Update Base Model",

                        value=True,

                        help="Whether to update the base model with new data"

                    )

                with right:
                    date_to = st.date_input(

                        "End Date:",

                        value=datetime.now().date(),

                        key="incr_date_to"

                    )

                    create_enhanced = st.checkbox(

                        "Create Enhanced Model",

                        value=True,

                        help="Create enhanced model with error correction"

                    )

                # Advanced options

                with st.expander("🔧 Advanced Options"):
                    limit_races = st.number_input(

                        "Limit Races (0 = no limit):",

                        min_value=0,

                        value=0,

                        help="Maximum number of races to process"

                    )

                    archive_after = st.checkbox(

                        "Archive Races After Training",

                        value=True,

                        help="Move processed races from daily_race to historical_races"

                    )

                # Show current state

                races_with_results = st.session_state.helper.get_races_with_results(

                    date_from.strftime('%Y-%m-%d'),

                    date_to.strftime('%Y-%m-%d')

                )

                if races_with_results:

                    st.success(f"Found {len(races_with_results)} races with predictions and results")

                    # Show sample of races

                    with st.expander("📋 Races Ready for Training"):

                        sample_df = pd.DataFrame(races_with_results[:10])  # Show first 10

                        if not sample_df.empty:

                            display_cols = ['comp', 'jour', 'hippo', 'prix', 'partant']

                            available_cols = [col for col in display_cols if col in sample_df.columns]

                            st.dataframe(sample_df[available_cols], hide_index=True)

                            if len(races_with_results) > 10:
                                st.info(f"... and {len(races_with_results) - 10} more races")

                else:

                    st.warning("No races with both predictions and results found for the selected date range")

                # Training execution

                if st.button("🚀 Start Incremental Training", key="incr_btn"):

                    if races_with_results:

                        # Create progress bar and status text

                        progress_bar = st.progress(0)

                        status_text = st.empty()

                        execute_incremental_training(

                            date_from.strftime('%Y-%m-%d'),

                            date_to.strftime('%Y-%m-%d'),

                            limit_races if limit_races > 0 else None,

                            update_model,

                            create_enhanced,

                            archive_after,

                            progress_bar,

                            status_text

                        )

                        # Display results if stored in session state

                        if 'incremental_results' in st.session_state:
                            results = st.session_state.incremental_results

                            if results.get('success'):
                                st.success("✅ Incremental training completed!")

                                # Display training metrics
                                training_results = results.get('training_results', {})

                                if training_results:
                                    st.markdown("### 📊 Training Results")

                                    # Performance metrics
                                    if 'performance_analysis' in training_results:
                                        perf = training_results['performance_analysis']

                                        left, center, right = st.columns(3)
                                        with left:
                                            st.metric("Sample Size", perf.get('sample_size', 0))
                                        with center:
                                            st.metric("MAE", f"{perf.get('overall_mae', 0):.4f}")
                                        with right:
                                            st.metric("RMSE", f"{perf.get('overall_rmse', 0):.4f}")

                                    # RF Model improvement metrics
                                    if 'rf_training' in training_results:
                                        rf_results = training_results['rf_training']

                                        if rf_results.get('status') == 'success':
                                            improvement = rf_results.get('improvement', {})
                                            st.markdown("### 🎯 RF Model Improvement")

                                            left, right = st.columns(2)
                                            with left:
                                                mae_improvement = improvement.get('mae_improvement_pct', 0)
                                                st.metric(
                                                    "RF MAE Improvement",
                                                    f"{mae_improvement:.2f}%",
                                                    delta=f"{mae_improvement:.2f}%"
                                                )
                                            with right:
                                                rmse_improvement = improvement.get('rmse_improvement_pct', 0)
                                                st.metric(
                                                    "RF RMSE Improvement",
                                                    f"{rmse_improvement:.2f}%",
                                                    delta=f"{rmse_improvement:.2f}%"
                                                )

                                            # Significance indicator
                                            if improvement.get('significant', False):
                                                st.success("🎉 RF model shows significant improvement!")
                                            else:
                                                st.info("ℹ️ RF improvement below threshold.")

                                    # LSTM Model improvement metrics
                                    if 'lstm_training' in training_results:
                                        lstm_results = training_results['lstm_training']

                                        if lstm_results.get('status') == 'success':
                                            lstm_improvement = lstm_results.get('improvement', {})
                                            st.markdown("### 🧠 LSTM Model Improvement")

                                            left, right = st.columns(2)
                                            with left:
                                                lstm_mae_improvement = lstm_improvement.get('mae_improvement_pct', 0)
                                                st.metric(
                                                    "LSTM MAE Improvement",
                                                    f"{lstm_mae_improvement:.2f}%",
                                                    delta=f"{lstm_mae_improvement:.2f}%"
                                                )
                                            with right:
                                                lstm_rmse_improvement = lstm_improvement.get('rmse_improvement_pct', 0)
                                                st.metric(
                                                    "LSTM RMSE Improvement",
                                                    f"{lstm_rmse_improvement:.2f}%",
                                                    delta=f"{lstm_rmse_improvement:.2f}%"
                                                )

                                            if lstm_improvement.get('significant', False):
                                                st.success("🎉 LSTM model shows significant improvement!")
                                            else:
                                                st.info("ℹ️ LSTM improvement below threshold.")

                                        elif lstm_results.get('status') == 'skipped':
                                            st.warning("⚠️ LSTM training was skipped")
                                            if lstm_results.get('message'):
                                                st.info(f"Reason: {lstm_results['message']}")

                                    # Model saving results
                                    if 'model_saved' in training_results:
                                        model_saved = training_results['model_saved']

                                        if model_saved.get('status') == 'success':
                                            st.markdown("### 💾 Model Saving")
                                            version = model_saved.get('version', 'unknown')
                                            st.success(f"✅ New model saved: **{version}**")

                                            # Show which models were updated
                                            models_updated = model_saved.get('models_updated', {})
                                            model_sources = model_saved.get('model_sources', {})

                                            left, right = st.columns(2)
                                            with left:
                                                if models_updated.get('rf', False):
                                                    st.info("🔄 RF model: Retrained")
                                                else:
                                                    st.info("📋 RF model: Copied from base")

                                            with right:
                                                if models_updated.get('lstm', False):
                                                    st.info("🔄 LSTM model: Retrained")
                                                else:
                                                    st.info("📋 LSTM model: Copied from base")

                                        else:
                                            st.warning("⚠️ Models were not saved")
                                            if model_saved.get('reason'):
                                                st.info(f"Reason: {model_saved['reason']}")

                                    # Archive results
                                    if 'races_archived' in training_results:
                                        archive = training_results['races_archived']

                                        if archive.get('status') == 'success':
                                            archived_count = archive.get('successful', 0)
                                            st.success(f"📁 Archived {archived_count} races to historical data")
                                        elif archive.get('status') == 'skipped':
                                            st.info("📁 Race archiving was skipped")
                                        else:
                                            st.warning("⚠️ Race archiving failed")

                                    # Execution summary
                                    execution_time = training_results.get('execution_time', 0)
                                    races_processed = training_results.get('races_processed', 0)
                                    training_samples = training_results.get('training_samples', 0)

                                    st.markdown("### ⏱️ Execution Summary")
                                    left, center, right = st.columns(3)
                                    with left:
                                        st.metric("Races Processed", races_processed)
                                    with center:
                                        st.metric("Training Samples", training_samples)
                                    with right:
                                        st.metric("Execution Time", f"{execution_time:.1f}s")

                            else:
                                st.error(f"❌ Training failed: {results.get('message', 'Unknown error')}")

    with col2:
        # Status and output panel
        st.markdown('''
                        <div class="output-panel">
                            <h3> 📋 System Status</h3>
                            <!-- Other content here -->
                        </div>
                        ''', unsafe_allow_html=True)


        # Load config if not already loaded for status
        if not st.session_state.config_loaded:
            load_config()

        # System status indicators
        last_training, model_version = st.session_state.helper.get_last_training_info()
        col_status1, col_status2 = st.columns(2)
        with col_status1:
            st.metric("Active DB", st.session_state.helper.get_active_db())
            st.metric("Last Training", last_training)
        with col_status2:
            st.metric("Model Version", model_version)
        
        # Alternative Models Status
        st.markdown("### 🤖 Model Status")
        model_info_result = st.session_state.helper.get_prediction_model_info()
        
        if model_info_result.get('success'):
            prediction_info = model_info_result['prediction_info']
        else:
            # Use fallback information if available
            st.warning(f"Model info error: {model_info_result.get('error', 'Unknown error')}")
            prediction_info = model_info_result.get('fallback_info', {})
            
            if not prediction_info:
                st.error("Could not retrieve any model information")
                return
        
        # Display model information (works with both success and fallback data)
        # Legacy Models Status
        st.markdown("**Legacy Models:**")
        legacy_models = prediction_info.get('legacy_models', {})
        legacy_status = legacy_models.get('model_status', {})
        
        col_rf, col_lstm, col_tabnet = st.columns(3)
        with col_rf:
            rf_status = "✅ Loaded" if legacy_status.get('rf_loaded', False) else "❌ Not Found"
            st.metric("Random Forest", rf_status)
        with col_lstm:
            lstm_status = "✅ Loaded" if legacy_status.get('lstm_loaded', False) else "❌ Not Found"
            st.metric("LSTM", lstm_status)
        with col_tabnet:
            tabnet_status = "✅ Loaded" if legacy_status.get('tabnet_loaded', False) else "❌ Not Found"
            st.metric("TabNet", tabnet_status)
        
        # Alternative Models Status
        alt_models = prediction_info.get('alternative_models', {})
        if alt_models.get('enabled'):
            st.markdown("**Alternative Models:**")
            col_trans, col_ens = st.columns(2)
            
            with col_trans:
                trans_info = alt_models.get('transformer', {})
                trans_status = "✅ Loaded" if trans_info.get('loaded', False) else ("⚙️ Configured" if trans_info.get('configured', False) else "❌ Disabled")
                st.metric("Transformer", trans_status)
            
            with col_ens:
                ens_info = alt_models.get('ensemble', {})
                ens_status = "✅ Loaded" if ens_info.get('loaded', False) else ("⚙️ Configured" if ens_info.get('configured', False) else "❌ Disabled")
                st.metric("Ensemble", ens_status)
            
            # System Status Summary
            system_status = prediction_info.get('system_status', 'unknown')
            status_emoji_map = {
                'full_operational': '🟢 Full Operational',
                'legacy_operational': '🟡 Legacy Only',
                'alternative_operational': '🔵 Alternative Only',
                'no_models_loaded': '🔴 No Models',
                'configuration_error': '⚠️ Config Error'
            }
            
            st.markdown(f"**Overall Status:** {status_emoji_map.get(system_status, '⚪ Unknown')}")
        else:
            alt_message = alt_models.get('message', "Alternative models not configured. Enable in config.yaml to see status.")
            st.info(alt_message)

        st.markdown('</div>', unsafe_allow_html=True)

        # Output logs
        if st.button("🗑️ Clear Logs"):
            clear_logs()

        display_logs()


if __name__ == "__main__":
    main()