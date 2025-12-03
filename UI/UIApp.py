import streamlit as st
import pandas as pd
from datetime import datetime, timedelta
import yaml
import json
import os
import time
import sys
from pathlib import Path
from UI.UIhelper import PipelineHelper
from utils.batch_job_manager import BatchJobManager
from utils.env_setup import AppConfig
import plotly.express as px
import plotly.graph_objects as go

# Add project root to path for calibration imports
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

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
def mysql_sqlite_sync(db, quinte=False):
    """Placeholder for MySQL to SQLite sync"""
    sync_type = "Quinte races only" if quinte else "All races"
    log_output(f"Starting MySQL to SQLite synchronization ({sync_type})...", "info")
    from core.orchestrators.mysql_sqlite_sync import sync_data
    sync_data(db, quinte=quinte)
    log_output(f"MySQL to SQLite sync completed successfully! ({sync_type})", "success")


def execute_full_training(progress_bar, status_text):
    """Execute full training using pipeline helper with async support"""

    # Check if training is already running
    if st.session_state.helper.is_training:
        # Training in progress - check for updates
        updates = st.session_state.helper.get_training_updates()

        for update in updates:
            if update['type'] == 'info':
                # Initial thread info
                log_output(f"✅ {update['message']}", "info")
                log_output(f"   Process ID: {update.get('process_id', 'N/A')}", "info")
                log_output(f"   Thread ID: {update.get('thread_id', 'N/A')}", "info")
                log_output(f"   Thread Name: {update.get('thread_name', 'N/A')}", "info")
                log_output(f"   Thread Alive: {update.get('is_alive', 'N/A')}", "info")

            elif update['type'] == 'worker_started':
                log_output(f"🔄 {update['message']}", "info")
                log_output(f"   Thread ID: {update.get('thread_id', 'N/A')}", "info")
                log_output(f"   Start Time: {update.get('start_time', 'N/A')}", "info")

            elif update['type'] == 'progress':
                # Update session state for persistence across reruns
                st.session_state.training_progress = update['percent']
                st.session_state.training_status = update['message']

                # Update UI
                progress_bar.progress(update['percent'] / 100)
                status_text.text(f"Progress: {update['percent']}% - {update['message']}")
                log_output(f"[Thread {update.get('thread_id', 'N/A')}] {update['message']}", "info")

            elif update['type'] == 'complete':
                duration = update.get('duration_seconds', 0)
                duration_str = f"{duration:.1f}s" if duration < 60 else f"{duration/60:.1f}min"

                if update['success']:
                    st.session_state.training_progress = 100
                    st.session_state.training_status = "Training completed!"

                    progress_bar.progress(1.0)
                    status_text.text("Training completed!")
                    log_output(f"✅ {update['message']} (Duration: {duration_str})", "success")
                else:
                    st.session_state.training_progress = 0
                    st.session_state.training_status = f"Training failed: {update.get('message', 'Unknown error')}"

                    progress_bar.progress(0)
                    status_text.text("Training failed!")
                    log_output(f"❌ {update['message']} (Duration: {duration_str})", "error")
                    if 'traceback' in update:
                        log_output(f"Error details: {update['error']}", "error")

                # Clear training state
                st.session_state.training_active = False

            elif update['type'] == 'worker_stopped':
                log_output(f"⏹️ {update['message']}", "info")

        # Keep training active if still running
        if st.session_state.helper.is_training:
            st.session_state.training_active = True
            return
    else:
        # Not training - start new training
        log_output("Starting background training...", "info")

        if st.session_state.helper.start_training_async():
            st.session_state.training_progress = 0
            st.session_state.training_status = "Training started in background..."
            st.session_state.training_active = True

            progress_bar.progress(0)
            status_text.text("Training started in background...")
            log_output("✅ Training started successfully in background", "success")

            # Get and display thread status
            thread_status = st.session_state.helper.get_training_status()
            log_output(f"📊 Training Thread Status:", "info")
            log_output(f"   Process ID: {thread_status.get('process_id', 'N/A')}", "info")
            log_output(f"   Thread ID: {thread_status.get('thread_id', 'N/A')}", "info")
            log_output(f"   Thread Name: {thread_status.get('thread_name', 'N/A')}", "info")
            log_output(f"   Thread Alive: {thread_status.get('thread_alive', 'N/A')}", "info")
        else:
            st.session_state.training_active = False
            log_output("❌ Failed to start training", "error")


def execute_re_blending(all_races=True, date=None, progress_bar=None, status_text=None):
    """Execute re-blending with dynamic weights"""
    if all_races:
        log_output(f"Starting re-blending for ALL races...", "info")
    else:
        log_output(f"Starting re-blending for {date}...", "info")

    def progress_callback(percentage, message):
        if progress_bar:
            progress_bar.progress(percentage / 100)
        if status_text:
            status_text.text(message)

    try:
        result = st.session_state.helper.reblend_with_dynamic_weights(
            date=date,
            all_races=all_races,
            progress_callback=progress_callback
        )

        if result["success"]:
            if progress_bar:
                progress_bar.progress(100)
            races_processed = result.get('races_processed', 0)
            horses_updated = result.get('horses_updated', 0)

            if races_processed == 0:
                if status_text:
                    status_text.text(f"No races found")
                log_output(f"⚠️  No races with predictions found.", "warning")
            else:
                if status_text:
                    status_text.text("Re-blending completed!")
                log_output(result["message"], "success")
                log_output(f"✅ Processed {races_processed} races, updated {horses_updated} horses", "info")

                # Show per-race details if available
                races_detail = result.get('races_detail', [])
                if races_detail:
                    log_output(f"\nWeight changes (first 5 races):", "info")
                    for race_info in races_detail[:5]:
                        race_id = race_info['race_id']
                        horses = race_info['horses_updated']
                        rf_w = race_info['rf_weight']
                        tabnet_w = race_info['tabnet_weight']
                        log_output(f"  {race_id}: {horses} horses (RF={rf_w:.2f}, TabNet={tabnet_w:.2f})", "info")

            st.session_state.reblend_results = result
        else:
            if progress_bar:
                progress_bar.progress(100)
            if status_text:
                status_text.text("Re-blending failed!")
            log_output(result["message"], "error")

    except Exception as e:
        if progress_bar:
            progress_bar.progress(100)
        if status_text:
            status_text.text("Error during re-blending!")
        log_output(f"Re-blending error: {str(e)}", "error")


def execute_predictions_background(selected_races, force_reprediction=False):
    """Execute predictions in background using batch job system"""
    prediction_type = "force reprediction" if force_reprediction else "standard prediction"
    log_output(f"Launching {prediction_type} job in background...", "info")

    try:
        # Initialize job manager
        config = AppConfig()
        db_path = config.get_sqlite_dbpath(config._config.base.active_db)
        job_manager = BatchJobManager(db_path)

        # Launch background job
        job_id = job_manager.launch_job(
            race_ids=selected_races if selected_races else None,
            force_reprediction=force_reprediction,
            workers=3,  # Use all CPU cores
            chunk_size=50,
            max_memory_mb=4096
        )

        log_output(f"✅ Job launched successfully!", "success")
        log_output(f"   Job ID: {job_id}", "info")
        log_output(f"   Monitor progress below...", "info")

        # Store job ID in session state
        if 'active_prediction_jobs' not in st.session_state:
            st.session_state.active_prediction_jobs = []
        st.session_state.active_prediction_jobs.append(job_id)

        return job_id

    except Exception as e:
        log_output(f"❌ Failed to launch prediction job: {str(e)}", "error")
        return None


def display_active_prediction_jobs():
    """Display active prediction jobs with progress"""
    # Initialize job manager
    config = AppConfig()
    db_path = config.get_sqlite_dbpath(config._config.base.active_db)
    job_manager = BatchJobManager(db_path)

    # Get active jobs
    active_jobs = job_manager.get_active_jobs()

    if not active_jobs:
        return False  # No active jobs

    st.markdown("### 🚀 Active Prediction Jobs")

    for job in active_jobs:
        with st.container():
            # Job header
            left, middle, right = st.columns([3, 1, 1])

            with left:
                st.markdown(f"**Job:** `{job.job_id}`")

            with middle:
                if job.duration_seconds:
                    duration_str = f"{job.duration_seconds:.1f}s" if job.duration_seconds < 60 else f"{job.duration_seconds/60:.1f}min"
                    st.markdown(f"**Duration:** {duration_str}")

            with right:
                throughput = job.processed_races / job.duration_seconds if job.duration_seconds and job.duration_seconds > 0 else 0
                if throughput > 0:
                    st.markdown(f"**Speed:** {throughput:.1f} races/s")

            # Progress bar
            st.progress(job.progress / 100.0)
            st.caption(f"{job.progress}% - {job.message}")

            # Stats
            stat1, stat2, stat3, stat4 = st.columns(4)
            with stat1:
                st.metric("Total Races", job.total_races)
            with stat2:
                st.metric("Processed", job.processed_races)
            with stat3:
                st.metric("Successful", job.successful_races)
            with stat4:
                remaining = job.total_races - job.processed_races
                st.metric("Remaining", remaining)

            st.markdown("---")

    return True  # Has active jobs


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

    stat1, stat2, stat3, stat4 = st.columns(4)
    with stat1:
        st.metric("Total Races", summary['total_races'])
    with stat2:
        st.metric("Winner Accuracy", f"{summary['winner_accuracy']:.1f}%")
    with stat3:
        st.metric("Podium Accuracy", f"{summary['podium_accuracy']:.1f}%")
    with stat4:
        st.metric("Total Winning Bets", summary['total_winning_bets'])

    # Bet Type Performance Chart
    if chart_data['bet_performance']:
        st.markdown("### 🎯 Bet Type Performance")

        bet_df = pd.DataFrame(chart_data['bet_performance'])

        # Create side-by-side charts
        col1,col2 = st.columns(2)

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


def execute_quinte_incremental_training(date_from, date_to, limit, focus_on_failures,
                                        progress_bar, status_text):
    """Execute quinté incremental training using specialized pipeline"""
    log_output("Starting quinté incremental training...", "info")

    def progress_callback(percentage, message):
        """Callback function to update progress"""
        progress_bar.progress(percentage / 100)
        status_text.text(f"Progress: {percentage}% - {message}")
        log_output(message, "info")

    try:
        result = st.session_state.helper.execute_quinte_incremental_training(
            date_from=date_from,
            date_to=date_to,
            limit=limit,
            focus_on_failures=focus_on_failures,
            progress_callback=progress_callback
        )

        if result["success"]:
            progress_bar.progress(1.0)
            status_text.text("Quinté incremental training completed!")
            log_output(result["message"], "success")
            st.session_state.quinte_incremental_results = result
        else:
            progress_bar.progress(0)
            status_text.text("Quinté training failed!")
            log_output(result["message"], "error")
            st.session_state.quinte_incremental_results = result

    except Exception as e:
        progress_bar.progress(0)
        status_text.text("Quinté training error!")
        log_output(f"Quinté incremental training error: {str(e)}", "error")
        import traceback
        st.session_state.quinte_incremental_results = {
            "success": False,
            "message": str(e),
            "error": traceback.format_exc()
        }


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
            "🏇 Quinté Predictions",
            "📈 Execute Evaluation",
            "⚖️ Model Weight Analysis",
            "🔄 Incremental Training",
            "🏆 Quinté Incremental Training",
            "🎯 Execute Full Training",
            "🔄 MySQL ↔ SQLite Sync",
            "⚙️ Update Config.yaml",
        ],
        index=0
    )

    # Main content area
    main, status = st.columns([2, 1])

    with main:
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

                # Quinte filter radio box
                sync_mode = st.radio(
                    "Sync Mode:",
                    ["All Races", "Quinte Races Only"],
                    help="Select whether to sync all races or only quinte races"
                )

                quinte_only = (sync_mode == "Quinte Races Only")

                if quinte_only:
                    st.info("🌟 This will synchronize ONLY quinte races from MySQL to the historical_quinte table.")
                else:
                    st.info("This will synchronize all race data from MySQL to the historical_races table.")

                if st.button("🚀 Start Sync", key="sync_btn"):
                    if mysql_db:
                        mysql_sqlite_sync(mysql_db, quinte_only)
                    else:
                        mysql_sqlite_sync(custom_db, quinte_only)

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

                        # TabNet Configuration
                        st.markdown("**TabNet Settings**")
                        col1_tabnet, col2_tabnet = st.columns(2)
                        with col1_tabnet:
                            if config_sections.get('tabnet'):
                                new_n_d = st.number_input(
                                    "Decision Width (n_d):",
                                    min_value=8, max_value=64,
                                    value=config_sections['tabnet'].get('n_d', 32),
                                    key="tabnet_n_d_input"
                                )

                        with col2_tabnet:
                            if config_sections.get('tabnet'):
                                new_n_a = st.number_input(
                                    "Attention Width (n_a):",
                                    min_value=8, max_value=64,
                                    value=config_sections['tabnet'].get('n_a', 32),
                                    key="tabnet_n_a_input"
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
                                if 'tabnet' in config_sections:
                                    updates['tabnet'] = {'n_d': new_n_d, 'n_a': new_n_a}
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

                # Show progress if training is active
                if st.session_state.get('training_active', False):
                    with training_container:
                        progress_bar = st.progress(st.session_state.get('training_progress', 0) / 100)
                        status_text = st.empty()
                        status_text.text(st.session_state.get('training_status', 'Training in progress...'))

                        # Check for updates
                        execute_full_training(progress_bar, status_text)

                        # Show current status
                        st.info(f"🔄 Training in progress - {st.session_state.get('training_progress', 0)}% complete")

                        # Auto-refresh button (acts as manual refresh)
                        if st.button("🔄 Refresh Progress", key="refresh_training_btn"):
                            st.rerun()

                   
                else:
                    # Show start button
                    if st.button("🚀 Start Training", key="train_btn"):
                        # Initialize session state for tracking
                        st.session_state.training_progress = 0
                        st.session_state.training_status = "Starting training..."

                        # Start training (this will set training_active to True)
                        with training_container:
                            progress_bar = st.progress(0)
                            status_text = st.empty()
                            execute_full_training(progress_bar, status_text)

                        # Trigger rerun to start polling
                        st.rerun()

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
                    if race.get('has_results', 0):
                        status_indicators.append("🏁 Results")

                    # Handle prediction results safely
                    predicted_arriv = "N/A"
                    if race.get("prediction_results"):
                        try:
                            pred_data = json.loads(race["prediction_results"])
                            predicted_arriv = pred_data.get("predicted_arriv", "N/A")
                        except (json.JSONDecodeError, TypeError):
                            predicted_arriv = "N/A"

                    races_df.append({
                        "Select": False,
                        "Date": race['jour'],
                        "Race ID": race['comp'],
                        "Track": race['hippo'],
                        "Race": race_name,
                        "Prediction": predicted_arriv,
                        "Type": race.get('typec', 'N/A'),
                        "Status": " | ".join(status_indicators) if status_indicators else "⏳ Pending"
                    })
                if races_df:
                    # Convert to DataFrame for data_editor
                    races_df_pd = pd.DataFrame(races_df)

                    # Display races table with selection
                    edited_df = st.data_editor(
                        races_df_pd,
                        column_config={
                            "Select": st.column_config.CheckboxColumn(
                                "Select",
                                help="Select races to predict",
                                default=False,
                            )
                        },
                        disabled=["Date", "Race ID", "Track", "Race", "Prediction", "Type", "Status"],
                        hide_index=True,
                        key="races_editor"
                    )

                    # Get selected races using pandas filtering
                    if "Select" in edited_df.columns:
                        selected_mask = edited_df["Select"] == True
                        selected_races = edited_df.loc[selected_mask, "Race ID"].tolist()
                    else:
                        selected_races = []

                    if selected_races:
                        st.info(f"📌 {len(selected_races)} race(s) selected")

                        # Add button to predict selected races
                        if st.button("🎯 Predict Selected Races", key="predict_selected", use_container_width=True):
                            execute_predictions_background(selected_races, force_reprediction=True)
                            time.sleep(1)  # Give job time to start
                            st.rerun()

                # Get races needing predictions
                races_needing_prediction = st.session_state.helper.get_races_needing_prediction()
                st.markdown(f"**{len(races_needing_prediction)} races need predictions**")

                left, right = st.columns(2)
                with left:
                    if st.button("🔮 Predict All New Races", key="predict_all", use_container_width=True):
                        if races_needing_prediction:
                            execute_predictions_background(None, force_reprediction=False)
                            time.sleep(1)  # Give job time to start
                            st.rerun()
                        else:
                            st.info("No races need predictions")

                with right:
                    if st.button("🔁 Force Reprediction All", key="force_repred", use_container_width=True):
                        execute_predictions_background(None, force_reprediction=True)
                        time.sleep(1)  # Give job time to start
                        st.rerun()

                # Display active prediction jobs
                st.markdown("---")
                has_active_jobs = display_active_prediction_jobs()

                # Auto-refresh if there are active jobs
                if has_active_jobs:
                    st.info("🔄 Auto-refreshing every 5 seconds...")
                    time.sleep(5)
                    st.rerun()

                # Re-blending section
                st.markdown("---")
                st.markdown("### ⚡ Quick Re-blending with Dynamic Weights")
                st.info("Re-apply new weights to existing predictions without re-running full prediction (much faster!)")

                # Show total races with predictions
                import sqlite3
                try:
                    conn = sqlite3.connect(st.session_state.helper.config_path.replace('config.yaml', 'data/hippique2.db'))
                    cursor = conn.cursor()
                    cursor.execute("""
                        SELECT COUNT(DISTINCT dr.comp)
                        FROM daily_race dr
                        JOIN race_predictions rp ON dr.comp = rp.race_id
                    """)
                    total_races = cursor.fetchone()[0]
                    conn.close()

                    if total_races:
                        st.caption(f"📊 Total races with predictions: {total_races}")
                except:
                    pass

                if st.button("⚡ Re-blend ALL Races with Dynamic Weights", key="reblend_all_btn", type="primary", help="Update ALL predictions with new weights without re-predicting"):
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    execute_re_blending(all_races=True, progress_bar=progress_bar, status_text=status_text)
                    st.rerun()

                st.markdown('</div>', unsafe_allow_html=True)

            elif operation == "🏇 Quinté Predictions":
                st.markdown('''
                <div class="config-panel">
                    <h3>🏇 Quinté Race Predictions</h3>
                    <p>Run predictions specifically optimized for Quinté races</p>
                </div>
                ''', unsafe_allow_html=True)

                # Tabs for single date vs batch
                tab_single, tab_batch = st.tabs(["📅 Single Date", "📊 Batch Prediction"])

                with tab_single:
                    # Date selection
                    col_date, col_mode = st.columns(2)
                    with col_date:
                        prediction_date = st.date_input(
                            "Select Date for Prediction",
                            value=datetime.now(),
                            key="quinte_date"
                        )
                    with col_mode:
                        store_to_db = st.checkbox("Store predictions to database", value=True, key="quinte_store_db")

                    # Run prediction button
                    if st.button("🏇 Run Quinté Prediction", key="run_quinte_pred", type="primary"):
                        progress_bar = st.progress(0)
                        status_text = st.empty()

                        log_output("Starting Quinté predictions...", "info")

                        try:
                            from race_prediction.predict_quinte import QuintePredictionEngine

                            # Format date
                            race_date = prediction_date.strftime('%Y-%m-%d')

                            # Create predictor
                            predictor = QuintePredictionEngine(verbose=True)

                            # Run prediction
                            status_text.text("Running predictions...")
                            progress_bar.progress(20)

                            result = predictor.run_prediction(
                                race_date=race_date,
                                output_dir='predictions',
                                store_to_db=store_to_db
                            )

                            progress_bar.progress(100)

                            if result['status'] == 'success':
                                # Store results in session state
                                st.session_state.quinte_prediction_result = result

                                log_output(f"✅ Quinté predictions completed!", "success")
                                log_output(f"  - {result['races']} races predicted", "info")
                                log_output(f"  - {result['horses']} horses analyzed", "info")
                                log_output(f"  - Time: {result['prediction_time']:.2f}s", "info")
                                log_output(f"  - Output: {result['files']['csv']}", "info")

                                st.success(f"✅ Predicted {result['races']} Quinté races in {result['prediction_time']:.2f}s")
                            else:
                                log_output(f"❌ Prediction failed: {result.get('message', 'Unknown error')}", "error")
                                st.error(f"Prediction failed: {result.get('message', 'Unknown error')}")

                        except Exception as e:
                            progress_bar.progress(0)
                            log_output(f"❌ Error running Quinté predictions: {str(e)}", "error")
                            st.error(f"Error: {str(e)}")

                        status_text.text("Complete")

                with tab_batch:
                    st.markdown("### 📊 Batch Quinté Predictions")
                    st.info("Predict all Quinté races across a date range")

                    # Date range selection
                    col_start, col_end = st.columns(2)
                    with col_start:
                        batch_start_date = st.date_input(
                            "Start Date",
                            value=datetime.now() - timedelta(days=30),
                            key="quinte_batch_start"
                        )
                    with col_end:
                        batch_end_date = st.date_input(
                            "End Date",
                            value=datetime.now(),
                            key="quinte_batch_end"
                        )

                    # Query Quinté races in date range
                    import sqlite3
                    try:
                        db_path = st.session_state.helper.config_path.replace('config.yaml', 'data/hippique2.db')
                        conn = sqlite3.connect(db_path)

                        # Get Quinté races
                        query = """
                        SELECT
                            dr.comp,
                            dr.jour,
                            dr.hippo,
                            dr.prixnom,
                            CASE
                                WHEN qp.race_id IS NOT NULL THEN 1
                                ELSE 0
                            END as has_quinte_prediction
                        FROM daily_race dr
                        LEFT JOIN quinte_predictions qp ON dr.comp = qp.race_id
                        WHERE dr.quinte = 1
                        AND dr.jour BETWEEN ? AND ?
                        GROUP BY dr.comp
                        ORDER BY dr.jour DESC, dr.comp
                        """

                        df_quinte_races = pd.read_sql_query(
                            query,
                            conn,
                            params=(batch_start_date.strftime('%Y-%m-%d'), batch_end_date.strftime('%Y-%m-%d'))
                        )
                        conn.close()

                        if len(df_quinte_races) > 0:
                            # Count races needing predictions
                            races_needing_pred = df_quinte_races[df_quinte_races['has_quinte_prediction'] == 0]
                            races_with_pred = df_quinte_races[df_quinte_races['has_quinte_prediction'] == 1]

                            st.markdown(f"**Found {len(df_quinte_races)} Quinté races:**")
                            st.markdown(f"- ✅ {len(races_with_pred)} already predicted")
                            st.markdown(f"- ⏳ {len(races_needing_pred)} need predictions")

                            # Display races table
                            display_df = df_quinte_races.copy()
                            display_df['Status'] = display_df['has_quinte_prediction'].apply(
                                lambda x: "✅ Predicted" if x == 1 else "⏳ Pending"
                            )
                            display_df = display_df[['jour', 'hippo', 'prixnom', 'Status']]
                            display_df.columns = ['Date', 'Track', 'Race Name', 'Status']

                            st.dataframe(display_df, use_container_width=True, hide_index=True)

                            # Prediction buttons
                            st.markdown("---")
                            col1, col2 = st.columns(2)

                            with col1:
                                if st.button("🏇 Predict All Quinté Races", key="predict_all_quinte", type="primary", use_container_width=True):
                                    if len(races_needing_pred) > 0:
                                        progress_bar = st.progress(0)
                                        status_text = st.empty()

                                        try:
                                            from race_prediction.predict_quinte import QuintePredictionEngine
                                            predictor = QuintePredictionEngine(verbose=True)

                                            total_races = 0
                                            total_horses = 0
                                            failed_dates = []

                                            # Get unique dates that need predictions
                                            dates_to_predict = races_needing_pred['jour'].unique()

                                            log_output(f"Starting batch Quinté predictions for {len(dates_to_predict)} dates...", "info")

                                            for idx, race_date in enumerate(dates_to_predict):
                                                progress = int((idx / len(dates_to_predict)) * 100)
                                                progress_bar.progress(progress)
                                                status_text.text(f"Processing {race_date} ({idx+1}/{len(dates_to_predict)})...")

                                                try:
                                                    result = predictor.run_prediction(
                                                        race_date=race_date,
                                                        output_dir='predictions',
                                                        store_to_db=True
                                                    )

                                                    if result['status'] == 'success':
                                                        total_races += result.get('races', 0)
                                                        total_horses += result.get('horses', 0)
                                                        log_output(f"  ✅ {race_date}: {result['races']} races, {result['horses']} horses", "success")
                                                    else:
                                                        failed_dates.append(race_date)
                                                        log_output(f"  ❌ {race_date}: Failed", "error")

                                                except Exception as e:
                                                    failed_dates.append(race_date)
                                                    log_output(f"  ❌ {race_date}: {str(e)}", "error")

                                            progress_bar.progress(100)
                                            status_text.text("Complete!")

                                            # Show summary
                                            st.success(f"✅ Batch prediction complete!")
                                            st.markdown(f"- **{total_races}** races predicted")
                                            st.markdown(f"- **{total_horses}** horses analyzed")
                                            if failed_dates:
                                                st.warning(f"⚠️ Failed dates: {', '.join(failed_dates)}")

                                            time.sleep(2)
                                            st.rerun()

                                        except Exception as e:
                                            log_output(f"❌ Batch prediction error: {str(e)}", "error")
                                            st.error(f"Error: {str(e)}")
                                    else:
                                        st.info("All Quinté races already have predictions!")

                            with col2:
                                if st.button("🔁 Force Reprediction", key="force_quinte_repred", use_container_width=True):
                                    progress_bar = st.progress(0)
                                    status_text = st.empty()

                                    try:
                                        from race_prediction.predict_quinte import QuintePredictionEngine
                                        predictor = QuintePredictionEngine(verbose=True)

                                        total_races = 0
                                        total_horses = 0

                                        # Get all unique dates
                                        all_dates = df_quinte_races['jour'].unique()

                                        log_output(f"Force repredicting {len(all_dates)} dates...", "info")

                                        for idx, race_date in enumerate(all_dates):
                                            progress = int((idx / len(all_dates)) * 100)
                                            progress_bar.progress(progress)
                                            status_text.text(f"Processing {race_date} ({idx+1}/{len(all_dates)})...")

                                            try:
                                                result = predictor.run_prediction(
                                                    race_date=race_date,
                                                    output_dir='predictions',
                                                    store_to_db=True
                                                )

                                                if result['status'] == 'success':
                                                    total_races += result.get('races', 0)
                                                    total_horses += result.get('horses', 0)

                                            except Exception as e:
                                                log_output(f"  ❌ {race_date}: {str(e)}", "error")

                                        progress_bar.progress(100)
                                        status_text.text("Complete!")

                                        st.success(f"✅ Repredicted {total_races} races!")
                                        time.sleep(2)
                                        st.rerun()

                                    except Exception as e:
                                        log_output(f"❌ Reprediction error: {str(e)}", "error")
                                        st.error(f"Error: {str(e)}")
                        else:
                            st.warning("No Quinté races found in selected date range")

                    except Exception as e:
                        st.error(f"Error loading Quinté races: {str(e)}")

                # Today's Predictions Section
                st.markdown("---")
                st.markdown("### 📅 Today's Quinté Predictions")
                st.info("Compare Quinté-specific and General model predictions for today's races")

                try:
                    today_date = datetime.now().strftime('%Y-%m-%d')
                    db_path = st.session_state.helper.config_path.replace('config.yaml', 'data/hippique2.db')
                    conn = sqlite3.connect(db_path)

                    # Query today's Quinté races
                    today_races_query = """
                    SELECT
                        dr.comp,
                        dr.jour,
                        dr.hippo,
                        dr.prixnom,
                        dr.prix,
                        dr.prediction_results,
                        dr.participants
                    FROM daily_race dr
                    WHERE dr.quinte = 1
                    AND dr.jour = ?
                    ORDER BY dr.prix
                    """

                    df_today_races = pd.read_sql_query(today_races_query, conn, params=(today_date,))

                    if len(df_today_races) > 0:
                        st.markdown(f"**Found {len(df_today_races)} Quinté race(s) today ({today_date})**")

                        # Get Quinté predictions for today
                        race_ids = df_today_races['comp'].tolist()
                        comp_placeholders = ','.join(['?' for _ in race_ids])

                        quinte_query = f"""
                        SELECT
                            race_id,
                            horse_number as numero,
                            horse_name,
                            final_prediction,
                            predicted_rank,
                            quinte_rf_prediction,
                            quinte_tabnet_prediction,
                            calibrated_rf_prediction,
                            calibrated_tabnet_prediction
                        FROM quinte_predictions
                        WHERE race_id IN ({comp_placeholders})
                        ORDER BY race_id, predicted_rank
                        """

                        df_quinte_today = pd.read_sql_query(quinte_query, conn, params=race_ids)
                        conn.close()

                        # Display each race
                        for _, race in df_today_races.iterrows():
                            race_comp = race['comp']
                            race_title = f"R{race['prix']} - {race['prixnom']} ({race['hippo']})"

                            # Create horse name lookup from participants
                            horse_name_lookup = {}
                            if race['participants']:
                                try:
                                    participants = json.loads(race['participants'])
                                    for p in participants:
                                        horse_name_lookup[p['numero']] = p.get('cheval', 'Unknown')
                                except (json.JSONDecodeError, KeyError):
                                    pass

                            with st.expander(f"🏇 {race_title}", expanded=True):
                                col_quinte, col_general = st.columns(2)

                                # Quinté Predictions
                                with col_quinte:
                                    st.markdown("#### 🏇 Quinté Model")

                                    quinte_race_preds = df_quinte_today[df_quinte_today['race_id'] == race_comp]

                                    if len(quinte_race_preds) > 0:
                                        top6_quinte = quinte_race_preds.head(6)

                                        for idx, horse in top6_quinte.iterrows():
                                            rank = int(horse['predicted_rank'])
                                            # Use emojis for ranks 1-6
                                            if rank == 1:
                                                emoji = "🥇"
                                            elif rank == 2:
                                                emoji = "🥈"
                                            elif rank == 3:
                                                emoji = "🥉"
                                            elif rank == 4:
                                                emoji = "4️⃣"
                                            elif rank == 5:
                                                emoji = "5️⃣"
                                            elif rank == 6:
                                                emoji = "6️⃣"
                                            else:
                                                emoji = f"#{rank}"

                                            # Get horse name from participants lookup (same as general model)
                                            # Fall back to database field if not in lookup
                                            horse_numero = int(horse['numero'])
                                            if horse_numero in horse_name_lookup:
                                                horse_name = horse_name_lookup[horse_numero]
                                            elif pd.notna(horse['horse_name']):
                                                horse_name = horse['horse_name']
                                            else:
                                                horse_name = 'Unknown'

                                            st.markdown(f"**{emoji} #{horse_numero} - {horse_name}**")
                                            st.caption(f"Predicted position: {horse['final_prediction']:.2f}")

                                            # Show model details
                                            if pd.notna(horse['quinte_rf_prediction']) and pd.notna(horse['quinte_tabnet_prediction']):
                                                st.caption(f"RF: {horse['quinte_rf_prediction']:.2f} | TabNet: {horse['quinte_tabnet_prediction']:.2f}")
                                    else:
                                        st.warning("⏳ No Quinté predictions yet - run prediction first")

                                # General Predictions
                                with col_general:
                                    st.markdown("#### 🎲 General Model")

                                    if race['prediction_results']:
                                        try:
                                            pred_data = json.loads(race['prediction_results'])
                                            predictions_list = pred_data.get('predictions', [])

                                            # Sort by predicted rank and get top 6
                                            sorted_preds = sorted(predictions_list, key=lambda x: x.get('predicted_rank', 999))[:6]

                                            for idx, horse in enumerate(sorted_preds, 1):
                                                # Use emojis for ranks 1-6
                                                if idx == 1:
                                                    emoji = "🥇"
                                                elif idx == 2:
                                                    emoji = "🥈"
                                                elif idx == 3:
                                                    emoji = "🥉"
                                                elif idx == 4:
                                                    emoji = "4️⃣"
                                                elif idx == 5:
                                                    emoji = "5️⃣"
                                                elif idx == 6:
                                                    emoji = "6️⃣"
                                                else:
                                                    emoji = f"#{idx}"

                                                # Get horse name from lookup
                                                horse_name = horse_name_lookup.get(horse['numero'], 'Unknown')

                                                st.markdown(f"**{emoji} #{horse['numero']} - {horse_name}**")
                                                st.caption(f"Predicted position: {horse.get('predicted_position', 'N/A'):.2f}")

                                                # Show model details if available
                                                if 'predicted_position_rf' in horse and 'predicted_position_tabnet' in horse:
                                                    st.caption(f"RF: {horse['predicted_position_rf']:.2f} | TabNet: {horse['predicted_position_tabnet']:.2f}")
                                        except (json.JSONDecodeError, KeyError, ValueError) as e:
                                            st.warning("⏳ No general predictions yet")
                                    else:
                                        st.warning("⏳ No general predictions yet")

                                # Comparison section
                                st.markdown("---")
                                if len(quinte_race_preds) > 0 and race['prediction_results']:
                                    try:
                                        # Compare top 6
                                        quinte_top6 = quinte_race_preds.head(6)['numero'].astype(int).tolist()

                                        pred_data = json.loads(race['prediction_results'])
                                        predictions_list = pred_data.get('predictions', [])
                                        sorted_preds = sorted(predictions_list, key=lambda x: x.get('predicted_rank', 999))[:6]
                                        general_top6 = [int(p['numero']) for p in sorted_preds]

                                        # Find common horses
                                        common = set(quinte_top6) & set(general_top6)

                                        st.markdown(f"**Agreement:** {len(common)}/6 horses in common")
                                        if common:
                                            st.caption(f"Common picks: {', '.join([f'#{n}' for n in sorted(common)])}")

                                        # Show differences
                                        quinte_only = set(quinte_top6) - set(general_top6)
                                        general_only = set(general_top6) - set(quinte_top6)

                                        if quinte_only:
                                            st.caption(f"🏇 Quinté-only: {', '.join([f'#{n}' for n in sorted(quinte_only)])}")
                                        if general_only:
                                            st.caption(f"🎲 General-only: {', '.join([f'#{n}' for n in sorted(general_only)])}")

                                    except Exception as e:
                                        pass

                    else:
                        st.info(f"No Quinté races found for today ({today_date})")

                except Exception as e:
                    st.error(f"Error loading today's predictions: {str(e)}")
                    import traceback
                    st.code(traceback.format_exc())

                # Display results if available
                if 'quinte_prediction_result' in st.session_state:
                    result = st.session_state.quinte_prediction_result

                    st.markdown("---")
                    st.markdown("### 📊 Prediction Results")

                    # Summary metrics
                    stat1, stat2, stat3, stat4 = st.columns(4)
                    with stat1:
                        st.metric("Races Predicted", result.get('races', 0))
                    with stat2:
                        st.metric("Horses Analyzed", result.get('horses', 0))
                    with stat3:
                        st.metric("Prediction Time", f"{result.get('prediction_time', 0):.2f}s")
                    with stat4:
                        st.metric("Status", "✅ Success" if result.get('status') == 'success' else "❌ Failed")

                    # Load and display predictions
                    if 'files' in result and 'csv' in result['files']:
                        try:
                            df_predictions = pd.read_csv(result['files']['csv'])

                            st.markdown("### 🏆 Top Predictions per Race")

                            # Group by race and show top 5
                            for race_comp, race_df in df_predictions.groupby('comp'):
                                with st.expander(f"🏁 {race_df.iloc[0]['prixnom']} - {race_df.iloc[0]['hippo']}"):
                                    # Sort by predicted rank
                                    race_df_sorted = race_df.sort_values('predicted_rank').head(5)

                                    # Display top 5
                                    cols = st.columns([1, 3, 2, 2])
                                    cols[0].write("**Rank**")
                                    cols[1].write("**Horse**")
                                    cols[2].write("**Predicted Position**")
                                    cols[3].write("**Competitive Score**")

                                    for _, horse in race_df_sorted.iterrows():
                                        cols = st.columns([1, 3, 2, 2])
                                        cols[0].write(f"#{int(horse['predicted_rank'])}")
                                        cols[1].write(horse.get('cheval', 'N/A'))
                                        cols[2].write(f"{horse['predicted_position']:.2f}")
                                        cols[3].write(f"{horse.get('competitive_score', 0):.2f}")

                        except Exception as e:
                            st.error(f"Error loading predictions: {str(e)}")

                # Model Comparison Section
                st.markdown("---")
                st.markdown("### 🔬 Quinté vs General Model Comparison")
                st.info("Compare Quinté-specific predictions against General model predictions with actual results")

                # Date range for comparison
                col_start, col_end = st.columns(2)
                with col_start:
                    compare_start_date = st.date_input(
                        "Start Date",
                        value=datetime.now() - timedelta(days=30),
                        key="compare_start_date"
                    )
                with col_end:
                    compare_end_date = st.date_input(
                        "End Date",
                        value=datetime.now(),
                        key="compare_end_date"
                    )

                if st.button("📊 Generate Comparison Report", key="generate_comparison"):

                    progress_bar = st.progress(0)
                    status_text = st.empty()

                    try:
                        status_text.text("Loading predictions and results...")
                        progress_bar.progress(20)

                        # Format dates
                        start_date_str = compare_start_date.strftime('%Y-%m-%d')
                        end_date_str = compare_end_date.strftime('%Y-%m-%d')

                        # Connect to database
                        db_path = st.session_state.helper.config_path.replace('config.yaml', 'data/hippique2.db')
                        conn = sqlite3.connect(db_path)

                        # Load races with results in date range
                        query = """
                        SELECT
                            dr.comp,
                            dr.jour,
                            dr.hippo,
                            dr.prixnom,
                            dr.quinte,
                            dr.actual_results,
                            dr.prediction_results
                        FROM daily_race dr
                        WHERE dr.jour BETWEEN ? AND ?
                        AND dr.actual_results IS NOT NULL
                        AND dr.actual_results != ''
                        AND dr.quinte = 1
                        ORDER BY dr.jour DESC, dr.comp
                        """

                        df_races = pd.read_sql_query(query, conn, params=(start_date_str, end_date_str))

                        progress_bar.progress(40)
                        status_text.text(f"Found {len(df_races)} Quinté races with results")

                        if len(df_races) == 0:
                            st.warning("No Quinté races with results found in selected date range")
                            conn.close()
                            progress_bar.progress(0)
                            status_text.text("")
                        else:
                            # Load predictions for these races
                            race_comps = df_races['comp'].tolist()
                            comp_placeholders = ','.join(['?' for _ in race_comps])

                            progress_bar.progress(60)
                            status_text.text("Loading Quinté predictions from database...")

                            # Load Quinté predictions from dedicated quinte_predictions table
                            quinte_query = f"""
                            SELECT
                                race_id,
                                horse_number as numero,
                                horse_id,
                                final_prediction as predicted_position,
                                predicted_rank
                            FROM quinte_predictions
                            WHERE race_id IN ({comp_placeholders})
                            ORDER BY race_id, predicted_rank
                            """
                            df_quinte_preds = pd.read_sql_query(quinte_query, conn, params=race_comps)

                            # Check if we have Quinté predictions
                            if len(df_quinte_preds) == 0:
                                st.warning("⚠️ No Quinté predictions found in database for selected date range. Please run Quinté predictions first.")
                                log_output(f"No Quinté predictions found for {len(race_comps)} races", "warning")

                            progress_bar.progress(80)
                            status_text.text("Analyzing results...")

                            # Build comparison data
                            comparison_data = []

                            for _, race in df_races.iterrows():
                                race_comp = race['comp']

                                # Parse actual results (format: "14-16-3-5-7")
                                try:
                                    actual_order = [int(x.strip()) for x in race['actual_results'].split('-') if x.strip().isdigit()]
                                    actual_top5 = actual_order[:5] if len(actual_order) >= 5 else actual_order
                                    actual_winner = actual_order[0] if len(actual_order) > 0 else None
                                except:
                                    continue

                                # Get general predictions for this race from prediction_results JSON
                                general_top5 = []
                                general_winner = None
                                general_horses_in_top5 = 0
                                general_winner_correct = False
                                general_order = False  # Quinté in exact order
                                general_desordre = False  # Quinté in any order (5/5)
                                general_bonus3 = False  # Top 3 actual in predicted top 5
                                general_bonus4 = False  # Top 4 actual in predicted top 5

                                if race['prediction_results']:
                                    try:
                                        pred_data = json.loads(race['prediction_results'])
                                        predictions_list = pred_data.get('predictions', [])

                                        # Sort by predicted_rank and get top 5 numeros
                                        sorted_preds = sorted(predictions_list, key=lambda x: x.get('predicted_rank', 999))
                                        general_top5 = [int(p['numero']) for p in sorted_preds[:5] if 'numero' in p]
                                        general_winner = general_top5[0] if len(general_top5) > 0 else None

                                        # Calculate general model performance
                                        general_horses_in_top5 = len(set(general_top5) & set(actual_top5))
                                        general_winner_correct = (general_winner == actual_winner) if general_winner and actual_winner else False

                                        # Calculate bet results
                                        actual_top3 = set(actual_order[:3]) if len(actual_order) >= 3 else set(actual_order)
                                        actual_top4 = set(actual_order[:4]) if len(actual_order) >= 4 else set(actual_order)
                                        predicted_top5_set = set(general_top5)

                                        general_order = (general_top5 == actual_top5) if len(general_top5) == 5 and len(actual_top5) == 5 else False
                                        general_desordre = (general_horses_in_top5 == 5)
                                        general_bonus3 = (len(actual_top3 & predicted_top5_set) == 3)
                                        general_bonus4 = (len(actual_top4 & predicted_top5_set) == 4)
                                    except (json.JSONDecodeError, KeyError, ValueError):
                                        pass

                                # Get quinté predictions for this race (from quinte_predictions table)
                                quinte_top5 = []
                                quinte_winner = None
                                quinte_horses_in_top5 = 0
                                quinte_winner_correct = False
                                quinte_order = False  # Quinté in exact order
                                quinte_desordre = False  # Quinté in any order (5/5)
                                quinte_bonus3 = False  # Top 3 actual in predicted top 5
                                quinte_bonus4 = False  # Top 4 actual in predicted top 5

                                if len(df_quinte_preds) > 0:
                                    quinte_race_preds = df_quinte_preds[df_quinte_preds['race_id'] == race_comp]
                                    if len(quinte_race_preds) > 0:
                                        # Already sorted by predicted_rank from query
                                        quinte_top5_nums = quinte_race_preds.head(5)['numero'].astype(int).tolist()
                                        quinte_winner = int(quinte_race_preds.iloc[0]['numero']) if len(quinte_race_preds) > 0 else None

                                        # Calculate quinté model performance
                                        quinte_horses_in_top5 = len(set(quinte_top5_nums) & set(actual_top5))
                                        quinte_winner_correct = (quinte_winner == actual_winner) if quinte_winner and actual_winner else False
                                        quinte_top5 = quinte_top5_nums

                                        # Calculate bet results
                                        actual_top3 = set(actual_order[:3]) if len(actual_order) >= 3 else set(actual_order)
                                        actual_top4 = set(actual_order[:4]) if len(actual_order) >= 4 else set(actual_order)
                                        predicted_top5_set = set(quinte_top5_nums)

                                        quinte_order = (quinte_top5_nums == actual_top5) if len(quinte_top5_nums) == 5 and len(actual_top5) == 5 else False
                                        quinte_desordre = (quinte_horses_in_top5 == 5)
                                        quinte_bonus3 = (len(actual_top3 & predicted_top5_set) == 3)
                                        quinte_bonus4 = (len(actual_top4 & predicted_top5_set) == 4)

                                # Determine which model won
                                if quinte_horses_in_top5 > general_horses_in_top5:
                                    better_model = "🏇 Quinté"
                                elif general_horses_in_top5 > quinte_horses_in_top5:
                                    better_model = "🎲 General"
                                elif quinte_horses_in_top5 == general_horses_in_top5 and quinte_horses_in_top5 > 0:
                                    better_model = "🤝 Tie"
                                else:
                                    better_model = "❌ Both Failed"

                                comparison_data.append({
                                    'Date': race['jour'],
                                    'Race': f"{race['prixnom']} ({race['hippo']})",
                                    'Comp': race_comp,
                                    'Actual Top 5': '-'.join(map(str, actual_top5)),
                                    'Actual Winner': actual_winner,
                                    # Quinté Model Results
                                    'Quinté Top 5': '-'.join(map(str, quinte_top5)) if quinte_top5 else 'N/A',
                                    'Quinté Winner': f"{'✅' if quinte_winner_correct else '❌'} {quinte_winner}" if quinte_winner else 'N/A',
                                    'Quinté Order': '✅' if quinte_order else '❌',
                                    'Quinté Désordre': '✅' if quinte_desordre else '❌',
                                    'Quinté B3': '✅' if quinte_bonus3 else '❌',
                                    'Quinté B4': '✅' if quinte_bonus4 else '❌',
                                    'Quinté Horses in Top 5': f"{quinte_horses_in_top5}/5",
                                    # General Model Results
                                    'General Top 5': '-'.join(map(str, general_top5)) if general_top5 else 'N/A',
                                    'General Winner': f"{'✅' if general_winner_correct else '❌'} {general_winner}" if general_winner else 'N/A',
                                    'General Order': '✅' if general_order else '❌',
                                    'General Désordre': '✅' if general_desordre else '❌',
                                    'General B3': '✅' if general_bonus3 else '❌',
                                    'General B4': '✅' if general_bonus4 else '❌',
                                    'General Horses in Top 5': f"{general_horses_in_top5}/5",
                                    'Better Model': better_model
                                })

                            conn.close()

                            if comparison_data:
                                df_comparison = pd.DataFrame(comparison_data)
                                st.session_state.comparison_results = df_comparison

                                progress_bar.progress(100)
                                status_text.text("Complete!")

                                log_output(f"✅ Comparison generated for {len(df_comparison)} races", "success")
                                st.success(f"✅ Analyzed {len(df_comparison)} Quinté races")
                            else:
                                st.warning("No comparison data could be generated")
                                progress_bar.progress(0)

                    except Exception as e:
                        progress_bar.progress(0)
                        log_output(f"❌ Comparison failed: {str(e)}", "error")
                        st.error(f"Error: {str(e)}")
                        status_text.text("")

                # Display comparison results
                if 'comparison_results' in st.session_state:
                    df_comp = st.session_state.comparison_results

                    st.markdown("### 📊 Comparison Summary")

                    # Calculate overall statistics
                    total_races = len(df_comp)
                    quinte_wins = len(df_comp[df_comp['Better Model'] == '🏇 Quinté'])
                    general_wins = len(df_comp[df_comp['Better Model'] == '🎲 General'])
                    ties = len(df_comp[df_comp['Better Model'] == '🤝 Tie'])

                    # Bet results for Quinté model
                    quinte_order_count = len(df_comp[df_comp['Quinté Order'] == '✅'])
                    quinte_desordre_count = len(df_comp[df_comp['Quinté Désordre'] == '✅'])
                    quinte_b3_count = len(df_comp[df_comp['Quinté B3'] == '✅'])
                    quinte_b4_count = len(df_comp[df_comp['Quinté B4'] == '✅'])
                    quinte_winners = len(df_comp[df_comp['Quinté Winner'].str.contains('✅', na=False)])

                    # Bet results for General model
                    general_order_count = len(df_comp[df_comp['General Order'] == '✅'])
                    general_desordre_count = len(df_comp[df_comp['General Désordre'] == '✅'])
                    general_b3_count = len(df_comp[df_comp['General B3'] == '✅'])
                    general_b4_count = len(df_comp[df_comp['General B4'] == '✅'])
                    general_winners = len(df_comp[df_comp['General Winner'].str.contains('✅', na=False)])

                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Total Races", total_races)
                    with col2:
                        st.metric("Quinté Model Wins", f"{quinte_wins} ({quinte_wins/total_races*100:.1f}%)" if total_races > 0 else "0")
                    with col3:
                        st.metric("General Model Wins", f"{general_wins} ({general_wins/total_races*100:.1f}%)" if total_races > 0 else "0")
                    with col4:
                        st.metric("Ties", f"{ties} ({ties/total_races*100:.1f}%)" if total_races > 0 else "0")

                    st.markdown("---")
                    st.markdown("#### 🏇 Quinté Model Bet Results")

                    col1, col2, col3, col4, col5 = st.columns(5)
                    with col1:
                        st.metric("Winners", f"{quinte_winners} ({quinte_winners/total_races*100:.1f}%)" if total_races > 0 else "0")
                    with col2:
                        st.metric("Quinté Order", f"{quinte_order_count} ({quinte_order_count/total_races*100:.1f}%)" if total_races > 0 else "0")
                    with col3:
                        st.metric("Quinté Désordre", f"{quinte_desordre_count} ({quinte_desordre_count/total_races*100:.1f}%)" if total_races > 0 else "0")
                    with col4:
                        st.metric("Bonus 3", f"{quinte_b3_count} ({quinte_b3_count/total_races*100:.1f}%)" if total_races > 0 else "0")
                    with col5:
                        st.metric("Bonus 4", f"{quinte_b4_count} ({quinte_b4_count/total_races*100:.1f}%)" if total_races > 0 else "0")

                    st.markdown("#### 🎲 General Model Bet Results")

                    col1, col2, col3, col4, col5 = st.columns(5)
                    with col1:
                        st.metric("Winners", f"{general_winners} ({general_winners/total_races*100:.1f}%)" if total_races > 0 else "0")
                    with col2:
                        st.metric("Quinté Order", f"{general_order_count} ({general_order_count/total_races*100:.1f}%)" if total_races > 0 else "0")
                    with col3:
                        st.metric("Quinté Désordre", f"{general_desordre_count} ({general_desordre_count/total_races*100:.1f}%)" if total_races > 0 else "0")
                    with col4:
                        st.metric("Bonus 3", f"{general_b3_count} ({general_b3_count/total_races*100:.1f}%)" if total_races > 0 else "0")
                    with col5:
                        st.metric("Bonus 4", f"{general_b4_count} ({general_b4_count/total_races*100:.1f}%)" if total_races > 0 else "0")

                    # Detailed results table
                    st.markdown("### 📋 Detailed Race Comparison")

                    # Color code the better model column
                    def highlight_better_model(row):
                        if '🏇 Quinté' in row['Better Model']:
                            return ['background-color: #E6483320'] * len(row)
                        elif '🎲 General' in row['Better Model']:
                            return ['background-color: #24485520'] * len(row)
                        elif '🤝 Tie' in row['Better Model']:
                            return ['background-color: #90AEAD20'] * len(row)
                        else:
                            return [''] * len(row)

                    # Display with styling
                    st.dataframe(
                        df_comp.style.apply(highlight_better_model, axis=1),
                        use_container_width=True,
                        hide_index=True,
                        height=600
                    )

                    # Add download button
                    csv = df_comp.to_csv(index=False)
                    st.download_button(
                        label="📥 Download Comparison CSV",
                        data=csv,
                        file_name=f"quinte_vs_general_comparison_{datetime.now().strftime('%Y%m%d')}.csv",
                        mime="text/csv"
                    )

                # Blend Optimization Section
                st.markdown("---")
                st.markdown("### ⚖️ Blend Optimization")
                st.info("Optimize the blend between Quinté-specific and General model predictions")

                # Find latest prediction file
                if st.button("🔍 Run Blend Optimization", key="run_blend_opt"):
                    progress_bar = st.progress(0)
                    status_text = st.empty()

                    log_output("Starting blend optimization...", "info")

                    try:
                        from race_prediction.optimize_quinte_general_blend import QuinteGeneralBlendOptimizer

                        status_text.text("Initializing optimizer...")
                        progress_bar.progress(10)

                        optimizer = QuinteGeneralBlendOptimizer(verbose=False)

                        # Find latest prediction file
                        status_text.text("Finding latest predictions...")
                        progress_bar.progress(20)
                        prediction_file = optimizer.find_latest_prediction_file('predictions')

                        log_output(f"Using prediction file: {prediction_file}", "info")

                        # Run optimization
                        status_text.text("Testing blend weights...")
                        progress_bar.progress(30)

                        df_results = optimizer.optimize(
                            quinte_prediction_file=prediction_file,
                            weight_step=0.1
                        )

                        progress_bar.progress(100)
                        status_text.text("Complete")

                        # Store results
                        st.session_state.blend_optimization_results = df_results

                        log_output("✅ Blend optimization completed!", "success")
                        st.success("✅ Blend optimization completed!")

                    except Exception as e:
                        progress_bar.progress(0)
                        log_output(f"❌ Error during optimization: {str(e)}", "error")
                        st.error(f"Error: {str(e)}")

                # Display optimization results
                if 'blend_optimization_results' in st.session_state:
                    df_results = st.session_state.blend_optimization_results

                    st.markdown("### 📈 Optimization Results")

                    # Find best combination
                    best_row = df_results.sort_values('quinte_desordre_pct', ascending=False).iloc[0]

                    st.success(f"🏆 Best weights: Quinté {best_row['quinte_weight']:.2f} / General {best_row['general_weight']:.2f}")

                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Quinté Désordre", f"{best_row['quinte_desordre_pct']:.1f}%")
                    with col2:
                        st.metric("Bonus 4", f"{best_row['bonus4_pct']:.1f}%")
                    with col3:
                        st.metric("Bonus 3", f"{best_row['bonus3_pct']:.1f}%")
                    with col4:
                        st.metric("Winner Accuracy", f"{best_row['winner_accuracy']:.1f}%")

                    # Plot results
                    st.markdown("### 📊 Performance by Weight Combination")

                    import plotly.graph_objects as go

                    fig = go.Figure()

                    fig.add_trace(go.Scatter(
                        x=df_results['quinte_weight'],
                        y=df_results['quinte_desordre_pct'],
                        mode='lines+markers',
                        name='Quinté Désordre',
                        line=dict(color='#E64833', width=3)
                    ))

                    fig.add_trace(go.Scatter(
                        x=df_results['quinte_weight'],
                        y=df_results['bonus4_pct'],
                        mode='lines+markers',
                        name='Bonus 4',
                        line=dict(color='#244855', width=2)
                    ))

                    fig.add_trace(go.Scatter(
                        x=df_results['quinte_weight'],
                        y=df_results['bonus3_pct'],
                        mode='lines+markers',
                        name='Bonus 3',
                        line=dict(color='#874F41', width=2)
                    ))

                    fig.update_layout(
                        title="Performance Metrics by Quinté Weight",
                        xaxis_title="Quinté Weight (0 = All General, 1 = All Quinté)",
                        yaxis_title="Success Rate (%)",
                        hovermode='x unified',
                        template='plotly_white',
                        height=500
                    )

                    st.plotly_chart(fig, use_container_width=True)

                    # Show detailed results table
                    with st.expander("📋 Detailed Results Table"):
                        st.dataframe(
                            df_results,
                            use_container_width=True,
                            hide_index=True
                        )

                # Re-blending section
                st.markdown("---")
                st.markdown("### ⚡ Quick Re-blending with Dynamic Weights")
                st.info("Re-apply new weights to existing predictions without re-running full prediction (much faster!)")

                # Show total races with predictions
                import sqlite3
                try:
                    conn = sqlite3.connect(st.session_state.helper.config_path.replace('config.yaml', 'data/hippique2.db'))
                    cursor = conn.cursor()
                    cursor.execute("""
                        SELECT COUNT(DISTINCT dr.comp)
                        FROM daily_race dr
                        JOIN race_predictions rp ON dr.comp = rp.race_id
                    """)
                    total_races = cursor.fetchone()[0]
                    conn.close()

                    if total_races:
                        st.caption(f"📊 Total races with predictions: {total_races}")
                except:
                    pass

                if st.button("⚡ Re-blend ALL Races with Dynamic Weights", key="reblend_all_btn", type="primary", help="Update ALL predictions with new weights without re-predicting"):
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    execute_re_blending(all_races=True, progress_bar=progress_bar, status_text=status_text)
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
            elif operation == "⚖️ Model Weight Analysis":
                st.markdown('''
                <div class="config-panel">
                    <h3>⚖️ Automated Weight & Pattern Analysis</h3>
                </div>
                ''', unsafe_allow_html=True)

                st.info("🤖 Automated analysis: Tests all RF/TabNet weight combinations (0.0-1.0 by 0.1), finds optimal weights, and detects patterns based on race features")

                # Initialize session state for analysis results
                if 'weight_patterns' not in st.session_state:
                    st.session_state.weight_patterns = None

                # Simple Configuration
                st.markdown("### ⚙️ Configuration")

                left, right = st.columns(2)

                with left:
                    date_from_weight = st.date_input(
                        "Start Date:",
                        value=(datetime.now() - timedelta(days=90)).date(),
                        key="weight_date_from",
                        help="Recommended: 30-90 days for reliable patterns"
                    )

                with right:
                    date_to_weight = st.date_input(
                        "End Date:",
                        value=datetime.now().date(),
                        key="weight_date_to"
                    )

                # Single-click automated analysis
                if st.button("🚀 Run Automated Analysis", key="run_auto_analysis", help="Tests all weights and detects patterns automatically"):
                    with st.spinner("📊 Loading race data..."):
                        # Load data
                        data_result = st.session_state.helper.load_weight_analysis_data(
                            date_from=date_from_weight.strftime('%Y-%m-%d'),
                            date_to=date_to_weight.strftime('%Y-%m-%d'),
                            race_filters=None
                        )

                        if data_result['success']:
                            log_output(data_result['message'], "success")

                            # Run automated pattern detection
                            with st.spinner("🔍 Testing all weight combinations (0.0-1.0 by 0.1) and detecting patterns..."):
                                pattern_result = st.session_state.helper.detect_weight_patterns(
                                    race_data=data_result['race_data'],
                                    weight_step=0.1
                                )

                                if pattern_result['success']:
                                    st.session_state.weight_patterns = pattern_result
                                    log_output(pattern_result['message'], "success")
                                else:
                                    log_output(pattern_result['message'], "error")
                        else:
                            log_output(data_result['message'], "error")

                        st.rerun()

                # Display Results
                if st.session_state.weight_patterns is not None:
                    import plotly.express as px

                    patterns = st.session_state.weight_patterns['patterns']

                    st.markdown("### 📊 Analysis Results")

                    # Summary insights
                    st.markdown("#### 💡 Key Findings")
                    for insight in patterns['summary']:
                        if insight['type'] == 'no_patterns':
                            st.success(f"✅ {insight['message']}")
                        elif insight['type'] == 'patterns_found':
                            st.warning(f"⚠️ {insight['message']}")
                        else:
                            st.info(f"📌 {insight['message']}")

                    # Overall Optimal Weights
                    overall = patterns['overall_best']

                    st.markdown("#### 🎯 Overall Best Weights")
                    tab1, tab2, tab3, tab4, tab5 = st.columns(5)
                    with tab1:
                        st.metric("RF Weight", f"{overall['rf_weight']:.1f}")
                    with tab2:
                        st.metric("TabNet Weight", f"{overall['tabnet_weight']:.1f}")
                    with tab3:
                        st.metric("Winner Accuracy", f"{overall['winner_accuracy']*100:.1f}%")
                    with tab4:
                        st.metric("Podium Accuracy", f"{overall['podium_accuracy']*100:.1f}%")
                    with tab5:
                        st.metric("MAE", f"{overall['mae']:.2f}")

                    # Pattern-specific recommendations
                    st.markdown("#### 🔍 Detected Patterns Requiring Custom Weights")

                    # Race Type Patterns
                    if patterns['by_race_type']:
                        st.markdown("##### 🏇 Race Type Patterns")

                        for pattern in patterns['by_race_type']:
                            with st.expander(f"**{pattern['typec']}** - Custom weights recommended"):
                                pcol1, pcol2, pcol3, pcol4 = st.columns(4)

                                with pcol1:
                                    st.metric("Optimal RF", f"{pattern['optimal_rf_weight']:.1f}")
                                with pcol2:
                                    st.metric("Optimal TabNet", f"{pattern['optimal_tabnet_weight']:.1f}")
                                with pcol3:
                                    st.metric("Winner Accuracy", f"{pattern['winner_accuracy']*100:.1f}%")
                                    if pattern['improvement_vs_overall'] > 0:
                                        st.success(f"+{pattern['improvement_vs_overall']*100:.1f}% vs overall")
                                with pcol4:
                                    st.metric("Races", pattern['total_races'])

                                st.info(f"💡 **Recommendation:** {pattern['recommendation']}")
                    else:
                        st.success("✅ No significant race type patterns detected - overall weights work well")

                    # Distance Range Patterns
                    if patterns['by_distance_range']:
                        st.markdown("##### 📏 Distance Range Patterns")

                        for pattern in patterns['by_distance_range']:
                            with st.expander(f"**{pattern['distance_range']}** - Custom weights recommended"):
                                pcol1, pcol2, pcol3, pcol4 = st.columns(4)

                                with pcol1:
                                    st.metric("Optimal RF", f"{pattern['optimal_rf_weight']:.1f}")
                                with pcol2:
                                    st.metric("Optimal TabNet", f"{pattern['optimal_tabnet_weight']:.1f}")
                                with pcol3:
                                    st.metric("Winner Accuracy", f"{pattern['winner_accuracy']*100:.1f}%")
                                    if pattern['improvement_vs_overall'] > 0:
                                        st.success(f"+{pattern['improvement_vs_overall']*100:.1f}% vs overall")
                                with pcol4:
                                    st.metric("Races", pattern['total_races'])

                                st.info(f"💡 **Recommendation:** {pattern['recommendation']}")
                    else:
                        st.success("✅ No significant distance patterns detected - overall weights work well")

                    # Field Size Patterns
                    if patterns['by_field_size']:
                        st.markdown("##### 👥 Field Size Patterns")

                        for pattern in patterns['by_field_size']:
                            with st.expander(f"**{pattern['field_size']}** - Custom weights recommended"):
                                pcol1, pcol2, pcol3, pcol4 = st.columns(4)

                                with pcol1:
                                    st.metric("Optimal RF", f"{pattern['optimal_rf_weight']:.1f}")
                                with pcol2:
                                    st.metric("Optimal TabNet", f"{pattern['optimal_tabnet_weight']:.1f}")
                                with pcol3:
                                    st.metric("Winner Accuracy", f"{pattern['winner_accuracy']*100:.1f}%")
                                    if pattern['improvement_vs_overall'] > 0:
                                        st.success(f"+{pattern['improvement_vs_overall']*100:.1f}% vs overall")
                                with pcol4:
                                    st.metric("Races", pattern['total_races'])

                                st.info(f"💡 **Recommendation:** {pattern['recommendation']}")
                    else:
                        st.success("✅ No significant field size patterns detected - overall weights work well")

                    # Weight performance visualization
                    with st.expander("📈 View All Weight Combinations Performance"):
                        all_results_df = pd.DataFrame(st.session_state.weight_patterns['all_weight_results'])

                        # Winner accuracy line chart
                        fig_winner = px.line(
                            all_results_df,
                            x='rf_weight',
                            y='winner_accuracy',
                            title='Winner Accuracy vs RF Weight (all tested combinations)',
                            markers=True
                        )
                        fig_winner.update_yaxes(title='Winner Accuracy', tickformat='.1%')
                        fig_winner.update_xaxes(title='RF Weight')
                        fig_winner.update_traces(line_color='#244855', marker_color='#E64833')
                        fig_winner.update_layout(
                            height=400,
                            paper_bgcolor='rgba(0,0,0,0)',
                            plot_bgcolor='rgba(0,0,0,0)'
                        )
                        st.plotly_chart(fig_winner, use_container_width=True)

                        # Data table
                        st.dataframe(all_results_df, hide_index=True, use_container_width=True)

                    # Export options
                    with st.expander("💾 Export Pattern Results"):
                        # Create comprehensive export
                        export_data = {
                            'overall_best': patterns['overall_best'],
                            'race_type_patterns': patterns['by_race_type'],
                            'distance_patterns': patterns['by_distance_range'],
                            'field_size_patterns': patterns['by_field_size']
                        }

                        json_str = json.dumps(export_data, indent=2)

                        st.download_button(
                            label="📥 Download Pattern Analysis (JSON)",
                            data=json_str,
                            file_name=f"weight_pattern_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                            mime="application/json"
                        )

                    # Save to Config.yaml
                    st.markdown("### 💾 Apply Findings to Config.yaml")
                    st.info("⚙️ This will update config.yaml with the optimal weights and detected patterns from this analysis")

                    col1, col2 = st.columns([1, 1])

                    with col1:
                        use_dynamic = st.checkbox(
                            "Enable Dynamic Weights",
                            value=True,
                            help="If checked, will enable pattern-specific weights. If unchecked, only overall weights will be used."
                        )

                    with col2:
                        min_improvement = st.number_input(
                            "Minimum Improvement (%)",
                            min_value=0.0,
                            max_value=20.0,
                            value=3.0,
                            step=0.5,
                            help="Only include patterns with at least this much improvement over overall weights"
                        )

                    if st.button("💾 Save to Config.yaml", key="save_weights_to_config", type="primary", use_container_width=True):
                        try:
                            # Load current config
                            st.session_state.helper.load_config()
                            config_data = st.session_state.helper._config_data

                            # Get overall best weights
                            overall = patterns['overall_best']

                            # Update blend section
                            if 'blend' not in config_data:
                                config_data['blend'] = {}

                            # Set overall weights
                            config_data['blend']['rf_weight'] = float(overall['rf_weight'])
                            config_data['blend']['tabnet_weight'] = float(overall['tabnet_weight'])
                            config_data['blend']['optimal_mae'] = float(overall['mae'])
                            config_data['blend']['use_dynamic_weights'] = use_dynamic

                            # Set default weights (same as overall)
                            config_data['blend']['default_weights'] = {
                                'rf_weight': float(overall['rf_weight']),
                                'tabnet_weight': float(overall['tabnet_weight']),
                                'accuracy': float(overall['winner_accuracy'] * 100),
                                'description': 'Default optimal weights from automated analysis'
                            }

                            # Build dynamic weights list
                            dynamic_weights = []

                            # Add race type patterns
                            for pattern in patterns['by_race_type']:
                                improvement_pct = pattern['improvement_vs_overall'] * 100
                                if improvement_pct >= min_improvement:
                                    dynamic_weights.append({
                                        'condition': {'typec': pattern['typec']},
                                        'weights': {
                                            'rf_weight': float(pattern['optimal_rf_weight']),
                                            'tabnet_weight': float(pattern['optimal_tabnet_weight'])
                                        },
                                        'accuracy': float(pattern['winner_accuracy'] * 100),
                                        'description': f"{pattern['typec']} races - {improvement_pct:.1f}% improvement"
                                    })

                            # Add distance patterns
                            for pattern in patterns['by_distance_range']:
                                improvement_pct = pattern['improvement_vs_overall'] * 100
                                if improvement_pct >= min_improvement:
                                    # Parse distance range like "2000-2500m"
                                    dist_range = pattern['distance_range'].replace('m', '')
                                    if '-' in dist_range:
                                        dist_min, dist_max = map(int, dist_range.split('-'))
                                        dynamic_weights.append({
                                            'condition': {
                                                'dist_min': dist_min,
                                                'dist_max': dist_max
                                            },
                                            'weights': {
                                                'rf_weight': float(pattern['optimal_rf_weight']),
                                                'tabnet_weight': float(pattern['optimal_tabnet_weight'])
                                            },
                                            'accuracy': float(pattern['winner_accuracy'] * 100),
                                            'description': f"Distance {pattern['distance_range']} - {improvement_pct:.1f}% improvement"
                                        })

                            # Add field size patterns
                            for pattern in patterns['by_field_size']:
                                improvement_pct = pattern['improvement_vs_overall'] * 100
                                if improvement_pct >= min_improvement:
                                    # Parse field size like "Small (≤8)", "Medium (9-12)", "Large (13-16)", "Very Large (≥17)"
                                    field_size = pattern['field_size']
                                    condition = {}

                                    if '≤' in field_size:
                                        max_val = int(field_size.split('≤')[1].split(')')[0])
                                        condition['partant_max'] = max_val
                                    elif '≥' in field_size:
                                        min_val = int(field_size.split('≥')[1].split(')')[0])
                                        condition['partant_min'] = min_val
                                    elif '-' in field_size:
                                        # Extract range like "9-12"
                                        range_part = field_size.split('(')[1].split(')')[0]
                                        min_val, max_val = map(int, range_part.split('-'))
                                        condition['partant_min'] = min_val
                                        condition['partant_max'] = max_val

                                    if condition:
                                        dynamic_weights.append({
                                            'condition': condition,
                                            'weights': {
                                                'rf_weight': float(pattern['optimal_rf_weight']),
                                                'tabnet_weight': float(pattern['optimal_tabnet_weight'])
                                            },
                                            'accuracy': float(pattern['winner_accuracy'] * 100),
                                            'description': f"Field size {field_size} - {improvement_pct:.1f}% improvement"
                                        })

                            # Sort by accuracy descending
                            dynamic_weights.sort(key=lambda x: x['accuracy'], reverse=True)

                            # Set dynamic weights
                            config_data['blend']['dynamic_weights'] = dynamic_weights
                            config_data['blend']['description'] = f"Optimal blend weights from automated analysis ({datetime.now().strftime('%Y-%m-%d')})"

                            # Save config
                            success = st.session_state.helper.save_config(config_data)

                            if success:
                                log_output(f"✅ Config updated successfully!", "success")
                                log_output(f"   • Overall weights: RF={overall['rf_weight']:.1f}, TabNet={overall['tabnet_weight']:.1f}", "info")
                                log_output(f"   • Dynamic weights: {len(dynamic_weights)} patterns added", "info")
                                log_output(f"   • Use dynamic weights: {use_dynamic}", "info")
                                st.success(f"✅ Config.yaml updated with {len(dynamic_weights)} dynamic weight patterns!")
                            else:
                                log_output("❌ Failed to save config", "error")
                                st.error("Failed to save config.yaml")

                        except Exception as e:
                            log_output(f"❌ Error saving to config: {str(e)}", "error")
                            st.error(f"Error: {str(e)}")
                            import traceback
                            st.code(traceback.format_exc())

                st.markdown('</div>', unsafe_allow_html=True)

            elif operation == "⚖️ Model Weight Analysis":
                st.markdown('''
                <div class="config-panel">
                    <h3>⚖️ Automated Weight & Pattern Analysis</h3>
                </div>
                ''', unsafe_allow_html=True)

                st.info("🤖 Automated analysis: Tests all RF/TabNet weight combinations (0.0-1.0 by 0.1), finds optimal weights, and detects patterns based on race features")

                # Initialize session state for analysis results
                if 'weight_patterns' not in st.session_state:
                    st.session_state.weight_patterns = None

                # Simple Configuration
                st.markdown("### ⚙️ Configuration")

                left, right = st.columns(2)

                with left:
                    date_from_weight = st.date_input(
                        "Start Date:",
                        value=(datetime.now() - timedelta(days=90)).date(),
                        key="weight_date_from",
                        help="Recommended: 30-90 days for reliable patterns"
                    )

                with right:
                    date_to_weight = st.date_input(
                        "End Date:",
                        value=datetime.now().date(),
                        key="weight_date_to"
                    )

                # Single-click automated analysis
                if st.button("🚀 Run Automated Analysis", key="run_auto_analysis", help="Tests all weights and detects patterns automatically"):
                    with st.spinner("📊 Loading race data..."):
                        # Load data
                        data_result = st.session_state.helper.load_weight_analysis_data(
                            date_from=date_from_weight.strftime('%Y-%m-%d'),
                            date_to=date_to_weight.strftime('%Y-%m-%d'),
                            race_filters=None
                        )

                        if data_result['success']:
                            log_output(data_result['message'], "success")

                            # Run automated pattern detection
                            with st.spinner("🔍 Testing all weight combinations (0.0-1.0 by 0.1) and detecting patterns..."):
                                pattern_result = st.session_state.helper.detect_weight_patterns(
                                    race_data=data_result['race_data'],
                                    weight_step=0.1
                                )

                                if pattern_result['success']:
                                    st.session_state.weight_patterns = pattern_result
                                    log_output(pattern_result['message'], "success")
                                else:
                                    log_output(pattern_result['message'], "error")
                        else:
                            log_output(data_result['message'], "error")

                        st.rerun()

                # Display Results
                if st.session_state.weight_patterns is not None:
                    patterns = st.session_state.weight_patterns['patterns']

                    st.markdown("### 📊 Analysis Results")

                    # Summary insights
                    st.markdown("#### 💡 Key Findings")
                    for insight in patterns['summary']:
                        if insight['type'] == 'no_patterns':
                            st.success(f"✅ {insight['message']}")
                        elif insight['type'] == 'patterns_found':
                            st.warning(f"⚠️ {insight['message']}")
                        else:
                            st.info(f"📌 {insight['message']}")

                    # Overall Optimal Weights
                    overall = patterns['overall_best']

                    st.markdown("#### 🎯 Overall Best Weights")
                    tab1, tab2, tab3, tab4, tab5 = st.columns(5)
                    with tab1:
                        st.metric("RF Weight", f"{overall['rf_weight']:.1f}")
                    with tab2:
                        st.metric("TabNet Weight", f"{overall['tabnet_weight']:.1f}")
                    with tab3:
                        st.metric("Winner Accuracy", f"{overall['winner_accuracy']*100:.1f}%")
                    with tab4:
                        st.metric("Podium Accuracy", f"{overall['podium_accuracy']*100:.1f}%")
                    with tab5:
                        st.metric("MAE", f"{overall['mae']:.2f}")

                    # Pattern-specific recommendations
                    st.markdown("#### 🔍 Detected Patterns Requiring Custom Weights")

                    # Race Type Patterns
                    if patterns['by_race_type']:
                        st.markdown("##### 🏇 Race Type Patterns")

                        for pattern in patterns['by_race_type']:
                            with st.expander(f"**{pattern['typec']}** - Custom weights recommended"):
                                pcol1, pcol2, pcol3, pcol4 = st.columns(4)

                                with pcol1:
                                    st.metric("Optimal RF", f"{pattern['optimal_rf_weight']:.1f}")
                                with pcol2:
                                    st.metric("Optimal TabNet", f"{pattern['optimal_tabnet_weight']:.1f}")
                                with pcol3:
                                    st.metric("Winner Accuracy", f"{pattern['winner_accuracy']*100:.1f}%")
                                    if pattern['improvement_vs_overall'] > 0:
                                        st.success(f"+{pattern['improvement_vs_overall']*100:.1f}% vs overall")
                                with pcol4:
                                    st.metric("Races", pattern['total_races'])

                                st.info(f"💡 **Recommendation:** {pattern['recommendation']}")
                    else:
                        st.success("✅ No significant race type patterns detected - overall weights work well")

                    # Distance Range Patterns
                    if patterns['by_distance_range']:
                        st.markdown("##### 📏 Distance Range Patterns")

                        for pattern in patterns['by_distance_range']:
                            with st.expander(f"**{pattern['distance_range']}** - Custom weights recommended"):
                                pcol1, pcol2, pcol3, pcol4 = st.columns(4)

                                with pcol1:
                                    st.metric("Optimal RF", f"{pattern['optimal_rf_weight']:.1f}")
                                with pcol2:
                                    st.metric("Optimal TabNet", f"{pattern['optimal_tabnet_weight']:.1f}")
                                with pcol3:
                                    st.metric("Winner Accuracy", f"{pattern['winner_accuracy']*100:.1f}%")
                                    if pattern['improvement_vs_overall'] > 0:
                                        st.success(f"+{pattern['improvement_vs_overall']*100:.1f}% vs overall")
                                with pcol4:
                                    st.metric("Races", pattern['total_races'])

                                st.info(f"💡 **Recommendation:** {pattern['recommendation']}")
                    else:
                        st.success("✅ No significant distance patterns detected - overall weights work well")

                    # Field Size Patterns
                    if patterns['by_field_size']:
                        st.markdown("##### 👥 Field Size Patterns")

                        for pattern in patterns['by_field_size']:
                            with st.expander(f"**{pattern['field_size']}** - Custom weights recommended"):
                                pcol1, pcol2, pcol3, pcol4 = st.columns(4)

                                with pcol1:
                                    st.metric("Optimal RF", f"{pattern['optimal_rf_weight']:.1f}")
                                with pcol2:
                                    st.metric("Optimal TabNet", f"{pattern['optimal_tabnet_weight']:.1f}")
                                with pcol3:
                                    st.metric("Winner Accuracy", f"{pattern['winner_accuracy']*100:.1f}%")
                                    if pattern['improvement_vs_overall'] > 0:
                                        st.success(f"+{pattern['improvement_vs_overall']*100:.1f}% vs overall")
                                with pcol4:
                                    st.metric("Races", pattern['total_races'])

                                st.info(f"💡 **Recommendation:** {pattern['recommendation']}")
                    else:
                        st.success("✅ No significant field size patterns detected - overall weights work well")

                    # Weight performance visualization
                    with st.expander("📈 View All Weight Combinations Performance"):
                        all_results_df = pd.DataFrame(st.session_state.weight_patterns['all_weight_results'])

                        # Winner accuracy line chart
                        fig_winner = px.line(
                            all_results_df,
                            x='rf_weight',
                            y='winner_accuracy',
                            title='Winner Accuracy vs RF Weight (all tested combinations)',
                            markers=True
                        )
                        fig_winner.update_yaxes(title='Winner Accuracy', tickformat='.1%')
                        fig_winner.update_xaxes(title='RF Weight')
                        fig_winner.update_traces(line_color='#244855', marker_color='#E64833')
                        fig_winner.update_layout(
                            height=400,
                            paper_bgcolor='rgba(0,0,0,0)',
                            plot_bgcolor='rgba(0,0,0,0)'
                        )
                        st.plotly_chart(fig_winner, use_container_width=True)

                        # Data table
                        st.dataframe(all_results_df, hide_index=True, use_container_width=True)

                    # Export options
                    with st.expander("💾 Export Pattern Results"):
                        # Create comprehensive export
                        export_data = {
                            'overall_best': patterns['overall_best'],
                            'race_type_patterns': patterns['by_race_type'],
                            'distance_patterns': patterns['by_distance_range'],
                            'field_size_patterns': patterns['by_field_size']
                        }

                        import json
                        json_str = json.dumps(export_data, indent=2)

                        st.download_button(
                            label="📥 Download Pattern Analysis (JSON)",
                            data=json_str,
                            file_name=f"weight_pattern_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                            mime="application/json"
                        )

                st.markdown('</div>', unsafe_allow_html=True)

            elif operation == "🔄 Incremental Training":

                st.markdown('<div class="config-panel">', unsafe_allow_html=True)

                st.markdown("### 🔄 Incremental Training & Calibration")

                # Create tabs for different training approaches
                tab_retrain, tab_calibrate = st.tabs(["🔄 Model Retraining", "🎯 Calibration System"])

                with tab_retrain:
                    st.info("Process completed races with predictions and results to improve model performance through retraining")

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

                                    # TabNet Model improvement metrics
                                    if 'tabnet_training' in training_results:
                                        tabnet_results = training_results['tabnet_training']

                                        if tabnet_results.get('status') == 'success':
                                            tabnet_improvement = tabnet_results.get('improvement', {})
                                            st.markdown("### 🧠 TabNet Model Improvement")

                                            left, right = st.columns(2)
                                            with left:
                                                tabnet_mae_improvement = tabnet_improvement.get('mae_improvement_pct', 0)
                                                st.metric(
                                                    "TabNet MAE Improvement",
                                                    f"{tabnet_mae_improvement:.2f}%",
                                                    delta=f"{tabnet_mae_improvement:.2f}%"
                                                )
                                            with right:
                                                tabnet_rmse_improvement = tabnet_improvement.get('rmse_improvement_pct', 0)
                                                st.metric(
                                                    "TabNet RMSE Improvement",
                                                    f"{tabnet_rmse_improvement:.2f}%",
                                                    delta=f"{tabnet_rmse_improvement:.2f}%"
                                                )

                                            if tabnet_improvement.get('significant', False):
                                                st.success("🎉 TabNet model shows significant improvement!")
                                            else:
                                                st.info("ℹ️ TabNet improvement below threshold.")

                                        elif tabnet_results.get('status') == 'skipped':
                                            st.warning("⚠️ TabNet training was skipped")
                                            if tabnet_results.get('message'):
                                                st.info(f"Reason: {tabnet_results['message']}")

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
                                                if models_updated.get('tabnet', False):
                                                    st.info("🔄 TabNet model: Retrained")
                                                else:
                                                    st.info("📋 TabNet model: Copied from base")

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

                with tab_calibrate:
                    st.info("🎯 Detect and correct systematic biases without retraining - fast, automatic, and effective")

                    # Calibration tabs
                    cal_tabs = st.tabs(["📊 Status & Check", "🔧 Manual Calibration", "📈 Reports"])

                    with cal_tabs[0]:  # Status & Check tab
                        st.markdown("### 📊 Calibration Status")

                        # Check if calibration files exist
                        general_cal_path = 'models/calibration/general_calibration.json'
                        quinte_cal_path = 'models/calibration/quinte_calibration.json'

                        general_exists = os.path.exists(general_cal_path)
                        quinte_exists = os.path.exists(quinte_cal_path)

                        col1, col2 = st.columns(2)

                        with col1:
                            st.markdown("#### 🎲 General Model")
                            if general_exists:
                                try:
                                    with open(general_cal_path, 'r') as f:
                                        cal_data = json.load(f)

                                    st.success("✅ Calibration active")
                                    st.metric("Version", cal_data.get('version', 'Unknown')[:19])
                                    st.metric("Corrections", len(cal_data.get('corrections', {})))

                                    if 'validation' in cal_data:
                                        val = cal_data['validation']
                                        improvement = val.get('improvement_pct', 0)
                                        st.metric("Improvement", f"{improvement:+.1f}%", delta=f"{improvement:+.1f}%")
                                except Exception as e:
                                    st.error(f"Error loading calibration: {e}")
                            else:
                                st.warning("⚠️ No calibration")
                                st.info("Run initial calibration to create")

                        with col2:
                            st.markdown("#### 🏇 Quinté Model")
                            if quinte_exists:
                                try:
                                    with open(quinte_cal_path, 'r') as f:
                                        cal_data = json.load(f)

                                    st.success("✅ Calibration active")
                                    st.metric("Version", cal_data.get('version', 'Unknown')[:19])
                                    st.metric("Corrections", len(cal_data.get('corrections', {})))

                                    if 'validation' in cal_data:
                                        val = cal_data['validation']
                                        improvement = val.get('improvement_pct', 0)
                                        st.metric("Improvement", f"{improvement:+.1f}%", delta=f"{improvement:+.1f}%")
                                except Exception as e:
                                    st.error(f"Error loading calibration: {e}")
                            else:
                                st.warning("⚠️ No calibration")
                                st.info("Run initial calibration to create")

                        st.markdown("---")
                        st.markdown("### 🔍 Health Check & Update")

                        check_model = st.selectbox(
                            "Select Model:",
                            ["Both", "General", "Quinté"],
                            help="Which model to check and update"
                        )

                        force_update = st.checkbox(
                            "Force Update",
                            value=False,
                            help="Force update even if calibration is healthy"
                        )

                        if st.button("🔍 Check Calibration Health", key="check_cal_btn", use_container_width=True):
                            with st.spinner("Checking calibration health..."):
                                try:
                                    from scripts.calibrate_models import check_and_update_calibration

                                    db_path = st.session_state.helper.config_path.replace('config.yaml', 'data/hippique2.db')

                                    models_to_check = []
                                    if check_model == "Both":
                                        models_to_check = ['general', 'quinte']
                                    elif check_model == "General":
                                        models_to_check = ['general']
                                    else:
                                        models_to_check = ['quinte']

                                    for model in models_to_check:
                                        st.markdown(f"#### {model.title()} Model")

                                        calibrator, detector, metrics = check_and_update_calibration(
                                            db_path,
                                            model_type=model,
                                            force=force_update
                                        )

                                        # Display metrics
                                        if metrics:
                                            m1, m2, m3 = st.columns(3)
                                            with m1:
                                                st.metric("Current MAE", f"{metrics.get('current_mae', 0):.3f}")
                                            with m2:
                                                st.metric("Baseline MAE", f"{metrics.get('baseline_mae', 0):.3f}")
                                            with m3:
                                                st.metric("Samples", metrics.get('n_samples', 0))

                                            if metrics.get('should_update'):
                                                st.success(f"✅ Updated: {metrics.get('reason')}")
                                            else:
                                                st.info(f"ℹ️ {metrics.get('reason')}")

                                except Exception as e:
                                    st.error(f"Error checking calibration: {e}")
                                    import traceback
                                    with st.expander("Error details"):
                                        st.code(traceback.format_exc())

                    with cal_tabs[1]:  # Manual Calibration tab
                        st.markdown("### 🔧 Manual Calibration")

                        st.info("Run initial calibration or force rebuild calibration from historical data")

                        calib_model = st.selectbox(
                            "Select Model:",
                            ["General", "Quinté", "Both"],
                            key="calib_model_select",
                            help="Which model to calibrate"
                        )

                        calib_days = st.slider(
                            "Training Days:",
                            min_value=30,
                            max_value=180,
                            value=90,
                            help="Number of days of historical data to use"
                        )

                        val_split = st.slider(
                            "Validation Split:",
                            min_value=0.1,
                            max_value=0.4,
                            value=0.2,
                            step=0.05,
                            help="Fraction of data to use for validation"
                        )

                        if st.button("🚀 Run Calibration", key="run_cal_btn", type="primary", use_container_width=True):
                            with st.spinner("Running calibration analysis..."):
                                try:
                                    from scripts.calibrate_models import calibrate_model

                                    db_path = st.session_state.helper.config_path.replace('config.yaml', 'data/hippique2.db')

                                    models_to_calibrate = []
                                    if calib_model == "Both":
                                        models_to_calibrate = ['general', 'quinte']
                                    elif calib_model == "General":
                                        models_to_calibrate = ['general']
                                    else:
                                        models_to_calibrate = ['quinte']

                                    for model in models_to_calibrate:
                                        st.markdown(f"#### {model.title()} Model")

                                        # Create progress placeholder
                                        progress_text = st.empty()
                                        progress_text.text(f"Loading {calib_days} days of predictions...")

                                        calibrator, detector = calibrate_model(
                                            db_path,
                                            model_type=model,
                                            validation_split=val_split
                                        )

                                        if calibrator:
                                            st.success(f"✅ Calibration complete for {model}")

                                            # Display results
                                            if 'validation' in calibrator.calibrations:
                                                val = calibrator.calibrations['validation']
                                                v1, v2, v3 = st.columns(3)
                                                with v1:
                                                    st.metric("Before MAE", f"{val['mae_before']:.3f}")
                                                with v2:
                                                    st.metric("After MAE", f"{val['mae_after']:.3f}")
                                                with v3:
                                                    improvement = val['improvement_pct']
                                                    st.metric("Improvement", f"{improvement:+.1f}%", delta=f"{improvement:+.1f}%")
                                        else:
                                            st.warning(f"No significant biases detected for {model}")

                                        st.markdown("---")

                                except Exception as e:
                                    st.error(f"Error during calibration: {e}")
                                    import traceback
                                    with st.expander("Error details"):
                                        st.code(traceback.format_exc())

                    with cal_tabs[2]:  # Reports tab
                        st.markdown("### 📈 Calibration Reports")

                        report_model = st.selectbox(
                            "Select Model:",
                            ["General", "Quinté"],
                            key="report_model_select"
                        )

                        report_path = f"models/calibration/{report_model.lower()}_calibration_report.txt"

                        if os.path.exists(report_path):
                            with open(report_path, 'r') as f:
                                report_content = f.read()

                            st.text_area(
                                "Calibration Report",
                                value=report_content,
                                height=400,
                                key=f"{report_model}_report"
                            )

                            # Download button
                            st.download_button(
                                label="📥 Download Report",
                                data=report_content,
                                file_name=f"{report_model.lower()}_calibration_report.txt",
                                mime="text/plain"
                            )
                        else:
                            st.warning(f"No report found for {report_model} model")
                            st.info("Run calibration to generate a report")

                        # Calibration JSON viewer
                        st.markdown("---")
                        st.markdown("### 🔧 Calibration Details (JSON)")

                        json_path = f"models/calibration/{report_model.lower()}_calibration.json"

                        if os.path.exists(json_path):
                            with open(json_path, 'r') as f:
                                cal_json = json.load(f)

                            st.json(cal_json)

                            # Download JSON
                            st.download_button(
                                label="📥 Download Calibration JSON",
                                data=json.dumps(cal_json, indent=2),
                                file_name=f"{report_model.lower()}_calibration.json",
                                mime="application/json"
                            )
                        else:
                            st.warning(f"No calibration found for {report_model} model")

            elif operation == "🏆 Quinté Incremental Training":

                st.markdown('<div class="config-panel">', unsafe_allow_html=True)

                st.markdown("### 🏆 Quinté Incremental Training")

                st.info("Specialized incremental training for quinté predictions - learns from failures and improves quinté accuracy")

                # Training parameters
                left, right = st.columns(2)

                with left:
                    quinte_date_from = st.date_input(
                        "Start Date:",
                        value=(datetime.now() - timedelta(days=60)).date(),
                        key="quinte_incr_date_from"
                    )

                    focus_on_failures = st.checkbox(
                        "Focus on Failures",
                        value=True,
                        help="Weight training samples by failure severity (quinté misses get 10x weight)"
                    )

                with right:
                    quinte_date_to = st.date_input(
                        "End Date:",
                        value=datetime.now().date(),
                        key="quinte_incr_date_to"
                    )

                # Advanced options
                with st.expander("🔧 Advanced Options"):
                    quinte_limit_races = st.number_input(
                        "Limit Races (0 = no limit):",
                        min_value=0,
                        value=0,
                        help="Maximum number of quinté races to process",
                        key="quinte_limit"
                    )

                # Fetch quinté races with results
                try:
                    from utils.env_setup import AppConfig
                    import sqlite3

                    config = AppConfig()
                    db_path = config.get_active_db_path()
                    conn = sqlite3.connect(db_path)
                    cursor = conn.cursor()

                    # Query quinté races with predictions and results
                    cursor.execute("""
                        SELECT COUNT(*)
                        FROM daily_race
                        WHERE quinte = 1
                        AND actual_results IS NOT NULL
                        AND actual_results != 'pending'
                        AND prediction_results IS NOT NULL
                        AND jour >= ? AND jour <= ?
                    """, (quinte_date_from.strftime('%Y-%m-%d'), quinte_date_to.strftime('%Y-%m-%d')))

                    quinte_races_count = cursor.fetchone()[0]
                    conn.close()

                    if quinte_races_count > 0:
                        st.success(f"Found {quinte_races_count} quinté races with predictions and results")

                        # Show quinté race stats
                        with st.expander("📋 Quinté Races Ready for Training"):
                            st.info(f"Total quinté races available: **{quinte_races_count}**")
                            if quinte_limit_races > 0 and quinte_limit_races < quinte_races_count:
                                st.warning(f"Will process only the first **{quinte_limit_races}** races")
                    else:
                        st.warning("No quinté races with both predictions and results found for the selected date range")

                except Exception as e:
                    st.error(f"Error fetching quinté races: {str(e)}")
                    quinte_races_count = 0

                # Training execution
                if st.button("🚀 Start Quinté Incremental Training", key="quinte_incr_btn"):

                    if quinte_races_count > 0:

                        # Create progress bar and status text
                        progress_bar = st.progress(0)
                        status_text = st.empty()

                        execute_quinte_incremental_training(
                            quinte_date_from.strftime('%Y-%m-%d'),
                            quinte_date_to.strftime('%Y-%m-%d'),
                            quinte_limit_races if quinte_limit_races > 0 else None,
                            focus_on_failures,
                            progress_bar,
                            status_text
                        )

                        # Display results if stored in session state
                        if 'quinte_incremental_results' in st.session_state:
                            results = st.session_state.quinte_incremental_results

                            if results.get('success'):
                                st.success("✅ Quinté incremental training completed!")

                                # Display training metrics
                                training_results = results.get('training_results', {})

                                if training_results:
                                    st.markdown("### 📊 Quinté Training Results")

                                    # Baseline vs Improved metrics
                                    baseline = training_results.get('baseline_metrics', {})
                                    improved = training_results.get('improved_metrics', {})

                                    if baseline and improved:
                                        st.markdown("### 🎯 Performance Improvement")

                                        col1, col2, col3 = st.columns(3)

                                        with col1:
                                            baseline_desordre = baseline.get('quinte_desordre_rate', 0) * 100
                                            improved_desordre = improved.get('quinte_desordre_rate', 0) * 100
                                            delta_desordre = improved_desordre - baseline_desordre

                                            st.metric(
                                                "Quinté Désordre Rate",
                                                f"{improved_desordre:.1f}%",
                                                delta=f"{delta_desordre:+.1f}%",
                                                help="Exact 5 horses in any order"
                                            )

                                        with col2:
                                            baseline_bonus4 = baseline.get('bonus_4_rate', 0) * 100
                                            improved_bonus4 = improved.get('bonus_4_rate', 0) * 100
                                            delta_bonus4 = improved_bonus4 - baseline_bonus4

                                            st.metric(
                                                "Bonus 4 Rate",
                                                f"{improved_bonus4:.1f}%",
                                                delta=f"{delta_bonus4:+.1f}%",
                                                help="4 of top 5 horses correct"
                                            )

                                        with col3:
                                            baseline_bonus3 = baseline.get('bonus_3_rate', 0) * 100
                                            improved_bonus3 = improved.get('bonus_3_rate', 0) * 100
                                            delta_bonus3 = improved_bonus3 - baseline_bonus3

                                            st.metric(
                                                "Bonus 3 Rate",
                                                f"{improved_bonus3:.1f}%",
                                                delta=f"{delta_bonus3:+.1f}%",
                                                help="3 of top 5 horses correct"
                                            )

                                        # MAE metric
                                        col4, col5 = st.columns(2)
                                        with col4:
                                            baseline_mae = baseline.get('avg_mae', 0)
                                            st.metric("Baseline MAE", f"{baseline_mae:.2f}")
                                        with col5:
                                            improved_mae = improved.get('avg_mae', 0)
                                            mae_improvement = baseline_mae - improved_mae
                                            st.metric("Improved MAE", f"{improved_mae:.2f}", delta=f"{-mae_improvement:+.2f}")

                                    # Failure patterns analysis
                                    if 'failure_patterns' in training_results:
                                        patterns = training_results['failure_patterns']

                                        st.markdown("### 📉 Failure Pattern Analysis")

                                        with st.expander("🔍 View Detailed Patterns"):
                                            col1, col2 = st.columns(2)

                                            with col1:
                                                st.markdown("**Pattern Distribution:**")
                                                missed_fav = patterns.get('missed_favorites_pct', 0)
                                                missed_long = patterns.get('missed_longshots_pct', 0)
                                                st.metric("Missed Favorites", f"{missed_fav:.1f}%")
                                                st.metric("Missed Longshots", f"{missed_long:.1f}%")

                                            with col2:
                                                st.markdown("**Common Issues:**")
                                                high_mae = patterns.get('high_mae_races', 0)
                                                st.metric("High MAE Races", high_mae)

                                    # Corrections applied
                                    if 'corrections_applied' in training_results:
                                        corrections = training_results['corrections_applied']

                                        if corrections:
                                            st.markdown("### 🔧 Corrections Applied")

                                            for correction in corrections:
                                                category = correction.get('category', 'Unknown')
                                                suggestion = correction.get('suggestion', '')
                                                priority = correction.get('priority', 'medium')

                                                priority_emoji = {'high': '🔴', 'medium': '🟡', 'low': '🟢'}.get(priority, '⚪')

                                                st.info(f"{priority_emoji} **{category}**: {suggestion}")

                                    # Model saved
                                    model_saved = training_results.get('model_saved', '')
                                    if model_saved:
                                        st.markdown("### 💾 Model Saved")
                                        model_name = model_saved.split('/')[-1] if '/' in model_saved else model_saved
                                        st.success(f"✅ New quinté model saved: **{model_name}**")

                                    # Execution summary
                                    races_processed = training_results.get('races_processed', 0)
                                    execution_time = training_results.get('execution_time', 0)

                                    st.markdown("### ⏱️ Execution Summary")
                                    col1, col2 = st.columns(2)
                                    with col1:
                                        st.metric("Quinté Races Processed", races_processed)
                                    with col2:
                                        st.metric("Execution Time", f"{execution_time:.1f}s")

                            else:
                                st.error(f"❌ Quinté training failed: {results.get('message', 'Unknown error')}")
                                if 'error' in results:
                                    with st.expander("🔍 View Error Details"):
                                        st.code(results['error'])

                    else:
                        st.warning("⚠️ No quinté races available for training")

                st.markdown('</div>', unsafe_allow_html=True)

    with status:
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
        
        # Display model information - simplified view
        st.markdown("**Models:**")

        # Get model paths from config
        model_paths = prediction_info.get('model_paths', {})

        # Check if models exist and their status
        col_rf, col_tabnet = st.columns(2)

        with col_rf:
            rf_path = model_paths.get('rf', '')
            if rf_path:
                # Check if RF model files exist (no hybrid prefix)
                rf_model_file = f"{rf_path}/rf_model.joblib"
                import os
                rf_exists = os.path.exists(rf_model_file)
                rf_status = "✅ Ready" if rf_exists else "⚠️ Missing Files"

                st.metric("Random Forest", rf_status)
                if not rf_exists:
                    st.caption(f"Expected: {rf_model_file}")
            else:
                st.metric("Random Forest", "❌ Not Configured")

        with col_tabnet:
            tabnet_path = model_paths.get('tabnet', '')
            if tabnet_path:
                # Check if TabNet model files exist (zip format)
                tabnet_model_file = f"{tabnet_path}/tabnet_model.zip"
                import os
                tabnet_exists = os.path.exists(tabnet_model_file)
                tabnet_status = "✅ Ready" if tabnet_exists else "⚠️ Missing Files"

                st.metric("TabNet", tabnet_status)
                if not tabnet_exists:
                    st.caption(f"Expected: {tabnet_model_file}")
            else:
                st.metric("TabNet", "❌ Not Configured")

        # Show blend weights
        blend_weights = prediction_info.get('blend_weights', {})
        if blend_weights:
            st.markdown("**Blend Configuration:**")
            col_w1, col_w2 = st.columns(2)
            with col_w1:
                st.metric("RF Weight", f"{blend_weights.get('rf_weight', 0):.1f}")
            with col_w2:
                st.metric("TabNet Weight", f"{blend_weights.get('tabnet_weight', 0):.1f}")

        # Overall status
        rf_ready = rf_path and os.path.exists(f"{rf_path}/rf_model.joblib") if rf_path else False
        tabnet_ready = tabnet_path and os.path.exists(f"{tabnet_path}/tabnet_model.zip") if tabnet_path else False

        if rf_ready and tabnet_ready:
            st.success("🟢 All models ready for predictions")
        elif rf_ready or tabnet_ready:
            st.warning("🟡 Some models ready - partial functionality available")
        else:
            st.error("🔴 No models available - training required")

        # Training Thread Status
        if hasattr(st.session_state, 'helper'):
            thread_status = st.session_state.helper.get_training_status()

            if thread_status.get('is_training', False):
                st.markdown("### 🔄 Training Status")

                # Show training progress if available
                if hasattr(st.session_state, 'training_progress'):
                    progress_val = st.session_state.get('training_progress', 0)
                    st.progress(progress_val / 100)
                    st.caption(st.session_state.get('training_status', 'Training in progress...'))

                # Show thread info in an expander
                with st.expander("📊 Thread Details", expanded=False):
                    col_t1, col_t2 = st.columns(2)
                    with col_t1:
                        st.metric("Process ID", thread_status.get('process_id', 'N/A'))
                        st.metric("Thread Name", thread_status.get('thread_name', 'N/A'))
                    with col_t2:
                        st.metric("Thread ID", thread_status.get('thread_id', 'N/A'))
                        thread_alive = thread_status.get('thread_alive', False)
                        alive_status = "✅ Running" if thread_alive else "❌ Stopped"
                        st.metric("Thread Status", alive_status)
            else:
                # Show idle status
                st.markdown("### ⏸️ Training Status")
                st.info("No training in progress")
        
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
        
        # Display model information - simplified view
        st.markdown("**Models:**")

        # Get model paths from config
        model_paths = prediction_info.get('model_paths', {})

        # Check if models exist and their status
        col_rf, col_tabnet = st.columns(2)

        with col_rf:
            rf_path = model_paths.get('rf', '')
            if rf_path:
                # Check if RF model files exist (no hybrid prefix)
                rf_model_file = f"{rf_path}/rf_model.joblib"
                rf_exists = os.path.exists(rf_model_file)
                rf_status = "✅ Ready" if rf_exists else "⚠️ Missing Files"

                st.metric("Random Forest", rf_status)
                if not rf_exists:
                    st.caption(f"Expected: {rf_model_file}")
            else:
                st.metric("Random Forest", "❌ Not Configured")

        with col_tabnet:
            tabnet_path = model_paths.get('tabnet', '')
            if tabnet_path:
                # Check if TabNet model files exist (zip format)
                tabnet_model_file = f"{tabnet_path}/tabnet_model.zip"
                tabnet_exists = os.path.exists(tabnet_model_file)
                tabnet_status = "✅ Ready" if tabnet_exists else "⚠️ Missing Files"

                st.metric("TabNet", tabnet_status)
                if not tabnet_exists:
                    st.caption(f"Expected: {tabnet_model_file}")
            else:
                st.metric("TabNet", "❌ Not Configured")

        # Show blend weights
        blend_weights = prediction_info.get('blend_weights', {})
        if blend_weights:
            st.markdown("**Blend Configuration:**")
            col_w1, col_w2 = st.columns(2)
            with col_w1:
                st.metric("RF Weight", f"{blend_weights.get('rf_weight', 0):.1f}")
            with col_w2:
                st.metric("TabNet Weight", f"{blend_weights.get('tabnet_weight', 0):.1f}")

        # Overall status
        rf_ready = rf_path and os.path.exists(f"{rf_path}/rf_model.joblib") if rf_path else False
        tabnet_ready = tabnet_path and os.path.exists(f"{tabnet_path}/tabnet_model.zip") if tabnet_path else False

        if rf_ready and tabnet_ready:
            st.success("🟢 All models ready for predictions")
        elif rf_ready or tabnet_ready:
            st.warning("🟡 Some models ready - partial functionality available")
        else:
            st.error("🔴 No models available - training required")

        st.markdown('</div>', unsafe_allow_html=True)

        # Output logs
        if st.button("🗑️ Clear Logs"):
            clear_logs()

        display_logs()


if __name__ == "__main__":
    main()