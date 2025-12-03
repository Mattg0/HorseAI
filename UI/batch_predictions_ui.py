"""
Batch Predictions UI Component for Streamlit

Provides UI for launching and monitoring batch prediction jobs.
Can be integrated into main UIApp.py or used as standalone page.
"""

import streamlit as st
from datetime import datetime, timedelta
import time
from pathlib import Path

from utils.batch_job_manager import BatchJobManager
from utils.env_setup import AppConfig


def format_duration(seconds: float) -> str:
    """Format duration in human-readable format."""
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        return f"{seconds/60:.1f}min"
    else:
        hours = int(seconds / 3600)
        minutes = int((seconds % 3600) / 60)
        return f"{hours}h {minutes}m"


def render_batch_predictions_ui():
    """Render the batch predictions UI."""

    # Initialize job manager
    config = AppConfig()
    db_path = config.get_sqlite_dbpath(config._config.base.active_db)
    job_manager = BatchJobManager(db_path)

    # Header
    st.markdown("""
    <div class="config-panel">
        <h2>🚀 Batch Predictions</h2>
        <p>Run predictions on hundreds or thousands of races in parallel with progress tracking.</p>
    </div>
    """, unsafe_allow_html=True)

    # Create tabs
    tab1, tab2, tab3 = st.tabs(["▶️ Launch Job", "📊 Active Jobs", "📜 Job History"])

    # TAB 1: Launch Job
    with tab1:
        st.markdown("### Configure Batch Job")

        col1, col2 = st.columns(2)

        with col1:
            # Job configuration
            job_mode = st.radio(
                "Prediction Mode:",
                ["Unpredicted Races Only", "All Races (Force Reprediction)", "Specific Date", "Custom Race IDs"],
                help="Choose which races to predict"
            )

            if job_mode == "Specific Date":
                prediction_date = st.date_input(
                    "Race Date:",
                    value=datetime.now().date(),
                    help="Predict all races from this date"
                )
            elif job_mode == "Custom Race IDs":
                race_ids_text = st.text_area(
                    "Race IDs (one per line):",
                    help="Enter race IDs, one per line"
                )

            limit = st.number_input(
                "Limit (optional):",
                min_value=0,
                value=0,
                help="Maximum number of races to predict (0 = no limit)"
            )

        with col2:
            # Performance settings
            st.markdown("**Performance Settings**")

            workers = st.number_input(
                "Workers:",
                min_value=-1,
                max_value=32,
                value=-1,
                help="-1 = all CPU cores, or specify number of parallel workers"
            )

            chunk_size = st.number_input(
                "Chunk Size:",
                min_value=10,
                max_value=200,
                value=50,
                help="Races per chunk (smaller = less memory, larger = faster)"
            )

            max_memory = st.number_input(
                "Max Memory (MB):",
                min_value=512,
                max_value=16384,
                value=4096,
                help="Maximum memory usage before cleanup"
            )

        # Launch button
        st.markdown("---")

        if st.button("🚀 Launch Batch Prediction", type="primary", use_container_width=True):
            # Prepare parameters
            kwargs = {
                'workers': workers,
                'chunk_size': chunk_size,
                'max_memory_mb': max_memory,
            }

            # Add mode-specific parameters
            if job_mode == "Specific Date":
                kwargs['date'] = prediction_date.strftime('%Y-%m-%d')
            elif job_mode == "Custom Race IDs":
                race_ids = [rid.strip() for rid in race_ids_text.split('\n') if rid.strip()]
                if not race_ids:
                    st.error("Please enter at least one race ID")
                    return
                kwargs['race_ids'] = race_ids
            elif job_mode == "All Races (Force Reprediction)":
                kwargs['force_reprediction'] = True

            if limit > 0:
                kwargs['limit'] = limit

            # Launch job
            try:
                job_id = job_manager.launch_job(**kwargs)
                st.success(f"✅ Job launched successfully!")
                st.info(f"Job ID: `{job_id}`")
                st.info("Switch to the 'Active Jobs' tab to monitor progress.")

                # Wait a moment for job to start
                time.sleep(1)
                st.rerun()

            except Exception as e:
                st.error(f"❌ Failed to launch job: {e}")

    # TAB 2: Active Jobs
    with tab2:
        st.markdown("### Currently Running Jobs")

        # Auto-refresh control
        col1, col2, col3 = st.columns([2, 1, 1])
        with col1:
            auto_refresh = st.checkbox("Auto-refresh (5 seconds)", value=True)
        with col2:
            if st.button("🔄 Refresh Now"):
                st.rerun()
        with col3:
            if auto_refresh:
                st.markdown(f"*Next refresh in {5}s*")

        # Get active jobs
        active_jobs = job_manager.get_active_jobs()

        if not active_jobs:
            st.info("No active jobs. Launch a new job from the 'Launch Job' tab.")
        else:
            for job in active_jobs:
                with st.container():
                    # Job header
                    col1, col2, col3 = st.columns([3, 1, 1])

                    with col1:
                        st.markdown(f"**Job:** `{job.job_id}`")

                    with col2:
                        if job.duration_seconds:
                            st.markdown(f"**Duration:** {format_duration(job.duration_seconds)}")

                    with col3:
                        throughput = job.processed_races / job.duration_seconds if job.duration_seconds and job.duration_seconds > 0 else 0
                        if throughput > 0:
                            st.markdown(f"**Speed:** {throughput:.1f} races/s")

                    # Progress bar
                    st.progress(job.progress / 100.0)
                    st.markdown(f"**Progress:** {job.progress}% - {job.message}")

                    # Stats
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Total Races", job.total_races)
                    with col2:
                        st.metric("Processed", job.processed_races)
                    with col3:
                        st.metric("Successful", job.successful_races)
                    with col4:
                        remaining = job.total_races - job.processed_races
                        st.metric("Remaining", remaining)

                    # Config details (collapsible)
                    with st.expander("Job Configuration"):
                        st.json(job.config)

                    st.markdown("---")

        # Auto-refresh mechanism
        if auto_refresh and active_jobs:
            time.sleep(5)
            st.rerun()

    # TAB 3: Job History
    with tab3:
        st.markdown("### Recent Jobs")

        # Controls
        col1, col2, col3 = st.columns([2, 1, 1])
        with col1:
            history_limit = st.selectbox("Show:", [10, 25, 50, 100], index=0)
        with col2:
            status_filter = st.selectbox("Status:", ["All", "Completed", "Failed", "Running"], index=0)
        with col3:
            if st.button("🧹 Cleanup Old Jobs", help="Delete completed/failed jobs older than 7 days"):
                deleted = job_manager.cleanup_old_jobs(days=7)
                st.success(f"Deleted {deleted} old jobs")
                time.sleep(1)
                st.rerun()

        # Get job history
        if status_filter == "All":
            jobs = job_manager.get_recent_jobs(limit=history_limit)
        else:
            jobs = job_manager.list_jobs(status=status_filter.lower(), limit=history_limit)

        if not jobs:
            st.info("No job history found.")
        else:
            for job in jobs:
                with st.container():
                    # Job header with status badge
                    col1, col2, col3 = st.columns([3, 1, 1])

                    with col1:
                        st.markdown(f"**Job:** `{job.job_id}`")

                    with col2:
                        if job.is_completed:
                            st.success("✅ Completed")
                        elif job.is_failed:
                            st.error("❌ Failed")
                        elif job.is_running:
                            st.info("🔄 Running")

                    with col3:
                        if job.duration_seconds:
                            st.markdown(f"**Duration:** {format_duration(job.duration_seconds)}")

                    # Job stats
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Total Races", job.total_races)
                    with col2:
                        st.metric("Successful", job.successful_races)
                    with col3:
                        st.metric("Failed", job.failed_races)
                    with col4:
                        success_rate = (job.successful_races / job.total_races * 100) if job.total_races > 0 else 0
                        st.metric("Success Rate", f"{success_rate:.1f}%")

                    # Message
                    st.markdown(f"**Message:** {job.message}")

                    # Error details if failed
                    if job.is_failed and job.error:
                        with st.expander("⚠️ Error Details"):
                            st.code(job.error, language="python")

                    # Actions
                    col1, col2, col3 = st.columns([1, 1, 4])
                    with col1:
                        if st.button(f"📜 View Log", key=f"log_{job.job_id}"):
                            log_content = job_manager.get_job_log(job.job_id)
                            if log_content:
                                st.text_area("Job Log", log_content, height=300)
                            else:
                                st.warning("Log file not found")

                    with col2:
                        if st.button(f"🗑️ Delete", key=f"del_{job.job_id}"):
                            if job_manager.delete_job(job.job_id):
                                st.success("Job deleted")
                                time.sleep(1)
                                st.rerun()
                            else:
                                st.error("Failed to delete job")

                    st.markdown("---")


# Standalone app entry point
if __name__ == "__main__":
    st.set_page_config(
        page_title="Batch Predictions",
        page_icon="🚀",
        layout="wide"
    )

    render_batch_predictions_ui()
