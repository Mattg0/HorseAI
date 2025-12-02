# Batch Predictions System Guide

## Overview

The new batch prediction system allows you to run predictions on hundreds or thousands of races efficiently with:

- ✅ **Background execution** - No UI blocking
- ✅ **Progress tracking** - Real-time updates in database
- ✅ **Memory management** - Chunked processing with automatic cleanup
- ✅ **Parallel processing** - Uses all CPU cores efficiently
- ✅ **Job monitoring** - Track multiple jobs from Streamlit UI
- ✅ **Job history** - View past jobs with logs

## Architecture

```
┌─────────────────┐
│   Streamlit UI  │  ← User Interface (no blocking!)
└────────┬────────┘
         │ Launches subprocess
         ↓
┌─────────────────┐
│  CLI Script     │  ← Runs in background
│ batch_predict.py│
└────────┬────────┘
         │ Writes progress
         ↓
┌─────────────────┐
│  Database       │  ← Progress tracking
│ prediction_jobs │
└────────┬────────┘
         │ Reads progress
         ↓
┌─────────────────┐
│   Streamlit UI  │  ← Displays real-time updates
└─────────────────┘
```

## Components

### 1. CLI Script (`scripts/batch_predict.py`)

Standalone script that runs batch predictions with progress tracking.

**Features:**
- Runs independently (can be launched from CLI or Streamlit)
- Writes progress to database table `prediction_jobs`
- Supports multiprocessing with memory management
- Logs output to file

### 2. Job Manager (`utils/batch_job_manager.py`)

Python API for launching and monitoring jobs.

**Key Methods:**
- `launch_job()` - Launch prediction job as background process
- `get_job_status()` - Get current status of a job
- `list_jobs()` - List jobs with optional filtering
- `get_active_jobs()` - Get all running jobs
- `delete_job()` - Delete job from history

### 3. Streamlit UI (`UI/batch_predictions_ui.py`)

Interactive UI for job management.

**Features:**
- **Launch Tab** - Configure and launch new jobs
- **Active Jobs Tab** - Monitor running jobs with auto-refresh
- **History Tab** - View completed/failed jobs with logs

## Usage

### Option 1: Streamlit UI (Recommended)

1. **Run standalone**:
```bash
streamlit run UI/batch_predictions_ui.py
```

2. **Or integrate into main UI** (see Integration section below)

3. **Launch a job**:
   - Go to "Launch Job" tab
   - Select prediction mode (unpredicted, all, date, or custom IDs)
   - Configure performance settings
   - Click "Launch Batch Prediction"

4. **Monitor progress**:
   - Switch to "Active Jobs" tab
   - Progress updates automatically every 5 seconds
   - View throughput and estimated time remaining

5. **View history**:
   - Go to "History" tab
   - View past jobs with success rates
   - Access logs for debugging

### Option 2: Command Line

```bash
# Predict all unpredicted races
python scripts/batch_predict.py

# Predict specific date
python scripts/batch_predict.py --date 2025-10-15

# Predict with custom settings
python scripts/batch_predict.py --limit 1000 --workers 8 --chunk-size 30

# Force reprediction of all races
python scripts/batch_predict.py --force-reprediction

# Custom memory limit
python scripts/batch_predict.py --max-memory 2048

# View all options
python scripts/batch_predict.py --help
```

### Option 3: Python API

```python
from utils.batch_job_manager import BatchJobManager
from utils.env_setup import AppConfig

# Initialize
config = AppConfig()
db_path = config.get_sqlite_dbpath(config._config.base.active_db)
manager = BatchJobManager(db_path)

# Launch job
job_id = manager.launch_job(
    date='2025-10-15',
    workers=-1,  # All cores
    chunk_size=50,
    max_memory_mb=4096
)

# Monitor progress
while True:
    status = manager.get_job_status(job_id)
    print(f"Progress: {status.progress}% - {status.message}")

    if not status.is_running:
        break

    time.sleep(5)

# Get result
if status.is_completed:
    print(f"Success! {status.successful_races}/{status.total_races} races")
else:
    print(f"Failed: {status.error}")
```

## Integration with Main UI (UIApp.py)

Add the batch predictions option to your main Streamlit app:

### Step 1: Import the UI component

```python
# At top of UIApp.py
from UI.batch_predictions_ui import render_batch_predictions_ui
```

### Step 2: Add to sidebar radio options

```python
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
        "🚀 Batch Predictions",  # NEW!
    ],
    index=0
)
```

### Step 3: Add the UI section

```python
# In the main content area, add:
elif operation == "🚀 Batch Predictions":
    render_batch_predictions_ui()
```

That's it! The batch predictions UI will now be available in your main app.

## Performance Settings Guide

### Workers (`-w`, `--workers`)

- **-1** (default): Use all CPU cores (recommended)
- **1**: Sequential processing (slow, but uses less memory)
- **4-16**: Specific number of parallel workers

**Recommendation**: Use `-1` for maximum speed.

### Chunk Size (`-c`, `--chunk-size`)

- **Small (10-30)**: Lower memory usage, but more overhead
- **Medium (50)**: Default, balanced approach
- **Large (100-200)**: Faster, but uses more memory

**Recommendation**:
- **< 8GB RAM**: Use 25-30
- **8-16GB RAM**: Use 50 (default)
- **> 16GB RAM**: Use 100

### Max Memory (`-m`, `--max-memory`)

Maximum memory usage in MB before forcing garbage collection.

- **Low memory systems**: 2048 (2GB)
- **Standard**: 4096 (4GB, default)
- **High memory systems**: 8192 (8GB)

**Note**: This is per Python process, not total system memory.

## Database Schema

The `prediction_jobs` table stores job information:

```sql
CREATE TABLE prediction_jobs (
    job_id TEXT PRIMARY KEY,          -- Unique job identifier
    status TEXT NOT NULL,             -- 'running', 'completed', 'failed'
    progress INTEGER DEFAULT 0,       -- 0-100
    message TEXT,                     -- Current status message
    total_races INTEGER,              -- Total races to predict
    processed_races INTEGER DEFAULT 0, -- Races processed so far
    successful_races INTEGER DEFAULT 0, -- Successfully predicted
    failed_races INTEGER DEFAULT 0,    -- Failed predictions
    start_time TIMESTAMP,             -- Job start time
    end_time TIMESTAMP,               -- Job end time
    error TEXT,                       -- Error message if failed
    config TEXT                       -- Job configuration (JSON)
);
```

## Logs

Logs are stored in `/logs/{job_id}.log`:

```
🚀 Starting batch prediction job: batch_20251103_120000
   Total races: 1000
   Workers: all CPU cores
   Chunk size: 50
   Memory limit: 4096MB
   Database: hippique2

[5%] Loading races from database...
[15%] Loaded 1000 races, starting predictions...
[20%] Processing chunk 1/20 (Mem: 1234MB)...
[23%] Processing chunk 2/20 (Mem: 1456MB)...
...
[95%] Storing predictions to database...
[100%] Complete! Processed 1000 races in 125.3s

============================================================
✅ BATCH PREDICTION COMPLETE
============================================================
  Job ID: batch_20251103_120000
  Total races: 1000
  Successful: 998
  Failed: 2
  Total time: 125.34s
  Throughput: 7.9 races/second
============================================================
```

## Troubleshooting

### Job stays at 0% progress

**Cause**: Job failed to start or crashed immediately.

**Solution**:
1. Check logs: `logs/{job_id}.log`
2. Run CLI directly to see errors: `python scripts/batch_predict.py --limit 10`
3. Check database connection and models are loaded

### Memory usage keeps growing

**Cause**: Chunk size too large for your system.

**Solution**:
1. Reduce chunk size: `--chunk-size 25`
2. Reduce max memory: `--max-memory 2048`
3. Reduce workers: `--workers 4`

### Predictions fail silently

**Cause**: Individual race prediction errors.

**Solution**:
1. Check `failed_races` count in job status
2. View logs to identify which races failed
3. Run failed races individually for debugging

### UI doesn't update

**Cause**: Auto-refresh disabled or Streamlit needs restart.

**Solution**:
1. Enable auto-refresh in Active Jobs tab
2. Click "Refresh Now" button
3. Restart Streamlit if stuck

## Best Practices

1. **Start small**: Test with `--limit 10` before running full batch
2. **Monitor first job**: Watch memory usage and adjust settings
3. **Clean up regularly**: Use "Cleanup Old Jobs" button
4. **Save logs**: Keep logs of important jobs for debugging
5. **Check success rate**: If < 95%, investigate failures

## Performance Benchmarks

Expected throughput on different systems:

| CPU Cores | RAM   | Chunk Size | Throughput (races/sec) |
|-----------|-------|------------|------------------------|
| 4         | 8GB   | 25         | ~3-4                   |
| 8         | 16GB  | 50         | ~6-8                   |
| 16        | 32GB  | 100        | ~12-15                 |

**Example**: Predicting 1000 races on 8-core system:
- **Old approach** (Streamlit blocking): ~50 minutes
- **New approach** (Background CLI): ~2-3 minutes

**Speedup**: ~16-25x faster! 🚀

## FAQ

**Q: Can I run multiple jobs simultaneously?**
A: Yes! Each job runs independently. You can launch multiple jobs and monitor them all in the Active Jobs tab.

**Q: Will closing Streamlit stop the job?**
A: No! Jobs run as independent background processes. You can close Streamlit and jobs will continue. Reopen to check progress.

**Q: How do I stop a running job?**
A: Currently jobs can't be stopped from UI. You need to find the process ID from logs and kill it manually. Future feature!

**Q: Can I use this for real-time predictions?**
A: No, this is for batch processing. For real-time predictions, use the standard prediction API.

**Q: What happens if the system crashes during a job?**
A: The job will remain in "running" status. You can delete it from history and relaunch.

## Future Improvements

- [ ] Job cancellation from UI
- [ ] Email notifications on job completion
- [ ] Retry failed races automatically
- [ ] Parallel job execution with priority queue
- [ ] Export results to CSV from UI
- [ ] Scheduled batch predictions (cron-like)

## Support

For issues or questions:
1. Check logs first
2. Review this guide
3. Try with small batch (`--limit 10`)
4. Report issue with full error logs
