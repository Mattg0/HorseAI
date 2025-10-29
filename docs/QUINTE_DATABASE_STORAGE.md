# Quinté Prediction Database Storage - Implementation Guide

## Overview

Quinté predictions are now stored in the **`race_predictions`** database table, replacing the JSON-based storage. This enables:
- ✅ Incremental training from database
- ✅ Efficient querying and analysis
- ✅ Production-ready prediction workflow
- ✅ Unified storage with general predictions

---

## 🏗️ Database Schema

### race_predictions Table

```sql
CREATE TABLE IF NOT EXISTS race_predictions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    race_id TEXT NOT NULL,              -- Race identifier (comp)
    horse_id INTEGER NOT NULL,          -- Horse identifier (idche)
    prediction_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

    -- Base Model Predictions
    rf_prediction REAL,                 -- Random Forest prediction
    tabnet_prediction REAL,             -- TabNet prediction

    -- Ensemble Weighting
    ensemble_weight_rf REAL,            -- RF weight (e.g., 0.40)
    ensemble_weight_tabnet REAL,        -- TabNet weight (e.g., 0.60)
    ensemble_prediction REAL,           -- Weighted ensemble prediction

    -- Competitive Analysis
    competitive_adjustment REAL,        -- Position adjustment applied
    primary_advantage_type TEXT,        -- 'speed', 'track', 'class', 'form', 'none'
    advantage_strength REAL,            -- Magnitude of competitive edge

    -- Final Result
    final_prediction REAL,              -- Final adjusted prediction

    -- Post-Race Data (populated later)
    actual_result INTEGER NULL,         -- Actual finishing position

    -- Constraints
    UNIQUE(race_id, horse_id)
);

-- Indexes for efficient queries
CREATE INDEX IF NOT EXISTS idx_race_horse ON race_predictions (race_id, horse_id);
CREATE INDEX IF NOT EXISTS idx_prediction_date ON race_predictions (prediction_date);
CREATE INDEX IF NOT EXISTS idx_race_analysis ON race_predictions (race_id, actual_result);
```

---

## 🔄 Prediction Storage Workflow

### 1. **predict_quinte.py** - Generate Predictions

**File**: [race_prediction/predict_quinte.py](race_prediction/predict_quinte.py)

#### Changes Made:

**Import SimplePredictionStorage** ([L26](race_prediction/predict_quinte.py#L26)):
```python
from race_prediction.simple_prediction_storage import SimplePredictionStorage
```

**Initialize Storage** ([L63](race_prediction/predict_quinte.py#L63)):
```python
# Initialize prediction storage
self.prediction_storage = SimplePredictionStorage(db_path=self.db_path, verbose=self.verbose)
```

**New Method: store_predictions_to_database()** ([L498-553](race_prediction/predict_quinte.py#L498-L553)):
```python
def store_predictions_to_database(self, df_predictions: pd.DataFrame) -> Dict[str, int]:
    """
    Store quinté predictions to race_predictions table.

    Replaces general predictions in the same table.
    """
    self.log_info("\nStoring predictions to database...")

    races_stored = 0
    horses_stored = 0

    # Group by race (comp)
    for race_comp, race_df in df_predictions.groupby('comp'):
        # Prepare prediction data
        predictions_data = []

        for _, horse_row in race_df.iterrows():
            horse_data = {
                'horse_id': horse_row.get('idche'),
                'rf_prediction': horse_row.get('predicted_position_rf'),
                'tabnet_prediction': horse_row.get('predicted_position_tabnet'),
                'ensemble_weight_rf': 0.4,  # Quinté default weights
                'ensemble_weight_tabnet': 0.6,
                'ensemble_prediction': horse_row.get('predicted_position_base'),
                'competitive_adjustment': horse_row.get('competitive_adjustment', 0.0),
                'primary_advantage_type': horse_row.get('primary_advantage_type', 'none'),
                'advantage_strength': horse_row.get('advantage_strength', 0.0),
                'final_prediction': horse_row.get('predicted_position')
            }
            predictions_data.append(horse_data)

        # Store race predictions
        stored_count = self.prediction_storage.store_race_predictions(
            race_id=race_comp,
            predictions_data=predictions_data
        )

        races_stored += 1
        horses_stored += stored_count

    return {'races_stored': races_stored, 'horses_stored': horses_stored}
```

**Updated run_prediction()** ([L555-621](race_prediction/predict_quinte.py#L555-L621)):
```python
def run_prediction(self, race_date: Optional[str] = None, output_dir: str = 'predictions',
                  store_to_db: bool = True) -> Dict:
    """
    Complete prediction workflow with database storage.

    Args:
        store_to_db: Whether to store in race_predictions table (default: True)
    """
    # ... existing prediction steps ...

    # Step 7: Store predictions to database (NEW)
    db_storage = {}
    if store_to_db:
        db_storage = self.store_predictions_to_database(df_formatted)

    return {
        'status': 'success',
        'files': saved_files,
        'db_storage': db_storage,  # NEW
        'predictions': df_formatted
    }
```

---

### 2. **quinte_incremental_trainer.py** - Read from Database

**File**: [model_training/regressions/quinte_incremental_trainer.py](model_training/regressions/quinte_incremental_trainer.py)

#### Changes Made:

**Updated get_completed_quinte_races()** ([L170-267](model_training/regressions/quinte_incremental_trainer.py#L170-L267)):

```python
def get_completed_quinte_races(self, date_from: str, date_to: str,
                               limit: int = None, use_db_predictions: bool = True) -> List[Dict]:
    """
    Fetch quinté races with predictions and results.

    Args:
        use_db_predictions: If True, fetch from race_predictions table (NEW).
                           If False, use legacy prediction_results JSON column.
    """
    if use_db_predictions:
        # NEW: Fetch races that have predictions in race_predictions table
        query = """
        SELECT DISTINCT
            dr.*,
            COUNT(rp.id) as prediction_count
        FROM daily_race dr
        INNER JOIN race_predictions rp ON dr.comp = rp.race_id
        WHERE dr.quinte = 1
        AND dr.actual_results IS NOT NULL
        AND dr.actual_results != 'pending'
        AND dr.jour BETWEEN ? AND ?
        GROUP BY dr.comp
        HAVING prediction_count > 0
        ORDER BY dr.jour DESC
        """

        # Fetch predictions from race_predictions table
        for race in races:
            cursor.execute("""
                SELECT
                    horse_id as idche,
                    rf_prediction as predicted_position_rf,
                    tabnet_prediction as predicted_position_tabnet,
                    ensemble_prediction as predicted_position_base,
                    final_prediction as predicted_position,
                    competitive_adjustment,
                    primary_advantage_type,
                    advantage_strength
                FROM race_predictions
                WHERE race_id = ?
                ORDER BY final_prediction ASC
            """, (race_id,))

            predictions = [dict(row) for row in cursor.fetchall()]

            # Embed predictions as JSON (same format as prediction_results)
            race['prediction_results'] = json.dumps(predictions)
    else:
        # LEGACY: Use prediction_results JSON column
        query = """
        SELECT * FROM daily_race
        WHERE prediction_results IS NOT NULL
        ...
        """

    return races
```

**Backward Compatibility**:
- ✅ `use_db_predictions=True` (default): Reads from `race_predictions` table
- ✅ `use_db_predictions=False`: Falls back to `prediction_results` JSON column

---

## 🎯 Usage Examples

### Generate and Store Quinté Predictions

```python
from race_prediction.predict_quinte import QuintePredictionEngine

# Initialize engine
engine = QuintePredictionEngine(verbose=True)

# Run predictions for today (stores to database by default)
result = engine.run_prediction(
    race_date='2025-10-22',
    output_dir='predictions',
    store_to_db=True  # NEW: Store in race_predictions table
)

print(f"Status: {result['status']}")
print(f"Races: {result['races']}")
print(f"DB Storage: {result['db_storage']}")
# Output:
# Status: success
# Races: 3
# DB Storage: {'races_stored': 3, 'horses_stored': 45}
```

### Run Incremental Training from Database

```python
from model_training.regressions.quinte_incremental_trainer import QuinteIncrementalTrainer

# Initialize trainer
trainer = QuinteIncrementalTrainer(verbose=True)

# Fetch races from database (uses race_predictions table by default)
races = trainer.get_completed_quinte_races(
    date_from='2025-09-01',
    date_to='2025-10-22',
    limit=50,
    use_db_predictions=True  # NEW: Read from race_predictions table
)

print(f"Found {len(races)} quinté races with predictions from database")

# Calculate baseline metrics
baseline = trainer.calculate_baseline_metrics(races)

# Extract failure data
training_df, analyses = trainer.extract_failure_data(races)

# Train on failures
results = trainer.train_on_failures(training_df, focus_on_failures=True)
```

### Query Predictions from Database

```python
import sqlite3
import pandas as pd

conn = sqlite3.connect('data/quinte_data.db')

# Get all predictions for a specific race
race_predictions = pd.read_sql_query("""
    SELECT
        rp.horse_id,
        rp.rf_prediction,
        rp.tabnet_prediction,
        rp.final_prediction,
        rp.competitive_adjustment,
        rp.actual_result
    FROM race_predictions rp
    WHERE rp.race_id = 'C7-20251022-6'
    ORDER BY rp.final_prediction ASC
""", conn)

print(race_predictions)

# Get quinté races with predictions
quinte_races = pd.read_sql_query("""
    SELECT DISTINCT
        dr.comp,
        dr.jour,
        dr.hippo,
        dr.prixnom,
        COUNT(rp.id) as prediction_count
    FROM daily_race dr
    INNER JOIN race_predictions rp ON dr.comp = rp.race_id
    WHERE dr.quinte = 1
    GROUP BY dr.comp
    ORDER BY dr.jour DESC
    LIMIT 10
""", conn)

print(f"Last 10 quinté races with predictions:\n{quinte_races}")
```

---

## 📊 Data Flow

```
┌─────────────────────────────────────────────────────────────┐
│                    Quinté Prediction Workflow                │
└─────────────────────────────────────────────────────────────┘

1. Generate Predictions
   ┌─────────────────────────┐
   │  predict_quinte.py      │
   │  - Load quinté models   │
   │  - Generate predictions │
   │  - Apply competitive    │
   └───────────┬─────────────┘
               │
               ▼
2. Store to Database
   ┌─────────────────────────┐
   │ SimplePredictionStorage │
   │  - INSERT OR REPLACE    │
   │  - race_predictions     │
   └───────────┬─────────────┘
               │
               ▼
3. Database Storage
   ┌─────────────────────────┐
   │   race_predictions      │
   │  ┌─────────────────┐    │
   │  │ race_id         │    │
   │  │ horse_id        │    │
   │  │ rf_prediction   │    │
   │  │ tabnet_predict. │    │
   │  │ final_predic.   │    │
   │  │ competitive_adj │    │
   │  └─────────────────┘    │
   └───────────┬─────────────┘
               │
               ▼
4. Incremental Training
   ┌─────────────────────────┐
   │ QuinteIncrementalTrainer│
   │  - Read from DB         │
   │  - Analyze failures     │
   │  - Train improvements   │
   └─────────────────────────┘
```

---

## 🔍 Verification

### Check Stored Predictions

```sql
-- Count quinté predictions
SELECT COUNT(*) as prediction_count
FROM race_predictions rp
INNER JOIN daily_race dr ON rp.race_id = dr.comp
WHERE dr.quinte = 1;

-- View recent quinté predictions
SELECT
    dr.comp,
    dr.jour,
    dr.hippo,
    COUNT(rp.id) as horses,
    MAX(rp.prediction_date) as last_updated
FROM race_predictions rp
INNER JOIN daily_race dr ON rp.race_id = dr.comp
WHERE dr.quinte = 1
GROUP BY dr.comp
ORDER BY dr.jour DESC
LIMIT 10;

-- Check prediction details for a race
SELECT
    horse_id,
    rf_prediction,
    tabnet_prediction,
    final_prediction,
    competitive_adjustment
FROM race_predictions
WHERE race_id = 'C7-20251022-6'
ORDER BY final_prediction ASC;
```

### Verify Incremental Training Data

```python
from model_training.regressions.quinte_incremental_trainer import QuinteIncrementalTrainer

trainer = QuinteIncrementalTrainer(verbose=True)

# Test database reading
races = trainer.get_completed_quinte_races(
    date_from='2025-10-01',
    date_to='2025-10-22',
    limit=5,
    use_db_predictions=True
)

print(f"Fetched {len(races)} races")

for race in races[:2]:
    print(f"\nRace: {race['comp']}")
    print(f"  Date: {race['jour']}")
    print(f"  Predictions: {len(json.loads(race['prediction_results']))} horses")
    print(f"  Actual results: {race['actual_results']}")
```

---

## 🚀 Migration Path

### Current State (Before Changes)
- ❌ Predictions stored in JSON files
- ❌ `prediction_results` JSON column in `daily_race`
- ❌ Manual data loading for incremental training

### New State (After Changes)
- ✅ Predictions stored in `race_predictions` table
- ✅ Automatic database storage on prediction
- ✅ Incremental training reads from database
- ✅ Backward compatible with JSON column

### Migration Steps

1. **Run New Predictions** (automatically stores to database):
   ```bash
   python race_prediction/predict_quinte.py --date 2025-10-22
   ```

2. **Verify Storage**:
   ```sql
   SELECT COUNT(*) FROM race_predictions;
   ```

3. **Test Incremental Training**:
   ```bash
   # Via UI: 🏆 Quinté Incremental Training
   # Or programmatically:
   python -c "from model_training.regressions.quinte_incremental_trainer import QuinteIncrementalTrainer; ..."
   ```

4. **Gradually Phase Out JSON Storage** (optional):
   - Keep JSON files for backup
   - Primary source is now database
   - Can disable JSON saving by setting `store_to_db=True, output_dir=None`

---

## 📝 Key Benefits

### 1. **Production Ready**
- Structured data in database
- Efficient querying and indexing
- ACID compliance

### 2. **Incremental Training Integration**
- Direct database reads
- No JSON parsing overhead
- Real-time training data availability

### 3. **Unified Storage**
- Quinté and general predictions in same table
- Consistent schema
- Shared analysis tools

### 4. **Backward Compatible**
- `use_db_predictions` flag allows fallback
- Existing JSON workflows still work
- Gradual migration possible

### 5. **Enhanced Analysis**
- SQL queries for performance analysis
- Join with daily_race for metadata
- Temporal analysis of predictions

---

## ⚙️ Configuration

### Enable/Disable Database Storage

**In predict_quinte.py**:
```python
# Enable database storage (default)
result = engine.run_prediction(store_to_db=True)

# Disable database storage (JSON only)
result = engine.run_prediction(store_to_db=False)
```

**In incremental training**:
```python
# Use database predictions (default)
races = trainer.get_completed_quinte_races(use_db_predictions=True)

# Use JSON predictions (legacy)
races = trainer.get_completed_quinte_races(use_db_predictions=False)
```

---

## 🎯 Next Steps

1. ✅ **Implemented**: Database storage in `predict_quinte.py`
2. ✅ **Implemented**: Database reading in `quinte_incremental_trainer.py`
3. ⏳ **Todo**: Run predictions to populate database
4. ⏳ **Todo**: Test incremental training with database
5. ⏳ **Todo**: Update UI to show database statistics
6. ⏳ **Todo**: Add database cleanup/archiving logic

---

## 📚 Related Documentation

- [QUINTE_MODEL_INTEGRATION.md](QUINTE_MODEL_INTEGRATION.md) - Model usage and integration
- [QUINTE_INCREMENTAL_PROGRESS.md](QUINTE_INCREMENTAL_PROGRESS.md) - Training progress
- [simple_prediction_storage.py](race_prediction/simple_prediction_storage.py) - Storage implementation

---

**Last Updated**: October 22, 2025
**Status**: ✅ Implemented and Ready for Testing
**Replaces**: JSON-based prediction storage for production use
