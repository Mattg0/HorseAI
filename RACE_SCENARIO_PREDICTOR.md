# Race Scenario Predictor - Configured Weight Predictions

**Script**: `race_prediction/predict_race_all_weights.py`

Generate all prediction scenarios for a single high-stakes race using **configured weight strategies** from `config.yaml`.

---

## Purpose

For important races (Prix d'Amérique, Arc de Triomphe, etc.), you want to see how different weight configurations from your `config.yaml` would predict the race. This script:

1. Loads race metadata (type, distance, field size)
2. Identifies which weight configurations from config.yaml apply to this race
3. Generates predictions using each eligible configuration
4. Shows you all scenarios sorted by historical accuracy

**This is NOT random weight testing** - it uses your carefully configured and validated weights from `config.yaml`.

---

## Quick Start

```bash
# Generate eligible scenarios for a race (only matching configurations)
python race_prediction/predict_race_all_weights.py --race-id 1621325

# Include ALL weight configurations (not just matching ones)
python race_prediction/predict_race_all_weights.py --race-id 1621325 --include-all

# Save to specific location
python race_prediction/predict_race_all_weights.py --race-id 1621325 --output scenarios/important_race
```

---

## What It Generates

### Configured Weight Scenarios

The script loads all weight configurations from `config.yaml` and applies those that match the race metadata.

**Example Config (from config.yaml)**:

```yaml
blend:
  default_weights:
    rf_weight: 1.0
    tabnet_weight: 0.0
    accuracy: 15.59%
    description: "Default optimal weights"

  dynamic_weights:
    - condition:
        typec: "Cross-country"
      weights:
        rf_weight: 0.9
        tabnet_weight: 0.1
      accuracy: 47.37%
      description: "Cross-country races - 31.8% improvement"

    - condition:
        dist_min: 1500
        dist_max: 2000
      weights:
        rf_weight: 0.7
        tabnet_weight: 0.3
      accuracy: 28.02%
      description: "Distance 1500-2000m - 12.4% improvement"

    - condition:
        partant_max: 8
      weights:
        rf_weight: 1.0
        tabnet_weight: 0.0
      accuracy: 25.33%
      description: "Small field (≤8 horses) - 9.7% improvement"
```

**For a 1700m Cross-country race**, the script would generate scenarios using:
1. ✓ **Default weights** (RF=1.0, TabNet=0.0) - Always included
2. ✓ **Cross-country weights** (RF=0.9, TabNet=0.1) - Matches typec
3. ✓ **Distance 1500-2000m weights** (RF=0.7, TabNet=0.3) - Matches distance

All scenarios are sorted by historical accuracy, so you see the best-performing strategy first.

---

## Output Files

### CSV File (Summary)
`predictions/race_scenarios_{race_id}_{timestamp}.csv`

Contains top 5 predictions for each scenario:

```csv
race_id,race_name,scenario_type,scenario_description,predicted_rank,horse_number,horse_name,rf_weight,tabnet_weight
1621325,Prix d'Amérique,RF_TabNet_Blend,"RF=0.0, TabNet=1.0",1,7,CHAMPION DU JOUR,0.0,1.0
1621325,Prix d'Amérique,RF_TabNet_Blend,"RF=0.0, TabNet=1.0",2,12,VAINQUEUR,0.0,1.0
...
```

### JSON File (Full Details)
`predictions/race_scenarios_{race_id}_{timestamp}.json`

Contains complete predictions for all horses in all scenarios:

```json
{
  "race_info": {
    "race_id": "1621325",
    "date": "2025-11-03",
    "track": "VINCENNES",
    "race_name": "Prix d'Amérique",
    "actual_results": "7-12-3-5-9-..."
  },
  "num_horses": 16,
  "total_scenarios": 33,
  "scenarios": {
    "rf_tabnet_blend": [
      {
        "scenario_type": "RF_TabNet_Blend",
        "rf_weight": 0.0,
        "tabnet_weight": 1.0,
        "description": "RF=0.0, TabNet=1.0",
        "top5": [...],
        "all_predictions": [...]
      },
      ...
    ],
    "competitive_adjustment": [...],
    "quinte_general_blend": [...]
  }
}
```

---

## Use Cases

### 1. High-Stakes Races - Matching Configurations Only
For major races where you want to see predictions using weights that match the race conditions:

```bash
# Only apply weight configs that match this race's metadata
python race_prediction/predict_race_all_weights.py --race-id 1621325
```

**Result**: Shows only the weight strategies configured for this type of race. Typically 2-4 scenarios.

### 2. High-Stakes Races - All Configurations
For major races where you want to see ALL possible configured strategies:

```bash
# Apply all weight configs regardless of match
python race_prediction/predict_race_all_weights.py --race-id 1621325 --include-all
```

**Result**: Shows predictions with every weight configuration in config.yaml. Typically 5-10 scenarios depending on your config.

### 3. Consensus Analysis
After generating all scenarios, analyze which horses appear most frequently in top 5:

```python
import pandas as pd

# Load CSV
df = pd.read_csv('predictions/race_scenarios_1621325_20251114.csv')

# Count how many times each horse appears in top 5 across all scenarios
top5_counts = df.groupby('horse_number').size().sort_values(ascending=False)

print("Horses appearing most frequently in top 5:")
print(top5_counts.head(10))

# Horses that appear in top 5 in >80% of scenarios are strong bets
total_scenarios = df['scenario_description'].nunique()
consensus_horses = top5_counts[top5_counts > total_scenarios * 0.8]
print(f"\nConsensus picks (in >80% of scenarios): {list(consensus_horses.index)}")
```

### 4. Compare to Actual Results
If race has finished, compare scenarios to actual results:

```python
import pandas as pd

df = pd.read_csv('predictions/race_scenarios_1621325_20251114.csv')

# Actual results: "7-12-3-5-9-..."
actual_top5 = [7, 12, 3, 5, 9]

# For each scenario, count how many of actual top 5 were predicted
scenarios = []
for scenario in df['scenario_description'].unique():
    scenario_df = df[df['scenario_description'] == scenario]
    predicted_top5 = scenario_df.head(5)['horse_number'].tolist()

    horses_in_quinte = len(set(predicted_top5) & set(actual_top5))

    scenarios.append({
        'scenario': scenario,
        'horses_in_quinte': horses_in_quinte,
        'predicted_top5': predicted_top5
    })

# Sort by best performance
scenarios_df = pd.DataFrame(scenarios).sort_values('horses_in_quinte', ascending=False)

print("Best performing scenarios:")
print(scenarios_df.head(10))
```

---

## Number of Scenarios

The number of scenarios depends on your `config.yaml`:

| Mode | Typical Scenarios | Description |
|------|-------------------|-------------|
| Default (matching only) | 2-4 | Only weight configs that match race metadata |
| `--include-all` | 5-10 | All weight configs in your config.yaml |

**Performance**: Very fast! Only applies pre-configured weights, no iterative testing.

```bash
# Recommended for most races: matching configs only
python race_prediction/predict_race_all_weights.py --race-id 1621325

# For maximum visibility: all configs
python race_prediction/predict_race_all_weights.py --race-id 1621325 --include-all
```

---

## Integration with Existing Tools

This script complements the existing optimization tools:

| Tool | Purpose | Scope |
|------|---------|-------|
| `predict_race_all_weights.py` | **Single race** - all weight scenarios | 1 race, many weights |
| `optimize_quinte_blend.py` | **Multiple races** - find best RF/TabNet weight | Many races, test weights |
| `competitive_weighting_check.py` | **Multiple races** - find best competitive weight | Many races, test weights |
| `optimize_quinte_general_blend.py` | **Multiple races** - find best quinté/general weight | Many races, test weights |

**Workflow**:
1. Use optimization tools to find best weights across historical races
2. For important upcoming race, use `predict_race_all_weights.py` to see all scenarios
3. Apply historical optimal weights + review other scenarios for high-stakes bets

---

## Example Output

```
================================================================================
PREDICTION SCENARIOS SUMMARY
================================================================================

Race: PRIX D'AMERIQUE
Date: 2025-01-26
Track: VINCENNES
Type: Trot attelé
Distance: 2700m
Field Size: 18
Horses: 18
Eligible Weight Configurations: 3
Total Scenarios: 3

Actual Results: 7-12-3-5-9-14-1-...

--------------------------------------------------------------------------------
CONFIGURED WEIGHT SCENARIOS (sorted by historical accuracy)
--------------------------------------------------------------------------------

1. Default optimal weights (Accuracy: 15.6%) ✓ MATCH
   Weights: RF=1.0, TabNet=0.0
   Condition: (always applies)
   Top 5: 7, 12, 3, 5, 9

2. Distance 1500-2000m - 12.4% improvement (Accuracy: 28.0%)
   Weights: RF=0.7, TabNet=0.3
   Condition: dist_min=1500, dist_max=2000
   Top 5: 7, 12, 3, 9, 5

3. Small field (≤8) - 9.7% improvement (Accuracy: 25.3%)
   Weights: RF=1.0, TabNet=0.0
   Condition: partant_max=8
   Top 5: 7, 12, 3, 5, 9

================================================================================
✓ All scenarios saved to predictions/race_scenarios_1621325_20251114.csv/json
================================================================================
```

**Note**: ✓ MATCH indicator shows which configs match this race's metadata.

---

## Advanced: Batch Processing Multiple Races

Process multiple important races:

```bash
#!/bin/bash
# Generate scenarios for all races on a date

RACE_IDS=(1621325 1621326 1621327)

for race_id in "${RACE_IDS[@]}"; do
    echo "Processing race $race_id..."
    python race_prediction/predict_race_all_weights.py \
        --race-id $race_id \
        --output scenarios/race_${race_id}
done

echo "All races processed!"
```

---

## Tips

1. **Start with matching configs** (default mode) for most races - shows only relevant strategies
2. **Use `--include-all`** for major races to see all possible configured strategies
3. **Trust the accuracy ranking** - scenarios are sorted by historical performance
4. **Look for ✓ MATCH indicators** - these configs are specifically tuned for this race type
5. **Check consensus** - if all scenarios predict similar top 5, it's a strong signal
6. **Compare scenarios** - if different configs produce very different top 5s, the race is uncertain
7. **Add new configs to config.yaml** - if you discover a pattern, add it to your config for future use

---

## Summary

This script gives you **strategic flexibility** for high-stakes races by showing predictions with all your validated weight configurations. Instead of relying on a single strategy, you can:

- See predictions from all configured weight strategies
- Identify which configurations best match this race
- Compare historical accuracy of different approaches
- Make informed decisions based on proven strategies

**Key Difference from Other Tools**:
- `optimize_quinte_blend.py` etc.: Find best weights across many races
- `predict_race_all_weights.py`: Apply configured weights to ONE important race

**Workflow**:
1. Use optimization tools to find and configure optimal weights → save in `config.yaml`
2. For regular races: Use standard prediction pipeline (automatically applies best matching config)
3. For high-stakes races: Use this script to see all configured strategies side-by-side
