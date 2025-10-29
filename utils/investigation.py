# 1. Load your quinté model's expected features
import json
with open('models/2025-10-20/2years_165822_quinte_rf/feature_columns.json') as f:
    expected_features = json.load(f)

print(f"Model expects: {len(expected_features)} features")
print(f"Expected features: {expected_features[:10]}")

# 2. Prepare a REAL quinté race for prediction
sample_quinte = get_next_quinte_race()  # Your actual prediction function
prepared_features = prepare_quinte_prediction(sample_quinte)

print(f"\nActual features provided: {len(prepared_features.columns)}")
print(f"Actual features: {list(prepared_features.columns)[:10]}")

# 3. Compare
missing = set(expected_features) - set(prepared_features.columns)
extra = set(prepared_features.columns) - set(expected_features)

print(f"\n❌ Missing features: {len(missing)}")
if missing:
    print(f"   Critical missing: {list(missing)[:10]}")

print(f"\n⚠️  Extra features: {len(extra)}")

# 4. Check bytype features specifically
bytype_expected = [f for f in expected_features if 'bytype' in f]
bytype_actual = [f for f in prepared_features.columns if 'bytype' in f]

print(f"\n🔍 Bytype features:")
print(f"   Expected: {len(bytype_expected)}")
print(f"   Actual: {len(bytype_actual)}")

if bytype_actual:
    print(f"   Sample values: {prepared_features[bytype_actual[0]].values[:5]}")
    print(f"   All zeros? {(prepared_features[bytype_actual] == 0).all().all()}")