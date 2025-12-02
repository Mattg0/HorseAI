#!/usr/bin/env python3
"""
Check Adaptive Calibrator Status

Quick script to check the current status of adaptive calibrators.

Usage:
    python scripts/check_calibrator_status.py
"""

import sys
from pathlib import Path
from datetime import datetime

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from model_training.regressions.adaptive_calibrator import AdaptiveCalibratorManager


def main():
    print("=" * 70)
    print("ADAPTIVE CALIBRATOR STATUS")
    print("=" * 70)
    print()

    manager = AdaptiveCalibratorManager()
    status = manager.get_calibrator_status()

    for model_type in ['rf', 'tabnet']:
        print(f"{model_type.upper()} Calibrator:")
        print("-" * 40)

        if status[model_type]['exists']:
            info = status[model_type]

            # Format last updated
            last_updated = datetime.fromisoformat(info['last_updated'])
            last_updated_str = last_updated.strftime('%Y-%m-%d %H:%M:%S')

            print(f"  Status: {'✅ Active' if info['is_valid'] else '⚠️  Inactive'}")
            print(f"  Last updated: {last_updated_str}")
            print(f"  Age: {info['days_old']} days")
            print(f"  Data points: {info['data_points']}")
            print(f"  MAE improvement: {info['mae_improvement']:.2f}%")

            if not info['is_valid']:
                print()
                if info['days_old'] > 90:
                    print(f"  ⚠️  Calibrator is too old (>{90} days)")
                    print(f"     Run: python scripts/update_calibrators.py")
                if info['data_points'] < 100:
                    print(f"  ⚠️  Not enough data points (<100)")
                    print(f"     Need more race results with predictions")
        else:
            print(f"  Status: ❌ Not found")
            print(f"  Run: python scripts/update_calibrators.py --days 30")

        print()

    print("=" * 70)

    # Overall status
    all_valid = all(status[m]['is_valid'] for m in ['rf', 'tabnet'])

    if all_valid:
        print("✅ All calibrators are active and up-to-date")
        print("   Predictions will use adaptive calibration")
    else:
        print("⚠️  Some calibrators are missing or outdated")
        print("   Predictions will use raw model outputs")
        print()
        print("   To activate calibration, run:")
        print("   python scripts/update_calibrators.py --days 30")

    return 0 if all_valid else 1


if __name__ == '__main__':
    sys.exit(main())
