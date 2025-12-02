"""
Daily Calibration Check Script

Designed to run as a scheduled task (cron job or system scheduler).
Checks calibration health and updates if needed.

Cron schedule example (run daily at 2 AM):
    0 2 * * * cd /path/to/HorseAIv2 && python scripts/daily_calibration_check.py >> logs/calibration.log 2>&1

Windows Task Scheduler:
    - Action: Start a program
    - Program: python
    - Arguments: scripts/daily_calibration_check.py
    - Start in: C:\path\to\HorseAIv2
    - Trigger: Daily at 2:00 AM
"""

import sys
from pathlib import Path
from datetime import datetime
import logging

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from scripts.calibrate_models import check_and_update_calibration

# Setup logging
log_dir = project_root / 'logs'
log_dir.mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_dir / 'calibration_checks.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)


def daily_calibration_check():
    """
    Run daily check for both models
    """

    logger.info("="*80)
    logger.info(f"Daily Calibration Check - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("="*80)

    db_path = project_root / 'data' / 'hippique2.db'

    if not db_path.exists():
        logger.error(f"Database not found at {db_path}")
        return

    results = {}

    # Check quinté model
    try:
        logger.info("\n1. QUINTÉ MODEL")
        logger.info("-"*80)

        calibrator, detector, metrics = check_and_update_calibration(
            str(db_path),
            model_type='quinte',
            force=False
        )

        results['quinte'] = {
            'success': True,
            'reason': metrics.get('reason', 'Unknown'),
            'mae': metrics.get('current_mae'),
            'updated': metrics.get('should_update', False)
        }

        logger.info(f"Quinté: {metrics.get('reason', 'Unknown')}")

        if metrics.get('should_update'):
            logger.info("✓ Quinté calibration updated")
        else:
            logger.info("✓ Quinté calibration healthy")

    except Exception as e:
        logger.error(f"Error checking quinté calibration: {e}")
        results['quinte'] = {
            'success': False,
            'error': str(e)
        }

    # Check general model
    try:
        logger.info("\n2. GENERAL MODEL")
        logger.info("-"*80)

        calibrator, detector, metrics = check_and_update_calibration(
            str(db_path),
            model_type='general',
            force=False
        )

        results['general'] = {
            'success': True,
            'reason': metrics.get('reason', 'Unknown'),
            'mae': metrics.get('current_mae'),
            'updated': metrics.get('should_update', False)
        }

        logger.info(f"General: {metrics.get('reason', 'Unknown')}")

        if metrics.get('should_update'):
            logger.info("✓ General calibration updated")
        else:
            logger.info("✓ General calibration healthy")

    except Exception as e:
        logger.error(f"Error checking general calibration: {e}")
        results['general'] = {
            'success': False,
            'error': str(e)
        }

    # Summary
    logger.info("\n" + "="*80)
    logger.info("DAILY CALIBRATION CHECK COMPLETE")
    logger.info("="*80)

    total_updated = sum(1 for r in results.values() if r.get('updated', False))
    total_errors = sum(1 for r in results.values() if not r.get('success', False))

    logger.info(f"Models checked: {len(results)}")
    logger.info(f"Updates applied: {total_updated}")
    logger.info(f"Errors: {total_errors}")

    if total_errors > 0:
        logger.warning("Some calibration checks failed - review logs")

    return results


if __name__ == "__main__":
    try:
        results = daily_calibration_check()

        # Exit with error code if there were failures
        if any(not r.get('success', False) for r in results.values()):
            sys.exit(1)
        else:
            sys.exit(0)

    except Exception as e:
        logger.error(f"Fatal error during calibration check: {e}")
        import traceback
        logger.error(traceback.format_exc())
        sys.exit(1)
