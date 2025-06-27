from pathlib import Path
from datetime import datetime
import time
from typing import List, Dict, Optional

# Import your existing modules
from core.orchestrators.prediction_orchestrator import PredictionOrchestrator
from utils.TelegramNotifier import TelegramNotifier
from utils.env_setup import AppConfig
#from core.database import Database


class PredictionBot:
    """Simplified bot for IDE usage - no command line needed."""

    def __init__(self):
        """Initialize with hardcoded values for simplicity."""
        # CONFIGURATION - Change these values as needed
        self.MODEL_PATH = "models/2years/hybrid/2years_full_v20250409"  # Your model path
        self.CONFIG_PATH = "config.yaml"

        # Load configuration
        self.config = setup_environment(self.CONFIG_PATH)
        self.db = Database()

        # Initialize prediction orchestrator
        print(f"Loading model from: {self.MODEL_PATH}")
        self.orchestrator = PredictionOrchestrator(
            model_path=self.MODEL_PATH,
            db_name=self.config.get('active_db'),
            verbose=True
        )

        # Initialize telegram
        self.telegram = TelegramNotifier(
            self.config['telegram']['bot_token'],
            self.config['telegram']['chat_id']
        )
        self.telegram.callback_handler = self.predict_race

    def predict_race(self, comp_id: int) -> None:
        """Generate and send prediction for a race."""
        print(f"\n{'=' * 50}")
        print(f"Predicting race {comp_id}")
        print(f"{'=' * 50}")

        # Get race details
        with self.db.get_connection() as conn:
            cursor = conn.execute('''
                SELECT hippodrome, heure, prixnom, quinte, jour
                FROM daily_races
                WHERE comp = ?
            ''', (comp_id,))
            race_data = cursor.fetchone()

        if not race_data:
            self.telegram.send_message(f"❌ Race {comp_id} not found")
            return

        hippo, heure, prixnom, is_quinte, jour = race_data

        # Run prediction
        result = self.orchestrator.predict_race(comp_id)

        if result['status'] == 'success':
            # Build message
            predictions = result['predictions']

            lines = [
                f"{'🌟' if is_quinte else '🏇'} <b>{hippo} - {heure}</b>",
                f"📍 {prixnom} ({jour})",
                "",
                "<b>Arrivée prédite:</b>"
            ]

            # Top 5 horses
            for i, horse in enumerate(predictions[:5], 1):
                lines.append(f"{i}. {horse['numero']} - {horse['cheval']}")

            # Full arrival string
            if result['metadata'].get('predicted_arriv'):
                lines.extend(["", f"📋 {result['metadata']['predicted_arriv']}"])

            self.telegram.send_message("\n".join(lines))
            print("✅ Prediction sent to Telegram")
        else:
            self.telegram.send_message(f"❌ Error: {result.get('error', 'Unknown')}")
            print(f"❌ Prediction failed: {result.get('error')}")

    def send_today_races(self, date: str = None) -> None:
        """Fetch today's races and send list to Telegram."""
        if not date:
            date = datetime.now().strftime("%Y-%m-%d")

        print(f"\nFetching races for {date}...")

        # Fetch from API and store
        fetch_result = self.orchestrator.race_fetcher.fetch_and_store_daily_races(date)

        if fetch_result['status'] == 'error':
            print(f"Error fetching races: {fetch_result['error']}")
            return

        print(f"Fetched {fetch_result['successful']} races")

        # Get races from DB for display
        with self.db.get_connection() as conn:
            cursor = conn.execute('''
                SELECT comp, hippodrome, reun, prix, heure, prixnom, quinte
                FROM daily_races
                WHERE jour = ?
                ORDER BY heure
            ''', (date,))

            races = []
            for row in cursor.fetchall():
                races.append({
                    'numcourse': {
                        'comp': row[0],
                        'hippo': row[1],
                        'reun': row[2],
                        'prix': row[3],
                        'heure': row[4],
                        'prixnom': row[5],
                        'quinte': row[6]
                    }
                })

        if races:
            print(f"Sending {len(races)} races to Telegram...")
            self.telegram.send_daily_races(races)
        else:
            print("No races to send")

    def run_bot(self):
        """Start the bot and keep it running."""
        # Send today's races
        self.send_today_races()

        # Start polling
        print("\n🤖 Bot is running! Click on races in Telegram to get predictions.")
        print("Press Ctrl+C to stop\n")

        self.telegram.start_polling()

        # Keep running
        while True:
            time.sleep(1)


# Direct usage - no command line needed
if __name__ == "__main__":
    # Create bot instance
    bot = PredictionBot()

    # Option 1: Run full bot (sends races and waits for clicks)
    bot.run_bot()

    # Option 2: Just send today's races (uncomment to use)
    # bot.send_today_races()

    # Option 3: Predict a specific race (uncomment to use)
    # bot.predict_race(12345)  # Replace with actual comp_id

    # Option 4: Send races for specific date (uncomment to use)
    # bot.send_today_races("2025-01-15")