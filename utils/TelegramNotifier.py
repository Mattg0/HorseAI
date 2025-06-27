import requests
from datetime import datetime
from typing import List, Dict, Optional, Any, Tuple
import json
import threading
import time
from queue import Queue
from pathlib import Path
import sys
import sqlite3
# First get the config

# Load configuration
##config = setup_environment('config.yaml')

# Get paths from config
##project_root = Path(config['rootdir'])
##model_paths = get_model_paths(config, 'claude')  # Get claude model paths

# Add necessary paths to Python path
##if str(project_root) not in sys.path:
##    sys.path.insert(0, str(project_root))
##if str(model_paths['base']) not in sys.path:
##    sys.path.insert(0, str(model_paths['base']))

class TelegramNotifier:
    def __init__(self, bot_token: str, chat_id: str):
        """Initialize Telegram notifier with bot token and chat ID."""
        self.bot_token = bot_token
        self.chat_id = chat_id
        self.base_url = f"https://api.telegram.org/bot{bot_token}"
        self.update_id = 0
        self.callback_queue = Queue()
        self.callback_handler = None  # Will be set by orchestrator

    def format_reunion_message(self, reun_races: List[Dict]) -> Tuple[str, List[List[Dict]]]:
        """Format races for a single reunion into a message with buttons."""
        if not reun_races:
            return "", []

        first_race = reun_races[0].get('numcourse', reun_races[0])
        hippo = first_race.get('hippo')
        reun = first_race.get('reun')

        # Build message header
        message = f"🏇 <b>Reunion {reun} - {hippo}</b>\n\n"
        keyboard = []

        # Sort races by time
        sorted_races = sorted(reun_races, key=lambda x: x.get('numcourse', x).get('heure', ''))

        for race in sorted_races:
            numcourse = race.get('numcourse', race)
            time = numcourse.get('heure')
            prix_name = numcourse.get('prixnom')
            is_quinte = bool(numcourse.get('quinte', 0))

            # Format race line
            race_line = (
                f"{'🌟' if is_quinte else '▫️'} {numcourse.get('comp')} / {time} - {prix_name}\n"
            )

            if is_quinte:
                race_line = f"<b>{race_line}</b>"

            message += race_line

            # Add button for this race
            callback_data = json.dumps({
                'action': 'predict',
                'comp': numcourse.get('comp')
            })

            button_text = f"{'🌟 ' if is_quinte else ''}{numcourse.get('reun')} - {numcourse.get('prixnom')} / {time}"
            keyboard.append([{
                'text': button_text,
                'callback_data': callback_data
            }])


        return message, keyboard

    def send_daily_races(self, races: List[Dict]) -> bool:
        """Send one message per reunion."""
        try:
            # Group races by reunion
            reunions = {}
            for race in races:
                numcourse = race.get('numcourse', race)
                reun = numcourse.get('reun')
                if reun not in reunions:
                    reunions[reun] = []
                reunions[reun].append(race)

            # Send a message for each reunion
            for reun_races in reunions.values():
                message, keyboard = self.format_reunion_message(reun_races)
                if message:  # Only send if we have a message
                    reply_markup = {"inline_keyboard": keyboard}
                    self.send_message(message, reply_markup)
                    time.sleep(1)  # Add small delay between messages

            return True

        except Exception as e:
            print(f"Error sending daily races: {e}")
            return False

    def send_message(self, message: str, reply_markup: Optional[Dict] = None) -> bool:
        """Send a message through Telegram bot."""
        url = f"{self.base_url}/sendMessage"
        data = {
            "chat_id": self.chat_id,
            "text": message,
            "parse_mode": "HTML"
        }

        if reply_markup:
            data["reply_markup"] = json.dumps(reply_markup)

        try:
            response = requests.post(url, json=data)
            response.raise_for_status()
            return True
        except requests.RequestException as e:
            print(f"Error sending Telegram message: {e}")
            return False

    def handle_callback_query(self, callback_query: Dict[str, Any]) -> None:
        """Handle callback queries from inline keyboard."""
        try:
            callback_data = json.loads(callback_query['data'])
            # Only process callback if it has the required action and comp_id
            if (callback_data.get('action') == 'predict' and
                'comp' in callback_data and
                self.callback_handler):
                # Convert comp_id to int for processing
                comp_id = int(callback_data['comp'])
                self.callback_handler(comp_id)
            self._answer_callback_query(callback_query['id'])
        except Exception as e:
            print(f"Error handling callback query: {e}")
            print(f"Callback data: {callback_query.get('data', 'No data')}")

    def _answer_callback_query(self, callback_query_id: str) -> None:
        """Answer callback query to remove loading state."""
        url = f"{self.base_url}/answerCallbackQuery"
        data = {"callback_query_id": callback_query_id}
        try:
            requests.post(url, json=data)
        except requests.RequestException as e:
            print(f"Error answering callback query: {e}")

    def start_polling(self) -> None:
        """Start polling for updates in a separate thread."""
        def poll_updates():
            while True:
                try:
                    updates = self._get_updates()
                    for update in updates:
                        if 'callback_query' in update:
                            self.handle_callback_query(update['callback_query'])
                    time.sleep(1)
                except Exception as e:
                    print(f"Error polling updates: {e}")
                    time.sleep(5)

        thread = threading.Thread(target=poll_updates, daemon=True)
        thread.start()

    def _get_updates(self) -> List[Dict[str, Any]]:
        """Get updates from Telegram."""
        url = f"{self.base_url}/getUpdates"
        params = {
            "offset": self.update_id + 1,
            "timeout": 30
        }
        try:
            response = requests.get(url, params=params)
            response.raise_for_status()
            updates = response.json()['result']
            if updates:
                self.update_id = updates[-1]['update_id']
            return updates
        except requests.RequestException as e:
            print(f"Error getting updates: {e}")
            return []