import json
import sqlite3

import requests
from typing import Dict, List, Any, Optional
from datetime import datetime
from pathlib import Path
import pandas as pd

from utils.env_setup import AppConfig


class BettingAdvisor:
    """
    Betting advisor that uses LM Studio APIs to analyze race predictions
    and evaluation results to provide intelligent betting recommendations.
    """

    def __init__(self,
                 lm_studio_url: str = None,
                 model_name: str = None,
                 verbose: bool = False):
        """
        Initialize the betting advisor.

        Args:
            lm_studio_url: LM Studio server URL (if None, will use config)
            model_name: Specific model to use (optional)
            verbose: Enable verbose logging
        """
        self.config = AppConfig()
        
        # Get LM Studio URL from config if not provided
        if lm_studio_url is None:
            self.lm_studio_url = self._get_lm_studio_url_from_config()
        else:
            self.lm_studio_url = lm_studio_url.rstrip('/')
        
        self.model_name = model_name
        self.verbose = verbose

        if self.verbose:
            print(f"[BettingAdvisor] Initialized with LM Studio at {self.lm_studio_url}")

    def _get_lm_studio_url_from_config(self) -> str:
        """Get LM Studio URL from configuration or environment variables."""
        # Try to get from config file
        try:
            config_data = self.config._config
            if config_data:
                # Try to access llm_url from the config
                if hasattr(config_data, 'llm_url') and hasattr(config_data.llm_url, 'local'):
                    return config_data.llm_url.local.rstrip('/')
                # Alternative access method in case the structure is different
                elif hasattr(config_data, '__dict__'):
                    config_dict = config_data.__dict__
                    if 'llm_url' in config_dict and isinstance(config_dict['llm_url'], dict):
                        if 'local' in config_dict['llm_url']:
                            return config_dict['llm_url']['local'].rstrip('/')
        except Exception as e:
            if self.verbose:
                print(f"Error accessing config: {e}")
        
        # Try environment variable
        import os
        env_url = os.getenv('LM_STUDIO_URL')
        if env_url:
            return env_url.rstrip('/')
        
        # Default fallback
        return "http://localhost:1234"

    def log_info(self, message: str):
        """Simple logging."""
        if self.verbose:
            print(f"[BettingAdvisor] {message}")

    def debug_connection(self):
        """Debug method to test LM Studio connection and model availability."""
        print(f"🔍 Debug: Testing LM Studio connection...")
        print(f"URL: {self.lm_studio_url}")
        print(f"Specified model: {self.model_name or 'None (will use first available)'}")
        
        try:
            # Test connection
            print("\n1. Testing connection to LM Studio...")
            health_response = requests.get(
                f"{self.lm_studio_url}/v1/models",
                timeout=10
            )
            
            if health_response.status_code != 200:
                print(f"❌ Connection failed: HTTP {health_response.status_code}")
                print(f"Response: {health_response.text}")
                return False
            
            print("✅ Connection successful!")
            
            # Check available models
            print("\n2. Checking available models...")
            models_data = health_response.json()
            available_models = [model.get('id', 'unknown') for model in models_data.get('data', [])]
            
            if not available_models:
                print("❌ No models loaded in LM Studio")
                print("Please load a model in LM Studio first")
                return False
            
            print(f"✅ Found {len(available_models)} model(s):")
            for i, model in enumerate(available_models, 1):
                print(f"  {i}. {model}")
            
            # Test a simple API call
            print("\n3. Testing API call...")
            test_payload = {
                "messages": [
                    {
                        "role": "user",
                        "content": "Hello, this is a test message. Please respond with 'Test successful!'"
                    }
                ],
                "model": self.model_name if self.model_name else available_models[0],
                "temperature": 0.1,
                "max_tokens": 50
            }
            
            print(f"Using model: {test_payload['model']}")
            
            response = requests.post(
                f"{self.lm_studio_url}/v1/chat/completions",
                headers={"Content-Type": "application/json"},
                json=test_payload,
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                if "choices" in result and len(result["choices"]) > 0:
                    message = result["choices"][0]["message"]["content"]
                    print(f"✅ API call successful!")
                    print(f"Response: {message}")
                    return True
                else:
                    print(f"❌ Invalid response format: {result}")
                    return False
            else:
                print(f"❌ API call failed: HTTP {response.status_code}")
                print(f"Response: {response.text}")
                return False
                
        except requests.exceptions.ConnectTimeout:
            print("❌ Connection timeout - LM Studio may not be running")
            return False
        except requests.exceptions.ConnectionError:
            print("❌ Connection error - Check if LM Studio is running on the specified URL")
            return False
        except Exception as e:
            print(f"❌ Unexpected error: {str(e)}")
            return False

    def get_pmu_odds(self, jour: str, reunion: str, course: str) -> Optional[Dict]:
        """
        Retrieve online odds from PMU API.
        
        Args:
            jour: Date in format YYYY-MM-DD
            reunion: Reunion number
            course: Course number
            
        Returns:
            Dictionary containing odds data or None if failed
        """
        try:
            # Format date for PMU API (remove dashes)
            formatted_date = jour.replace('-', '')
            if not isinstance(reunion, (int, float)):
                return None
            url = f"https://online.turfinfo.api.pmu.fr/rest/client/61/programme/{formatted_date}/{reunion}/{course}/participants"
            params = {"specialisation": "INTERNET"}
            
            self.log_info(f"Fetching PMU odds from: {url}")
            
            response = requests.get(url, params=params, timeout=30)
            
            if response.status_code == 200:
                data = response.json()
                self.log_info(f"Successfully retrieved PMU odds data")
                return data
            else:
                self.log_info(f"PMU API request failed with status {response.status_code}")
                return None
                
        except Exception as e:
            self.log_info(f"Error fetching PMU odds: {str(e)}")
            return None

    def extract_odds_from_pmu_data(self, pmu_data: Dict) -> Dict[str, Dict]:
        """
        Extract odds information from PMU API response.
        
        Args:
            pmu_data: Raw PMU API response
            
        Returns:
            Dictionary mapping horse numbers to odds data
        """
        odds_data = {}
        
        if not pmu_data or 'participants' not in pmu_data:
            return odds_data
            
        for participant in pmu_data['participants']:
            numero = str(participant.get('numero', ''))
            if not numero:
                continue
                
            horse_odds = {
                'numero': numero,
                'direct_odds': None,
                'reference_odds': None
            }
            
            # Extract direct odds
            if 'dernierRapportDirect' in participant and participant['dernierRapportDirect']:
                direct_rapport = participant['dernierRapportDirect'].get('rapport')
                if direct_rapport:
                    horse_odds['direct_odds'] = float(direct_rapport)
            
            # Extract reference odds
            if 'dernierRapportReference' in participant and participant['dernierRapportReference']:
                reference_rapport = participant['dernierRapportReference'].get('rapport')
                if reference_rapport:
                    horse_odds['reference_odds'] = float(reference_rapport)
            
            odds_data[numero] = horse_odds
            
        return odds_data

    def _call_lm_studio(self, prompt: str, temperature: float = 0.3) -> str:
        """
        Make API call to LM Studio with enhanced error handling.

        Args:
            prompt: The prompt to send
            temperature: Sampling temperature

        Returns:
            Response text from the model
        """
        headers = {
            "Content-Type": "application/json"
        }

        payload = {
            "messages": [
                {
                    "role": "system",
                    "content": "You are an expert horse racing analyst and betting advisor. Provide clear, actionable betting recommendations based on the prediction data and historical performance metrics provided."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            "temperature": temperature,
            "max_tokens": 1000
        }

        if self.model_name:
            payload["model"] = self.model_name

        if self.verbose:
            print("🔍 DEBUG: Request payload structure:")
            print(f"  URL: {self.lm_studio_url}/v1/chat/completions")
            print(f"  Headers: {headers}")
            print(f"  Payload keys: {list(payload.keys())}")
            print(f"  Messages count: {len(payload['messages'])}")
            print(f"  Temperature: {payload['temperature']}")
            print(f"  Max tokens: {payload['max_tokens']}")
            if 'model' in payload:
                print(f"  Model: {payload['model']}")
            print(f"  Prompt length: {len(prompt)} characters")
            print(f"  First 200 chars of prompt: {prompt[:200]}...")
            print("\n📤 Full JSON payload:")
            print(json.dumps(payload, indent=2, ensure_ascii=False))

        try:
            # Test connection and get available models
            health_response = requests.get(
                f"{self.lm_studio_url}/v1/models",
                timeout=10
            )
            
            if health_response.status_code != 200:
                self.log_info(f"LM Studio not available: {health_response.status_code}")
                return "LM Studio server is not available. Please check if it's running and accessible."

            # Check available models
            models_data = health_response.json()
            available_models = [model.get('id', 'unknown') for model in models_data.get('data', [])]
            
            if not available_models:
                return "No models are loaded in LM Studio. Please load a model first."
            
            # If no model specified, use the first available model
            if not self.model_name:
                payload["model"] = available_models[0]
                self.log_info(f"Using default model: {available_models[0]}")
            else:
                # Check if specified model is available
                if self.model_name not in available_models:
                    return f"Model '{self.model_name}' not available. Available models: {', '.join(available_models)}"
                payload["model"] = self.model_name
                self.log_info(f"Using specified model: {self.model_name}")

            response = requests.post(
                f"{self.lm_studio_url}/v1/chat/completions",
                headers=headers,
                json=payload,
                timeout=180
            )

            if self.verbose:
                print(f"\n📥 Response received:")
                print(f"  Status Code: {response.status_code}")
                print(f"  Headers: {dict(response.headers)}")
                print(f"  Response length: {len(response.text)} characters")

            if response.status_code == 200:
                result = response.json()
                if self.verbose:
                    print(f"\n✅ Success response structure:")
                    print(f"  Response keys: {list(result.keys())}")
                    if "choices" in result:
                        print(f"  Choices count: {len(result['choices'])}")
                        if len(result["choices"]) > 0:
                            choice = result["choices"][0]
                            print(f"  Choice keys: {list(choice.keys())}")
                            if "message" in choice:
                                message = choice["message"]
                                print(f"  Message keys: {list(message.keys())}")
                                if "content" in message:
                                    content = message["content"]
                                    print(f"  Content length: {len(content)} characters")
                                    print(f"  First 200 chars: {content[:200]}...")
                    print(f"\n📤 Full JSON response:")
                    print(json.dumps(result, indent=2, ensure_ascii=False))
                
                if "choices" in result and len(result["choices"]) > 0:
                    return result["choices"][0]["message"]["content"]
                else:
                    self.log_info(f"Invalid response format: {result}")
                    return "Invalid response format from LM Studio"
            elif response.status_code == 400:
                if self.verbose:
                    print(f"\n❌ Bad Request (400) response:")
                    print(f"  Raw response: {response.text}")
                    try:
                        error_json = response.json()
                        print(f"  JSON response: {json.dumps(error_json, indent=2)}")
                    except:
                        print("  Response is not valid JSON")
                self.log_info(f"Bad request to LM Studio: {response.text}")
                return "Bad request to LM Studio - check model configuration"
            elif response.status_code == 404:
                if self.verbose:
                    print(f"\n❌ Not Found (404) response:")
                    print(f"  Raw response: {response.text}")
                self.log_info(f"LM Studio endpoint not found: {response.text}")
                return "LM Studio endpoint not found - check URL configuration"
            elif response.status_code == 500:
                if self.verbose:
                    print(f"\n❌ Internal Server Error (500) response:")
                    print(f"  Raw response: {response.text}")
                self.log_info(f"LM Studio internal error: {response.text}")
                return "LM Studio internal server error - check model loading"
            else:
                if self.verbose:
                    print(f"\n❌ Error response ({response.status_code}):")
                    print(f"  Raw response: {response.text}")
                self.log_info(f"API call failed with status {response.status_code}: {response.text}")
                return f"API call failed with status {response.status_code}"

        except requests.exceptions.ConnectTimeout:
            self.log_info("Connection timeout to LM Studio")
            return "Connection timeout to LM Studio - check if server is running"
        except requests.exceptions.ConnectionError:
            self.log_info("Connection error to LM Studio")
            return "Cannot connect to LM Studio - check URL and server status"
        except requests.exceptions.ReadTimeout:
            self.log_info("Read timeout from LM Studio")
            return "Read timeout from LM Studio - request took too long"
        except Exception as e:
            self.log_info(f"Error calling LM Studio API: {str(e)}")
            return f"Error connecting to LM Studio: {str(e)}"

    def analyze_daily_results(self, evaluation_results: Dict) -> str:
        """
        Analyze daily evaluation results and provide betting advice.

        Args:
            evaluation_results: Results from daily evaluation

        Returns:
            Betting advice and analysis
        """
        self.log_info("Analyzing daily results for betting advice...")

        # Format the evaluation data for analysis
        analysis_data = self._format_evaluation_data(evaluation_results)

        prompt = f"""
Analyze the following horse racing prediction results and provide betting advice for today:

{analysis_data}

Please provide:
1. Overall prediction accuracy assessment
2. Best bet types to consider based on recent performance
3. Specific race recommendations if any stand out
4. Risk management advice
5. Expected value considerations

Focus on actionable advice for today's betting decisions.
"""

        return self._call_lm_studio(prompt)

    def analyze_race_prediction(self, race_data: Dict, prediction_results: Dict, previous_results: Dict = None) -> str:
        """
        Analyze a specific race prediction and provide betting advice with odds tuning.

        Args:
            race_data: Race information and metadata
            prediction_results: Prediction results for the race
            previous_results: Previous prediction results and outcomes for tuning

        Returns:
            Race-specific betting advice with odds-tuned predictions
        """
        self.log_info(f"Analyzing race prediction with odds integration...")

        # Fetch PMU odds
        jour = race_data.get('jour', '') if hasattr(race_data, 'get') else race_data['jour'] if 'jour' in race_data else ''
        reunion = race_data.get('reunion', '') if hasattr(race_data, 'get') else race_data['reunion'] if 'reunion' in race_data else ''
        course = race_data.get('course', '') if hasattr(race_data, 'get') else race_data['course'] if 'course' in race_data else ''
        
        pmu_odds = None
        if jour and reunion and course:
            pmu_data = self.get_pmu_odds(jour, reunion, course)
            if pmu_data:
                pmu_odds = self.extract_odds_from_pmu_data(pmu_data)

        race_info = self._format_race_data_with_odds(race_data, prediction_results, pmu_odds, previous_results)

        prompt = f"""
You are an expert horse racing analyst. Analyze the ML predictions alongside current market odds to provide tuned predictions and betting advice.

{race_info}

Your task is to:
1. Compare ML predictions with market odds to identify value opportunities
2. Tune the ML predictions based on market sentiment and odds movements
3. Provide final ranking recommendations that consider both ML confidence and market odds
4. Recommend specific bet types and stakes based on the tuned analysis
5. Identify any discrepancies between ML predictions and market that suggest betting opportunities

Focus on actionable betting advice that combines both ML insights and market intelligence.
"""

        return self._call_lm_studio(prompt)

    def analyze_quinte_betting_strategy(self, evaluation_results: Dict) -> str:
        """
        Analyze evaluation results with specific focus on quinte racing and provide 3 refined betting recommendations.

        Args:
            evaluation_results: Results from daily evaluation with emphasis on quinte data

        Returns:
            Quinte-specific betting advice with 3 refined recommendations
        """
        self.log_info("Analyzing quinte betting strategy with 3 refined recommendations...")

        # Format the evaluation data for quinte-focused analysis
        quinte_analysis_data = self._format_quinte_evaluation_data(evaluation_results)

        prompt = f"""
You are an expert French horse racing analyst. Analyze the following historical prediction results and current odds to provide a revised horse ranking that corrects for past prediction errors.

{quinte_analysis_data}

Your task is to analyze the prediction errors, consider current market odds, and provide a REVISED HORSE RANKING:

**ANALYSIS STEPS:**
1. Compare predicted vs actual orders from historical races
2. Identify systematic prediction biases (horses consistently over/under-predicted)
3. Look for patterns with outsiders (high odds horses) that performed better than predicted
4. Consider if favorites (low odds) are being over-relied upon
5. Balance ML predictions with market wisdom from odds

**REVISED RANKING:**
Provide your corrected ranking as: [horse_number_1]-[horse_number_2]-[horse_number_3]-[horse_number_4]-[horse_number_5]

**REASONING:**
- Which horses were moved up/down from original prediction and why
- How odds influenced your corrections (e.g., "outsider #7 at 15.0 odds historically outperforms prediction")
- Any favorites that might be overvalued by the model
- Specific pattern observations from historical errors

**OUTSIDER CONSIDERATION:**
- Identify any long shots (high odds) that could surprise based on historical patterns
- Note if the model systematically underestimates certain types of outsiders

Focus on learning from past mistakes while respecting both ML predictions and market odds to create a more accurate ranking.
"""

        return self._call_lm_studio(prompt)

    def compare_model_performance(self, historical_metrics: Dict) -> str:
        """
        Analyze model performance trends and provide strategic advice.

        Args:
            historical_metrics: Historical performance data

        Returns:
            Strategic betting advice based on model performance
        """
        self.log_info("Analyzing model performance trends...")

        performance_summary = self._format_performance_data(historical_metrics)

        prompt = f"""
Analyze the following model performance trends and provide strategic betting advice:

{performance_summary}

Provide insights on:
1. Model reliability trends
2. Best performing bet types over time
3. When to increase or decrease bet sizes
4. Market conditions where the model performs best
5. Warning signs to watch for

Focus on long-term betting strategy optimization.
"""

        return self._call_lm_studio(prompt)

    def _format_evaluation_data(self, evaluation_results: Dict) -> str:
        """Format evaluation results for LM Studio analysis."""
        formatted = []

        # Summary metrics
        if 'summary_metrics' in evaluation_results:
            metrics = evaluation_results['summary_metrics']
            formatted.append("=== DAILY PERFORMANCE SUMMARY ===")
            formatted.append(f"Total races evaluated: {metrics.get('total_races', 'N/A')}")
            formatted.append(f"Winner accuracy: {metrics.get('winner_accuracy', 0):.2%}")
            formatted.append(f"Podium accuracy: {metrics.get('podium_accuracy', 0):.2%}")
            formatted.append(f"Mean rank error: {metrics.get('mean_rank_error', 'N/A'):.2f}")
            formatted.append("")

        # PMU betting performance
        if 'pmu_summary' in evaluation_results:
            pmu = evaluation_results['pmu_summary']
            # Ensure pmu is a dictionary before accessing it
            if isinstance(pmu, dict):
                formatted.append("=== BETTING PERFORMANCE ===")

                bet_types = [
                    ('winner_rate', 'Winner'),
                    ('exacta_rate', 'Exacta'),
                    ('trifecta_rate', 'Trifecta'),
                    ('quinte_exact_rate', 'Quinté+ Exact'),
                    ('quinte_desordre_rate', 'Quinté+ Désordre'),
                    ('bonus4_rate', 'Bonus 4'),
                    ('bonus3_rate', 'Bonus 3'),
                    ('multi4_rate', 'Multi 4')
                ]

                for rate_key, name in bet_types:
                    if rate_key in pmu:
                        formatted.append(f"{name}: {pmu[rate_key]:.2%}")
                formatted.append("")
            else:
                formatted.append("=== BETTING PERFORMANCE ===")
                formatted.append(f"PMU summary error: {pmu}")
                formatted.append("")

        # Quinte analysis
        if 'quinte_analysis' in evaluation_results:
            quinte = evaluation_results['quinte_analysis']
            # Ensure quinte is a dictionary before calling .get()
            if isinstance(quinte, dict):
                formatted.append("=== QUINTÉ+ SPECIFIC ANALYSIS ===")
                formatted.append(f"Total Quinté+ races: {quinte.get('total_quinte_races', 0)}")

                if 'betting_scenarios' in quinte and isinstance(quinte['betting_scenarios'], dict):
                    scenarios = quinte['betting_scenarios']
                    for scenario, data in scenarios.items():
                        # Ensure data is a dictionary before calling .get()
                        if isinstance(data, dict):
                            wins = data.get('wins', 0)
                            total = data.get('total', 0)
                            rate = data.get('win_rate', 0)
                            formatted.append(f"{scenario}: {wins}/{total} ({rate:.2%})")
                        else:
                            formatted.append(f"{scenario}: {data}")
                formatted.append("")
            else:
                formatted.append("=== QUINTÉ+ SPECIFIC ANALYSIS ===")
                formatted.append(f"Quinte analysis error: {quinte}")
                formatted.append("")

        # Individual race details
        if 'race_details' in evaluation_results:
            races = evaluation_results['race_details']
            if isinstance(races, list):
                winning_races = [r for r in races if isinstance(r, dict) and r.get('winning_bet_count', 0) > 0]

                if winning_races:
                    formatted.append("=== SUCCESSFUL RACES TODAY ===")
                    for race in winning_races[:5]:  # Top 5 successful races
                        formatted.append(f"Race: {race.get('hippo', '')} - {race.get('prix', '')}")
                        formatted.append(f"  Winning bets: {', '.join(race.get('winning_bets', []))}")
                        formatted.append(f"  Winner correct: {race.get('winner_correct', False)}")
                    formatted.append("")
            else:
                formatted.append("=== SUCCESSFUL RACES TODAY ===")
                formatted.append(f"Race details error: {races}")
                formatted.append("")

        return "\n".join(formatted)

    def _format_quinte_evaluation_data(self, race_to_predict: Dict) -> str:
        """Format race data for quinte analysis - Historical races + current race to predict."""
        
        # Import here to avoid circular imports
        from datetime import datetime, timedelta
        
        formatted = []
        # Connect to database
        db = self.config.get_active_db_path()
        conn = sqlite3.connect(db)
        conn.row_factory = sqlite3.Row  # Enable dict-like row access

        # Calculate date range for last 5 days
        end_date = datetime.now().strftime("%Y-%m-%d")
        start_date = (datetime.now() - timedelta(days=4)).strftime("%Y-%m-%d")

        if self.verbose:
            print(f"🔍 Fetching historical races from {start_date} to {end_date}")

        # Query for historical races with completed results
        query = """
            SELECT comp, hippo, prix, jour, reun, quinte, 
                   participants, prediction_results, actual_results
            FROM daily_race
            WHERE jour >= ? AND jour <= ?
            AND prediction_results IS NOT NULL 
            AND actual_results IS NOT NULL
            AND actual_results != 'pending'
            AND quinte=1
            ORDER BY jour DESC, comp DESC LIMIT 0,8
        """

        cursor = conn.execute(query, (start_date, end_date))
        historical_races = cursor.fetchall()
        #historical_races = pd.DataFrame(data)
        conn.close()

        if self.verbose:
            print(f"🔍 Found {len(historical_races)} historical races with completed results")

        # Format historical data
        formatted.append("=== HISTORICAL RACE DATA ===")

        for index, race in enumerate(historical_races):
            # Convert Row object to a dictionary for easier access
            race_data = dict(race)

            participants = race_data.get('participants')
            if isinstance(participants, str):
                participants = json.loads(participants)

            prediction_results = race_data.get('prediction_results', {})
            if isinstance(prediction_results, str):
                prediction_results = json.loads(prediction_results)

            # Extract ML prediction order
            ml_prediction = ""
            if 'predicted_arriv' in prediction_results:
                ml_prediction = prediction_results['predicted_arriv']
            elif 'predictions' in prediction_results:
                predictions = prediction_results['predictions']
                if isinstance(predictions, list):
                    sorted_predictions = sorted(predictions, key=lambda x: (
                    x.get('predicted_rank', 0), x.get('predicted_position', 0)))
                    ml_prediction = '-'.join([str(p['numero']) for p in sorted_predictions])

            # Prepare formatted data for the current race
            race_details = {
                "race": len(formatted) + 1,
                "quinte": race_data.get('quinte'),
                "participants": [],
                "ml_prediction": ml_prediction,
                "actual_results": "" if 'actual_results' not in race_data else race_data['actual_results'],
            }

            for participant in participants:
                participant_info = {
                    "number": participant['numero'],
                    "result": participant.get('cl', 'N/A'),
                    "odds": participant.get('cotedirect', 'N/A')
                }
                race_details["participants"].append(participant_info)

            # Append formatted data to the list
            formatted.append(race_details)

        # Close the connection after use
        conn.close()

        # Convert the list of dictionaries into a JSON object
        formatted_json = json.dumps(formatted, indent=2)
        print(formatted_json)

        # Format current race to predict
        formatted.append("=== CURRENT RACE TO PREDICT ===")

        # Get current race data
        conn = sqlite3.connect(db)
        conn.row_factory = sqlite3.Row

        query = """
        SELECT comp, hippo, prix, jour, reun, quinte, 
               participants, prediction_results
        FROM daily_race
        WHERE quinte = 1 and actual_results='pending' ORDER BY jour DESC LIMIT 0,1
        """

        cursor = conn.execute(query)
        current_race = cursor.fetchone()
        conn.close()
        race_predict = dict(race)
        if race_predict:
            # Parse participants
            participants = race_predict['participants'] if 'participants' in race_predict else '[]'
            if isinstance(participants, str):
                participants = json.loads(participants)

            # Parse prediction results
            prediction_results = race_predict['prediction_results'] if 'prediction_results' in race_predict else '{}'
            if isinstance(prediction_results, str):
                prediction_results = json.loads(prediction_results)

            # Extract ML prediction
            ml_prediction = ""
            if isinstance(prediction_results, dict):
                if 'predicted_arriv' in prediction_results:
                    ml_prediction = prediction_results['predicted_arriv']
                elif 'predictions' in prediction_results:
                    predictions = prediction_results['predictions']
                    if predictions and isinstance(predictions, list):
                        sorted_preds = sorted(predictions, key=lambda x: x.get('predicted_rank', x.get('predicted_position', 999)))
                        ml_prediction = '-'.join([str(p.get('numero', '')) for p in sorted_preds])

            # Get PMU odds for current race
            pmu_odds = {}
            jour = race_predict['jour'] if 'jour' in current_race else ''
            reunion = race_predict['reunion'] if 'reunion' in current_race else ''
            course = race_predict['course'] if 'course' in current_race else ''

            if jour and reunion and course:
                pmu_data = self.get_pmu_odds(jour, reunion, course)
                if pmu_data:
                    pmu_odds = self.extract_odds_from_pmu_data(pmu_data)

            is_quinte = current_race['quinte'] if 'quinte' in current_race else 0
            race_type = "QUINTÉ+" if is_quinte == 1 else "Regular"

            hippo = current_race['hippo'] if 'hippo' in current_race else 'N/A'
            jour_display = current_race['jour'] if 'jour' in current_race else 'N/A'
            formatted.append(f"Race ({race_type}) - {hippo} {jour_display}:")

            # Show participants with odds
            formatted.append("  Participants + Odds:")
            for participant in participants:
                numero = str(participant.get('numero', ''))
                if numero in pmu_odds:
                    odds = pmu_odds[numero].get('direct_odds', 'N/A')
                else:
                    # Try multiple fields for odds from stored participants
                    odds = participant.get('cotedirect') or participant.get('cote') or participant.get('odds') or 'N/A'
                formatted.append(f"    #{numero}: {odds}")

            # Show ML prediction - ensure it's displayed
            if ml_prediction:
                formatted.append(f"  ML Prediction: {ml_prediction}")
            else:
                formatted.append("  ML Prediction: Not available")

            formatted.append("")
        else:
            formatted.append("  No current race data found in database")
            formatted.append("")

        return formatted

    def _format_race_data(self, race_data: Dict, prediction_results: Dict) -> str:
        """Format individual race data for analysis."""
        return self._format_race_data_with_odds(race_data, prediction_results, None, None)

    def _format_race_data_with_odds(self, race_data: Dict, prediction_results: Dict, pmu_odds: Optional[Dict], previous_results: Optional[Dict]) -> str:
        """Format race data - MINIMAL VERSION with only horse numbers and odds."""
        formatted = []

        # Only show historical data for context
        if previous_results:
            formatted.append("=== HISTORICAL PERFORMANCE ===")
            formatted.append(f"Recent quinte predictions from past races:")
            
            # Show only the last few quinte results
            if 'race_details' in previous_results:
                races = previous_results['race_details']
                if isinstance(races, list):
                    quinte_races = [r for r in races if isinstance(r, dict) and r.get('quinte', False)]
                    for race in quinte_races[:3]:  # Last 3 quinte races
                        predicted = race.get('predicted_arriv', '')
                        actual = race.get('actual_arriv', '')
                        if predicted and actual:
                            formatted.append(f"Predicted: {predicted}")
                            formatted.append(f"Actual: {actual}")
                            formatted.append("")

        # Current race prediction with odds
        if 'predictions' in prediction_results:
            predictions = prediction_results['predictions']
            formatted.append("=== TODAY'S RACE PREDICTION ===")

            # Sort by predicted rank/position
            if predictions and isinstance(predictions[0], dict):
                if 'predicted_rank' in predictions[0]:
                    sorted_preds = sorted(predictions, key=lambda x: x.get('predicted_rank', 999))
                elif 'predicted_position' in predictions[0]:
                    sorted_preds = sorted(predictions, key=lambda x: x.get('predicted_position', 999))
                else:
                    sorted_preds = predictions

                # Show predicted order
                predicted_order = [str(pred.get('numero', 'N/A')) for pred in sorted_preds]
                formatted.append(f"Predicted order: {'-'.join(predicted_order)}")
                formatted.append("")

                # Show odds for each horse
                if pmu_odds:
                    formatted.append("Horse odds:")
                    for pred in sorted_preds:
                        numero = str(pred.get('numero', 'N/A'))
                        if numero in pmu_odds:
                            odds_data = pmu_odds[numero]
                            direct_odds = odds_data.get('direct_odds')
                            if direct_odds:
                                formatted.append(f"#{numero}: {direct_odds:.1f}")
                    formatted.append("")

        return "\n".join(formatted)

    def _format_performance_data(self, metrics: Dict) -> str:
        """Format historical performance data."""
        formatted = []

        formatted.append("=== HISTORICAL PERFORMANCE TRENDS ===")

        # Add any available historical metrics
        for key, value in metrics.items():
            if isinstance(value, (int, float)):
                if 'rate' in key or 'accuracy' in key:
                    formatted.append(f"{key.replace('_', ' ').title()}: {value:.2%}")
                else:
                    formatted.append(f"{key.replace('_', ' ').title()}: {value:.2f}")
            else:
                formatted.append(f"{key.replace('_', ' ').title()}: {value}")

        return "\n".join(formatted)


def get_betting_advice(evaluation_results: Dict,
                       lm_studio_url: str = "http://localhost:1234",
                       verbose: bool = False) -> str:
    """
    Convenience function to get betting advice from evaluation results.

    Args:
        evaluation_results: Daily evaluation results
        lm_studio_url: LM Studio server URL
        verbose: Enable verbose output

    Returns:
        Betting advice string
    """
    advisor = BettingAdvisor(lm_studio_url=lm_studio_url, verbose=verbose)
    return advisor.analyze_daily_results(evaluation_results)


def get_race_advice(race_data: Dict,
                    prediction_results: Dict,
                    previous_results: Dict = None,
                    lm_studio_url: str = "http://localhost:1234",
                    verbose: bool = False) -> str:
    """
    Convenience function to get advice for a specific race with odds tuning.

    Args:
        race_data: Race information
        prediction_results: Prediction results
        previous_results: Previous performance results for tuning
        lm_studio_url: LM Studio server URL
        verbose: Enable verbose output

    Returns:
        Race-specific betting advice with odds integration
    """
    advisor = BettingAdvisor(lm_studio_url=lm_studio_url, verbose=verbose)
    return advisor.analyze_race_prediction(race_data, prediction_results, previous_results)


# Debug function to test LM Studio connection
def test_lm_studio_connection():
    """Test function to run in IDE for debugging LM Studio connection."""
    print("🧪 Testing LM Studio Connection...")
    advisor = BettingAdvisor(verbose=True)
    result = advisor.debug_connection()
    
    if result:
        print("\n✅ All tests passed! LM Studio is working correctly.")
    else:
        print("\n❌ LM Studio connection failed. Check the output above for details.")
    
    return result


if __name__ == "__main__":
    test_lm_studio_connection()
