from core.connectors.api_daily_sync import RaceFetcher
from core.orchestrators.prediction_orchestrator import PredictionOrchestrator
from race_prediction.daily_predictor import DailyPredictor, predict_race, evaluate_today
import datetime

def main():
    predictor = DailyPredictor()
    result = predict_race(1593465)
    print(result)
if __name__ == "__main__":
    main()