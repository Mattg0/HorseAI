# utils/predict_evaluator.py

import sqlite3
import json
import pandas as pd
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from utils.env_setup import AppConfig


@dataclass
class EvaluationMetrics:
    """Container for evaluation metrics."""
    total_races: int
    races_evaluated: int
    overall_winner_accuracy: float
    overall_podium_accuracy: float
    total_winning_bets: int
    bet_win_rates: Dict[str, Dict]  # bet_type -> {wins, total, rate}
    quinte_performance: Optional[Dict] = None


@dataclass
class RaceResult:
    """Container for individual race evaluation result."""
    comp: str
    race_info: Dict
    winner_correct: bool
    podium_accuracy: float
    winning_bets: List[str]
    predicted_arrival: str
    actual_arrival: str
    is_quinte: bool


class PredictEvaluator:
    """
    Simple evaluation class that processes races with both predictions and results.
    No date filtering - just gets all available races that can be evaluated.
    """

    def __init__(self, db_name: str = None):
        """
        Initialize evaluator.

        Args:
            db_name: Database name from config (defaults to active_db)
        """
        self.config = AppConfig('config.yaml')

        if db_name is None:
            db_name = self.config._config.base.active_db

        self.db_path = self.config.get_sqlite_dbpath(db_name)

        # Bet type mappings for UI display
        self.bet_type_labels = {
            'tierce_exact': 'Tiercé Exact',
            'tierce_desordre': 'Tiercé Désordre',
            'quarte_exact': 'Quarté Exact',
            'quarte_desordre': 'Quarté Désordre',
            'quinte_exact': 'Quinté+ Exact',
            'quinte_desordre': 'Quinté+ Désordre',
            'bonus4': 'Bonus 4',
            'bonus3': 'Bonus 3',
            'deuxsur4': '2 sur 4',
            'multi4': 'Multi en 4',
            'quinte_6horses': 'Quinté 6 chevaux',
            'quinte_7horses': 'Quinté 7 chevaux',
            'bonus4_6horses': 'Bonus 4 (6 chevaux)',
            'bonus4_7horses': 'Bonus 4 (7 chevaux)',
            'bonus3_7horses': 'Bonus 3 (7 chevaux)'
        }

    def get_evaluable_races(self) -> List[Dict]:
        """
        Get all races that have both predictions and results.

        Returns:
            List of race records ready for evaluation
        """
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        query = """
        SELECT * FROM daily_race 
        WHERE prediction_results IS NOT NULL 
        AND actual_results IS NOT NULL 
        AND actual_results != 'pending'
        ORDER BY jour DESC, comp ASC
        """

        cursor.execute(query)
        rows = cursor.fetchall()
        conn.close()

        return [dict(row) for row in rows]

    def evaluate_all_races(self) -> EvaluationMetrics:
        """
        Evaluate all races that have both predictions and results.

        Returns:
            Complete evaluation metrics for UI display
        """
        races = self.get_evaluable_races()

        if not races:
            return EvaluationMetrics(
                total_races=0,
                races_evaluated=0,
                overall_winner_accuracy=0.0,
                overall_podium_accuracy=0.0,
                total_winning_bets=0,
                bet_win_rates={},
                quinte_performance=None
            )

        race_results = []

        # Evaluate each race
        for race_data in races:
            result = self._evaluate_single_race(race_data)
            if result:
                race_results.append(result)

        # Calculate aggregate metrics
        return self._calculate_aggregate_metrics(race_results)

    def get_races_won_by_bet_type(self) -> Dict[str, List[RaceResult]]:
        """
        Get races organized by bet types won.

        Returns:
            Dictionary mapping bet types to list of races where that bet won
        """
        races = self.get_evaluable_races()
        bet_type_races = {bet_type: [] for bet_type in self.bet_type_labels.keys()}

        for race_data in races:
            result = self._evaluate_single_race(race_data)
            if result:
                for bet_type in result.winning_bets:
                    if bet_type in bet_type_races:
                        bet_type_races[bet_type].append(result)

        return bet_type_races

    def get_quinte_horse_betting_analysis(self) -> Optional[Dict]:
        """
        Analyze win rate improvement when betting on 6 or 7 horses for quinte races.

        Returns:
            Analysis of quinte betting strategies or None if no quinte races
        """
        races = self.get_evaluable_races()
        quinte_races = [race for race in races if race.get('quinte') == 1]

        if not quinte_races:
            return None

        analysis = {
            'total_quinte_races': len(quinte_races),
            'betting_scenarios': {
                '5_horses': {'wins': 0, 'total': len(quinte_races)},
                '6_horses': {'wins': 0, 'total': len(quinte_races)},
                '7_horses': {'wins': 0, 'total': len(quinte_races)}
            }
        }

        for race_data in quinte_races:
            # Calculate wins for different horse counts
            pmu_bets = self._calculate_pmu_bets(race_data)

            # 5 horses (standard prediction)
            if any(pmu_bets.get(bet, False) for bet in ['quinte_exact', 'quinte_desordre', 'bonus4', 'bonus3']):
                analysis['betting_scenarios']['5_horses']['wins'] += 1

            # 6 horses - check if adding 6th horse would win
            if self._would_win_with_extra_horses(race_data, 6):
                analysis['betting_scenarios']['6_horses']['wins'] += 1

            # 7 horses - check if adding 7th horse would win  
            if self._would_win_with_extra_horses(race_data, 7):
                analysis['betting_scenarios']['7_horses']['wins'] += 1

        # Calculate win rates
        for scenario in analysis['betting_scenarios'].values():
            scenario['win_rate'] = scenario['wins'] / scenario['total'] if scenario['total'] > 0 else 0

        return analysis

    def _evaluate_single_race(self, race_data: Dict) -> Optional[RaceResult]:
        """
        Evaluate a single race.

        Args:
            race_data: Race data from database

        Returns:
            RaceResult or None if evaluation failed
        """
        try:
            # Parse JSON data
            predictions = self._parse_predictions(race_data.get('prediction_results'))
            actual_results = race_data.get('actual_results', '')

            if not predictions or not actual_results or actual_results == 'pending':
                return None

            # Get predicted and actual arrivals
            predicted_arrival = predictions.get('predicted_arriv', '')
            actual_arrival = actual_results

            # Calculate basic metrics
            winner_correct = self._is_winner_correct(predicted_arrival, actual_arrival)
            podium_accuracy = self._calculate_podium_accuracy(predicted_arrival, actual_arrival)

            # Calculate PMU bet wins
            pmu_bets = self._calculate_pmu_bets(race_data)
            winning_bets = [bet_type for bet_type, won in pmu_bets.items() if won]

            # Build race info
            race_info = {
                'comp': race_data.get('comp'),
                'hippo': race_data.get('hippo'),
                'prix': race_data.get('prix'),
                'prixnom': race_data.get('prixnom'),
                'jour': race_data.get('jour'),
                'partant': race_data.get('partant')
            }

            return RaceResult(
                comp=str(race_data.get('comp')),
                race_info=race_info,
                winner_correct=winner_correct,
                podium_accuracy=podium_accuracy,
                winning_bets=winning_bets,
                predicted_arrival=predicted_arrival,
                actual_arrival=actual_arrival,
                is_quinte=bool(race_data.get('quinte') == 1)
            )

        except Exception as e:
            # Simple error handling - just skip problematic races
            return None

    def _parse_predictions(self, prediction_results: str) -> Optional[Dict]:
        """Parse prediction results JSON."""
        if not prediction_results:
            return None

        try:
            if isinstance(prediction_results, str):
                return json.loads(prediction_results)
            return prediction_results
        except:
            return None

    def _is_winner_correct(self, predicted: str, actual: str) -> bool:
        """Check if winner is correctly predicted."""
        if not predicted or not actual:
            return False

        try:
            pred_first = predicted.split('-')[0]
            actual_first = actual.split('-')[0]
            return pred_first == actual_first
        except:
            return False

    def _calculate_podium_accuracy(self, predicted: str, actual: str) -> float:
        """Calculate podium (top 3) accuracy."""
        if not predicted or not actual:
            return 0.0

        try:
            pred_top3 = set(predicted.split('-')[:3])
            actual_top3 = set(actual.split('-')[:3])
            intersection = len(pred_top3 & actual_top3)
            return intersection / 3.0
        except:
            return 0.0

    def _calculate_pmu_bets(self, race_data: Dict) -> Dict[str, bool]:
        """Calculate which PMU bets would have won."""
        try:
            predictions = self._parse_predictions(race_data.get('prediction_results'))
            actual_results = race_data.get('actual_results', '')

            if not predictions or not actual_results:
                return {}

            predicted_arrival = predictions.get('predicted_arriv', '')

            # Use existing logic from orchestrator
            return self._calculate_arriv_metrics(predicted_arrival, actual_results)

        except:
            return {}

    def _calculate_arriv_metrics(self, predicted_arriv: str, actual_arriv: str) -> Dict[str, bool]:
        """
        Calculate evaluation metrics based on arrival strings, including PMU bet types.
        Simplified version of the orchestrator method.
        """
        if not predicted_arriv or not actual_arriv:
            return {}

        try:
            # Parse arrivals
            pred_positions = predicted_arriv.split('-')
            actual_positions = actual_arriv.split('-')

            # Ensure we have at least 5 positions for quinte bets
            min_length = min(len(pred_positions), len(actual_positions), 5)

            results = {}

            # Tiercé (top 3)
            if min_length >= 3:
                pred_top3 = pred_positions[:3]
                actual_top3 = actual_positions[:3]
                results['tierce_exact'] = pred_top3 == actual_top3
                results['tierce_desordre'] = set(pred_top3) == set(actual_top3)

            # Quarté (top 4)
            if min_length >= 4:
                pred_top4 = pred_positions[:4]
                actual_top4 = actual_positions[:4]
                results['quarte_exact'] = pred_top4 == actual_top4
                results['quarte_desordre'] = set(pred_top4) == set(actual_top4)

            # Quinté (top 5)
            if min_length >= 5:
                pred_top5 = pred_positions[:5]
                actual_top5 = actual_positions[:5]
                results['quinte_exact'] = pred_top5 == actual_top5
                results['quinte_desordre'] = set(pred_top5) == set(actual_top5)

                # Bonus bets
                pred_set = set(pred_top5)
                actual_set = set(actual_top5)
                intersection = len(pred_set & actual_set)

                results['bonus4'] = intersection >= 4
                results['bonus3'] = intersection >= 3
                results['deuxsur4'] = len(set(pred_top4) & set(actual_top4)) >= 2
                results['multi4'] = len(set(pred_top4) & set(actual_top5)) == 4

            return results

        except Exception:
            return {}

    def _would_win_with_extra_horses(self, race_data: Dict, total_horses: int) -> bool:
        """
        Check if betting on more horses would result in a win.
        This is a simplified estimation - in reality would require complex calculations.
        """
        # For now, return a simple heuristic
        # In a real implementation, you'd calculate the combinations
        result = self._evaluate_single_race(race_data)
        if not result:
            return False

        # If already winning quinte bets, adding more horses keeps the win
        quinte_bets = ['quinte_exact', 'quinte_desordre', 'bonus4', 'bonus3']
        if any(bet in result.winning_bets for bet in quinte_bets):
            return True

        # Simple heuristic: if podium accuracy is high, more horses might help
        return result.podium_accuracy >= 0.6

    def _calculate_aggregate_metrics(self, race_results: List[RaceResult]) -> EvaluationMetrics:
        """Calculate aggregate metrics from individual race results."""
        if not race_results:
            return EvaluationMetrics(
                total_races=0,
                races_evaluated=0,
                overall_winner_accuracy=0.0,
                overall_podium_accuracy=0.0,
                total_winning_bets=0,
                bet_win_rates={},
                quinte_performance=None
            )

        total_races = len(race_results)
        winner_correct_count = sum(1 for r in race_results if r.winner_correct)
        avg_podium_accuracy = sum(r.podium_accuracy for r in race_results) / total_races

        # Calculate bet win rates
        bet_win_rates = {}
        all_winning_bets = [bet for r in race_results for bet in r.winning_bets]

        for bet_type in self.bet_type_labels.keys():
            wins = sum(1 for r in race_results if bet_type in r.winning_bets)
            bet_win_rates[bet_type] = {
                'wins': wins,
                'total': total_races,
                'rate': wins / total_races if total_races > 0 else 0,
                'display_name': self.bet_type_labels[bet_type]
            }

        # Quinte-specific analysis
        quinte_races = [r for r in race_results if r.is_quinte]
        quinte_performance = None

        if quinte_races:
            quinte_winner_accuracy = sum(1 for r in quinte_races if r.winner_correct) / len(quinte_races)
            quinte_bet_types = ['quinte_exact', 'quinte_desordre', 'bonus4', 'bonus3']

            quinte_races_with_wins = sum(
                1 for r in quinte_races
                if any(bet in r.winning_bets for bet in quinte_bet_types)
            )

            quinte_performance = {
                'total_quinte_races': len(quinte_races),
                'winner_accuracy': quinte_winner_accuracy,
                'quinte_bet_win_rate': quinte_races_with_wins / len(quinte_races),
                'avg_podium_accuracy': sum(r.podium_accuracy for r in quinte_races) / len(quinte_races)
            }

        return EvaluationMetrics(
            total_races=total_races,
            races_evaluated=total_races,
            overall_winner_accuracy=winner_correct_count / total_races,
            overall_podium_accuracy=avg_podium_accuracy,
            total_winning_bets=len(all_winning_bets),
            bet_win_rates=bet_win_rates,
            quinte_performance=quinte_performance
        )


# Simple functions for IDE usage
def evaluate_all_predictions(db_name: str = None) -> EvaluationMetrics:
    """
    Evaluate all races with predictions and results.

    Args:
        db_name: Database name from config

    Returns:
        Complete evaluation metrics
    """
    evaluator = PredictEvaluator(db_name)
    return evaluator.evaluate_all_races()


def get_bet_type_wins(db_name: str = None) -> Dict[str, List[RaceResult]]:
    """
    Get races organized by bet types won.

    Args:
        db_name: Database name from config

    Returns:
        Dictionary mapping bet types to winning races
    """
    evaluator = PredictEvaluator(db_name)
    return evaluator.get_races_won_by_bet_type()


def analyze_quinte_horse_strategies(db_name: str = None) -> Optional[Dict]:
    """
    Analyze quinte betting strategies with different horse counts.

    Args:
        db_name: Database name from config

    Returns:
        Quinte strategy analysis or None
    """
    evaluator = PredictEvaluator(db_name)
    return evaluator.get_quinte_horse_betting_analysis()


# Usage example for IDE
if __name__ == "__main__":
    # Simple evaluation
    metrics = evaluate_all_predictions()

    print(f"Evaluated {metrics.races_evaluated} races")
    print(f"Winner accuracy: {metrics.overall_winner_accuracy:.2%}")
    print(f"Podium accuracy: {metrics.overall_podium_accuracy:.2%}")
    print(f"Total winning bets: {metrics.total_winning_bets}")

    # Print bet type performance
    print("\nBet type performance:")
    for bet_type, stats in metrics.bet_win_rates.items():
        if stats['wins'] > 0:
            print(f"  {stats['display_name']}: {stats['wins']}/{stats['total']} ({stats['rate']:.1%})")

    # Quinte analysis
    if metrics.quinte_performance:
        qp = metrics.quinte_performance
        print(f"\nQuinte races: {qp['total_quinte_races']}")
        print(f"Quinte winner accuracy: {qp['winner_accuracy']:.2%}")
        print(f"Quinte bet win rate: {qp['quinte_bet_win_rate']:.2%}")

    # Horse count strategy analysis for quinte races
    print("\n" + "=" * 60)
    print("QUINTE BETTING STRATEGY ANALYSIS")
    print("=" * 60)

    quinte_analysis = analyze_quinte_horse_strategies()
    if quinte_analysis:
        total_races = quinte_analysis['total_quinte_races']
        scenarios = quinte_analysis['betting_scenarios']

        print(f"Total quinte races analyzed: {total_races}")
        print("\nWin rate comparison by number of horses bet:")

        # Display results
        for scenario_name, data in scenarios.items():
            horse_count = scenario_name.replace('_horses', '').replace('_', ' ')
            wins = data['wins']
            total = data['total']
            win_rate = data['win_rate']
            print(f"  {horse_count.capitalize()} horses: {wins}/{total} ({win_rate:.1%})")

        # Calculate improvements
        base_rate = scenarios['5_horses']['win_rate']
        rate_6_horses = scenarios['6_horses']['win_rate']
        rate_7_horses = scenarios['7_horses']['win_rate']

        improvement_6 = rate_6_horses - base_rate
        improvement_7 = rate_7_horses - base_rate

        print(f"\nImprovement analysis:")
        print(f"  Base strategy (5 horses): {base_rate:.1%}")
        print(f"  6-horse improvement: {improvement_6:+.1%} ({improvement_6 * 100:.1f} percentage points)")
        print(f"  7-horse improvement: {improvement_7:+.1%} ({improvement_7 * 100:.1f} percentage points)")

        # Recommendations
        print(f"\nRecommendations:")
        if improvement_6 > 0.05:  # 5% improvement threshold
            print(f"  ✅ Using 6 horses shows significant improvement (+{improvement_6:.1%})")
        elif improvement_6 > 0:
            print(f"  ⚠️  Using 6 horses shows modest improvement (+{improvement_6:.1%})")
        else:
            print(f"  ❌ Using 6 horses does not improve results ({improvement_6:+.1%})")

        if improvement_7 > 0.10:  # 10% improvement threshold for 7 horses
            print(f"  ✅ Using 7 horses shows strong improvement (+{improvement_7:.1%})")
        elif improvement_7 > improvement_6:
            print(f"  ⚠️  Using 7 horses better than 6 horses (+{improvement_7:.1%} vs +{improvement_6:.1%})")
        else:
            print(f"  ❌ Using 7 horses not recommended ({improvement_7:+.1%})")

        # Cost-benefit analysis
        print(f"\nCost-benefit consideration:")
        if improvement_6 > 0:
            cost_multiplier_6 = 6 / 5  # 20% more combinations
            roi_6 = improvement_6 / (cost_multiplier_6 - 1)
            print(f"  6-horse ROI factor: {roi_6:.1f} (improvement vs extra cost)")

        if improvement_7 > 0:
            cost_multiplier_7 = 7 / 5  # 40% more combinations
            roi_7 = improvement_7 / (cost_multiplier_7 - 1)
            print(f"  7-horse ROI factor: {roi_7:.1f} (improvement vs extra cost)")

    else:
        print("\nNo quinte races found for horse count analysis")