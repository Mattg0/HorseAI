class EvaluationReporter:
    """Simple class to hold evaluation reporting functions."""

    @staticmethod
    def format_bet_name(bet_type):
        """Format bet type names for display"""
        name_mapping = {
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
        return name_mapping.get(bet_type, bet_type)

    @staticmethod
    def report_evaluation_results(results):
        """Generate a formatted report for race evaluation results."""
        if not results or 'status' not in results:
            return "Invalid results format"

        if results['status'] != 'success':
            return f"Error: {results.get('error', 'Unknown error')}"

        # Extract metrics
        metrics = results.get('metrics', {})
        race_info = metrics.get('race_info', {})

        # Format race information
        race_header = (
            f"\nEvaluation for race {race_info.get('comp')}: "
            f"{race_info.get('hippo')} - {race_info.get('prix')} "
            f"({race_info.get('jour')})"
        )

        # Format basic metrics
        basic_metrics = (
            f"Basic metrics:\n"
            f"  Winner correctly predicted: {'✓' if metrics.get('winner_correct') else '✗'}\n"
            f"  Podium accuracy: {metrics.get('podium_accuracy', 0):.2f}\n"
            f"  Mean rank error: {metrics.get('mean_rank_error', 'N/A'):.2f}"
        )

        # Format PMU bet results
        winning_bets = metrics.get('winning_bets', [])

        if not winning_bets:
            bet_results = "PMU bet results: No winning bets"
        else:
            bet_results = "PMU bet results: ✓ " + ", ".join([
                EvaluationReporter.format_bet_name(bet_type) for bet_type in winning_bets
            ])

        # Format arrival orders
        arrival_info = (
            f"Arrival orders:\n"
            f"  Predicted: {metrics.get('predicted_arriv', 'N/A')}\n"
            f"  Actual: {metrics.get('actual_arriv', 'N/A')}"
        )

        # Combine all sections
        report = f"{race_header}\n\n{basic_metrics}\n\n{bet_results}\n\n{arrival_info}"
        return report

    @staticmethod
    def report_quinte_evaluation(quinte_analysis):
        """Generate a formatted report for quinte race evaluation results."""
        if not quinte_analysis or quinte_analysis.get('quinte_races', 0) == 0:
            return "No quinte races were evaluated."

        quinte_count = quinte_analysis['quinte_races']
        winner_accuracy = quinte_analysis['winner_accuracy']
        quinte_bet_win_rate = quinte_analysis['quinte_bet_win_rate']
        any_bet_win_rate = quinte_analysis['any_bet_win_rate']

        # Format race count and winner accuracy
        race_header = (
            f"\n===== QUINTE RACES ANALYSIS ({quinte_count} races) =====\n"
            f"Winner accuracy: {winner_accuracy:.2f} "
            f"({int(winner_accuracy * quinte_count)}/{quinte_count})\n"
            f"Races with any quinte bet won: {quinte_bet_win_rate:.2f} "
            f"({int(quinte_bet_win_rate * quinte_count)}/{quinte_count})\n"
            f"Races with any bet type won: {any_bet_win_rate:.2f} "
            f"({int(any_bet_win_rate * quinte_count)}/{quinte_count})"
        )

        # Format performance by bet type
        bet_type_labels = {
            'tierce_exact': 'Tiercé Exact',
            'tierce_desordre': 'Tiercé Désordre',
            'quarte_exact': 'Quarté Exact',
            'quarte_desordre': 'Quarté Désordre',
            'quinte_exact': 'Quinté+ Exact',
            'quinte_desordre': 'Quinté+ Désordre',
            'bonus4': 'Bonus 4',
            'bonus3': 'Bonus 3',
            'deuxsur4': '2 sur 4',
            'multi4': 'Multi en 4'
        }

        bet_details = []
        bet_details.append("\nPerformance by bet type:")

        # First sort by success rate
        bet_types = sorted(
            quinte_analysis['bet_type_details'].items(),
            key=lambda x: x[1]['rate'],
            reverse=True
        )

        for bet_type, stats in bet_types:
            label = bet_type_labels.get(bet_type, bet_type)
            bet_details.append(
                f"  {label}: {stats['wins']}/{stats['total_races']} ({stats['rate']:.2f})"
            )

        # Format distribution of bets won per race
        bets_per_race = quinte_analysis.get('bets_per_race', {})
        if bets_per_race:
            bet_dist = ["\nBet types won per race:"]
            for bet_count, race_count in sorted(bets_per_race.items()):
                if bet_count == 0:
                    text = "No bets won"
                elif bet_count == 1:
                    text = "1 bet type won"
                else:
                    text = f"{bet_count} bet types won"
                bet_dist.append(f"  {text}: {race_count} races")
        else:
            bet_dist = []

        # Format race details for races with winning bets
        race_details = quinte_analysis.get('race_details', [])

        # Sort races by number of winning bets (highest first)
        race_details.sort(key=lambda x: x.get('winning_bet_count', 0), reverse=True)

        races_with_wins = [r for r in race_details if r.get('winning_bet_count', 0) > 0]

        if races_with_wins:
            race_output = ["\nQuinte races with winning bets:"]
            for race in races_with_wins:
                win_indicator = "✓" if race['winner_correct'] else "✗"
                bet_count = race.get('winning_bet_count', 0)

                # Format winning bets more concisely
                bet_names = []
                for bet in race.get('winning_bets', []):
                    short_name = bet_type_labels.get(bet, bet)
                    # Use shorter names
                    short_name = short_name.replace('Désordre', 'Dés')
                    short_name = short_name.replace('Exact', 'Ex')
                    bet_names.append(short_name)

                bet_text = ", ".join(bet_names)

                race_output.append(
                    f"  [{win_indicator}] {race['jour']} {race['hippo']} - {race['prix']}: "
                    f"{bet_count} bets ({bet_text})"
                )
        else:
            race_output = ["\nNo quinte races with winning bets"]

        # Combine all sections
        report = "\n".join([race_header] + bet_details + bet_dist + race_output)
        return report

    @staticmethod
    def report_summary_evaluation(summary):
        """Generate a formatted report for evaluation summary."""
        if not summary or 'summary_metrics' not in summary:
            return "No summary metrics available"

        metrics = summary['summary_metrics']
        races_evaluated = metrics.get('races_evaluated', 0)

        if races_evaluated == 0:
            return "No races evaluated"

        # Basic statistics
        basic_stats = (
            f"\nEvaluation Summary ({races_evaluated} races):\n"
            f"  Winner accuracy: {metrics.get('winner_accuracy', 0):.2f} "
            f"({int(metrics.get('winner_accuracy', 0) * races_evaluated)}/{races_evaluated})\n"
            f"  Average podium accuracy: {metrics.get('avg_podium_accuracy', 0):.2f}\n"
            f"  Average mean rank error: {metrics.get('avg_mean_rank_error', 'N/A'):.2f}"
        )

        # PMU bet statistics
        bet_stats = metrics.get('bet_statistics', {})
        pmu_summary = (
            f"\nOverall PMU Bet Performance:\n"
            f"  Races with at least one winning bet: {bet_stats.get('races_with_wins', 0)} "
            f"({bet_stats.get('win_rate', 0):.2f})\n"
            f"  Races with no winning bets: {bet_stats.get('races_with_no_wins', 0)}"
        )

        # Distribution of races by bet count
        bet_counts = bet_stats.get('races_by_bet_count', {})
        if bet_counts:
            bet_count_lines = [
                f"  Races winning {count} bet types: {bet_counts.get(count, 0)}"
                for count in sorted(bet_counts.keys())
            ]
            bet_count_summary = "\n" + "\n".join(bet_count_lines)
        else:
            bet_count_summary = ""

        # Detailed bet type performance
        bet_type_summary = metrics.get('bet_type_summary', {})
        if bet_type_summary:
            bet_type_lines = []
            for bet_type, stats in bet_type_summary.items():
                bet_type_lines.append(
                    f"  {EvaluationReporter.format_bet_name(bet_type)}: "
                    f"{stats.get('wins', 0)}/{stats.get('total_races', 0)} "
                    f"({stats.get('success_rate', 0):.1f}%)"
                )
            bet_type_report = "\n\nPMU Bet Type Success Rates:\n" + "\n".join(bet_type_lines)
        else:
            bet_type_report = ""

        # Complete report
        report = f"{basic_stats}\n{pmu_summary}{bet_count_summary}{bet_type_report}"
        return report