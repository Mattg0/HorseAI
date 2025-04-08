#!/usr/bin/env python
# model_training/regression_enhancement.py

import os
import sys
import pandas as pd
import numpy as np
import json
import sqlite3
import argparse
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
import joblib
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Add parent directory to path if needed
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import project modules
from utils.env_setup import AppConfig, get_sqlite_dbpath
from core.orchestrators.embedding_feature import FeatureEmbeddingOrchestrator
from model_training.regressions.isotonic_calibration import (
    CalibratedRegressor, regression_metrics_report, plot_prediction_vs_actual,
    plot_calibration_effect, plot_histogram_of_errors
)


class CombinedModel:
    def __init__(self, base_model, correction_models, blend_weight=0.5):
        self.base_model = base_model
        self.correction_models = correction_models
        self.blend_weight = blend_weight

    def predict(self, X, race_type=None):
        # First get base predictions
        base_preds = self.base_model.predict(X)

        # Create DataFrame for correction models
        X_corrector = pd.DataFrame(X, columns=X.columns if hasattr(X, 'columns') else None)
        X_corrector['predicted_position'] = base_preds

        # Determine which correction model to use
        if race_type and race_type in self.correction_models:
            correction_model = self.correction_models[race_type]
        else:
            correction_model = self.correction_models.get('global')

        # Apply correction model if available
        if correction_model is not None:
            # Get feature names required by the correction model
            if hasattr(correction_model, 'feature_names_in_'):
                required_features = correction_model.feature_names_in_
            else:
                # Assume it needs at least predicted_position
                required_features = ['predicted_position']

            # Ensure all required features are present
            for feature in required_features:
                if feature not in X_corrector.columns:
                    X_corrector[feature] = 0  # Default value

            # Get corrected predictions
            corrected_preds = correction_model.predict(X_corrector[required_features])

            # Blend predictions
            final_preds = (base_preds * self.blend_weight +
                           corrected_preds * (1 - self.blend_weight))
        else:
            # If no correction model, use base predictions
            final_preds = base_preds

        return final_preds

    def predict_with_type(self, X, race_types):
        """Predict with race types specified for each sample."""
        # Get base predictions for all samples
        base_preds = self.base_model.predict(X)

        # Initialize final predictions with base predictions
        final_preds = base_preds.copy()

        # Group samples by race type
        unique_types = set(race_types)

        # Create DataFrame for correction
        X_corrector = pd.DataFrame(X, columns=X.columns if hasattr(X, 'columns') else None)
        X_corrector['predicted_position'] = base_preds

        # Apply correction for each race type
        for race_type in unique_types:
            # Find indices for this race type
            indices = [i for i, rt in enumerate(race_types) if rt == race_type]

            if not indices:
                continue

            # Get the appropriate correction model
            correction_model = (self.correction_models.get(race_type) or
                                self.correction_models.get('global'))

            if correction_model is None:
                continue

            # Get feature names required by the correction model
            if hasattr(correction_model, 'feature_names_in_'):
                required_features = correction_model.feature_names_in_
            else:
                # Assume it needs at least predicted_position
                required_features = ['predicted_position']

            # Ensure all required features are present
            for feature in required_features:
                if feature not in X_corrector.columns:
                    X_corrector[feature] = 0  # Default value

            # Get the subset for this race type
            X_subset = X_corrector.iloc[indices]

            # Get corrected predictions
            corrected_preds = correction_model.predict(X_subset[required_features])

            # Blend predictions
            for i, index in enumerate(indices):
                final_preds[index] = (base_preds[index] * self.blend_weight +
                                      corrected_preds[i] * (1 - self.blend_weight))

        return final_preds

class RegressionEnhancer:
    """
    Analyzes prediction gaps and enhances models using regression analysis.

    This module focuses on:
    1. Identifying systematic prediction errors
    2. Building regression models to understand these gaps
    3. Updating models to correct for identified biases
    4. Verifying improved performance
    """

    def __init__(self, model_path: str, db_name: str = None,
                 output_dir: str = None, verbose: bool = False):
        """
        Initialize the regression enhancer.

        Args:
            model_path: Path to the model directory
            db_name: Database name from config (defaults to active_db)
            output_dir: Directory for output files (defaults to model_path/analysis)
            verbose: Whether to enable verbose output
        """
        # Initialize config
        self.config = AppConfig()

        # Set database
        if db_name is None:
            self.db_name = self.config._config.base.active_db
        else:
            self.db_name = db_name

        # Get database path
        self.db_path = self.config.get_sqlite_dbpath(self.db_name)

        # Model path should be a directory
        self.model_path = Path(model_path)
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model path {self.model_path} does not exist")

        # Find latest version if it exists
        versions = [d for d in self.model_path.iterdir()
                    if d.is_dir() and d.name.startswith('v')]

        if versions:
            versions.sort(reverse=True)
            self.version_path = versions[0]
        else:
            self.version_path = self.model_path

        # Set up output directory
        if output_dir is None:
            self.output_dir = self.version_path / "analysis"
        else:
            self.output_dir = Path(output_dir)

        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Set verbosity
        self.verbose = verbose

        # Set up logging
        self._setup_logging()

        # Set up paths
        self.rf_model_path = self.version_path / "hybrid_rf_model.joblib"
        self.feature_config_path = self.version_path / "hybrid_feature_engineer.joblib"
        self.model_config_path = self.version_path / "model_config.json"

        # Initialize components
        self.orchestrator = FeatureEmbeddingOrchestrator(
            sqlite_path=self.db_path,
            verbose=verbose
        )

        # Load models
        self._load_models()

        # Initialize bias correction models
        self.bias_models = {}

        self.logger.info(f"Regression Enhancer initialized with model at {self.version_path}")

    def _setup_logging(self):
        """Set up logging."""
        log_file = self.output_dir / f"regression_analysis_{datetime.now().strftime('%Y%m%d')}.log"

        logging.basicConfig(
            level=logging.DEBUG if self.verbose else logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ],
            force=True
        )

        self.logger = logging.getLogger("RegressionEnhancer")
        self.logger.info(f"Logging initialized to {log_file}")

    def _load_models(self):
        """Load existing models and configurations."""
        try:
            with open(self.model_config_path, 'r') as f:
                self.model_config = json.load(f)
                self.logger.info(f"Loaded model config: {self.model_config.get('version', 'unknown')}")
        except Exception as e:
            self.logger.error(f"Error loading model config: {str(e)}")
            self.model_config = {}

        try:
            self.feature_config = joblib.load(self.feature_config_path)
            self.logger.info("Loaded feature engineering configuration")

            # Update orchestrator with feature configuration
            if isinstance(self.feature_config, dict):
                if 'preprocessing_params' in self.feature_config:
                    self.orchestrator.preprocessing_params.update(
                        self.feature_config['preprocessing_params']
                    )
                if 'embedding_dim' in self.feature_config:
                    self.orchestrator.embedding_dim = self.feature_config['embedding_dim']
        except Exception as e:
            self.logger.error(f"Error loading feature config: {str(e)}")
            self.feature_config = {}

        try:
            rf_data = joblib.load(self.rf_model_path)

            # Handle different saving formats
            if isinstance(rf_data, dict) and 'model' in rf_data:
                self.rf_model = rf_data['model']
            elif isinstance(rf_data, CalibratedRegressor):
                self.rf_model = rf_data
            elif hasattr(rf_data, 'predict') and callable(getattr(rf_data, 'predict')):
                self.rf_model = rf_data
            else:
                self.rf_model = rf_data

            self.logger.info(f"Loaded RF model: {type(self.rf_model)}")
        except Exception as e:
            self.logger.error(f"Error loading RF model: {str(e)}")
            self.rf_model = None

    def collect_prediction_data(self, date_from: str = None, date_to: str = None,
                                limit: int = None) -> pd.DataFrame:
        """
        Collect prediction data from completed races.

        Args:
            date_from: Start date (YYYY-MM-DD)
            date_to: End date (YYYY-MM-DD)
            limit: Maximum number of races to collect

        Returns:
            DataFrame with prediction data
        """
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        # Build query with optional date filters
        query = """
        SELECT * FROM daily_race 
        WHERE actual_results IS NOT NULL 
        AND actual_results != 'pending'
        AND prediction_results IS NOT NULL
        """

        params = []
        if date_from:
            query += " AND jour >= ?"
            params.append(date_from)
        if date_to:
            query += " AND jour <= ?"
            params.append(date_to)

        query += " ORDER BY jour DESC"

        if limit:
            query += " LIMIT ?"
            params.append(limit)

        cursor.execute(query, params)
        rows = cursor.fetchall()

        # Process races to extract prediction vs actual results
        prediction_data = []

        for row in rows:
            race = dict(row)
            race_id = race['comp']

            try:
                # Parse JSON for prediction_results and participants
                for field in ['prediction_results', 'participants']:
                    if race.get(field) and isinstance(race[field], str):
                        try:
                            race[field] = json.loads(race[field])
                        except json.JSONDecodeError as e:
                            self.logger.error(f"Error parsing JSON for {field} in race {race_id}: {str(e)}")
                            race[field] = []

                # Process actual_results - always a hyphen-separated string
                actual_results = race.get('actual_results', '')
                if not actual_results or actual_results == 'pending':
                    # Skip races without results
                    continue

                # Parse the hyphen-separated finishing order
                try:
                    # Format: "1-4-2-3" (numeros in order of finish)
                    numeros = actual_results.split('-')
                    actual_positions = {numeros[i]: i + 1 for i in range(len(numeros))}
                except Exception as e:
                    self.logger.error(f"Error parsing actual_results '{actual_results}' for race {race_id}: {str(e)}")
                    continue

                # Extract predictions
                predictions = []
                pred_results = race.get('prediction_results', {})

                if isinstance(pred_results, dict) and 'predictions' in pred_results:
                    predictions = pred_results['predictions']
                elif isinstance(pred_results, list):
                    predictions = pred_results
                else:
                    # Skip if we can't get predictions
                    continue

                # Extract participant info if available
                participants_dict = {}
                participants = race.get('participants', [])
                if isinstance(participants, list):
                    for p in participants:
                        if isinstance(p, dict) and 'numero' in p:
                            participants_dict[str(p['numero'])] = p

                # Add race features
                race_features = {
                    'race_id': race_id,
                    'jour': race.get('jour'),
                    'hippo': race.get('hippo'),
                    'typec': race.get('typec'),
                    'dist': race.get('dist'),
                    'natpis': race.get('natpis'),
                    'meteo': race.get('meteo'),
                    'temperature': race.get('temperature'),
                    'corde': race.get('corde'),
                    'quinte': 1 if race.get('quinte') else 0,
                    'partant': race.get('partant')
                }

                # Match predictions with actual results
                for pred in predictions:
                    if not isinstance(pred, dict):
                        continue

                    numero = str(pred.get('numero', ''))
                    if not numero or numero not in actual_positions:
                        continue

                    # Get predicted position
                    pred_pos = pred.get('predicted_position')
                    if pred_pos is None:
                        continue

                    try:
                        pred_pos = float(pred_pos)
                    except (ValueError, TypeError):
                        continue

                    # Build data entry
                    entry = {
                        **race_features,
                        'numero': numero,
                        'predicted_position': pred_pos,
                        'actual_position': actual_positions[numero],
                        'prediction_error': pred_pos - actual_positions[numero],
                        'abs_error': abs(pred_pos - actual_positions[numero])
                    }

                    # Add participant features if available
                    participant = participants_dict.get(numero, {})
                    for key in ['cotedirect', 'age', 'poidmont', 'ratio_victoires',
                                'ratio_places', 'victoirescheval', 'placescheval']:
                        if key in participant:
                            entry[key] = participant[key]

                    prediction_data.append(entry)

            except Exception as e:
                self.logger.error(f"Error processing race {race_id}: {str(e)}")
                import traceback
                self.logger.error(traceback.format_exc())

        conn.close()

        # Convert to DataFrame
        if prediction_data:
            df = pd.DataFrame(prediction_data)
            self.logger.info(f"Collected prediction data for {len(df)} horses from {df['race_id'].nunique()} races")
            return df
        else:
            self.logger.warning("No prediction data collected")
            return pd.DataFrame()

    def analyze_prediction_gaps(self, prediction_df: pd.DataFrame) -> Dict[str, Any]:
        """
        Perform detailed analysis of prediction gaps.

        Args:
            prediction_df: DataFrame with prediction data

        Returns:
            Dictionary with analysis results
        """
        if len(prediction_df) == 0:
            return {"status": "error", "message": "No prediction data provided"}

        self.logger.info(f"Analyzing prediction gaps for {len(prediction_df)} samples")

        # 1. Overall error metrics
        overall_metrics = {
            "count": len(prediction_df),
            "mean_error": prediction_df['prediction_error'].mean(),
            "median_error": prediction_df['prediction_error'].median(),
            "mean_abs_error": prediction_df['abs_error'].mean(),
            "rmse": np.sqrt(np.mean(prediction_df['prediction_error'] ** 2)),
            "error_std": prediction_df['prediction_error'].std()
        }

        # Check for bias (systematic over/under prediction)
        if abs(overall_metrics["mean_error"]) > 0.2:  # Threshold for considering it biased
            bias_direction = "over-prediction" if overall_metrics["mean_error"] < 0 else "under-prediction"
            overall_metrics["bias"] = {
                "direction": bias_direction,
                "magnitude": abs(overall_metrics["mean_error"])
            }

        # 2. Error distribution visualization
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

        # Histogram of prediction errors
        ax1.hist(prediction_df['prediction_error'], bins=30, alpha=0.7)
        ax1.axvline(0, color='black', linestyle='-', linewidth=1)
        ax1.axvline(overall_metrics["mean_error"], color='red', linestyle='--',
                    label=f'Mean: {overall_metrics["mean_error"]:.2f}')
        ax1.set_title('Prediction Error Distribution')
        ax1.set_xlabel('Prediction Error (Predicted - Actual)')
        ax1.set_ylabel('Frequency')
        ax1.legend()

        # Scatter plot of predicted vs actual
        ax2.scatter(prediction_df['actual_position'], prediction_df['predicted_position'],
                    alpha=0.5, s=20)
        min_val = min(prediction_df['actual_position'].min(), prediction_df['predicted_position'].min())
        max_val = max(prediction_df['actual_position'].max(), prediction_df['predicted_position'].max())
        ax2.plot([min_val, max_val], [min_val, max_val], 'r--')
        ax2.set_title('Predicted vs Actual Positions')
        ax2.set_xlabel('Actual Position')
        ax2.set_ylabel('Predicted Position')

        # Save plot
        plt.tight_layout()
        fig.savefig(self.output_dir / 'prediction_error_analysis.png', dpi=300)
        plt.close(fig)

        # 3. Error analysis by race and horse factors
        factor_analyses = {}

        # Analysis by race type
        if 'typec' in prediction_df.columns:
            type_analysis = prediction_df.groupby('typec').agg({
                'prediction_error': ['count', 'mean', 'std'],
                'abs_error': 'mean',
                'actual_position': 'count'
            })

            # Filter to include only types with enough data
            type_analysis = type_analysis[type_analysis[('prediction_error', 'count')] >= 10]

            # Unstack the MultiIndex columns
            type_analysis.columns = [f"{col[0]}_{col[1]}" for col in type_analysis.columns]

            # Calculate RMSE for each type
            for race_type in type_analysis.index:
                type_data = prediction_df[prediction_df['typec'] == race_type]
                type_analysis.loc[race_type, 'rmse'] = np.sqrt(np.mean(type_data['prediction_error'] ** 2))

            # Create visualization
            fig, ax = plt.subplots(figsize=(10, 6))
            bars = ax.bar(type_analysis.index, type_analysis['prediction_error_mean'])

            # Color bars based on bias direction
            for i, bar in enumerate(bars):
                color = 'red' if type_analysis['prediction_error_mean'].iloc[i] < 0 else 'blue'
                bar.set_color(color)

            ax.axhline(0, color='black', linestyle='-', linewidth=1)
            ax.set_title('Prediction Bias by Race Type')
            ax.set_xlabel('Race Type')
            ax.set_ylabel('Mean Prediction Error (Predicted - Actual)')

            # Add count labels
            for i, v in enumerate(type_analysis['prediction_error_mean']):
                ax.text(i, v + (0.1 if v >= 0 else -0.2),
                        f"n={type_analysis['prediction_error_count'].iloc[i]}",
                        ha='center', fontsize=9)

            plt.tight_layout()
            fig.savefig(self.output_dir / 'bias_by_race_type.png', dpi=300)
            plt.close(fig)

            factor_analyses['race_type'] = type_analysis.to_dict()

        # Analysis by hippo
        if 'hippo' in prediction_df.columns:
            hippo_analysis = prediction_df.groupby('hippo').agg({
                'prediction_error': ['count', 'mean', 'std'],
                'abs_error': 'mean'
            })

            # Filter to include only tracks with enough data
            hippo_analysis = hippo_analysis[hippo_analysis[('prediction_error', 'count')] >= 15]

            # Unstack the MultiIndex columns
            hippo_analysis.columns = [f"{col[0]}_{col[1]}" for col in hippo_analysis.columns]

            # Calculate RMSE for each track
            for track in hippo_analysis.index:
                track_data = prediction_df[prediction_df['hippo'] == track]
                hippo_analysis.loc[track, 'rmse'] = np.sqrt(np.mean(track_data['prediction_error'] ** 2))

            # Sort by absolute bias
            hippo_analysis['abs_bias'] = hippo_analysis['prediction_error_mean'].abs()
            hippo_analysis = hippo_analysis.sort_values('abs_bias', ascending=False)

            # Create visualization for top 10 tracks with highest bias
            top_tracks = hippo_analysis.head(10).copy()

            if len(top_tracks) > 0:
                fig, ax = plt.subplots(figsize=(12, 6))
                bars = ax.bar(top_tracks.index, top_tracks['prediction_error_mean'])

                # Color bars based on bias direction
                for i, bar in enumerate(bars):
                    color = 'red' if top_tracks['prediction_error_mean'].iloc[i] < 0 else 'blue'
                    bar.set_color(color)

                ax.axhline(0, color='black', linestyle='-', linewidth=1)
                ax.set_title('Top 10 Tracks by Prediction Bias')
                ax.set_xlabel('Track')
                ax.set_ylabel('Mean Prediction Error (Predicted - Actual)')
                plt.xticks(rotation=45, ha='right')

                # Add count labels
                for i, v in enumerate(top_tracks['prediction_error_mean']):
                    ax.text(i, v + (0.1 if v >= 0 else -0.2),
                            f"n={top_tracks['prediction_error_count'].iloc[i]}",
                            ha='center', fontsize=9)

                plt.tight_layout()
                fig.savefig(self.output_dir / 'bias_by_track.png', dpi=300)
                plt.close(fig)

                factor_analyses['track'] = hippo_analysis.to_dict()

        # Analysis by odds (cotedirect) if available
        if 'cotedirect' in prediction_df.columns:
            # Create odds bins
            prediction_df['odds_bin'] = pd.cut(
                prediction_df['cotedirect'],
                bins=[0, 2, 4, 7, 10, 15, 30, float('inf')],
                labels=['1-2', '2-4', '4-7', '7-10', '10-15', '15-30', '30+']
            )

            odds_analysis = prediction_df.groupby('odds_bin').agg({
                'prediction_error': ['count', 'mean', 'std'],
                'abs_error': 'mean'
            })

            # Unstack the MultiIndex columns
            odds_analysis.columns = [f"{col[0]}_{col[1]}" for col in odds_analysis.columns]

            # Calculate RMSE for each odds bin
            for odds_bin in odds_analysis.index:
                bin_data = prediction_df[prediction_df['odds_bin'] == odds_bin]
                odds_analysis.loc[odds_bin, 'rmse'] = np.sqrt(np.mean(bin_data['prediction_error'] ** 2))

            # Create visualization
            fig, ax = plt.subplots(figsize=(10, 6))
            bars = ax.bar(odds_analysis.index, odds_analysis['prediction_error_mean'])

            # Color bars based on bias direction
            for i, bar in enumerate(bars):
                color = 'red' if odds_analysis['prediction_error_mean'].iloc[i] < 0 else 'blue'
                bar.set_color(color)

            ax.axhline(0, color='black', linestyle='-', linewidth=1)
            ax.set_title('Prediction Bias by Odds')
            ax.set_xlabel('Odds Range')
            ax.set_ylabel('Mean Prediction Error (Predicted - Actual)')

            # Add count labels
            for i, v in enumerate(odds_analysis['prediction_error_mean']):
                ax.text(i, v + (0.1 if v >= 0 else -0.2),
                        f"n={odds_analysis['prediction_error_count'].iloc[i]}",
                        ha='center', fontsize=9)

            plt.tight_layout()
            fig.savefig(self.output_dir / 'bias_by_odds.png', dpi=300)
            plt.close(fig)

            factor_analyses['odds'] = odds_analysis.to_dict()

        # 4. Identify strongest predictors of bias
        bias_predictors = self._identify_bias_predictors(prediction_df)

        # Compile analysis results
        analysis_results = {
            "status": "success",
            "metrics": overall_metrics,
            "factor_analyses": factor_analyses,
            "bias_predictors": bias_predictors
        }

        # Save detailed results to file
        with open(self.output_dir / 'prediction_gap_analysis.json', 'w') as f:
            json.dump(analysis_results, f, indent=4, default=str)

        return analysis_results

    def _identify_bias_predictors(self, prediction_df: pd.DataFrame) -> Dict[str, Any]:
        """
        Identify features that best predict systematic bias.

        Args:
            prediction_df: DataFrame with prediction data

        Returns:
            Dictionary with bias predictor analysis
        """
        # Prepare data for modeling
        # We want to predict prediction_error using available features

        # First, select only numeric columns
        numeric_cols = prediction_df.select_dtypes(include=np.number).columns.tolist()

        # Exclude target and direct prediction columns
        excluded_cols = ['prediction_error', 'abs_error', 'predicted_position',
                         'actual_position', 'numero']
        feature_cols = [col for col in numeric_cols if col not in excluded_cols]

        # Handle missing values
        X = prediction_df[feature_cols].fillna(0)
        y = prediction_df['prediction_error']

        # Check if we have enough data
        if len(X) < 30 or len(feature_cols) < 3:
            return {"status": "insufficient_data"}

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42
        )

        # Try different regression models to find predictors
        models = {
            'linear': LinearRegression(),
            'ridge': Ridge(alpha=1.0),
            'lasso': Lasso(alpha=0.1)
        }

        results = {}

        for name, model in models.items():
            # Train model
            model.fit(X_train, y_train)

            # Evaluate
            y_pred = model.predict(X_test)
            mse = mean_squared_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)

            # Get feature importance
            if hasattr(model, 'coef_'):
                coefficients = model.coef_
                importance = np.abs(coefficients)
                # Sort features by importance
                indices = np.argsort(importance)[::-1]

                # Store top features
                top_features = [
                    {"feature": feature_cols[i],
                     "coefficient": float(coefficients[i]),
                     "importance": float(importance[i])}
                    for i in indices[:10]  # Top 10 features
                ]
            else:
                top_features = []

            results[name] = {
                "mse": float(mse),
                "r2": float(r2),
                "top_features": top_features
            }

        # Select best model based on R²
        best_model = max(results.items(), key=lambda x: x[1]["r2"])
        best_model_name = best_model[0]

        self.logger.info(f"Best bias predictor model: {best_model_name} with R²={best_model[1]['r2']:.4f}")

        # Plot feature importance for best model
        if results[best_model_name]["top_features"]:
            top_features = results[best_model_name]["top_features"]

            # Sort by absolute coefficient
            top_features.sort(key=lambda x: abs(x["coefficient"]), reverse=True)

            # Create visualization
            fig, ax = plt.subplots(figsize=(10, 8))

            feature_names = [f["feature"] for f in top_features]
            coefficients = [f["coefficient"] for f in top_features]

            # Create horizontal bar chart
            bars = ax.barh(range(len(feature_names)), coefficients)

            # Color bars based on coefficient sign
            for i, bar in enumerate(bars):
                color = 'red' if coefficients[i] < 0 else 'blue'
                bar.set_color(color)

            ax.set_yticks(range(len(feature_names)))
            ax.set_yticklabels(feature_names)
            ax.set_xlabel('Coefficient')
            ax.set_title(f'Top Features in {best_model_name.title()} Model for Predicting Bias')
            ax.axvline(0, color='black', linestyle='-', linewidth=0.5)

            plt.tight_layout()
            fig.savefig(self.output_dir / 'bias_feature_importance.png', dpi=300)
            plt.close(fig)

        return {
            "models": results,
            "best_model": best_model_name
        }

    def build_correction_model(
            base_model_path: str = None,
            new_data_db: str = "dev",
            use_latest_base: bool = False,
            output_dir: str = None,
            verbose: bool = True
    ):
        """
        Build an incremental model by updating a base model with new data.

        Args:
            base_model_path: Path to base model (if not using latest_base)
            new_data_db: Database to use for new data
            use_latest_base: Whether to use latest base model from config
            output_dir: Custom output directory (optional)
            verbose: Whether to print verbose output

        Returns:
            Dictionary with paths to saved artifacts
        """
        # Import model manager
        from utils.model_manager import get_model_manager

        if verbose:
            print("\n===== BUILDING INCREMENTAL MODEL =====")

        # Initialize model manager
        model_manager = get_model_manager()

        # Resolve base model path
        if not base_model_path and not use_latest_base:
            raise ValueError("Either base_model_path or use_latest_base must be provided")

        try:
            base_path = model_manager.resolve_model_path(
                model_path=base_model_path,
                use_latest_base=use_latest_base
            )

            if verbose:
                print(f"Using base model from: {base_path}")
        except ValueError as e:
            raise ValueError(f"Could not resolve base model path: {str(e)}")

        # Load base model
        artifacts = model_manager.load_model_artifacts(
            base_path=base_path,
            load_rf=True,
            load_lstm=True,
            load_feature_config=True,
            verbose=verbose
        )

        base_rf_model = artifacts.get('rf_model')
        base_feature_config = artifacts.get('feature_config')

        if not base_rf_model:
            raise ValueError("Base RF model could not be loaded")

        # Load new data using orchestrator
        from core.orchestrators.embedding_feature import FeatureEmbeddingOrchestrator

        # Create orchestrator with same configuration as base model
        if base_feature_config and isinstance(base_feature_config, dict):
            embedding_dim = base_feature_config.get('embedding_dim', 8)
            sequence_length = base_feature_config.get('sequence_length', 5)
        else:
            embedding_dim = 8  # Default
            sequence_length = 5  # Default

        orchestrator = FeatureEmbeddingOrchestrator(
            sqlite_path=None,  # Will be set based on db_name
            db_name=new_data_db,
            embedding_dim=embedding_dim,
            sequence_length=sequence_length,
            verbose=verbose
        )

        # Load recent data
        # Note: We're only using the last month of data for incremental update
        from datetime import datetime, timedelta
        end_date = datetime.now().strftime('%Y-%m-%d')
        start_date = (datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d')

        date_filter = f"jour BETWEEN '{start_date}' AND '{end_date}'"

        if verbose:
            print(f"Loading recent data from {start_date} to {end_date}")

        # Load and prepare data
        X_train, X_val, X_test, y_train, y_val, y_test = orchestrator.run_pipeline(
            limit=None,
            date_filter=date_filter,
            use_cache=True,
            clean_embeddings=True
        )

        if verbose:
            print(f"Loaded {len(X_train) + len(X_val) + len(X_test)} samples for incremental update")

        # Create a combined dataset for training
        import pandas as pd
        X_combined = pd.concat([X_train, X_val, X_test])
        y_combined = pd.concat([y_train, y_val, y_test])

        if verbose:
            print(f"Combined dataset has {len(X_combined)} samples")

        # Create and train incremental model
        from model_training.regressions.isotonic_calibration import CalibratedRegressor

        # Option 1: Train new model directly on new data
        # incremental_rf = clone(base_rf_model)
        # incremental_rf.fit(X_combined, y_combined)

        # Option 2: Build upon existing model (better for continuous learning)
        # This depends on your model type, here's a simple example
        if hasattr(base_rf_model, 'base_regressor'):
            # This is a CalibratedRegressor with a base model
            base_regressor = base_rf_model.base_regressor

            # For now, just retrain on new data
            # In a real implementation, you might have more sophisticated
            # incremental learning approaches
            base_regressor.n_estimators += 20  # Add more trees
            base_regressor.fit(X_combined, y_combined)

            # Create a new calibrated regressor
            incremental_rf = CalibratedRegressor(
                base_regressor=base_regressor,
                clip_min=1.0,
                clip_max=None
            )

            # Fit the calibrator on validation data
            incremental_rf.fit(X_combined, y_combined)
        else:
            # Direct model, just fit it again
            base_rf_model.fit(X_combined, y_combined)
            incremental_rf = base_rf_model

        # Evaluate the incremental model
        from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
        predictions = incremental_rf.predict(X_test)

        eval_results = {
            'rmse': float(np.sqrt(mean_squared_error(y_test, predictions))),
            'mae': float(mean_absolute_error(y_test, predictions)),
            'r2': float(r2_score(y_test, predictions)),
            'test_size': len(X_test),
            'date_range': f"{start_date} to {end_date}"
        }

        if verbose:
            print("\n===== INCREMENTAL MODEL EVALUATION =====")
            print(f"RMSE: {eval_results['rmse']:.4f}")
            print(f"MAE: {eval_results['mae']:.4f}")
            print(f"R²: {eval_results['r2']:.4f}")
            print(f"Test samples: {eval_results['test_size']}")

        # Save the incremental model
        from race_prediction.daily_training import DailyModelTrainer

        # Create a trainer instance
        daily_trainer = DailyModelTrainer(db_name=new_data_db, verbose=verbose)

        # Call the new save_incremental_model method
        saved_paths = daily_trainer.save_incremental_model(
            rf_model=incremental_rf,
            lstm_model=None,  # We're not updating LSTM in this example
            evaluation_results=eval_results
        )

        if verbose:
            print(f"Incremental model saved to: {saved_paths}")

        return {
            'model': incremental_rf,
            'evaluation': eval_results,
            'paths': saved_paths,
            'feature_orchestrator': orchestrator
        }
    def _train_correction_model(self, df: pd.DataFrame, model_name: str) -> Dict[str, Any]:
        """
        Train a model to correct prediction errors.

        Args:
            df: DataFrame with prediction data
            model_name: Name for the model

        Returns:
            Dictionary with model and performance metrics
        """
        # Prepare data
        # Input: predicted_position and other features
        # Output: actual_position (or correction factor)

        # Determine if we should predict actual position or correction factor
        # In this implementation, we'll predict the actual position

        # First, select features that might help improve predictions
        numeric_cols = df.select_dtypes(include=np.number).columns.tolist()

        # We always include the original prediction
        features = ['predicted_position']

        # Add horse-related features if available
        potential_features = [
            'cotedirect', 'age', 'poidmont', 'ratio_victoires', 'ratio_places',
            'victoirescheval', 'placescheval'
        ]

        # Add race-related features
        race_features = ['dist', 'partant', 'quinte', 'temperature']

        # Combine all potential features
        potential_features.extend(race_features)

        # Filter to include only available columns
        for col in potential_features:
            if col in numeric_cols:
                features.append(col)

        # Add available embedding features (first dimensions of each type)
        for i in range(2):  # Just first 2 dimensions
            for embed_type in ['horse_emb_', 'jockey_emb_', 'couple_emb_']:
                key = f"{embed_type}{i}"
                if key in numeric_cols:
                    features.append(key)

        # Filter features to only include those in the dataframe
        features = [f for f in features if f in df.columns]

        # Handle missing values
        X = df[features].fillna(0)
        y = df['actual_position']

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        # Train model - Random Forest for correction
        model = RandomForestRegressor(
            n_estimators=100,
            max_depth=10,
            min_samples_leaf=2,
            random_state=42
        )

        model.fit(X_train, y_train)

        # Evaluate
        # Original predictions (baseline)
        original_rmse = np.sqrt(mean_squared_error(y_test, X_test['predicted_position']))
        original_mae = mean_absolute_error(y_test, X_test['predicted_position'])

        # Corrected predictions
        corrected_pred = model.predict(X_test)
        corrected_rmse = np.sqrt(mean_squared_error(y_test, corrected_pred))
        corrected_mae = mean_absolute_error(y_test, corrected_pred)

        # Calculate improvement
        rmse_improvement = ((original_rmse - corrected_rmse) / original_rmse) * 100
        mae_improvement = ((original_mae - corrected_mae) / original_mae) * 100

        # Average improvement
        avg_improvement = (rmse_improvement + mae_improvement) / 2

        self.logger.info(f"Model {model_name}: RMSE improved by {rmse_improvement:.2f}%, MAE by {mae_improvement:.2f}%")

        # Feature importance
        if hasattr(model, 'feature_importances_'):
            importance = model.feature_importances_
            indices = np.argsort(importance)[::-1]

            feature_importance = [
                {"feature": features[i], "importance": float(importance[i])}
                for i in indices
            ]

            # Create visualization if we have enough features
            if len(features) >= 3:
                fig, ax = plt.subplots(figsize=(10, 6))

                top_n = min(10, len(features))
                top_indices = indices[:top_n]
                top_importance = importance[top_indices]
                top_features = [features[i] for i in top_indices]

                ax.bar(range(top_n), top_importance)
                ax.set_xticks(range(top_n))
                ax.set_xticklabels(top_features, rotation=45, ha='right')
                ax.set_title(f'Feature Importance for {model_name} Correction Model')
                ax.set_ylabel('Importance')

                plt.tight_layout()
                fig.savefig(self.output_dir / f'correction_model_{model_name}_importance.png', dpi=300)
                plt.close(fig)
        else:
            feature_importance = []

        return {
            "model": model,
            "features": features,
            "performance": {
                "original_rmse": float(original_rmse),
                "corrected_rmse": float(corrected_rmse),
                "original_mae": float(original_mae),
                "corrected_mae": float(corrected_mae),
                "rmse_improvement": float(rmse_improvement),
                "mae_improvement": float(mae_improvement),
                "improvement": float(avg_improvement),
                "test_size": len(y_test)
            },
            "feature_importance": feature_importance
        }

    def save_incremental_model(self, rf_model, lstm_model=None, evaluation_results=None):
        """
        Save models trained incrementally on daily data.

        Args:
            rf_model: Random Forest model to save
            lstm_model: LSTM model to save (optional)
            evaluation_results: Evaluation metrics (optional)

        Returns:
            Dictionary with paths to saved artifacts
        """
        from utils.model_manager import get_model_manager

        print("===== SAVING INCREMENTAL MODEL =====")

        # Get the model manager
        model_manager = get_model_manager()

        # Create version string based on date, database type, and training type (incremental)
        version = model_manager.get_version_path(self.db_type, train_type='incremental')

        # Resolve the base path
        save_dir = model_manager.get_model_path('hybrid') / version

        print(f"Saving incremental model to: {save_dir}")

        # Prepare orchestrator state (simplified for incremental training)
        orchestrator_state = {
            'preprocessing_params': self.feature_orchestrator.preprocessing_params,
            'embedding_dim': self.feature_orchestrator.embedding_dim,
            'sequence_length': 5,  # Default value for incremental training
            'target_info': {
                'column': 'final_position',
                'type': 'regression'
            }
        }

        # Prepare model configuration
        model_config = {
            'version': version,
            'model_name': 'hybrid',
            'db_type': self.db_type,
            'train_type': 'incremental',  # Explicitly mark as incremental training
            'training_date': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S'),
            'base_model': model_manager.get_latest_base_model(),  # Reference to base model
            'models_trained': {
                'rf': rf_model is not None,
                'lstm': lstm_model is not None
            },
            'evaluation_results': evaluation_results or {}
        }

        # Save all artifacts at once
        saved_paths = model_manager.save_model_artifacts(
            base_path=save_dir,
            rf_model=rf_model,
            lstm_model=lstm_model,
            orchestrator_state=orchestrator_state,
            model_config=model_config,
            db_type=self.db_type,
            train_type='incremental',
            update_config=True  # Update config.yaml with reference to this incremental model
        )

        print(f"Incremental model saved successfully to {save_dir}")
        return saved_paths

    def update_base_model(self, training_data: pd.DataFrame, blend_weight: float = 0.7) -> Dict[str, Any]:
        """
        Update the base prediction model with new training data.

        Args:
            training_data: DataFrame with fresh training data
            blend_weight: Weight for blending old and new predictions

        Returns:
            Dictionary with update results
        """
        if len(training_data) == 0:
            return {"status": "error", "message": "No training data provided"}

        self.logger.info(f"Updating base model with {len(training_data)} samples")

        # Prepare data with orchestrator
        try:
            # We need to convert the prediction data to a format that the orchestrator can process
            # Step 1: Format data for the orchestrator
            self.logger.info("Preparing data for feature embedding")

            # Make a copy we can modify
            processed_df = training_data.copy()

            # Add a comp column if it doesn't exist (orchestrator expects this)
            if 'comp' not in processed_df.columns and 'race_id' in processed_df.columns:
                processed_df['comp'] = processed_df['race_id']

            # Use actual_position as final_position (target column)
            if 'actual_position' in processed_df.columns and 'final_position' not in processed_df.columns:
                processed_df['final_position'] = processed_df['actual_position']

            # Add required columns that might be missing for embedding
            for required_col in ['idche', 'idJockey', 'numero', 'typec']:
                if required_col not in processed_df.columns:
                    if required_col == 'idche' and 'horse_id' in processed_df.columns:
                        processed_df['idche'] = processed_df['horse_id']
                    elif required_col == 'numero' and 'horse_number' in processed_df.columns:
                        processed_df['numero'] = processed_df['horse_number']
                    elif required_col == 'typec' and 'race_type' in processed_df.columns:
                        processed_df['typec'] = processed_df['race_type']
                    else:
                        self.logger.warning(f"Required column {required_col} not found, using default values")
                        processed_df[required_col] = 0  # Default value

            # Step 2: Apply full preprocessing pipeline
            self.logger.info("Applying feature embedding and preprocessing")

            # Apply feature engineering
            processed_df = self.orchestrator.prepare_features(processed_df)

            # Apply embeddings with full cleaning
            processed_df = self.orchestrator.apply_embeddings(
                processed_df,
                clean_after_embedding=True,
                keep_identifiers=False
            )

            # Step 3: Now prepare for training using the fully processed data
            self.logger.info("Preparing training dataset from embedded features")
            X, y = self.orchestrator.prepare_training_dataset(processed_df)

            # Make sure all features are numeric
            # Check for any non-numeric columns
            non_numeric_cols = [col for col in X.columns if not pd.api.types.is_numeric_dtype(X[col])]
            if non_numeric_cols:
                self.logger.warning(f"Found {len(non_numeric_cols)} non-numeric columns: {non_numeric_cols}")
                # Drop non-numeric columns
                X = X.drop(columns=non_numeric_cols)
                self.logger.info(f"Dropped non-numeric columns, {X.shape[1]} features remaining")

            # Split into train/validation/test
            X_train, X_val, X_test, y_train, y_val, y_test = self.orchestrator.split_dataset(
                X, y, test_size=0.2, val_size=0.1
            )

            self.logger.info(f"Training set: {len(X_train)} samples with {X_train.shape[1]} features")
            self.logger.info(f"Validation set: {len(X_val)} samples")
            self.logger.info(f"Test set: {len(X_test)} samples")

            # Evaluate original model on test data
            if self.rf_model is not None and hasattr(self.rf_model, 'predict'):
                try:
                    # Get predictions from original model
                    orig_preds = self.rf_model.predict(X_test)
                    orig_rmse = np.sqrt(mean_squared_error(y_test, orig_preds))
                    orig_mae = mean_absolute_error(y_test, orig_preds)

                    self.logger.info(f"Original model performance - RMSE: {orig_rmse:.4f}, MAE: {orig_mae:.4f}")
                except Exception as e:
                    self.logger.error(f"Error evaluating original model: {str(e)}")
                    orig_rmse, orig_mae = float('inf'), float('inf')
            else:
                self.logger.warning("Original model not available for comparison")
                orig_rmse, orig_mae = float('inf'), float('inf')

            # Create and train a new model or update existing
            from sklearn.ensemble import RandomForestRegressor

            # First try to upgrade existing model if it's a CalibratedRegressor
            if isinstance(self.rf_model, CalibratedRegressor):
                self.logger.info("Updating existing CalibratedRegressor")

                # Retrain with new data
                self.rf_model.fit(X_train, y_train, X_val, y_val)

                # Evaluate updated model
                updated_preds = self.rf_model.predict(X_test)
                updated_rmse = np.sqrt(mean_squared_error(y_test, updated_preds))
                updated_mae = mean_absolute_error(y_test, updated_preds)

                updated_model = self.rf_model
            else:
                # Create a new model
                self.logger.info("Creating new CalibratedRegressor")

                # Create base model
                base_rf = RandomForestRegressor(
                    n_estimators=100,
                    max_depth=None,
                    min_samples_leaf=2,
                    random_state=42
                )

                # Create and fit calibrated model
                updated_model = CalibratedRegressor(
                    base_regressor=base_rf,
                    clip_min=1.0  # Min position is 1
                )

                updated_model.fit(X_train, y_train, X_val, y_val)

                # Evaluate updated model
                updated_preds = updated_model.predict(X_test)
                updated_rmse = np.sqrt(mean_squared_error(y_test, updated_preds))
                updated_mae = mean_absolute_error(y_test, updated_preds)

            self.logger.info(f"Updated model performance - RMSE: {updated_rmse:.4f}, MAE: {updated_mae:.4f}")

            # Calculate improvement
            if orig_rmse != float('inf'):
                rmse_improve = ((orig_rmse - updated_rmse) / orig_rmse) * 100
                mae_improve = ((orig_mae - updated_mae) / orig_mae) * 100
                avg_improve = (rmse_improve + mae_improve) / 2

                self.logger.info(f"Model improvement - RMSE: {rmse_improve:.2f}%, MAE: {mae_improve:.2f}%")
            else:
                rmse_improve, mae_improve, avg_improve = None, None, None

            # Save the updated model (with a new version)
            new_version = f"v{datetime.now().strftime('%Y%m%d')}"
            new_version_path = self.model_path / new_version
            new_version_path.mkdir(exist_ok=True)

            # Save model
            model_file = new_version_path / "hybrid_rf_model.joblib"
            joblib.dump(updated_model, model_file)

            # Copy feature configuration
            if self.feature_config_path.exists():
                joblib.dump(self.feature_config, new_version_path / "hybrid_feature_engineer.joblib")

            # Create updated model config
            model_config = {
                "version": new_version,
                "parent_version": self.model_config.get('version', 'unknown'),
                "training_date": datetime.now().isoformat(),
                "training_samples": len(X_train),
                "model_type": "CalibratedRegressor",
                "performance": {
                    "original_rmse": float(orig_rmse) if orig_rmse != float('inf') else None,
                    "updated_rmse": float(updated_rmse),
                    "original_mae": float(orig_mae) if orig_mae != float('inf') else None,
                    "updated_mae": float(updated_mae),
                    "improvement": float(avg_improve) if avg_improve is not None else None
                }
            }

            # Save model config
            with open(new_version_path / "model_config.json", 'w') as f:
                json.dump(model_config, f, indent=4)

            # Create logs directory
            logs_dir = new_version_path / "logs"
            logs_dir.mkdir(exist_ok=True)

            self.logger.info(f"Updated model saved as version {new_version}")

            return {
                "status": "success",
                "new_version": new_version,
                "model_path": str(new_version_path),
                "improvement": {
                    "rmse": float(rmse_improve) if rmse_improve is not None else None,
                    "mae": float(mae_improve) if mae_improve is not None else None,
                    "average": float(avg_improve) if avg_improve is not None else None
                },
                "performance": {
                    "original_rmse": float(orig_rmse) if orig_rmse != float('inf') else None,
                    "updated_rmse": float(updated_rmse),
                    "original_mae": float(orig_mae) if orig_mae != float('inf') else None,
                    "updated_mae": float(updated_mae)
                }
            }

        except Exception as e:
            self.logger.error(f"Error updating base model: {str(e)}")
            import traceback
            self.logger.error(traceback.format_exc())

            return {
                "status": "error",
                "message": str(e)
            }

    def create_combined_model(self, blend_weight: float = 0.5) -> Dict[str, Any]:
        """
        Create a combined model that incorporates both the base model and correction models.

        Args:
            blend_weight: Weight for blending base and correction predictions

        Returns:
            Dictionary with combined model information
        """
        if not self.bias_models:
            return {"status": "error", "message": "No correction models available"}

        self.logger.info(f"Creating combined model with blend weight {blend_weight}")

        # Define the combined model structure

        # Create the combined model
        combined_model = CombinedModel(
            base_model=self.rf_model,
            correction_models=self.bias_models,
            blend_weight=blend_weight
        )

        # Save the combined model
        model_path = self.output_dir / 'combined_model.joblib'
        joblib.dump(combined_model, model_path)

        # Save metadata
        metadata = {
            "model_type": "CombinedModel",
            "base_model_version": self.model_config.get('version', 'unknown'),
            "correction_models": list(self.bias_models.keys()),
            "blend_weight": blend_weight,
            "created_at": datetime.now().isoformat()
        }

        with open(self.output_dir / 'combined_model_metadata.json', 'w') as f:
            json.dump(metadata, f, indent=4)

        self.logger.info(f"Created and saved combined model with {len(self.bias_models)} correction models")

        return {
            "status": "success",
            "model_path": str(model_path),
            "metadata": metadata
        }

    def run_regression_pipeline(self, date_from: str = None, date_to: str = None,
                                limit: int = None, update_model: bool = True,
                                create_combined: bool = True) -> Dict[str, Any]:
        """
        Run the full regression analysis and enhancement pipeline.

        Args:
            date_from: Start date for collecting data (YYYY-MM-DD)
            date_to: End date for collecting data (YYYY-MM-DD)
            limit: Maximum number of races to process
            update_model: Whether to update the base model
            create_combined: Whether to create a combined model

        Returns:
            Dictionary with pipeline results
        """
        start_time = datetime.now()
        self.logger.info(f"Starting regression enhancement pipeline for dates: {date_from} to {date_to}")

        # 1. Collect prediction data
        prediction_df = self.collect_prediction_data(date_from, date_to, limit)

        if len(prediction_df) == 0:
            return {
                "status": "error",
                "message": "No prediction data found for the specified dates"
            }

        # 2. Analyze prediction gaps
        analysis_results = self.analyze_prediction_gaps(prediction_df)

        # 3. Build correction models
        correction_results = self.build_correction_models(prediction_df)

        # 4. Prepare training data for model update
        if update_model:
            # Use the prediction data directly as training data
            # Don't try to transform it again - it's already in the right format
            update_results = self.update_base_model(prediction_df)
        else:
            update_results = {"status": "skipped", "message": "Model update was disabled"}

        # 5. Create combined model if requested
        if create_combined and correction_results.get("status") == "success":
            combined_results = self.create_combined_model(blend_weight=0.5)
        else:
            combined_results = {"status": "skipped", "message": "Combined model creation was disabled"}

        # Calculate execution time
        execution_time = (datetime.now() - start_time).total_seconds()

        # Compile pipeline results
        pipeline_results = {
            "status": "success",
            "execution_time": execution_time,
            "data_processed": {
                "races": prediction_df['race_id'].nunique(),
                "predictions": len(prediction_df)
            },
            "analysis_results": analysis_results,
            "correction_models": correction_results,
            "model_update": update_results,
            "combined_model": combined_results
        }

        # Save pipeline results summary
        with open(self.output_dir / 'regression_pipeline_results.json', 'w') as f:
            json.dump(pipeline_results, f, indent=4, default=str)

        self.logger.info(f"Regression enhancement pipeline completed in {execution_time:.2f} seconds")

        return pipeline_results


def main():
    """Command-line interface for regression enhancement."""
    parser = argparse.ArgumentParser(description='Regression analysis and model enhancement')
    parser.add_argument('--model', required=True, help='Path to model directory')
    parser.add_argument('--db', help='Database name from config (defaults to active_db)')
    parser.add_argument('--output', help='Output directory for analysis files')
    parser.add_argument('--from-date', help='Start date for analysis (YYYY-MM-DD)')
    parser.add_argument('--to-date', help='End date for analysis (YYYY-MM-DD)')
    parser.add_argument('--limit', type=int, help='Maximum number of races to analyze')
    parser.add_argument('--skip-update', action='store_true', help='Skip model update')
    parser.add_argument('--skip-combined', action='store_true', help='Skip combined model creation')
    parser.add_argument('--verbose', action='store_true', help='Enable verbose output')

    args = parser.parse_args()

    # Set default dates if not provided
    if not args.from_date:
        # Default to 30 days ago
        from_date = (datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d')
    else:
        from_date = args.from_date

    if not args.to_date:
        # Default to today
        to_date = datetime.now().strftime('%Y-%m-%d')
    else:
        to_date = args.to_date

    # Create enhancer
    enhancer = RegressionEnhancer(
        model_path=args.model,
        db_name=args.db,
        output_dir=args.output,
        verbose=args.verbose
    )

    # Run pipeline
    results = enhancer.run_regression_pipeline(
        date_from=from_date,
        date_to=to_date,
        limit=args.limit,
        update_model=not args.skip_update,
        create_combined=not args.skip_combined
    )

    # Print summary results
    if results["status"] == "success":
        print("\nRegression Enhancement Pipeline Results:")
        print(
            f"Processed {results['data_processed']['races']} races with {results['data_processed']['predictions']} predictions")

        if 'metrics' in results['analysis_results']:
            print("\nPrediction Analysis:")
            metrics = results['analysis_results']['metrics']
            print(f"  Mean Error: {metrics['mean_error']:.4f}")
            print(f"  RMSE: {metrics['rmse']:.4f}")
            print(f"  MAE: {metrics['mean_abs_error']:.4f}")

        if results['correction_models']['status'] == 'success':
            print("\nCorrection Models:")
            print(f"  Models built: {results['correction_models']['models_built']}")
            if 'improvement_summary' in results['correction_models']:
                global_imp = results['correction_models']['improvement_summary']['global']
                print(f"  Global model improvement: {global_imp:.2f}%")

        if results['model_update']['status'] == 'success':
            print("\nModel Update:")
            print(f"  New version: {results['model_update']['new_version']}")
            if 'improvement' in results['model_update'] and results['model_update']['improvement'][
                'average'] is not None:
                print(f"  Average improvement: {results['model_update']['improvement']['average']:.2f}%")

        if results['combined_model']['status'] == 'success':
            print("\nCombined Model:")
            print(f"  Model saved to: {results['combined_model']['model_path']}")

        print(f"\nExecution time: {results['execution_time']:.2f} seconds")
        print(f"All analysis files saved to: {enhancer.output_dir}")
    else:
        print(f"Error: {results.get('message', 'Unknown error')}")

    return 0


if __name__ == "__main__":
    # For debugging in IDE - uncomment and modify these lines as needed
    # import sys
    sys.argv = [sys.argv[0], "--model", "models/hybrid/v20250407" , "--to-date", "2025-04-04"]
    sys.exit(main())