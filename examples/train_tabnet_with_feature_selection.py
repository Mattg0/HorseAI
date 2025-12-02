#!/usr/bin/env python3
"""
Example: Train TabNet with Automatic Feature Selection

This example shows how to train TabNet models with automatic feature selection
that optimizes performance without affecting RF models.

Usage:
    python examples/train_tabnet_with_feature_selection.py --model-type general
    python examples/train_tabnet_with_feature_selection.py --model-type quinte
"""

import argparse
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from model_training.tabnet.tabnet_trainer_with_selection import TabNetTrainerWithSelection


def train_general_tabnet():
    """Train general TabNet model with feature selection"""

    print("\n" + "="*80)
    print("TRAINING GENERAL TABNET MODEL WITH AUTOMATIC FEATURE SELECTION")
    print("="*80)

    # Initialize trainer
    trainer = TabNetTrainerWithSelection(config_path='config.yaml', verbose=True)

    # Load data (all features)
    print("\n1. Loading data...")
    trainer.load_and_prepare_data(
        limit=None,  # Use all data
        race_filter=None,  # All race types
        date_filter=None  # All dates
    )

    # Train with automatic feature selection
    print("\n2. Training with 3-phase approach...")
    results = trainer.train(
        test_size=0.2,
        validation_size=0.1,
        sparse_threshold=0.7,  # Remove features with >70% zeros
        correlation_threshold=0.95,  # Remove highly correlated features
        target_features=46,  # Target ~46 features (currently optimal)
        initial_epochs=50,  # Quick training for importance
        final_epochs=200,  # Full training on selected features
        batch_size=256,
        # TabNet architecture
        n_d=64,
        n_a=64,
        n_steps=5,
        gamma=1.5,
        n_independent=2,
        n_shared=2,
        lambda_sparse=1e-4,
        lr=2e-2
    )

    # Save model
    print("\n3. Saving model...")
    model_path = trainer.save_model(model_type='general')

    print("\n" + "="*80)
    print("TRAINING COMPLETE")
    print("="*80)
    print(f"\nModel saved to: {model_path}")
    print(f"\nResults:")
    print(f"  Test MAE:  {results['test_mae']:.3f}")
    print(f"  Test RMSE: {results['test_rmse']:.3f}")
    print(f"  Test R²:   {results['test_r2']:.4f}")
    print(f"\nFeature Selection:")
    print(f"  Original: {results['original_features']} features")
    print(f"  Selected: {results['selected_features']} features")
    print(f"  Reduction: {results['feature_reduction_pct']:.1f}%")

    return trainer, model_path


def train_quinte_tabnet():
    """Train Quinte TabNet model with feature selection"""

    print("\n" + "="*80)
    print("TRAINING QUINTE TABNET MODEL WITH AUTOMATIC FEATURE SELECTION")
    print("="*80)

    # Initialize trainer
    trainer = TabNetTrainerWithSelection(config_path='config.yaml', verbose=True)

    # Load Quinte-specific data
    print("\n1. Loading Quinte+ races...")
    trainer.load_and_prepare_data(
        limit=None,
        race_filter="type_course = 'Quinté+'",  # Quinte races only
        date_filter=None
    )

    # Train with automatic feature selection
    print("\n2. Training with 3-phase approach...")
    results = trainer.train(
        test_size=0.2,
        validation_size=0.1,
        sparse_threshold=0.7,
        correlation_threshold=0.95,
        target_features=45,  # Target ~45 features
        initial_epochs=50,
        final_epochs=200,
        batch_size=256,
        # TabNet architecture
        n_d=64,
        n_a=64,
        n_steps=5,
        gamma=1.5,
        n_independent=2,
        n_shared=2,
        lambda_sparse=1e-4,
        lr=2e-2
    )

    # Save model
    print("\n3. Saving model...")
    model_path = trainer.save_model(model_type='quinte')

    print("\n" + "="*80)
    print("TRAINING COMPLETE")
    print("="*80)
    print(f"\nModel saved to: {model_path}")
    print(f"\nResults:")
    print(f"  Test MAE:  {results['test_mae']:.3f}")
    print(f"  Test RMSE: {results['test_rmse']:.3f}")
    print(f"  Test R²:   {results['test_r2']:.4f}")
    print(f"\nFeature Selection:")
    print(f"  Original: {results['original_features']} features")
    print(f"  Selected: {results['selected_features']} features")
    print(f"  Reduction: {results['feature_reduction_pct']:.1f}%")

    return trainer, model_path


def train_both_models():
    """Train both General and Quinte TabNet models"""

    print("\n" + "="*80)
    print("TRAINING ALL TABNET MODELS")
    print("="*80)

    # Train General model
    print("\n\nPART 1: GENERAL MODEL")
    general_trainer, general_path = train_general_tabnet()

    # Train Quinte model
    print("\n\nPART 2: QUINTE MODEL")
    quinte_trainer, quinte_path = train_quinte_tabnet()

    print("\n\n" + "="*80)
    print("ALL MODELS TRAINED SUCCESSFULLY")
    print("="*80)
    print(f"\nGeneral Model: {general_path}")
    print(f"Quinte Model:  {quinte_path}")

    return {
        'general': (general_trainer, general_path),
        'quinte': (quinte_trainer, quinte_path)
    }


def main():
    """Main entry point"""

    parser = argparse.ArgumentParser(
        description='Train TabNet models with automatic feature selection'
    )
    parser.add_argument(
        '--model-type',
        choices=['general', 'quinte', 'both'],
        default='both',
        help='Which model to train'
    )

    args = parser.parse_args()

    if args.model_type == 'general':
        train_general_tabnet()
    elif args.model_type == 'quinte':
        train_quinte_tabnet()
    else:
        train_both_models()


if __name__ == "__main__":
    main()
