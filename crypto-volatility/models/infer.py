#!/usr/bin/env python3
"""
Inference Script
Real-time inference for volatility spike detection.
"""

import json
import logging
import os
import pickle
import sys
import time
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd
import yaml
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class VolatilityPredictor:
    """Predictor for volatility spikes."""
    
    def __init__(self, model_path: str, scaler_path: Optional[str] = None, model_type: str = "ml"):
        """Initialize predictor with saved model."""
        self.model_type = model_type
        
        # Load model
        with open(model_path, 'rb') as f:
            self.model = pickle.load(f)
        
        # Load scaler if provided
        self.scaler = None
        if scaler_path and os.path.exists(scaler_path):
            with open(scaler_path, 'rb') as f:
                self.scaler = pickle.load(f)
        
        logger.info(f"Loaded {model_type} model from {model_path}")
    
    def predict(self, features: pd.DataFrame) -> np.ndarray:
        """Predict volatility spikes."""
        # Select feature columns
        feature_cols = [
            'price', 'midprice', 'return_1s', 'return_5s', 'return_30s', 'return_60s',
            'volatility', 'trade_intensity', 'spread_abs', 'spread_rel', 'order_book_imbalance'
        ]
        
        # Filter to only columns that exist
        feature_cols = [col for col in feature_cols if col in features.columns]
        X = features[feature_cols].copy()
        
        # Handle NaN values
        X = X.fillna(0)
        
        # Scale if scaler available
        if self.scaler is not None:
            X_scaled = self.scaler.transform(X)
        else:
            X_scaled = X.values
        
        # Predict
        predictions = self.model.predict(X_scaled)
        
        return predictions
    
    def predict_proba(self, features: pd.DataFrame) -> np.ndarray:
        """Predict probabilities of volatility spikes."""
        # Select feature columns
        feature_cols = [
            'price', 'midprice', 'return_1s', 'return_5s', 'return_30s', 'return_60s',
            'volatility', 'trade_intensity', 'spread_abs', 'spread_rel', 'order_book_imbalance'
        ]
        
        # Filter to only columns that exist
        feature_cols = [col for col in feature_cols if col in features.columns]
        X = features[feature_cols].copy()
        
        # Handle NaN values
        X = X.fillna(0)
        
        # Scale if scaler available
        if self.scaler is not None:
            X_scaled = self.scaler.transform(X)
        else:
            X_scaled = X.values
        
        # Predict probabilities
        if hasattr(self.model, 'predict_proba'):
            probabilities = self.model.predict_proba(X_scaled)
        else:
            # For baseline model, use predict_proba if available
            probabilities = self.model.predict_proba(X)
        
        return probabilities


def benchmark_inference(predictor: VolatilityPredictor, test_features: pd.DataFrame, n_iterations: int = 100):
    """Benchmark inference speed."""
    logger.info(f"Benchmarking inference on {len(test_features)} samples...")
    
    # Warmup
    _ = predictor.predict(test_features.head(10))
    
    # Benchmark
    start_time = time.time()
    for _ in range(n_iterations):
        _ = predictor.predict(test_features)
    end_time = time.time()
    
    total_time = end_time - start_time
    avg_time = total_time / n_iterations
    time_per_sample = avg_time / len(test_features)
    
    logger.info(f"Benchmark results:")
    logger.info(f"  Total time: {total_time:.4f}s for {n_iterations} iterations")
    logger.info(f"  Average time per batch: {avg_time:.4f}s")
    logger.info(f"  Time per sample: {time_per_sample*1000:.4f}ms")
    logger.info(f"  Throughput: {len(test_features) / avg_time:.2f} samples/second")
    
    # Check if meets requirement (< 2x real-time)
    window_size = 300  # 5 minutes = 300 seconds
    real_time_per_sample = window_size / len(test_features)  # seconds per sample
    speedup = real_time_per_sample / time_per_sample
    
    logger.info(f"  Real-time window: {real_time_per_sample:.4f}s per sample")
    logger.info(f"  Speedup: {speedup:.2f}x")
    
    if speedup >= 2.0:
        logger.info("✓ Inference meets requirement (< 2x real-time)")
    else:
        logger.warning(f"⚠ Inference is {speedup:.2f}x slower than required (need < 2x)")
    
    return {
        'avg_time_per_batch': avg_time,
        'time_per_sample': time_per_sample,
        'throughput': len(test_features) / avg_time,
        'speedup': speedup
    }


def main():
    """Main inference function."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Run inference for volatility spike detection')
    parser.add_argument(
        '--model-path',
        type=str,
        default='models/artifacts/xgboost_model.pkl',
        help='Path to model file'
    )
    parser.add_argument(
        '--scaler-path',
        type=str,
        default='models/artifacts/xgboost_scaler.pkl',
        help='Path to scaler file (optional)'
    )
    parser.add_argument(
        '--model-type',
        type=str,
        default='ml',
        choices=['ml', 'baseline'],
        help='Model type'
    )
    parser.add_argument(
        '--features-file',
        type=str,
        default=None,
        help='Path to features file for inference (default: test set)'
    )
    parser.add_argument(
        '--benchmark',
        action='store_true',
        help='Run inference benchmark'
    )
    parser.add_argument(
        '--config',
        type=str,
        default='config.yaml',
        help='Path to config file'
    )
    
    args = parser.parse_args()
    config_path = os.getenv('CONFIG_PATH', args.config)
    
    try:
        # Load config
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        # Load features
        if args.features_file:
            features_path = Path(args.features_file)
        else:
            # Use test set
            features_path = Path(config['features']['data_dir']) / "test_set.parquet"
        
        if not features_path.exists():
            logger.error(f"Features file not found: {features_path}")
            sys.exit(1)
        
        logger.info(f"Loading features from {features_path}")
        df = pd.read_parquet(features_path)
        logger.info(f"Loaded {len(df)} samples")
        
        # Initialize predictor
        model_path = Path(args.model_path)
        scaler_path = Path(args.scaler_path) if args.scaler_path else None
        
        if not model_path.exists():
            logger.error(f"Model file not found: {model_path}")
            sys.exit(1)
        
        # Determine model name from path
        model_name = model_path.stem.replace('_model', '').upper()
        logger.info(f"Using {model_name} model for inference")
        
        predictor = VolatilityPredictor(
            str(model_path),
            str(scaler_path) if scaler_path and scaler_path.exists() else None,
            args.model_type
        )
        
        # Prepare features
        feature_cols = [
            'price', 'midprice', 'return_1s', 'return_5s', 'return_30s', 'return_60s',
            'volatility', 'trade_intensity', 'spread_abs', 'spread_rel', 'order_book_imbalance'
        ]
        feature_cols = [col for col in feature_cols if col in df.columns]
        features = df[feature_cols].copy()
        
        # Run inference
        logger.info("Running inference...")
        probabilities = predictor.predict_proba(features)
        prob_values = probabilities[:, 1] if probabilities.shape[1] > 1 else probabilities.flatten()
        
        # Try to load optimal threshold from MLflow, otherwise use percentile-based threshold
        optimal_threshold = None
        try:
            import mlflow
            mlflow.set_tracking_uri(config['mlflow']['tracking_uri'])
            experiment = mlflow.get_experiment_by_name(config['mlflow']['experiment_name'])
            if experiment:
                # Get the latest run for this model type
                runs = mlflow.search_runs(
                    experiment_ids=[experiment.experiment_id],
                    filter_string=f"tags.mlflow.runName = '{model_name.lower()}_model'",
                    order_by=["start_time DESC"],
                    max_results=1
                )
                if not runs.empty:
                    run_id = runs.iloc[0]['run_id']
                    run = mlflow.get_run(run_id)
                    if 'val_best_threshold' in run.data.metrics:
                        optimal_threshold = run.data.metrics['val_best_threshold']
                        logger.info(f"Loaded optimal threshold from MLflow: {optimal_threshold:.4f}")
        except Exception as e:
            logger.debug(f"Could not load threshold from MLflow: {e}")
        
        # Use optimal threshold if available, otherwise use 95th percentile
        if optimal_threshold is not None:
            threshold = optimal_threshold
            threshold_method = "optimal (from MLflow)"
        else:
            # Calculate 95th percentile threshold as fallback
            threshold = np.percentile(prob_values, 95)
            threshold_method = "95th percentile"
            logger.warning(f"Using {threshold_method} threshold: {threshold:.4f} (consider training with threshold tuning)")
        
        logger.info(f"Using {threshold_method} threshold: {threshold:.4f}")
        logger.info(f"Probability statistics: min={prob_values.min():.4f}, max={prob_values.max():.4f}, "
                   f"mean={prob_values.mean():.4f}, median={np.median(prob_values):.4f}")
        
        # Use threshold to make binary predictions
        predictions = (prob_values >= threshold).astype(int)
        
        # Add predictions to dataframe
        df['prediction'] = predictions
        df['probability'] = prob_values
        df['threshold'] = threshold
        df['threshold_method'] = threshold_method
        
        # Save results
        output_path = Path(config['features']['data_dir']) / "predictions.parquet"
        df.to_parquet(output_path, index=False)
        logger.info(f"Saved predictions to {output_path}")
        
        # Print summary
        spike_count = predictions.sum()
        logger.info(f"Predicted {spike_count} volatility spikes ({spike_count/len(predictions)*100:.2f}%)")
        
        # Calculate metrics if labels are available
        if 'label' in df.columns:
            from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix, classification_report
            
            y_true = df['label']
            y_pred = df['prediction']
            
            accuracy = (y_pred == y_true).mean()
            precision = precision_score(y_true, y_pred, zero_division=0)
            recall = recall_score(y_true, y_pred, zero_division=0)
            f1 = f1_score(y_true, y_pred, zero_division=0)
            
            cm = confusion_matrix(y_true, y_pred)
            
            logger.info("\n" + "="*60)
            logger.info(f"PERFORMANCE METRICS ({threshold_method} threshold)")
            logger.info("="*60)
            logger.info(f"Accuracy:  {accuracy:.4f}")
            logger.info(f"Precision: {precision:.4f}")
            logger.info(f"Recall:    {recall:.4f}")
            logger.info(f"F1 Score:  {f1:.4f}")
            logger.info(f"\nConfusion Matrix:")
            logger.info(f"  True Negatives:  {cm[0,0]}")
            logger.info(f"  False Positives: {cm[0,1]}")
            logger.info(f"  False Negatives: {cm[1,0]}")
            logger.info(f"  True Positives:  {cm[1,1]}")
            logger.info(f"\nActual spike rate: {y_true.mean()*100:.2f}%")
            logger.info(f"Predicted spike rate: {y_pred.mean()*100:.2f}%")
            logger.info("="*60)
        
        # Benchmark if requested
        if args.benchmark:
            benchmark_inference(predictor, features)
        
        logger.info("Inference complete!")
        
    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()

