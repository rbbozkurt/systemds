"""
Basic usage example for Isolation Forest.

This example demonstrates how to use the Isolation Forest algorithm
for anomaly detection on synthetic data with proper train/test split.
"""
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score
from isoforest import IsolationForest


def main():
    """Run basic Isolation Forest example with train/test split."""
    print("=" * 70)
    print("ISOLATION FOREST - Basic Usage Example")
    print("=" * 70)

    # Generate larger synthetic dataset
    np.random.seed(42)

    # Normal instances: 5000 samples clustered around origin
    print("\nGenerating synthetic data...")
    X_normal = np.random.randn(5000, 5) * 0.5
    y_normal = np.zeros(5000)

    # Anomalies: 250 samples scattered far from origin (5% contamination)
    X_anomalies = np.random.uniform(-4, 4, (250, 5))
    y_anomalies = np.ones(250)

    # Combine data
    X = np.vstack([X_normal, X_anomalies])
    y = np.hstack([y_normal, y_anomalies])

    print(f"Dataset: {X.shape[0]} samples, {X.shape[1]} features")
    print(f"Normal instances: {np.sum(y == 0)} ({np.sum(y == 0)/len(y)*100:.1f}%)")
    print(f"Anomalies: {np.sum(y == 1)} ({np.sum(y == 1)/len(y)*100:.1f}%)")

    # Split into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, shuffle=True, random_state=42
    )

    print(f"\nTrain set: {len(X_train)} samples")
    print(f"Test set: {len(X_test)} samples")

    # Train Isolation Forest
    print("\n" + "-" * 70)
    print("Training Isolation Forest...")
    print("-" * 70)
    iforest = IsolationForest(n_estimators=100, max_samples=256, random_state=42)
    iforest.fit(X_train)
    print("✓ Training complete")

    # Evaluate on test set
    print("\n" + "-" * 70)
    print("Evaluating on test set...")
    print("-" * 70)

    # Get anomaly scores
    test_scores = iforest.score_samples(X_test)

    # Get predictions
    test_predictions = iforest.predict(X_test)

    # Convert predictions: -1 (anomaly) -> 1, 1 (normal) -> 0 for sklearn metrics
    y_pred_binary = np.where(test_predictions == -1, 1, 0)

    # Calculate metrics using sklearn
    true_anomalies = np.sum(y_test == 1)
    predicted_anomalies = np.sum(test_predictions == -1)

    # Confusion matrix
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred_binary).ravel()

    # Calculate metrics
    precision = precision_score(y_test, y_pred_binary, zero_division=0)
    recall = recall_score(y_test, y_pred_binary, zero_division=0)
    f1 = f1_score(y_test, y_pred_binary, zero_division=0)

    print(f"\nResults:")
    print(f"  True anomalies in test set: {true_anomalies}")
    print(f"  Predicted anomalies: {predicted_anomalies}")
    print(f"\nConfusion Matrix:")
    print(f"  True Positives:  {tp:4d}  (anomalies correctly detected)")
    print(f"  False Positives: {fp:4d}  (normal flagged as anomaly)")
    print(f"  False Negatives: {fn:4d}  (anomalies missed)")
    print(f"  True Negatives:  {tn:4d}  (normal correctly identified)")
    print(f"\nMetrics:")
    print(f"  Precision: {precision:.3f}")
    print(f"  Recall:    {recall:.3f}")
    print(f"  F1-Score:  {f1:.3f}")

    # Score interpretation
    print("\n" + "-" * 70)
    print("Score Interpretation:")
    print("-" * 70)
    print("Anomaly scores range from 0.0 to 1.0:")
    print("  • Scores close to 1.0: Strong anomalies")
    print("  • Scores close to 0.5: Ambiguous instances")
    print("  • Scores close to 0.0: Normal instances")

    # Show top anomalies
    print(f"\nTop 5 anomaly scores in test set: {np.sort(test_scores)[-5:][::-1]}")

    # Show sample predictions
    print("\n" + "-" * 70)
    print("Sample Predictions (first 5 from test set):")
    print("-" * 70)
    for i in range(min(5, len(X_test))):
        actual = "Anomaly" if y_test[i] == 1 else "Normal"
        predicted = "Anomaly" if test_predictions[i] == -1 else "Normal"
        match = "✓" if actual == predicted else "✗"
        print(f"{match} Actual: {actual:7s} | Predicted: {predicted:7s} | Score: {test_scores[i]:.4f}")

    print("\n" + "=" * 70)
    print("Example complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()

