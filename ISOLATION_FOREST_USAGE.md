# Isolation Forest Usage Guide

## DML Usage

### Training a Model

```dml
# Load your data
X = read("data.csv")

# Train an isolation forest model
# Parameters:
#   - n_trees: Number of trees in the forest (e.g., 100)
#   - subsampling_size: Sample size for each tree (e.g., 256)
#   - seed: Random seed for reproducibility (optional, default: -1)
model = outlierByIsolationForest(X=X, n_trees=100, subsampling_size=256, seed=42)

# The model is a List containing:
#   - 'model': The trained forest (Matrix[Double])
#   - 'subsampling_size': The subsampling size used (Double)
```

### Detecting Outliers

```dml
# Apply the model to detect outliers
# Can be used on the same data or new data
anomaly_scores = outlierByIsolationForestApply(iForestModel=model, X=X)

# Scores range from 0 to 1
# Scores > 0.5 typically indicate outliers
# Higher scores = more likely to be an outlier

# Example: Filter outliers
threshold = 0.5
is_outlier = anomaly_scores > threshold
outliers = removeEmpty(target=X, margin="rows", select=is_outlier)
```

## Python Usage

### Training a Model

```python
from systemds.context import SystemDSContext
from systemds.operator.algorithm import outlierByIsolationForest, outlierByIsolationForestApply
import numpy as np

# Create SystemDS context
sds = SystemDSContext()

# Prepare your data
data = np.random.randn(1000, 10)  # 1000 samples, 10 features
# Add some outliers
data[0:10] = data[0:10] + 5

# Convert to SystemDS matrix
X = sds.from_numpy(data)

# Train the model
model = outlierByIsolationForest(X, n_trees=100, subsampling_size=256, seed=42)

# Detect outliers
scores = outlierByIsolationForestApply(model, X)

# Compute and retrieve results
anomaly_scores = scores.compute()

# Find outliers (scores > 0.5)
outlier_indices = np.where(anomaly_scores > 0.5)[0]
print(f"Found {len(outlier_indices)} outliers")
```

## Parameters Guide

### outlierByIsolationForest

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| X | Matrix[Double] | Yes | - | Numerical feature matrix to train on |
| n_trees | Integer | Yes | - | Number of isolation trees to build (recommended: 100-200) |
| subsampling_size | Integer | Yes | - | Size of subsample for each tree (recommended: 256 or min(256, nrow(X))) |
| seed | Integer | No | -1 | Random seed for reproducibility. -1 = random |

### outlierByIsolationForestApply

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| iForestModel | List[Unknown] | Yes | Trained model from outlierByIsolationForest |
| X | Matrix[Double] | Yes | Samples to calculate anomaly scores for |

**Returns:** Matrix[Double] - Column vector of anomaly scores (one per row in X)

## Interpreting Anomaly Scores

- **Score = 1.0**: Definitely an outlier
- **Score > 0.5**: Likely an outlier (recommended threshold)
- **Score ≈ 0.5**: Normal point (near the expected average path length)
- **Score < 0.5**: Normal point
- **Score → 0**: Definitely normal

The score is based on how quickly the algorithm can "isolate" a point:
- Outliers are easier to isolate (shorter paths) → higher scores
- Normal points require more splits (longer paths) → lower scores

## Best Practices

1. **Subsampling Size**: 
   - Use 256 for most cases (as per original paper)
   - For small datasets: min(256, nrow(X))
   - Must be > 1 and ≤ nrow(X)

2. **Number of Trees**:
   - 100 trees is usually sufficient
   - More trees = more stable results but slower
   - 50-200 is a reasonable range

3. **Seed**:
   - Set a specific seed for reproducible results
   - Use -1 for different results each run

4. **Data Requirements**:
   - Works with numerical features only
   - No need to normalize (algorithm is invariant to scaling)
   - Handles high-dimensional data well

## Example: Complete Workflow

```dml
# 1. Load and prepare data
X = read("sensor_data.csv")
print("Training on " + nrow(X) + " samples with " + ncol(X) + " features")

# 2. Train model
n_samples = nrow(X)
subsample_size = min(256, n_samples)
model = outlierByIsolationForest(
    X=X, 
    n_trees=100, 
    subsampling_size=subsample_size, 
    seed=42
)

# 3. Detect outliers
scores = outlierByIsolationForestApply(iForestModel=model, X=X)

# 4. Analyze results
threshold = 0.6  # Adjust based on your needs
outliers = scores > threshold
n_outliers = sum(outliers)
print("Found " + n_outliers + " outliers (" + (n_outliers/n_samples*100) + "%)")

# 5. Extract outliers and normal points
outlier_data = removeEmpty(target=X, margin="rows", select=outliers)
normal_data = removeEmpty(target=X, margin="rows", select=!outliers)

# 6. Save results
write(scores, "anomaly_scores.csv")
write(outlier_data, "outliers.csv")
write(normal_data, "normal_data.csv")
```

## References

Liu, F. T., Ting, K. M., & Zhou, Z. H. (2008, December). 
Isolation forest. 
In 2008 eighth ieee international conference on data mining (pp. 413-422). 
IEEE.
