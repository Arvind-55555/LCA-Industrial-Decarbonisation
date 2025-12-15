# Indian Industrial Decarbonisation Platform - Deployment Guide

## Overview

This guide covers deployment of the Indian Industrial Decarbonisation LCA Platform with ML model serving capabilities.

## ML Model Training for Indian Data

### Training Indian Models

Train ML models on Indian industrial data:

```bash
# Train all sectors
python scripts/train_indian_models.py --sectors steel cement aluminium

# Train specific states
python scripts/train_indian_models.py --sectors steel --states Maharashtra Gujarat Odisha

# With transfer learning from global models
python scripts/train_indian_models.py --sectors steel --transfer-from models/trained/pinn_model.pt
```

### Data Preparation

1. **Download Indian Data**:
   ```bash
   # Data will be automatically downloaded/generated from:
   # - data.gov.in (industrial emissions)
   # - CEA (grid carbon intensity)
   # - CPCB (emissions data)
   ```

2. **Data Location**:
   - Indian data: `data/raw/indian/`
   - Trained models: `models/trained/indian/`

## ML Model Serving

### Option 1: TorchServe

Deploy models using TorchServe:

```bash
# Start TorchServe
torchserve --start --model-store /models/indian --models indian_steel_pinn=indian_steel_pinn.pt

# Test inference
curl http://localhost:8080/predictions/indian_steel_pinn -T input.json
```

### Option 2: FastAPI Custom Serving

Deploy via FastAPI with Indian-specific endpoints:

```bash
# Start API server
python run_api.py --indian-mode

# Endpoints:
# POST /indian/lca/steel - Indian steel LCA
# POST /indian/lca/cement - Indian cement LCA
# POST /indian/lca/aluminium - Indian aluminium LCA
```

### Option 3: Kubernetes Deployment

Deploy using Kubernetes:

```bash
# Apply ML model serving configuration
kubectl apply -f deployment/ml_model_serving.yaml

# Check deployment
kubectl get pods -n lca-optimizer
kubectl get services -n lca-optimizer
```

## Indian-Specific Configuration

### Environment Variables

```bash
# Indian data settings
INDIAN_DATA_DIR=data/raw/indian
DEFAULT_STATE=Maharashtra
ENABLE_STATE_SPECIFIC_CI=true

# ML model settings
ENABLE_TRANSFER_LEARNING=true
INDIAN_DATA_AUGMENTATION=true
PHYSICS_REGULARIZATION_WEIGHT=0.15

# Indian policy constraints
ENABLE_POLICY_CONSTRAINTS=true
PERFORM_ACHIEVEMENT_TRAJECTORY=true
```

### State-Specific Settings

Configure state-specific parameters in `lca_optimizer/config/indian_settings.py`:

- Grid carbon intensity baselines
- Renewable share
- Industrial policy strictness
- Power reliability
- Material availability

## Model Versioning and A/B Testing

### Model Versioning

```python
from lca_optimizer.models.indian_pinn import IndianPhysicsInformedNN

# Load specific model version
model_v1 = IndianPhysicsInformedNN.load("models/trained/indian/steel_pinn_v1.pt")
model_v2 = IndianPhysicsInformedNN.load("models/trained/indian/steel_pinn_v2.pt")

# A/B testing
if use_v2:
    predictions = model_v2.predict(inputs)
else:
    predictions = model_v1.predict(inputs)
```

## Performance Monitoring

### Drift Detection

Monitor model performance on Indian operational data:

```python
from lca_optimizer.monitoring.metrics import detect_drift

# Check for data drift
drift_score = detect_drift(
    reference_data=historical_data,
    current_data=recent_data,
    model=indian_pinn_model
)

if drift_score > 0.3:
    logger.warning("Data drift detected - retraining recommended")
```

### Model Performance Metrics

- **RMSE**: Root Mean Squared Error on Indian validation data
- **Physics Constraint Violation**: % of predictions violating Indian constraints
- **Regional Accuracy**: State-specific prediction accuracy
- **Transfer Learning Effectiveness**: Improvement over baseline global model

## Indian Data Sources

### Primary Sources

1. **data.gov.in**: Industrial emissions, energy consumption
2. **CEA (Central Electricity Authority)**: Grid carbon intensity by state
3. **CPCB (Central Pollution Control Board)**: Industrial emissions data
4. **Ministry of Steel/Cement**: Sector-specific process data

### Data Format

Indian data is stored in:
- `data/raw/indian/industrial_emissions.csv`
- `data/raw/indian/grid_ci_{state}.csv`
- `data/raw/indian/{sector}_process_data.csv`

## Deployment Checklist

- [ ] Train Indian ML models on available data
- [ ] Configure state-specific settings
- [ ] Set up model serving infrastructure
- [ ] Configure monitoring and drift detection
- [ ] Test API endpoints with Indian data
- [ ] Validate physics constraints on Indian processes
- [ ] Set up model versioning
- [ ] Configure A/B testing framework

## Troubleshooting

### Limited Indian Data

If Indian data is limited:
1. Enable data augmentation: `INDIAN_DATA_AUGMENTATION=true`
2. Use transfer learning: `ENABLE_TRANSFER_LEARNING=true`
3. Increase physics regularization: `PHYSICS_REGULARIZATION_WEIGHT=0.2`

### Model Performance Issues

1. Check state-specific configurations
2. Validate physics constraints
3. Review data quality and completeness
4. Consider retraining with more data

## Next Steps

1. **Collect More Indian Data**: Integrate additional data sources
2. **Fine-tune Models**: Retrain with operational data
3. **Expand Sectors**: Add chemicals and refining sectors
4. **Regional Calibration**: Fine-tune state-specific models

