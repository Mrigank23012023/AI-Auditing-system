# FraudScope - Advanced Fraud Detection System

A comprehensive machine learning-based fraud detection system with multiple detection algorithms, interactive visualization, and real-time analysis capabilities.

## Features

### Detection Methods
- **Unsupervised Learning**: Isolation Forest and DBSCAN clustering for anomaly detection
- **Supervised Learning**: Random Forest, Gradient Boosting, Logistic Regression, and Neural Networks
- **Risk-Based Analysis**: Intelligent risk scoring based on transaction patterns
- **Ensemble Methods**: Combine multiple models for improved accuracy

### Key Capabilities
- Real-time transaction analysis
- Batch processing for large datasets
- Automatic feature engineering
- Interactive web dashboard
- Comprehensive reporting and exports
- Model training and deployment

## Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Setup

1. Clone the repository
```bash
git clone <repository-url>
cd fraud-detection-system
```

2. Install dependencies
```bash
pip install -r requirements.txt
```

## Usage

### 1. Training Models (auditing.py)

Train fraud detection models on your dataset:

```python
from auditing import FraudDetectionTrainer

trainer = FraudDetectionTrainer(output_dir='trained_models')

df = trainer.load_data('your_fraud_data.csv', target_column='is_fraud')

X_train, X_test, y_train, y_test = trainer.preprocess_data(
    handle_imbalance='oversample',
    test_size=0.2
)

all_metrics = trainer.train_all_models(optimize_ensemble=True)

ensemble = trainer.create_ensemble_predictor()

timestamp = trainer.save_models(save_metrics=True)
```

**Features:**
- Automatic feature engineering (time-based, amount transformations, interactions)
- Class imbalance handling (oversampling, undersampling, SMOTE)
- Hyperparameter optimization
- Multiple model training and comparison
- Model serialization for deployment

### 2. Making Predictions (detection.py)

Use trained models to detect fraud in new transactions:

```python
from detection import FraudDetector

detector = FraudDetector(
    model_path='trained_models/random_forest_20241223_120000.pkl',
    scaler_path='trained_models/scaler_20241223_120000.pkl',
    feature_names_path='trained_models/feature_names_20241223_120000.json'
)

transaction = {
    'amount': 1500.00,
    'hour': 23,
    'day_of_week': 6,
    'merchant_id': 12345,
}

result = detector.analyze_transaction(transaction, detailed=True)
print(f"Fraud Probability: {result['fraud_probability']:.2%}")
print(f"Risk Level: {result['risk_level']}")
print(f"Recommendation: {result['recommendation']}")

results_df = detector.batch_predict('transactions.csv', 'results.csv')
```

**Features:**
- Single transaction analysis
- Batch processing
- Risk level classification (LOW/MEDIUM/HIGH)
- Top risk factors identification
- Actionable recommendations

### 3. Interactive Dashboard (fraud.py)

Launch the web-based fraud detection interface:

```bash
streamlit run fraud.py
```

**Dashboard Features:**
- Upload CSV/Excel datasets
- Choose detection methods (Unsupervised/Supervised/Hybrid)
- Real-time fraud detection
- Interactive visualizations
- Statistical analysis and pattern discovery
- Export results and reports
- Synthetic data generation for testing

## Project Structure

```
fraud-detection-system/
├── auditing.py          # Model training and evaluation
├── detection.py         # Inference and prediction
├── fraud.py            # Streamlit web dashboard
├── requirements.txt    # Python dependencies
├── README.md          # Documentation
└── trained_models/    # Saved models and artifacts (created on first run)
```

## Data Format

### Training Data
Your dataset should include:
- **Required**: Numerical features (transaction amounts, counts, etc.)
- **Recommended**: Timestamps, user IDs, merchant information
- **For Supervised**: Binary fraud label column (0/1, True/False)

### Example Dataset Structure
```csv
transaction_id,amount,timestamp,user_id,merchant_id,is_fraud
TXN001,150.50,2024-01-15 14:30:00,USER123,MERCH456,0
TXN002,5000.00,2024-01-15 23:45:00,USER789,MERCH012,1
```

## Models and Algorithms

### Supervised Models
- **Random Forest**: Ensemble of decision trees with feature importance
- **Gradient Boosting**: Sequential boosting for high accuracy
- **Logistic Regression**: Linear model with probability outputs
- **Neural Network**: Multi-layer perceptron for complex patterns

### Unsupervised Models
- **Isolation Forest**: Anomaly detection through isolation
- **DBSCAN**: Density-based clustering for outlier detection

### Ensemble
- Weighted combination of multiple models for improved performance

## Performance Metrics

The system provides comprehensive evaluation metrics:
- Accuracy
- Precision, Recall, F1-Score
- ROC AUC Score
- Confusion Matrix
- Feature Importance
- Classification Report

## Configuration

### Training Parameters
- `handle_imbalance`: 'oversample', 'undersample', or 'smote'
- `test_size`: Proportion of data for testing (default: 0.2)
- `optimize_ensemble`: Enable hyperparameter tuning (slower but better)

### Detection Parameters
- `contamination`: Expected fraud rate for unsupervised methods (0.01-0.2)
- `threshold`: Classification threshold (default: 0.5)
- `min_confidence`: Minimum confidence for alerts (0.5-0.95)

## Output Files

### Training Outputs (in `trained_models/`)
- `{model_name}_{timestamp}.pkl`: Trained model
- `scaler_{timestamp}.pkl`: Feature scaler
- `encoders_{timestamp}.pkl`: Categorical encoders
- `feature_names_{timestamp}.json`: Feature list
- `training_metrics_{timestamp}.json`: Performance metrics
- `deployment_config_{timestamp}.json`: Deployment configuration

## Best Practices

1. **Data Quality**: Ensure clean, complete data for best results
2. **Feature Engineering**: Let the system automatically engineer features
3. **Multiple Methods**: Use hybrid approach to compare results
4. **Validation**: Always review flagged transactions manually
5. **Threshold Tuning**: Adjust based on your risk tolerance
6. **Regular Retraining**: Update models with new fraud patterns

## Troubleshooting

### Common Issues

**ImportError: No module named 'sklearn'**
```bash
pip install scikit-learn
```

**Too Many False Positives**
- Lower the contamination rate
- Adjust classification threshold
- Review feature engineering

**Poor Model Performance**
- Check data quality and completeness
- Ensure sufficient training data
- Enable feature engineering
- Try ensemble methods

**Streamlit Connection Error**
```bash
streamlit run fraud.py --server.port 8501
```

## Advanced Usage

### Custom Feature Engineering
```python
trainer.preprocess_data(handle_imbalance='oversample')
```

### Model Selection
```python
best_model, metrics = trainer.train_random_forest(optimize=True)
```

### Batch Processing
```python
results_df = detector.batch_predict('large_dataset.csv', 'output.csv')
```

## License

This project is provided as-is for educational and commercial use.

## Support

For issues, questions, or contributions, please open an issue in the repository.

## Acknowledgments

Built with:
- scikit-learn for machine learning
- Streamlit for web interface
- Plotly for interactive visualizations
- pandas for data manipulation
