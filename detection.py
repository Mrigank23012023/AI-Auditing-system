import pandas as pd
import numpy as np
import pickle
import json
from pathlib import Path

class FraudDetector:
    
    def __init__(self, model_path, scaler_path, feature_names_path=None):
        print("Loading fraud detection model...")
        
        with open(model_path, 'rb') as f:
            self.model = pickle.load(f)
        print(f"✓ Model loaded from {model_path}")
        
        with open(scaler_path, 'rb') as f:
            self.scaler = pickle.load(f)
        print(f"✓ Scaler loaded from {scaler_path}")
        
        if feature_names_path:
            if feature_names_path.endswith('.json'):
                with open(feature_names_path, 'r') as f:
                    self.feature_names = json.load(f)
            else:
                with open(feature_names_path, 'r') as f:
                    self.feature_names = [line.strip() for line in f.readlines()]
            print(f"✓ Feature names loaded: {len(self.feature_names)} features")
        else:
            self.feature_names = None
        
        print("✓ Fraud detector ready!\n")
    
    def preprocess_transaction(self, transaction_data):
        if isinstance(transaction_data, dict):
            df = pd.DataFrame([transaction_data])
        else:
            df = transaction_data.copy()
        
        numerical_cols = df.select_dtypes(include=[np.number]).columns
        X = df[numerical_cols]
        
        X = X.fillna(X.median())
        
        if self.feature_names:
            missing_features = set(self.feature_names) - set(X.columns)
            if missing_features:
                for feat in missing_features:
                    X[feat] = 0
            
            X = X[self.feature_names]
        
        return X
    
    def predict(self, transaction_data, threshold=0.5):
        X = self.preprocess_transaction(transaction_data)
        X_scaled = self.scaler.transform(X)
        predictions = self.model.predict(X_scaled)
        return predictions
    
    def predict_proba(self, transaction_data):
        X = self.preprocess_transaction(transaction_data)
        X_scaled = self.scaler.transform(X)
        probabilities = self.model.predict_proba(X_scaled)[:, 1]
        return probabilities
    
    def analyze_transaction(self, transaction_data, detailed=True):
        fraud_prob = self.predict_proba(transaction_data)[0]
        is_fraud = self.predict(transaction_data)[0]
        
        if fraud_prob < 0.3:
            risk_level = "LOW"
        elif fraud_prob < 0.7:
            risk_level = "MEDIUM"
        else:
            risk_level = "HIGH"
        
        result = {
            'is_fraud': bool(is_fraud),
            'fraud_probability': float(fraud_prob),
            'risk_level': risk_level,
            'recommendation': self._get_recommendation(fraud_prob)
        }
        
        if detailed and hasattr(self.model, 'feature_importances_'):
            X = self.preprocess_transaction(transaction_data)
            X_scaled = self.scaler.transform(X)
            
            feature_contributions = {}
            for i, feat in enumerate(self.feature_names):
                importance = self.model.feature_importances_[i]
                value = X_scaled[0, i]
                contribution = importance * abs(value)
                feature_contributions[feat] = float(contribution)
            
            top_factors = sorted(feature_contributions.items(), 
                               key=lambda x: x[1], reverse=True)[:5]
            result['top_risk_factors'] = [
                {'feature': feat, 'contribution': contrib} 
                for feat, contrib in top_factors
            ]
        
        return result
    
    def _get_recommendation(self, fraud_prob):
        if fraud_prob < 0.3:
            return "APPROVE - Low risk transaction"
        elif fraud_prob < 0.5:
            return "REVIEW - Medium risk, manual review suggested"
        elif fraud_prob < 0.8:
            return "CHALLENGE - High risk, additional verification required"
        else:
            return "BLOCK - Very high risk, decline transaction"
    
    def batch_predict(self, data_path, output_path=None):
        print(f"Processing batch file: {data_path}")
        
        df = pd.read_csv(data_path)
        print(f"Loaded {len(df):,} transactions")
        
        print("Making predictions...")
        df['fraud_probability'] = self.predict_proba(df)
        df['is_fraud_predicted'] = self.predict(df)
        
        def get_risk_level(prob):
            if prob < 0.3:
                return "LOW"
            elif prob < 0.7:
                return "MEDIUM"
            else:
                return "HIGH"
        
        df['risk_level'] = df['fraud_probability'].apply(get_risk_level)
        
        fraud_count = df['is_fraud_predicted'].sum()
        fraud_rate = (fraud_count / len(df)) * 100
        
        print(f"\n✓ Predictions complete!")
        print(f"  Fraudulent transactions: {fraud_count:,} ({fraud_rate:.2f}%)")
        print(f"  High risk: {len(df[df['risk_level']=='HIGH']):,}")
        print(f"  Medium risk: {len(df[df['risk_level']=='MEDIUM']):,}")
        print(f"  Low risk: {len(df[df['risk_level']=='LOW']):,}")
        
        if output_path:
            df.to_csv(output_path, index=False)
            print(f"\n✓ Results saved to {output_path}")
        
        return df


if __name__ == "__main__":
    
    detector = FraudDetector(
        model_path='fraud_detection_model.pkl',
        scaler_path='fraud_detection_scaler.pkl',
        feature_names_path='feature_names.txt'
    )
    
    print("="*60)
    print("EXAMPLE 1: Single Transaction Analysis")
    print("="*60)
    
    transaction = {
        'amount': 1500.00,
        'hour': 23,
        'day_of_week': 6,
        'merchant_id': 12345,
    }
    
    result = detector.analyze_transaction(transaction, detailed=True)
    
    print(f"\nTransaction Analysis:")
    print(f"  Fraud: {result['is_fraud']}")
    print(f"  Probability: {result['fraud_probability']:.2%}")
    print(f"  Risk Level: {result['risk_level']}")
    print(f"  Recommendation: {result['recommendation']}")
    
    if 'top_risk_factors' in result:
        print(f"\n  Top Risk Factors:")
        for factor in result['top_risk_factors']:
            print(f"    - {factor['feature']}: {factor['contribution']:.4f}")
    
    print("\n" + "="*60)
    print("EXAMPLE 2: Batch Processing")
    print("="*60 + "\n")
    
    print("\n" + "="*60)
    print("EXAMPLE 3: Real-time Monitoring Simulation")
    print("="*60)
    
    print("\nProcessing incoming transactions...")
    
    test_transactions = [
        {'amount': 50.00, 'hour': 14, 'day_of_week': 2},
        {'amount': 5000.00, 'hour': 3, 'day_of_week': 6},
        {'amount': 120.00, 'hour': 10, 'day_of_week': 1},
        {'amount': 8500.00, 'hour': 1, 'day_of_week': 0},
    ]
    
    for i, txn in enumerate(test_transactions, 1):
        prob = detector.predict_proba(txn)[0]
        is_fraud = detector.predict(txn)[0]
        
        status = "🚨 FRAUD" if is_fraud else "✓ OK"
        print(f"\nTransaction #{i}: {status}")
        print(f"  Amount: ${txn['amount']:.2f}")
        print(f"  Fraud Probability: {prob:.2%}")
    
    print("\n" + "="*60)
    print("✅ INFERENCE EXAMPLES COMPLETED")
    print("="*60)