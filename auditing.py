import pandas as pd
import numpy as np
import pickle
import json
from datetime import datetime
from pathlib import Path

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, IsolationForest
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (classification_report, confusion_matrix, roc_auc_score, 
                            precision_recall_curve, f1_score, accuracy_score)
from sklearn.utils import resample
import warnings
warnings.filterwarnings('ignore')

class FraudDetectionTrainer:
    
    def __init__(self, output_dir='trained_models'):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        self.models = {}
        self.scalers = {}
        self.encoders = {}
        self.feature_names = []
        self.training_history = []
        
    def load_data(self, filepath, target_column='is_fraud'):
        print(f"Loading data from {filepath}...")
        
        if filepath.endswith('.csv'):
            self.df = pd.read_csv(filepath)
        elif filepath.endswith('.xlsx'):
            self.df = pd.read_excel(filepath)
        else:
            raise ValueError("Unsupported file format. Use CSV or XLSX")
        
        print(f"Dataset loaded: {len(self.df)} records, {len(self.df.columns)} features")
        
        if target_column not in self.df.columns:
            fraud_cols = [col for col in self.df.columns 
                         if any(term in col.lower() for term in ['fraud', 'label', 'target', 'class'])]
            if fraud_cols:
                target_column = fraud_cols[0]
                print(f"Using detected fraud column: {target_column}")
            else:
                raise ValueError(f"Target column '{target_column}' not found")
        
        self.target_column = target_column
        
        fraud_count = self.df[target_column].sum()
        fraud_rate = (fraud_count / len(self.df)) * 100
        print(f"Fraud cases: {fraud_count} ({fraud_rate:.2f}%)")
        
        return self.df
    
    def preprocess_data(self, handle_imbalance='smote', test_size=0.2):
        print("\nPreprocessing data...")
        
        df = self.df.copy()
        
        y = df[self.target_column]
        X = df.drop(columns=[self.target_column])
        
        print("Engineering features...")
        
        for col in X.columns:
            if X[col].dtype == 'object':
                try:
                    X[col] = pd.to_datetime(X[col])
                    X[f'{col}_hour'] = X[col].dt.hour
                    X[f'{col}_day_of_week'] = X[col].dt.dayofweek
                    X[f'{col}_is_weekend'] = (X[col].dt.dayofweek >= 5).astype(int)
                    X[f'{col}_day'] = X[col].dt.day
                    X[f'{col}_month'] = X[col].dt.month
                    X = X.drop(columns=[col])
                    print(f"  - Extracted time features from {col}")
                except:
                    pass
        
        categorical_cols = X.select_dtypes(include=['object', 'category']).columns
        for col in categorical_cols:
            if X[col].nunique() < 50:
                le = LabelEncoder()
                X[f'{col}_encoded'] = le.fit_transform(X[col].astype(str))
                self.encoders[col] = le
                print(f"  - Encoded categorical feature: {col}")
            X = X.drop(columns=[col])
        
        numerical_cols = X.select_dtypes(include=[np.number]).columns
        for col in numerical_cols:
            if X[col].isnull().sum() > 0:
                X[col].fillna(X[col].median(), inplace=True)
        
        amount_cols = [col for col in numerical_cols if 'amount' in col.lower()]
        for col in amount_cols:
            X[f'{col}_log'] = np.log1p(X[col])
            X[f'{col}_sqrt'] = np.sqrt(X[col])
            X[f'{col}_squared'] = X[col] ** 2
            print(f"  - Created transformation features for {col}")
        
        if len(numerical_cols) >= 2:
            for i, col1 in enumerate(numerical_cols[:3]):
                for col2 in numerical_cols[i+1:4]:
                    X[f'{col1}_x_{col2}'] = X[col1] * X[col2]
        
        self.feature_names = X.columns.tolist()
        print(f"Total features after engineering: {len(self.feature_names)}")
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )
        
        if handle_imbalance == 'oversample':
            print("Applying oversampling to balance classes...")
            X_train_majority = X_train[y_train == 0]
            X_train_minority = X_train[y_train == 1]
            y_train_majority = y_train[y_train == 0]
            y_train_minority = y_train[y_train == 1]
            
            X_train_minority_upsampled, y_train_minority_upsampled = resample(
                X_train_minority, y_train_minority,
                replace=True,
                n_samples=len(X_train_majority),
                random_state=42
            )
            
            X_train = pd.concat([X_train_majority, X_train_minority_upsampled])
            y_train = pd.concat([y_train_majority, y_train_minority_upsampled])
        
        elif handle_imbalance == 'undersample':
            print("Applying undersampling to balance classes...")
            X_train_majority = X_train[y_train == 0]
            X_train_minority = X_train[y_train == 1]
            y_train_majority = y_train[y_train == 0]
            y_train_minority = y_train[y_train == 1]
            
            X_train_majority_downsampled, y_train_majority_downsampled = resample(
                X_train_majority, y_train_majority,
                replace=False,
                n_samples=len(X_train_minority),
                random_state=42
            )
            
            X_train = pd.concat([X_train_majority_downsampled, X_train_minority])
            y_train = pd.concat([y_train_majority_downsampled, y_train_minority])
        
        print("Scaling features...")
        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        X_train_scaled = pd.DataFrame(X_train_scaled, columns=self.feature_names)
        X_test_scaled = pd.DataFrame(X_test_scaled, columns=self.feature_names)
        
        self.X_train = X_train_scaled
        self.X_test = X_test_scaled
        self.y_train = y_train.reset_index(drop=True)
        self.y_test = y_test.reset_index(drop=True)
        
        print(f"Training set: {len(X_train_scaled)} samples")
        print(f"Test set: {len(X_test_scaled)} samples")
        print(f"Training fraud rate: {(y_train.sum() / len(y_train) * 100):.2f}%")
        
        return X_train_scaled, X_test_scaled, y_train, y_test
    
    def train_random_forest(self, optimize=True):
        print("\n" + "="*50)
        print("Training Random Forest Classifier...")
        print("="*50)
        
        if optimize:
            print("Performing hyperparameter optimization...")
            param_grid = {
                'n_estimators': [100, 200, 300],
                'max_depth': [10, 20, 30, None],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4],
                'class_weight': ['balanced', 'balanced_subsample']
            }
            
            rf = RandomForestClassifier(random_state=42)
            grid_search = GridSearchCV(
                rf, param_grid, cv=3, scoring='f1', n_jobs=-1, verbose=1
            )
            grid_search.fit(self.X_train, self.y_train)
            
            best_model = grid_search.best_estimator_
            print(f"Best parameters: {grid_search.best_params_}")
        else:
            best_model = RandomForestClassifier(
                n_estimators=200,
                max_depth=20,
                min_samples_split=5,
                min_samples_leaf=2,
                class_weight='balanced',
                random_state=42,
                n_jobs=-1
            )
            best_model.fit(self.X_train, self.y_train)
        
        metrics = self._evaluate_model(best_model, 'Random Forest')
        self.models['random_forest'] = best_model
        
        return best_model, metrics
    
    def train_gradient_boosting(self, optimize=True):
        print("\n" + "="*50)
        print("Training Gradient Boosting Classifier...")
        print("="*50)
        
        if optimize:
            print("Performing hyperparameter optimization...")
            param_grid = {
                'n_estimators': [100, 200],
                'learning_rate': [0.01, 0.1, 0.2],
                'max_depth': [3, 5, 7],
                'min_samples_split': [2, 5],
                'subsample': [0.8, 1.0]
            }
            
            gb = GradientBoostingClassifier(random_state=42)
            grid_search = GridSearchCV(
                gb, param_grid, cv=3, scoring='f1', n_jobs=-1, verbose=1
            )
            grid_search.fit(self.X_train, self.y_train)
            
            best_model = grid_search.best_estimator_
            print(f"Best parameters: {grid_search.best_params_}")
        else:
            best_model = GradientBoostingClassifier(
                n_estimators=200,
                learning_rate=0.1,
                max_depth=5,
                min_samples_split=5,
                subsample=0.8,
                random_state=42
            )
            best_model.fit(self.X_train, self.y_train)
        
        metrics = self._evaluate_model(best_model, 'Gradient Boosting')
        self.models['gradient_boosting'] = best_model
        
        return best_model, metrics
    
    def train_logistic_regression(self):
        print("\n" + "="*50)
        print("Training Logistic Regression...")
        print("="*50)
        
        model = LogisticRegression(
            class_weight='balanced',
            max_iter=1000,
            random_state=42
        )
        model.fit(self.X_train, self.y_train)
        
        metrics = self._evaluate_model(model, 'Logistic Regression')
        self.models['logistic_regression'] = model
        
        return model, metrics
    
    def train_neural_network(self):
        print("\n" + "="*50)
        print("Training Neural Network...")
        print("="*50)
        
        model = MLPClassifier(
            hidden_layer_sizes=(100, 50, 25),
            activation='relu',
            solver='adam',
            max_iter=500,
            random_state=42,
            early_stopping=True,
            validation_fraction=0.1
        )
        model.fit(self.X_train, self.y_train)
        
        metrics = self._evaluate_model(model, 'Neural Network')
        self.models['neural_network'] = model
        
        return model, metrics
    
    def train_isolation_forest(self, contamination=0.05):
        print("\n" + "="*50)
        print("Training Isolation Forest (Unsupervised)...")
        print("="*50)
        
        model = IsolationForest(
            contamination=contamination,
            n_estimators=100,
            random_state=42,
            n_jobs=-1
        )
        model.fit(self.X_train)
        
        y_pred = model.predict(self.X_test)
        y_pred = np.where(y_pred == -1, 1, 0)
        
        metrics = {
            'accuracy': accuracy_score(self.y_test, y_pred),
            'precision': precision_recall_curve(self.y_test, y_pred)[0][0],
            'recall': precision_recall_curve(self.y_test, y_pred)[1][0],
            'f1': f1_score(self.y_test, y_pred)
        }
        
        print(f"Accuracy: {metrics['accuracy']:.4f}")
        print(f"F1 Score: {metrics['f1']:.4f}")
        
        self.models['isolation_forest'] = model
        
        return model, metrics
    
    def _evaluate_model(self, model, model_name):
        y_pred = model.predict(self.X_test)
        y_pred_proba = model.predict_proba(self.X_test)[:, 1] if hasattr(model, 'predict_proba') else None
        
        accuracy = accuracy_score(self.y_test, y_pred)
        f1 = f1_score(self.y_test, y_pred)
        
        print(f"\n{model_name} Results:")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"F1 Score: {f1:.4f}")
        
        if y_pred_proba is not None:
            roc_auc = roc_auc_score(self.y_test, y_pred_proba)
            print(f"ROC AUC: {roc_auc:.4f}")
        
        print("\nClassification Report:")
        print(classification_report(self.y_test, y_pred))
        
        print("\nConfusion Matrix:")
        print(confusion_matrix(self.y_test, y_pred))
        
        if hasattr(model, 'feature_importances_'):
            feature_importance = dict(zip(self.feature_names, model.feature_importances_))
            top_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)[:10]
            print("\nTop 10 Important Features:")
            for feat, imp in top_features:
                print(f"  {feat}: {imp:.4f}")
        
        metrics = {
            'model_name': model_name,
            'accuracy': accuracy,
            'f1_score': f1,
            'roc_auc': roc_auc if y_pred_proba is not None else None,
            'classification_report': classification_report(self.y_test, y_pred, output_dict=True),
            'confusion_matrix': confusion_matrix(self.y_test, y_pred).tolist(),
            'feature_importance': feature_importance if hasattr(model, 'feature_importances_') else None
        }
        
        self.training_history.append(metrics)
        
        return metrics
    
    def train_all_models(self, optimize_ensemble=False):
        print("\n" + "="*60)
        print("TRAINING ALL MODELS")
        print("="*60)
        
        all_metrics = {}
        
        _, metrics = self.train_random_forest(optimize=optimize_ensemble)
        all_metrics['random_forest'] = metrics
        
        _, metrics = self.train_gradient_boosting(optimize=optimize_ensemble)
        all_metrics['gradient_boosting'] = metrics
        
        _, metrics = self.train_logistic_regression()
        all_metrics['logistic_regression'] = metrics
        
        _, metrics = self.train_neural_network()
        all_metrics['neural_network'] = metrics
        
        _, metrics = self.train_isolation_forest()
        all_metrics['isolation_forest'] = metrics
        
        print("\n" + "="*60)
        print("MODEL COMPARISON")
        print("="*60)
        
        comparison = []
        for name, metrics in all_metrics.items():
            comparison.append({
                'Model': name,
                'Accuracy': f"{metrics['accuracy']:.4f}",
                'F1 Score': f"{metrics['f1_score']:.4f}",
                'ROC AUC': f"{metrics['roc_auc']:.4f}" if metrics['roc_auc'] else 'N/A'
            })
        
        comparison_df = pd.DataFrame(comparison)
        print(comparison_df.to_string(index=False))
        
        best_model_name = max(all_metrics.items(), key=lambda x: x[1]['f1_score'])[0]
        print(f"\nBest Model: {best_model_name}")
        
        return all_metrics
    
    def save_models(self, save_metrics=True):
        print("\n" + "="*60)
        print("SAVING MODELS AND ARTIFACTS")
        print("="*60)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        for name, model in self.models.items():
            model_path = self.output_dir / f'{name}_{timestamp}.pkl'
            with open(model_path, 'wb') as f:
                pickle.dump(model, f)
            print(f"Saved {name} to {model_path}")
        
        scaler_path = self.output_dir / f'scaler_{timestamp}.pkl'
        with open(scaler_path, 'wb') as f:
            pickle.dump(self.scaler, f)
        print(f"Saved scaler to {scaler_path}")
        
        if self.encoders:
            encoders_path = self.output_dir / f'encoders_{timestamp}.pkl'
            with open(encoders_path, 'wb') as f:
                pickle.dump(self.encoders, f)
            print(f"Saved encoders to {encoders_path}")
        
        features_path = self.output_dir / f'feature_names_{timestamp}.json'
        with open(features_path, 'w') as f:
            json.dump(self.feature_names, f)
        print(f"Saved feature names to {features_path}")
        
        if save_metrics:
            metrics_path = self.output_dir / f'training_metrics_{timestamp}.json'
            with open(metrics_path, 'w') as f:
                json.dump(self.training_history, f, indent=2)
            print(f"Saved training metrics to {metrics_path}")
        
        config = {
            'timestamp': timestamp,
            'feature_names': self.feature_names,
            'target_column': self.target_column,
            'models': list(self.models.keys()),
            'scaler_file': f'scaler_{timestamp}.pkl',
            'encoders_file': f'encoders_{timestamp}.pkl' if self.encoders else None
        }
        
        config_path = self.output_dir / f'deployment_config_{timestamp}.json'
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
        print(f"Saved deployment config to {config_path}")
        
        print("\nAll models and artifacts saved successfully!")
        
        return timestamp
    
    def create_ensemble_predictor(self):
        print("\nCreating ensemble predictor...")
        
        class EnsemblePredictor:
            def __init__(self, models, weights=None):
                self.models = models
                self.weights = weights if weights else [1/len(models)] * len(models)
            
            def predict_proba(self, X):
                predictions = []
                for model in self.models.values():
                    if hasattr(model, 'predict_proba'):
                        pred = model.predict_proba(X)[:, 1]
                    else:
                        pred = model.decision_function(X)
                    predictions.append(pred)
                
                ensemble_pred = np.average(predictions, axis=0, weights=self.weights)
                return ensemble_pred
            
            def predict(self, X, threshold=0.5):
                proba = self.predict_proba(X)
                return (proba >= threshold).astype(int)
        
        ensemble = EnsemblePredictor(self.models)
        
        y_pred_proba = ensemble.predict_proba(self.X_test)
        y_pred = ensemble.predict(self.X_test)
        
        print("Ensemble Model Results:")
        print(f"Accuracy: {accuracy_score(self.y_test, y_pred):.4f}")
        print(f"F1 Score: {f1_score(self.y_test, y_pred):.4f}")
        print(f"ROC AUC: {roc_auc_score(self.y_test, y_pred_proba):.4f}")
        
        return ensemble


if __name__ == "__main__":
    trainer = FraudDetectionTrainer(output_dir='trained_models')
    
    df = trainer.load_data('your_fraud_data.csv', target_column='is_fraud')
    
    X_train, X_test, y_train, y_test = trainer.preprocess_data(
        handle_imbalance='oversample',
        test_size=0.2
    )
    
    all_metrics = trainer.train_all_models(optimize_ensemble=True)
    
    ensemble = trainer.create_ensemble_predictor()
    
    timestamp = trainer.save_models(save_metrics=True)
    
    print(f"\n✅ Training completed! Models saved with timestamp: {timestamp}")
    print(f"📁 Output directory: {trainer.output_dir}")