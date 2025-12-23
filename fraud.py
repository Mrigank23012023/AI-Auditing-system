import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
from datetime import datetime, timedelta
import io
import time
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.cluster import DBSCAN
import warnings
warnings.filterwarnings('ignore')

# Set page configuration
st.set_page_config(page_title="FraudScope - Advanced Fraud Detection", layout="wide", page_icon="🔍")

# Initialize session state
if 'dataset' not in st.session_state:
    st.session_state.dataset = None
if 'processed_data' not in st.session_state:
    st.session_state.processed_data = None
if 'fraud_predictions' not in st.session_state:
    st.session_state.fraud_predictions = None
if 'trained_models' not in st.session_state:
    st.session_state.trained_models = {}
if 'feature_importance' not in st.session_state:
    st.session_state.feature_importance = None

# Helper functions
def convert_to_native_types(obj):
    """Convert NumPy data types to native Python types for Plotly compatibility"""
    if isinstance(obj, pd.DataFrame):
        df = obj.copy()
        for col in df.select_dtypes(include=[np.number]).columns:
            df[col] = df[col].astype(float)
        return df
    elif isinstance(obj, dict):
        return {k: convert_to_native_types(v) for k, v in obj.items()}
    elif isinstance(obj, (np.floating, np.integer)):
        return float(obj) if isinstance(obj, np.floating) else int(obj)
    elif isinstance(obj, (list, tuple)):
        return [convert_to_native_types(item) for item in obj]
    elif isinstance(obj, pd.Series):
        if obj.dtype in [np.float64, np.float32, np.int64, np.int32]:
            return obj.astype(float) if obj.dtype in [np.float64, np.float32] else obj.astype(int)
        return obj
    return obj

def preprocess_data(df):
    """Preprocess data for fraud detection"""
    processed_df = df.copy()
    
    # Feature engineering
    numerical_cols = processed_df.select_dtypes(include=['number']).columns.tolist()
    categorical_cols = processed_df.select_dtypes(include=['object', 'category']).columns.tolist()
    
    # Handle missing values
    for col in numerical_cols:
        processed_df[col].fillna(processed_df[col].median(), inplace=True)
    
    for col in categorical_cols:
        processed_df[col].fillna(processed_df[col].mode()[0] if not processed_df[col].mode().empty else 'Unknown', inplace=True)
    
    # Encode categorical variables
    encoders = {}
    for col in categorical_cols:
        if processed_df[col].nunique() < 50:  # Only encode if not too many unique values
            le = LabelEncoder()
            processed_df[col + '_encoded'] = le.fit_transform(processed_df[col])
            encoders[col] = le
    
    # Create time-based features if timestamp columns exist
    time_cols = []
    for col in processed_df.columns:
        if any(term in col.lower() for term in ['time', 'date', 'timestamp']):
            try:
                processed_df[col] = pd.to_datetime(processed_df[col])
                processed_df[col + '_hour'] = processed_df[col].dt.hour
                processed_df[col + '_day_of_week'] = processed_df[col].dt.dayofweek
                processed_df[col + '_is_weekend'] = (processed_df[col].dt.dayofweek >= 5).astype(int)
                time_cols.append(col)
            except:
                continue
    
    # Create amount-based features
    amount_cols = []
    for col in numerical_cols:
        if any(term in col.lower() for term in ['amount', 'price', 'value', 'transaction', 'payment']):
            amount_cols.append(col)
            # Create log transformation
            processed_df[col + '_log'] = np.log1p(processed_df[col])
            # Create z-score
            processed_df[col + '_zscore'] = (processed_df[col] - processed_df[col].mean()) / processed_df[col].std()
    
    # Create velocity features (transaction frequency)
    if time_cols and 'user_id' in [col.lower() for col in processed_df.columns]:
        user_col = next((col for col in processed_df.columns if col.lower() == 'user_id'), None)
        if user_col and time_cols:
            time_col = time_cols[0]
            processed_df = processed_df.sort_values([user_col, time_col])
            processed_df['time_since_last_transaction'] = processed_df.groupby(user_col)[time_col].diff().dt.total_seconds()
            processed_df['transactions_last_hour'] = processed_df.groupby(user_col).rolling('1H', on=time_col).size().values
    
    return processed_df, encoders

def detect_anomalies_isolation_forest(df, contamination=0.1):
    """Detect anomalies using Isolation Forest"""
    # Select numerical features for anomaly detection
    numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    if len(numerical_cols) < 2:
        return None, None
    
    # Prepare data
    X = df[numerical_cols].fillna(df[numerical_cols].median())
    
    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Train Isolation Forest
    iso_forest = IsolationForest(contamination=contamination, random_state=42, n_estimators=100)
    anomaly_labels = iso_forest.fit_predict(X_scaled)
    
    # Convert to binary (1 for normal, -1 for anomaly)
    anomaly_scores = iso_forest.decision_function(X_scaled)
    
    return anomaly_labels, anomaly_scores

def detect_anomalies_dbscan(df, eps=0.5, min_samples=5):
    """Detect anomalies using DBSCAN clustering"""
    numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    if len(numerical_cols) < 2:
        return None
    
    # Prepare data
    X = df[numerical_cols].fillna(df[numerical_cols].median())
    
    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Apply DBSCAN
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    cluster_labels = dbscan.fit_predict(X_scaled)
    
    # Points labeled as -1 are considered anomalies
    return cluster_labels

def train_supervised_model(df, target_col):
    """Train supervised fraud detection model"""
    # Prepare features
    feature_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    feature_cols = [col for col in feature_cols if col != target_col]
    
    if len(feature_cols) < 2:
        return None, None, None
    
    X = df[feature_cols].fillna(df[feature_cols].median())
    y = df[target_col]
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Train Random Forest
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
    rf_model.fit(X_train, y_train)
    
    # Make predictions
    y_pred = rf_model.predict(X_test)
    y_pred_proba = rf_model.predict_proba(X_test)[:, 1]
    
    # Calculate metrics
    metrics = {
        'accuracy': rf_model.score(X_test, y_test),
        'classification_report': classification_report(y_test, y_pred, output_dict=True),
        'confusion_matrix': confusion_matrix(y_test, y_pred),
        'roc_auc': roc_auc_score(y_test, y_pred_proba),
        'feature_importance': dict(zip(feature_cols, rf_model.feature_importances_))
    }
    
    return rf_model, metrics, feature_cols

def generate_synthetic_fraud_features(df):
    """Generate synthetic features that might indicate fraud"""
    synth_df = df.copy()
    
    # Risk score based on transaction patterns
    numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    if len(numerical_cols) > 0:
        # Create composite risk score
        risk_features = []
        
        # Amount-based risk
        amount_cols = [col for col in numerical_cols if any(term in col.lower() for term in ['amount', 'price', 'value'])]
        if amount_cols:
            amount_col = amount_cols[0]
            synth_df['amount_risk'] = (synth_df[amount_col] - synth_df[amount_col].mean()) / synth_df[amount_col].std()
            synth_df['amount_risk'] = np.abs(synth_df['amount_risk'])  # Absolute z-score
            risk_features.append('amount_risk')
        
        # Time-based risk (if time data available)
        time_cols = [col for col in df.columns if any(term in col.lower() for term in ['time', 'date', 'timestamp'])]
        if time_cols:
            try:
                time_col = time_cols[0]
                synth_df[time_col] = pd.to_datetime(synth_df[time_col])
                hour = synth_df[time_col].dt.hour
                # Higher risk for unusual hours (late night/early morning)
                synth_df['time_risk'] = np.where((hour < 6) | (hour > 22), 1, 0)
                risk_features.append('time_risk')
            except:
                pass
        
        # Velocity risk (transaction frequency)
        if len(synth_df) > 100:
            # Simple velocity: number of transactions in rolling window
            synth_df['transaction_velocity'] = range(len(synth_df))  # Simplified
            velocity_threshold = np.percentile(synth_df['transaction_velocity'], 95)
            synth_df['velocity_risk'] = (synth_df['transaction_velocity'] > velocity_threshold).astype(int)
            risk_features.append('velocity_risk')
        
        # Composite risk score
        if risk_features:
            if len(risk_features) == 1:
                synth_df['fraud_risk_score'] = synth_df[risk_features[0]]
            else:
                synth_df['fraud_risk_score'] = synth_df[risk_features].mean(axis=1)
        
        # Generate synthetic fraud labels based on risk score
        if 'fraud_risk_score' in synth_df.columns:
            fraud_threshold = np.percentile(synth_df['fraud_risk_score'], 95)  # Top 5% as potential fraud
            synth_df['predicted_fraud'] = (synth_df['fraud_risk_score'] > fraud_threshold).astype(int)
    
    return synth_df

# App Header
st.title("🔍 FraudScope - Advanced Fraud Detection System")
st.markdown("### Upload your transaction data and let AI detect fraudulent patterns")

# Sidebar
with st.sidebar:
    st.header("🔧 Configuration")
    
    # File upload
    uploaded_file = st.file_uploader("📁 Upload Dataset", type=['csv', 'xlsx'], help="Upload CSV or Excel file with transaction data")
    
    if uploaded_file is not None:
        try:
            if uploaded_file.name.endswith('.csv'):
                df = pd.read_csv(uploaded_file)
            else:
                df = pd.read_excel(uploaded_file)
            
            st.session_state.dataset = df
            st.success(f"✅ Dataset loaded: {uploaded_file.name}")
            st.metric("Records", f"{len(df):,}")
            st.metric("Features", len(df.columns))
            
        except Exception as e:
            st.error(f"❌ Error loading file: {str(e)}")
    
    # Detection options
    if st.session_state.dataset is not None:
        st.header("🎯 Detection Methods")
        
        detection_mode = st.radio(
            "Choose detection approach:",
            ["Unsupervised (No labels needed)", "Supervised (With fraud labels)", "Hybrid (Both methods)"]
        )
        
        if detection_mode in ["Unsupervised", "Hybrid"]:
            st.subheader("Unsupervised Settings")
            contamination = st.slider("Expected fraud rate (%)", 1, 20, 5) / 100
            use_isolation_forest = st.checkbox("Isolation Forest", value=True)
            use_dbscan = st.checkbox("DBSCAN Clustering", value=True)
        
        if detection_mode in ["Supervised", "Hybrid"]:
            st.subheader("Supervised Settings")
            # Try to auto-detect fraud column
            potential_fraud_cols = []
            for col in st.session_state.dataset.columns:
                if any(term in col.lower() for term in ['fraud', 'label', 'flag', 'target', 'class']):
                    potential_fraud_cols.append(col)
            
            if potential_fraud_cols:
                fraud_column = st.selectbox("Select fraud label column:", potential_fraud_cols)
            else:
                fraud_column = st.selectbox("Select fraud label column:", st.session_state.dataset.columns)
        
        # Advanced options
        with st.expander("⚙️ Advanced Options"):
            preprocess_data_flag = st.checkbox("Enable feature engineering", value=True)
            generate_risk_score = st.checkbox("Generate risk scores", value=True)
            min_confidence = st.slider("Minimum confidence threshold", 0.5, 0.95, 0.8)
        
        # Run detection
        if st.button("🚀 Start Fraud Detection", type="primary"):
            with st.spinner("🔍 Analyzing transactions for fraud patterns..."):
                
                # Preprocess data
                if preprocess_data_flag:
                    processed_df, encoders = preprocess_data(st.session_state.dataset)
                    st.session_state.processed_data = processed_df
                else:
                    st.session_state.processed_data = st.session_state.dataset.copy()
                
                results = {}
                
                # Unsupervised detection
                if detection_mode in ["Unsupervised", "Hybrid"]:
                    if use_isolation_forest:
                        anomaly_labels, anomaly_scores = detect_anomalies_isolation_forest(
                            st.session_state.processed_data, contamination=contamination
                        )
                        if anomaly_labels is not None:
                            results['isolation_forest'] = {
                                'labels': anomaly_labels,
                                'scores': anomaly_scores,
                                'fraud_count': sum(anomaly_labels == -1)
                            }
                    
                    if use_dbscan:
                        cluster_labels = detect_anomalies_dbscan(st.session_state.processed_data)
                        if cluster_labels is not None:
                            results['dbscan'] = {
                                'labels': cluster_labels,
                                'fraud_count': sum(cluster_labels == -1)
                            }
                
                # Supervised detection
                if detection_mode in ["Supervised", "Hybrid"] and fraud_column:
                    model, metrics, feature_cols = train_supervised_model(
                        st.session_state.processed_data, fraud_column
                    )
                    if model is not None:
                        results['supervised'] = {
                            'model': model,
                            'metrics': metrics,
                            'feature_cols': feature_cols
                        }
                        st.session_state.trained_models['random_forest'] = model
                        st.session_state.feature_importance = metrics['feature_importance']
                
                # Generate synthetic features and risk scores
                if generate_risk_score:
                    synth_df = generate_synthetic_fraud_features(st.session_state.processed_data)
                    st.session_state.processed_data = synth_df
                    results['synthetic'] = True
                
                st.session_state.fraud_predictions = results
                st.success("✅ Fraud detection completed!")

# Main content
if st.session_state.dataset is not None:
    
    # Dataset overview
    st.header("📊 Dataset Overview")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Transactions", f"{len(st.session_state.dataset):,}")
    with col2:
        st.metric("Features", len(st.session_state.dataset.columns))
    with col3:
        missing_pct = (st.session_state.dataset.isnull().sum().sum() / 
                      (len(st.session_state.dataset) * len(st.session_state.dataset.columns))) * 100
        st.metric("Missing Data", f"{missing_pct:.1f}%")
    with col4:
        if st.session_state.fraud_predictions:
            total_detected = 0
            for method, result in st.session_state.fraud_predictions.items():
                if 'fraud_count' in result:
                    total_detected = max(total_detected, result['fraud_count'])
            st.metric("🚨 Fraud Detected", f"{total_detected:,}", delta=f"{(total_detected/len(st.session_state.dataset)*100):.1f}%")
        else:
            st.metric("Status", "Ready for analysis")

# Display results if fraud detection has been run
if st.session_state.fraud_predictions:
    results = st.session_state.fraud_predictions
    
    # Detection Results Summary
    st.header("🎯 Fraud Detection Results")
    
    tabs = []
    if 'isolation_forest' in results:
        tabs.append("Isolation Forest")
    if 'dbscan' in results:
        tabs.append("DBSCAN")
    if 'supervised' in results:
        tabs.append("Supervised ML")
    if 'synthetic' in results:
        tabs.append("Risk Analysis")
    
    if tabs:
        tab_objects = st.tabs(tabs)
        
        tab_idx = 0
        
        # Isolation Forest Results
        if 'isolation_forest' in results:
            with tab_objects[tab_idx]:
                st.subheader("🌲 Isolation Forest Anomaly Detection")
                
                iso_results = results['isolation_forest']
                
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Anomalies Detected", f"{iso_results['fraud_count']:,}")
                    st.metric("Detection Rate", f"{(iso_results['fraud_count']/len(st.session_state.dataset)*100):.2f}%")
                
                with col2:
                    # Anomaly score distribution
                    scores_df = pd.DataFrame({'Anomaly Score': iso_results['scores']})
                    fig = px.histogram(scores_df, x='Anomaly Score', title="Anomaly Score Distribution")
                    st.plotly_chart(fig, use_container_width=True)
                
                # Show detected anomalies
                if iso_results['fraud_count'] > 0:
                    anomaly_df = st.session_state.dataset.copy()
                    anomaly_df['Anomaly'] = iso_results['labels'] == -1
                    anomaly_df['Anomaly_Score'] = iso_results['scores']
                    
                    st.subheader("🚨 Detected Anomalous Transactions")
                    anomalous_transactions = anomaly_df[anomaly_df['Anomaly']].sort_values('Anomaly_Score')
                    st.dataframe(anomalous_transactions.head(20))
            
            tab_idx += 1
        
        # DBSCAN Results
        if 'dbscan' in results:
            with tab_objects[tab_idx]:
                st.subheader("🎯 DBSCAN Cluster Analysis")
                
                dbscan_results = results['dbscan']
                
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Outliers Detected", f"{dbscan_results['fraud_count']:,}")
                    unique_clusters = len(set(dbscan_results['labels'])) - (1 if -1 in dbscan_results['labels'] else 0)
                    st.metric("Clusters Found", unique_clusters)
                
                with col2:
                    # Cluster distribution
                    cluster_counts = pd.Series(dbscan_results['labels']).value_counts().sort_index()
                    cluster_df = pd.DataFrame({'Cluster': cluster_counts.index, 'Count': cluster_counts.values})
                    cluster_df['Cluster'] = cluster_df['Cluster'].astype(str)
                    cluster_df.loc[cluster_df['Cluster'] == '-1', 'Cluster'] = 'Outliers'
                    
                    fig = px.bar(cluster_df, x='Cluster', y='Count', title="Cluster Distribution")
                    st.plotly_chart(fig, use_container_width=True)
            
            tab_idx += 1
        
        # Supervised ML Results
        if 'supervised' in results:
            with tab_objects[tab_idx]:
                st.subheader("🤖 Supervised Machine Learning Results")
                
                sup_results = results['supervised']
                metrics = sup_results['metrics']
                
                # Model performance metrics
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Model Accuracy", f"{metrics['accuracy']:.3f}")
                with col2:
                    st.metric("ROC AUC Score", f"{metrics['roc_auc']:.3f}")
                with col3:
                    precision = metrics['classification_report']['1']['precision']
                    st.metric("Precision", f"{precision:.3f}")
                
                # Feature importance
                if st.session_state.feature_importance:
                    st.subheader("📊 Feature Importance")
                    
                    # Sort features by importance
                    importance_df = pd.DataFrame({
                        'Feature': list(st.session_state.feature_importance.keys()),
                        'Importance': list(st.session_state.feature_importance.values())
                    }).sort_values('Importance', ascending=True)
                    
                    fig = px.bar(importance_df.tail(15), x='Importance', y='Feature', 
                               orientation='h', title="Top 15 Most Important Features")
                    st.plotly_chart(fig, use_container_width=True)
                
                # Confusion Matrix
                col1, col2 = st.columns(2)
                with col1:
                    st.subheader("📈 Classification Report")
                    report_df = pd.DataFrame(metrics['classification_report']).transpose()
                    st.dataframe(report_df.round(3))
                
                with col2:
                    st.subheader("🎯 Confusion Matrix")
                    cm = metrics['confusion_matrix']
                    fig, ax = plt.subplots(figsize=(6, 4))
                    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
                    ax.set_ylabel('Actual')
                    ax.set_xlabel('Predicted')
                    st.pyplot(fig)
            
            tab_idx += 1
        
        # Risk Analysis
        if 'synthetic' in results and 'fraud_risk_score' in st.session_state.processed_data.columns:
            with tab_objects[tab_idx]:
                st.subheader("⚠️ Risk Score Analysis")
                
                risk_df = st.session_state.processed_data
                
                col1, col2 = st.columns(2)
                with col1:
                    # Risk score distribution
                    fig = px.histogram(risk_df, x='fraud_risk_score', title="Risk Score Distribution")
                    st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    # High risk transactions
                    high_risk_threshold = risk_df['fraud_risk_score'].quantile(0.95)
                    high_risk_count = len(risk_df[risk_df['fraud_risk_score'] > high_risk_threshold])
                    
                    st.metric("High Risk Transactions", f"{high_risk_count:,}")
                    st.metric("Risk Threshold (95th percentile)", f"{high_risk_threshold:.3f}")
                
                # Show high-risk transactions
                st.subheader("🚨 Highest Risk Transactions")
                high_risk_transactions = risk_df.nlargest(20, 'fraud_risk_score')
                st.dataframe(high_risk_transactions)
    
    # Interactive Fraud Explorer
    st.header("🔍 Interactive Fraud Explorer")
    
    if st.session_state.processed_data is not None:
        # Feature selection for visualization
        numerical_cols = st.session_state.processed_data.select_dtypes(include=[np.number]).columns.tolist()
        
        if len(numerical_cols) >= 2:
            col1, col2 = st.columns(2)
            with col1:
                x_feature = st.selectbox("Select X-axis feature:", numerical_cols, index=0)
            with col2:
                y_feature = st.selectbox("Select Y-axis feature:", numerical_cols, index=1)
            
            # Create scatter plot with fraud detection results
            plot_df = st.session_state.processed_data[[x_feature, y_feature]].copy()
            
            # Add fraud indicators from different methods
            if 'isolation_forest' in results:
                plot_df['Isolation_Forest_Anomaly'] = results['isolation_forest']['labels'] == -1
            if 'predicted_fraud' in st.session_state.processed_data.columns:
                plot_df['Risk_Based_Fraud'] = st.session_state.processed_data['predicted_fraud']
            
            # Choose coloring method
            color_by = st.selectbox("Color points by:", 
                                  [col for col in plot_df.columns if col not in [x_feature, y_feature]])
            
            if color_by:
                fig = px.scatter(plot_df, x=x_feature, y=y_feature, color=color_by,
                               title=f"Transaction Analysis: {x_feature} vs {y_feature}")
                st.plotly_chart(fig, use_container_width=True)
    
    # Export Results
    st.header("📤 Export Results")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("📊 Export Fraud Report"):
            # Create comprehensive report
            report_lines = []
            report_lines.append("FRAUDSCOPE - FRAUD DETECTION REPORT")
            report_lines.append("=" * 50)
            report_lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            report_lines.append(f"Dataset: {len(st.session_state.dataset):,} transactions")
            report_lines.append("")
            
            # Add results from each method
            for method, result in results.items():
                if 'fraud_count' in result:
                    report_lines.append(f"{method.upper()} DETECTION:")
                    report_lines.append(f"  Fraud cases detected: {result['fraud_count']:,}")
                    report_lines.append(f"  Detection rate: {(result['fraud_count']/len(st.session_state.dataset)*100):.2f}%")
                    report_lines.append("")
            
            report_text = "\n".join(report_lines)
            
            st.download_button(
                label="📄 Download Report",
                data=report_text,
                file_name=f"fraud_detection_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                mime="text/plain"
            )
    
    with col2:
        if st.button("🔢 Export Flagged Transactions"):
            # Combine all fraud flags
            export_df = st.session_state.dataset.copy()
            
            if 'isolation_forest' in results:
                export_df['Isolation_Forest_Flag'] = results['isolation_forest']['labels'] == -1
                export_df['Anomaly_Score'] = results['isolation_forest']['scores']
            
            if 'predicted_fraud' in st.session_state.processed_data.columns:
                export_df['Risk_Based_Flag'] = st.session_state.processed_data['predicted_fraud']
                export_df['Risk_Score'] = st.session_state.processed_data['fraud_risk_score']
            
            csv_data = export_df.to_csv(index=False)
            
            st.download_button(
                label="📋 Download CSV",
                data=csv_data,
                file_name=f"flagged_transactions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )
    
    with col3:
        if st.button("🧠 Export Model"):
            if 'supervised' in results:
                st.info("Model export functionality would save the trained model for future use")
            else:
                st.warning("No trained model available to export")

else:
    # Welcome screen
    st.markdown("""
    ## 🎯 Welcome to FraudScope Advanced Fraud Detection
    
    Upload your transaction dataset to get started with AI-powered fraud detection:
    
    ### 🔍 Detection Methods Available:
    - **Unsupervised Learning**: Detect anomalies without labeled data
      - Isolation Forest for outlier detection
      - DBSCAN clustering for pattern analysis
    
    - **Supervised Learning**: Train models with labeled fraud data
      - Random Forest classifier with feature importance
      - Advanced performance metrics and validation
    
    - **Risk-Based Analysis**: Generate intelligent risk scores
      - Transaction velocity analysis
      - Amount-based anomaly detection
      - Time-pattern analysis
    
    ### 📊 Key Features:
    - **Real-time Detection**: Process transactions instantly
    - **Feature Engineering**: Automatic creation of fraud indicators
    - **Interactive Visualization**: Explore fraud patterns visually
    - **Comprehensive Reports**: Detailed analysis and export options
    - **Multiple Algorithms**: Compare results across different methods
    
    ### 🚀 Get Started:
    1. Upload your transaction dataset (CSV or Excel)
    2. Choose your detection method
    3. Configure detection parameters
    4. Run analysis and explore results
    5. Export flagged transactions and reports
    
    ### 📋 Supported Data Formats:
    - Transaction amounts, timestamps, user IDs
    - Merchant information, transaction types
    - Geographic data, device information
    - Any numerical or categorical features
    """)

# Help and Documentation
with st.expander("📚 Help & Documentation"):
    st.markdown("""
    ## 🔧 How to Use FraudScope
    
    ### Data Preparation
    - **Required**: Numerical transaction data (amounts, counts, etc.)
    - **Recommended**: Timestamps, user IDs, merchant info
    - **Optional**: Known fraud labels for supervised learning
    
    ### Detection Methods Explained
    
    **🌲 Isolation Forest**
    - Best for: General anomaly detection
    - How it works: Isolates anomalies by randomly selecting features
    - Good for: Datasets without fraud labels
    - Contamination parameter: Expected % of fraud (1-20%)
    
    **🎯 DBSCAN Clustering**
    - Best for: Finding transaction clusters
    - How it works: Groups similar transactions, flags outliers
    - Good for: Identifying fraud patterns and networks
    - Parameters: Epsilon (distance) and minimum samples
    
    **🤖 Supervised Learning**
    - Best for: When you have labeled fraud data
    - How it works: Trains on known fraud/legitimate transactions
    - Good for: High accuracy predictions with ground truth
    - Requires: A column indicating fraud (0/1, True/False, etc.)
    
    **⚠️ Risk Scoring**
    - Best for: Generating fraud probability scores
    - How it works: Combines multiple risk factors
    - Good for: Ranking transactions by fraud likelihood
    - Features: Amount patterns, time analysis, velocity checks
    
    ### 📊 Interpreting Results
    
    **Anomaly Scores**: Lower scores = higher fraud probability
    **Risk Scores**: Higher scores = higher fraud probability  
    **Confidence**: Model's certainty in prediction
    **Feature Importance**: Which factors matter most for detection
    
    ### 💡 Best Practices
    
    1. **Start with Unsupervised**: If you don't have fraud labels
    2. **Use Multiple Methods**: Compare results across algorithms
    3. **Validate Results**: Review flagged transactions manually
    4. **Adjust Thresholds**: Based on your risk tolerance
    5. **Monitor Performance**: Track false positives/negatives
    
    ### 🚨 Common Issues
    
    - **Too Many False Positives**: Lower contamination rate or threshold
    - **Missing Fraud**: Increase contamination rate or check data quality
    - **Poor Model Performance**: More features or better data preprocessing
    - **Slow Processing**: Reduce dataset size or feature count
    """)

# Real-time Transaction Simulator
st.header("🔄 Real-time Transaction Simulator")

with st.expander("⚡ Simulate Live Transaction Processing"):
    st.markdown("### Generate synthetic transactions to test fraud detection")
    
    col1, col2 = st.columns(2)
    
    with col1:
        num_transactions = st.number_input("Number of transactions to generate", 100, 10000, 1000)
        fraud_rate = st.slider("Fraud rate (%)", 1, 20, 5)
    
    with col2:
        include_timestamps = st.checkbox("Include timestamps", True)
        include_user_ids = st.checkbox("Include user IDs", True)
    
    if st.button("🎲 Generate Synthetic Data"):
        with st.spinner("Generating synthetic transactions..."):
            # Generate synthetic transaction data
            np.random.seed(42)
            
            # Base transaction amounts (log-normal distribution)
            amounts = np.random.lognormal(mean=3, sigma=1, size=num_transactions)
            amounts = np.round(amounts, 2)
            
            # Generate fraud labels
            fraud_labels = np.random.choice([0, 1], size=num_transactions, 
                                          p=[1-fraud_rate/100, fraud_rate/100])
            
            # Fraudulent transactions tend to have higher amounts
            fraud_multiplier = np.where(fraud_labels == 1, 
                                      np.random.uniform(2, 10, size=num_transactions), 1)
            amounts = amounts * fraud_multiplier
            
            # Create synthetic dataset
            synthetic_data = {
                'transaction_id': [f'TXN_{i:06d}' for i in range(num_transactions)],
                'amount': amounts,
                'is_fraud': fraud_labels
            }
            
            if include_timestamps:
                # Generate timestamps over the last 30 days
                start_time = datetime.now() - timedelta(days=30)
                timestamps = [start_time + timedelta(
                    seconds=np.random.randint(0, 30*24*3600)) for _ in range(num_transactions)]
                synthetic_data['timestamp'] = timestamps
            
            if include_user_ids:
                # Generate user IDs (some users have multiple transactions)
                user_ids = np.random.choice([f'USER_{i:04d}' for i in range(num_transactions//5)], 
                                          size=num_transactions)
                synthetic_data['user_id'] = user_ids
            
            # Add merchant categories
            merchants = ['grocery', 'gas_station', 'restaurant', 'online', 'retail', 'pharmacy']
            synthetic_data['merchant_category'] = np.random.choice(merchants, size=num_transactions)
            
            # Create DataFrame
            synthetic_df = pd.DataFrame(synthetic_data)
            
            # Store in session state
            st.session_state.dataset = synthetic_df
            
            st.success(f"✅ Generated {num_transactions:,} synthetic transactions with {fraud_rate}% fraud rate")
            st.dataframe(synthetic_df.head(10))

# Performance Monitoring
if st.session_state.fraud_predictions:
    st.header("📈 Performance Monitoring")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🎯 Detection Summary")
        
        # Create summary metrics
        total_transactions = len(st.session_state.dataset)
        detection_summary = []
        
        for method, result in st.session_state.fraud_predictions.items():
            if 'fraud_count' in result:
                detection_rate = (result['fraud_count'] / total_transactions) * 100
                detection_summary.append({
                    'Method': method.replace('_', ' ').title(),
                    'Fraud Detected': result['fraud_count'],
                    'Detection Rate (%)': round(detection_rate, 2)
                })
        
        if detection_summary:
            summary_df = pd.DataFrame(detection_summary)
            st.dataframe(summary_df, use_container_width=True)
    
    with col2:
        st.subheader("⏱️ Processing Stats")
        
        # Show processing information
        st.metric("Dataset Size", f"{total_transactions:,} transactions")
        st.metric("Features Used", len(st.session_state.processed_data.columns) if st.session_state.processed_data is not None else 0)
        
        # Memory usage estimation
        if st.session_state.processed_data is not None:
            memory_usage = st.session_state.processed_data.memory_usage(deep=True).sum() / 1024**2
            st.metric("Memory Usage", f"{memory_usage:.1f} MB")

# Advanced Analytics
st.header("🧠 Advanced Analytics")

if st.session_state.fraud_predictions and st.session_state.processed_data is not None:
    
    tab1, tab2, tab3 = st.tabs(["📊 Statistical Analysis", "🔍 Pattern Discovery", "🎯 Model Insights"])
    
    with tab1:
        st.subheader("Statistical Distribution Analysis")
        
        # Analyze distributions by fraud status
        numerical_cols = st.session_state.processed_data.select_dtypes(include=[np.number]).columns.tolist()
        
        if 'predicted_fraud' in st.session_state.processed_data.columns:
            fraud_col = 'predicted_fraud'
        elif any('fraud' in col.lower() for col in st.session_state.processed_data.columns):
            fraud_col = next(col for col in st.session_state.processed_data.columns if 'fraud' in col.lower())
        else:
            fraud_col = None
        
        if fraud_col and len(numerical_cols) > 0:
            selected_feature = st.selectbox("Select feature for analysis:", numerical_cols)
            
            # Create distribution comparison
            fraud_data = st.session_state.processed_data[st.session_state.processed_data[fraud_col] == 1][selected_feature]
            normal_data = st.session_state.processed_data[st.session_state.processed_data[fraud_col] == 0][selected_feature]
            
            fig = go.Figure()
            fig.add_trace(go.Histogram(x=normal_data, name='Normal', opacity=0.7, nbinsx=50))
            fig.add_trace(go.Histogram(x=fraud_data, name='Fraud', opacity=0.7, nbinsx=50))
            fig.update_layout(
                title=f'Distribution Comparison: {selected_feature}',
                xaxis_title=selected_feature,
                yaxis_title='Count',
                barmode='overlay'
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Statistical tests
            from scipy import stats
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Normal Transactions Mean", f"{normal_data.mean():.2f}")
                st.metric("Normal Transactions Std", f"{normal_data.std():.2f}")
            
            with col2:
                st.metric("Fraud Transactions Mean", f"{fraud_data.mean():.2f}")
                st.metric("Fraud Transactions Std", f"{fraud_data.std():.2f}")
            
            # T-test
            if len(fraud_data) > 1 and len(normal_data) > 1:
                t_stat, p_value = stats.ttest_ind(fraud_data.dropna(), normal_data.dropna())
                st.metric("T-test P-value", f"{p_value:.6f}")
                if p_value < 0.05:
                    st.success("✅ Statistically significant difference between fraud and normal transactions")
                else:
                    st.warning("⚠️ No significant statistical difference detected")
    
    with tab2:
        st.subheader("Fraud Pattern Discovery")
        
        # Time-based patterns
        if any('time' in col.lower() or 'date' in col.lower() for col in st.session_state.processed_data.columns):
            time_cols = [col for col in st.session_state.processed_data.columns 
                        if 'time' in col.lower() or 'date' in col.lower()]
            
            if time_cols:
                time_col = time_cols[0]
                try:
                    temp_df = st.session_state.processed_data.copy()
                    temp_df[time_col] = pd.to_datetime(temp_df[time_col])
                    temp_df['hour'] = temp_df[time_col].dt.hour
                    temp_df['day_of_week'] = temp_df[time_col].dt.day_name()
                    
                    if fraud_col:
                        # Fraud by hour
                        hourly_fraud = temp_df.groupby('hour')[fraud_col].agg(['count', 'sum']).reset_index()
                        hourly_fraud['fraud_rate'] = (hourly_fraud['sum'] / hourly_fraud['count']) * 100
                        
                        fig = px.line(hourly_fraud, x='hour', y='fraud_rate', 
                                    title='Fraud Rate by Hour of Day')
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Fraud by day of week
                        daily_fraud = temp_df.groupby('day_of_week')[fraud_col].agg(['count', 'sum']).reset_index()
                        daily_fraud['fraud_rate'] = (daily_fraud['sum'] / daily_fraud['count']) * 100
                        
                        # Reorder days
                        day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
                        daily_fraud['day_of_week'] = pd.Categorical(daily_fraud['day_of_week'], categories=day_order)
                        daily_fraud = daily_fraud.sort_values('day_of_week')
                        
                        fig = px.bar(daily_fraud, x='day_of_week', y='fraud_rate',
                                   title='Fraud Rate by Day of Week')
                        st.plotly_chart(fig, use_container_width=True)
                
                except Exception as e:
                    st.error(f"Error in time analysis: {str(e)}")
        
        # Amount-based patterns
        amount_cols = [col for col in numerical_cols if 'amount' in col.lower()]
        if amount_cols and fraud_col:
            amount_col = amount_cols[0]
            
            # Fraud rate by amount ranges
            temp_df = st.session_state.processed_data.copy()
            temp_df['amount_range'] = pd.cut(temp_df[amount_col], bins=10, precision=2)
            amount_fraud = temp_df.groupby('amount_range')[fraud_col].agg(['count', 'sum']).reset_index()
            amount_fraud['fraud_rate'] = (amount_fraud['sum'] / amount_fraud['count']) * 100
            amount_fraud['range_str'] = amount_fraud['amount_range'].astype(str)
            
            fig = px.bar(amount_fraud, x='range_str', y='fraud_rate',
                       title='Fraud Rate by Transaction Amount Range')
            fig.update_xaxes(tickangle=45)
            st.plotly_chart(fig, use_container_width=True)
    
    with tab3:
        st.subheader("Model Performance Insights")
        
        if 'supervised' in st.session_state.fraud_predictions:
            metrics = st.session_state.fraud_predictions['supervised']['metrics']
            
            # ROC Curve would go here (simplified for this example)
            st.write("### 📊 Model Performance Summary")
            
            performance_data = {
                'Metric': ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC AUC'],
                'Score': [
                    metrics['accuracy'],
                    metrics['classification_report']['1']['precision'],
                    metrics['classification_report']['1']['recall'],
                    metrics['classification_report']['1']['f1-score'],
                    metrics['roc_auc']
                ]
            }
            
            perf_df = pd.DataFrame(performance_data)
            fig = px.bar(perf_df, x='Metric', y='Score', title='Model Performance Metrics')
            fig.update_layout(yaxis=dict(range=[0, 1]))
            st.plotly_chart(fig, use_container_width=True)
            
            # Feature importance radar chart
            if st.session_state.feature_importance:
                top_features = dict(sorted(st.session_state.feature_importance.items(), 
                                         key=lambda x: x[1], reverse=True)[:8])
                
                fig = go.Figure()
                fig.add_trace(go.Scatterpolar(
                    r=list(top_features.values()),
                    theta=list(top_features.keys()),
                    fill='toself',
                    name='Feature Importance'
                ))
                fig.update_layout(
                    polar=dict(
                        radialaxis=dict(visible=True, range=[0, max(top_features.values())])
                    ),
                    title="Top Features Radar Chart"
                )
                st.plotly_chart(fig, use_container_width=True)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; padding: 20px;'>
    <h4>🔍 FraudScope - Advanced Fraud Detection System</h4>
    <p>Powered by Machine Learning | Built with Streamlit</p>
    <p><em>Protecting your transactions with AI-driven intelligence</em></p>
</div>
""", unsafe_allow_html=True)