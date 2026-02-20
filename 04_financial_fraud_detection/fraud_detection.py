"""
Financial Fraud Detection
=========================
This script implements machine learning models to detect fraudulent credit card transactions
using imbalanced classification techniques.

Author: PREETHI R
Project: Finance Analytics Portfolio

Note: Place the Credit Card Fraud Detection dataset (creditcard.csv) in the 'data' subdirectory
Download from: https://www.kaggle.com/mlg-ulb/creditcardfraud
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import IsolationForest
import xgboost as xgb
from sklearn.metrics import (classification_report, precision_recall_curve, roc_curve,
                            auc, average_precision_score, confusion_matrix, roc_auc_score)
import warnings
warnings.filterwarnings('ignore')

# Set style
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (12, 8)


def load_fraud_data(file_path='data/creditcard.csv'):
    """Load and explore credit card fraud dataset"""
    try:
        print("Loading credit card fraud dataset...")
        data = pd.read_csv(file_path)
        print(f"✓ Loaded {len(data)} transactions")
        
        print(f"\nDataset Shape: {data.shape}")
        print(f"\nClass Distribution:")
        print(data['Class'].value_counts())
        print(f"\nFraud Rate: {data['Class'].mean():.4%}")
        
        return data
    except FileNotFoundError:
        print(f"❌ Error: File not found at {file_path}")
        print("\nTo run this analysis:")
        print("1. Download dataset from Kaggle: Credit Card Fraud Detection")
        print("2. Place 'creditcard.csv' in the 'data' subdirectory")
        print("3. Run this script again")
        return None


def visualize_fraud_data(data):
    """Visualize fraud data characteristics"""
    if data is None:
        return
    
    print("\nGenerating fraud data visualizations...")
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Plot 1: Class distribution
    class_counts = data['Class'].value_counts()
    axes[0, 0].bar(['Legitimate', 'Fraud'], class_counts.values, color=['green', 'red'])
    axes[0, 0].set_title('Transaction Class Distribution', fontsize=12, fontweight='bold')
    axes[0, 0].set_ylabel('Count')
    axes[0, 0].set_yscale('log')
    for i, v in enumerate(class_counts.values):
        axes[0, 0].text(i, v, f'{v:,}', ha='center', va='bottom')
    
    # Plot 2: Amount distribution by class
    axes[0, 1].hist([data[data['Class'] == 0]['Amount'], 
                     data[data['Class'] == 1]['Amount']], 
                    bins=50, label=['Legitimate', 'Fraud'], color=['green', 'red'], alpha=0.7)
    axes[0, 1].set_title('Transaction Amount Distribution', fontsize=12, fontweight='bold')
    axes[0, 1].set_xlabel('Amount')
    axes[0, 1].set_ylabel('Frequency')
    axes[0, 1].set_xlim([0, 500])
    axes[0, 1].legend()
    
    # Plot 3: Amount boxplot by class
    data.boxplot(column='Amount', by='Class', ax=axes[1, 0])
    axes[1, 0].set_title('Transaction Amount by Class', fontsize=12, fontweight='bold')
    axes[1, 0].set_xlabel('Class (0=Legitimate, 1=Fraud)')
    axes[1, 0].set_ylabel('Amount')
    axes[1, 0].set_ylim([0, 1000])
    
    # Plot 4: Time distribution
    axes[1, 1].hist([data[data['Class'] == 0]['Time'], 
                     data[data['Class'] == 1]['Time']], 
                    bins=50, label=['Legitimate', 'Fraud'], color=['green', 'red'], alpha=0.7)
    axes[1, 1].set_title('Transaction Time Distribution', fontsize=12, fontweight='bold')
    axes[1, 1].set_xlabel('Time (seconds from first transaction)')
    axes[1, 1].set_ylabel('Frequency')
    axes[1, 1].legend()
    
    plt.tight_layout()
    plt.savefig('fraud_data_exploration.png', dpi=300, bbox_inches='tight')
    plt.show()
    print("✓ Fraud data exploration saved as 'fraud_data_exploration.png'")


def preprocess_fraud_data(data):
    """Preprocess and prepare fraud data for modeling"""
    if data is None:
        return None, None, None, None
    
    print("\nPreprocessing fraud data...")
    
    # Scale Time and Amount features
    scaler = StandardScaler()
    data['scaled_amount'] = scaler.fit_transform(data['Amount'].values.reshape(-1, 1))
    data['scaled_time'] = scaler.fit_transform(data['Time'].values.reshape(-1, 1))
    
    # Drop original Time and Amount
    data_processed = data.drop(['Time', 'Amount'], axis=1)
    
    # Separate features and target
    X = data_processed.drop('Class', axis=1)
    y = data_processed['Class']
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"✓ Preprocessing complete")
    print(f"  Training samples: {len(X_train)} ({y_train.sum()} frauds)")
    print(f"  Testing samples: {len(X_test)} ({y_test.sum()} frauds)")
    print(f"  Features: {X_train.shape[1]}")
    
    return X_train, X_test, y_train, y_test


def train_fraud_models(X_train, X_test, y_train, y_test):
    """Train multiple fraud detection models"""
    print("\nTraining fraud detection models...")
    
    models = {}
    predictions = {}
    
    # 1. Logistic Regression with class weights
    print("  Training Logistic Regression...")
    log_reg = LogisticRegression(class_weight='balanced', random_state=42, max_iter=1000)
    log_reg.fit(X_train, y_train)
    models['Logistic Regression'] = log_reg
    predictions['Logistic Regression'] = log_reg.predict_proba(X_test)[:, 1]
    
    # 2. XGBoost with scale_pos_weight
    print("  Training XGBoost...")
    scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
    xgb_clf = xgb.XGBClassifier(
        objective='binary:logistic',
        scale_pos_weight=scale_pos_weight,
        random_state=42,
        n_estimators=100,
        max_depth=5,
        n_jobs=-1
    )
    xgb_clf.fit(X_train, y_train)
    models['XGBoost'] = xgb_clf
    predictions['XGBoost'] = xgb_clf.predict_proba(X_test)[:, 1]
    
    # 3. Isolation Forest (Unsupervised Anomaly Detection)
    print("  Training Isolation Forest...")
    contamination = y_train.sum() / len(y_train)
    iso_forest = IsolationForest(contamination=contamination, random_state=42, n_jobs=-1)
    iso_forest.fit(X_train)
    # Predict returns -1 for outliers, 1 for inliers
    iso_predictions = iso_forest.predict(X_test)
    # Convert to 1 for fraud, 0 for legitimate
    predictions['Isolation Forest'] = np.array([1 if x == -1 else 0 for x in iso_predictions])
    models['Isolation Forest'] = iso_forest
    
    print("✓ All models trained successfully")
    
    return models, predictions


def evaluate_fraud_models(models, predictions, y_test):
    """Evaluate fraud detection models"""
    print("\n" + "="*80)
    print("FRAUD DETECTION MODEL EVALUATION")
    print("="*80)
    
    results = []
    
    for name, pred in predictions.items():
        print(f"\n{name}:")
        print("="*60)
        
        if name == 'Isolation Forest':
            # Binary predictions for Isolation Forest
            y_pred = pred
            print(classification_report(y_test, y_pred, target_names=['Legitimate', 'Fraud']))
            results.append({
                'Model': name,
                'Type': 'Unsupervised',
                'AUC-ROC': 'N/A',
                'AP Score': 'N/A'
            })
        else:
            # Probability predictions
            auc_score = roc_auc_score(y_test, pred)
            ap_score = average_precision_score(y_test, pred)
            y_pred = (pred > 0.5).astype(int)
            
            print(f"AUC-ROC Score: {auc_score:.4f}")
            print(f"Average Precision Score: {ap_score:.4f}\n")
            print(classification_report(y_test, y_pred, target_names=['Legitimate', 'Fraud']))
            
            results.append({
                'Model': name,
                'Type': 'Supervised',
                'AUC-ROC': f'{auc_score:.4f}',
                'AP Score': f'{ap_score:.4f}'
            })
    
    return pd.DataFrame(results)


def plot_confusion_matrices(models, predictions, y_test):
    """Plot confusion matrices for all models"""
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    
    for idx, (name, pred) in enumerate(predictions.items()):
        if name == 'Isolation Forest':
            y_pred = pred
        else:
            y_pred = (pred > 0.5).astype(int)
        
        cm = confusion_matrix(y_test, y_pred)
        
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[idx],
                   xticklabels=['Legitimate', 'Fraud'],
                   yticklabels=['Legitimate', 'Fraud'])
        axes[idx].set_title(f'{name}\nConfusion Matrix', fontsize=12, fontweight='bold')
        axes[idx].set_ylabel('True Label')
        axes[idx].set_xlabel('Predicted Label')
    
    plt.tight_layout()
    plt.savefig('confusion_matrices.png', dpi=300, bbox_inches='tight')
    plt.show()
    print("✓ Confusion matrices saved as 'confusion_matrices.png'")


def plot_roc_pr_curves(predictions, y_test):
    """Plot ROC and Precision-Recall curves"""
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # ROC Curve
    for name, pred in predictions.items():
        if name != 'Isolation Forest':
            fpr, tpr, _ = roc_curve(y_test, pred)
            auc_score = roc_auc_score(y_test, pred)
            axes[0].plot(fpr, tpr, linewidth=2, label=f'{name} (AUC = {auc_score:.3f})')
    
    axes[0].plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random')
    axes[0].set_xlabel('False Positive Rate', fontsize=12)
    axes[0].set_ylabel('True Positive Rate', fontsize=12)
    axes[0].set_title('ROC Curves', fontsize=14, fontweight='bold')
    axes[0].legend(loc='lower right')
    axes[0].grid(True, alpha=0.3)
    
    # Precision-Recall Curve
    for name, pred in predictions.items():
        if name != 'Isolation Forest':
            precision, recall, _ = precision_recall_curve(y_test, pred)
            ap_score = average_precision_score(y_test, pred)
            axes[1].plot(recall, precision, linewidth=2, label=f'{name} (AP = {ap_score:.3f})')
    
    axes[1].set_xlabel('Recall', fontsize=12)
    axes[1].set_ylabel('Precision', fontsize=12)
    axes[1].set_title('Precision-Recall Curves', fontsize=14, fontweight='bold')
    axes[1].legend(loc='best')
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('roc_pr_curves.png', dpi=300, bbox_inches='tight')
    plt.show()
    print("✓ ROC and PR curves saved as 'roc_pr_curves.png'")


def analyze_fraud_costs(y_test, predictions, false_positive_cost=10, false_negative_cost=100):
    """Analyze business costs of different models"""
    print("\n" + "="*80)
    print("FRAUD COST ANALYSIS")
    print("="*80)
    print(f"Assumptions:")
    print(f"  - Cost of False Positive (blocking legitimate transaction): ${false_positive_cost}")
    print(f"  - Cost of False Negative (missing fraud): ${false_negative_cost}")
    print("="*80)
    
    for name, pred in predictions.items():
        if name == 'Isolation Forest':
            y_pred = pred
        else:
            y_pred = (pred > 0.5).astype(int)
        
        cm = confusion_matrix(y_test, y_pred)
        tn, fp, fn, tp = cm.ravel()
        
        total_cost = (fp * false_positive_cost) + (fn * false_negative_cost)
        
        print(f"\n{name}:")
        print(f"  False Positives: {fp} × ${false_positive_cost} = ${fp * false_positive_cost:,}")
        print(f"  False Negatives: {fn} × ${false_negative_cost} = ${fn * false_negative_cost:,}")
        print(f"  Total Cost: ${total_cost:,}")


def main():
    """Main execution function"""
    print("\n" + "="*80)
    print("FINANCIAL FRAUD DETECTION")
    print("="*80 + "\n")
    
    # Step 1: Load data
    data = load_fraud_data()
    
    if data is None:
        return
    
    # Step 2: Visualize
    visualize_fraud_data(data)
    
    # Step 3: Preprocess
    X_train, X_test, y_train, y_test = preprocess_fraud_data(data)
    
    if X_train is None:
        return
    
    # Step 4: Train models
    models, predictions = train_fraud_models(X_train, X_test, y_train, y_test)
    
    # Step 5: Evaluate
    results_df = evaluate_fraud_models(models, predictions, y_test)
    
    # Step 6: Visualizations
    plot_confusion_matrices(models, predictions, y_test)
    plot_roc_pr_curves(predictions, y_test)
    
    # Step 7: Cost analysis
    analyze_fraud_costs(y_test, predictions)
    
    # Summary
    print("\n" + "="*80)
    print("MODEL COMPARISON SUMMARY")
    print("="*80)
    print(results_df.to_string(index=False))
    print("="*80)
    
    print("\n📊 Generated Files:")
    print("  - fraud_data_exploration.png")
    print("  - confusion_matrices.png")
    print("  - roc_pr_curves.png")
    print("\n✅ Fraud detection analysis completed successfully!")


if __name__ == "__main__":
    main()
