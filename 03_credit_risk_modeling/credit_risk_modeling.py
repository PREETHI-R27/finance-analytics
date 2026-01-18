"""
Credit Risk Modeling and Loan Default Prediction
=================================================
This script builds machine learning models to predict loan defaults
and assess credit risk using classification algorithms.

Author: PREETHI R
Project: Finance Analytics Portfolio

Note: Place your loan dataset (e.g., loan_data.csv) in the 'data' subdirectory
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
from sklearn.metrics import (roc_auc_score, roc_curve, precision_recall_curve, 
                            auc, classification_report, confusion_matrix)
import warnings
warnings.filterwarnings('ignore')

# Set style
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (12, 8)


def load_and_explore_data(file_path='data/loan_data.csv'):
    """Load and perform initial exploration of loan data"""
    try:
        print("Loading loan dataset...")
        data = pd.read_csv(file_path)
        print(f"✓ Loaded {len(data)} records with {len(data.columns)} columns")
        
        print(f"\nDataset Shape: {data.shape}")
        print(f"\nColumn Types:\n{data.dtypes.value_counts()}")
        print(f"\nMissing Values:\n{data.isnull().sum()[data.isnull().sum() > 0]}")
        
        return data
    except FileNotFoundError:
        print(f"❌ Error: File not found at {file_path}")
        print("Please download loan dataset and place it in the 'data' subdirectory")
        print("Suggested datasets:")
        print("  - Lending Club Loan Data from Kaggle")
        print("  - 'Give Me Some Credit' from Kaggle")
        return None


def clean_and_preprocess_data(data, target_col='loan_status'):
    """Clean and preprocess the loan data"""
    if data is None:
        return None, None, None, None, None, None
    
    print("\nCleaning and preprocessing data...")
    
    # Handle missing values
    for col in data.select_dtypes(include=np.number).columns:
        data[col].fillna(data[col].median(), inplace=True)
    
    for col in data.select_dtypes(include='object').columns:
        if col != target_col:
            data[col].fillna(data[col].mode()[0] if len(data[col].mode()) > 0 else 'Unknown', inplace=True)
    
    # Encode target variable (assuming binary: default=1, paid=0)
    if target_col in data.columns:
        if data[target_col].dtype == 'object':
            # Adjust based on your dataset's target encoding
            data[target_col] = data[target_col].map(lambda x: 1 if 'default' in str(x).lower() or 'charged' in str(x).lower() else 0)
    
    # Separate features and target
    X = data.drop(target_col, axis=1)
    y = data[target_col]
    
    # Encode categorical variables
    categorical_cols = X.select_dtypes(include='object').columns
    X_encoded = pd.get_dummies(X, columns=categorical_cols, drop_first=True)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X_encoded, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    print(f"✓ Preprocessing complete")
    print(f"  Training samples: {len(X_train_scaled)}")
    print(f"  Testing samples: {len(X_test_scaled)}")
    print(f"  Features: {X_train_scaled.shape[1]}")
    print(f"  Default rate: {y_train.mean():.2%}")
    
    return X_train_scaled, X_test_scaled, y_train, y_test, X_train.columns, data


def visualize_eda(data, target_col='loan_status'):
    """Perform exploratory data analysis with visualizations"""
    if data is None:
        return
    
    print("\nGenerating EDA visualizations...")
    
    # Target distribution
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Plot 1: Target distribution
    data[target_col].value_counts().plot(kind='bar', ax=axes[0, 0], color=['green', 'red'])
    axes[0, 0].set_title('Loan Status Distribution', fontsize=12, fontweight='bold')
    axes[0, 0].set_xlabel('Status (0=Paid, 1=Default)')
    axes[0, 0].set_ylabel('Count')
    
    # Plot 2: Loan amount distribution by status (if column exists)
    if 'loan_amnt' in data.columns or 'loan_amount' in data.columns:
        loan_col = 'loan_amnt' if 'loan_amnt' in data.columns else 'loan_amount'
        data.boxplot(column=loan_col, by=target_col, ax=axes[0, 1])
        axes[0, 1].set_title('Loan Amount by Status', fontsize=12, fontweight='bold')
        axes[0, 1].set_xlabel('Status (0=Paid, 1=Default)')
        axes[0, 1].set_ylabel('Loan Amount')
    
    # Plot 3: Grade distribution (if exists)
    if 'grade' in data.columns:
        grade_default = pd.crosstab(data['grade'], data[target_col], normalize='index')
        grade_default[1].plot(kind='bar', ax=axes[1, 0], color='coral')
        axes[1, 0].set_title('Default Rate by Grade', fontsize=12, fontweight='bold')
        axes[1, 0].set_xlabel('Grade')
        axes[1, 0].set_ylabel('Default Rate')
        axes[1, 0].tick_params(axis='x', rotation=0)
    
    # Plot 4: Interest rate vs default (if exists)
    if 'int_rate' in data.columns:
        data.boxplot(column='int_rate', by=target_col, ax=axes[1, 1])
        axes[1, 1].set_title('Interest Rate by Status', fontsize=12, fontweight='bold')
        axes[1, 1].set_xlabel('Status (0=Paid, 1=Default)')
        axes[1, 1].set_ylabel('Interest Rate')
    
    plt.tight_layout()
    plt.savefig('eda_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    print("✓ EDA visualizations saved as 'eda_analysis.png'")


def train_models(X_train, X_test, y_train, y_test):
    """Train multiple classification models"""
    print("\nTraining classification models...")
    
    models = {}
    predictions = {}
    
    # 1. Logistic Regression
    print("  Training Logistic Regression...")
    log_reg = LogisticRegression(random_state=42, class_weight='balanced', max_iter=1000)
    log_reg.fit(X_train, y_train)
    models['Logistic Regression'] = log_reg
    predictions['Logistic Regression'] = log_reg.predict_proba(X_test)[:, 1]
    
    # 2. Random Forest
    print("  Training Random Forest...")
    rf_clf = RandomForestClassifier(n_estimators=100, random_state=42, 
                                   class_weight='balanced', n_jobs=-1)
    rf_clf.fit(X_train, y_train)
    models['Random Forest'] = rf_clf
    predictions['Random Forest'] = rf_clf.predict_proba(X_test)[:, 1]
    
    # 3. XGBoost
    print("  Training XGBoost...")
    scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
    xgb_clf = xgb.XGBClassifier(
        objective='binary:logistic',
        scale_pos_weight=scale_pos_weight,
        random_state=42,
        n_jobs=-1
    )
    xgb_clf.fit(X_train, y_train)
    models['XGBoost'] = xgb_clf
    predictions['XGBoost'] = xgb_clf.predict_proba(X_test)[:, 1]
    
    print("✓ All models trained successfully")
    
    return models, predictions


def evaluate_models(models, predictions, y_test):
    """Evaluate and compare model performance"""
    print("\n" + "="*80)
    print("MODEL EVALUATION RESULTS")
    print("="*80)
    
    results = []
    
    for name, y_pred_proba in predictions.items():
        auc_score = roc_auc_score(y_test, y_pred_proba)
        precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)
        pr_auc = auc(recall, precision)
        
        y_pred = (y_pred_proba > 0.5).astype(int)
        
        print(f"\n{name}:")
        print(f"  AUC-ROC Score: {auc_score:.4f}")
        print(f"  PR-AUC Score:  {pr_auc:.4f}")
        print(f"\nClassification Report:")
        print(classification_report(y_test, y_pred))
        
        results.append({
            'Model': name,
            'AUC-ROC': auc_score,
            'PR-AUC': pr_auc
        })
    
    return pd.DataFrame(results)


def plot_roc_curves(predictions, y_test):
    """Plot ROC curves for all models"""
    plt.figure(figsize=(10, 8))
    
    for name, y_pred_proba in predictions.items():
        fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
        auc_score = roc_auc_score(y_test, y_pred_proba)
        plt.plot(fpr, tpr, linewidth=2, label=f'{name} (AUC = {auc_score:.3f})')
    
    plt.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random Classifier')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title('ROC Curves - Credit Risk Models', fontsize=14, fontweight='bold')
    plt.legend(loc='lower right', fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('roc_curves.png', dpi=300, bbox_inches='tight')
    plt.show()
    print("✓ ROC curves saved as 'roc_curves.png'")


def plot_precision_recall_curves(predictions, y_test):
    """Plot Precision-Recall curves"""
    plt.figure(figsize=(10, 8))
    
    for name, y_pred_proba in predictions.items():
        precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)
        pr_auc = auc(recall, precision)
        plt.plot(recall, precision, linewidth=2, label=f'{name} (AUC = {pr_auc:.3f})')
    
    plt.xlabel('Recall', fontsize=12)
    plt.ylabel('Precision', fontsize=12)
    plt.title('Precision-Recall Curves - Credit Risk Models', fontsize=14, fontweight='bold')
    plt.legend(loc='best', fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('precision_recall_curves.png', dpi=300, bbox_inches='tight')
    plt.show()
    print("✓ Precision-Recall curves saved as 'precision_recall_curves.png'")


def plot_feature_importance(model, feature_names, top_n=20):
    """Plot feature importance for tree-based models"""
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
        indices = np.argsort(importances)[::-1][:top_n]
        
        plt.figure(figsize=(10, 8))
        plt.barh(range(top_n), importances[indices], color='steelblue')
        plt.yticks(range(top_n), [feature_names[i] for i in indices])
        plt.xlabel('Importance Score', fontsize=12)
        plt.title(f'Top {top_n} Feature Importances', fontsize=14, fontweight='bold')
        plt.gca().invert_yaxis()
        plt.tight_layout()
        plt.savefig('feature_importance.png', dpi=300, bbox_inches='tight')
        plt.show()
        print("✓ Feature importance chart saved as 'feature_importance.png'")


def main():
    """Main execution function"""
    print("\n" + "="*80)
    print("CREDIT RISK MODELING AND LOAN DEFAULT PREDICTION")
    print("="*80 + "\n")
    
    # Step 1: Load data
    data = load_and_explore_data()
    
    if data is None:
        print("\n⚠️ Cannot proceed without data. Please provide loan dataset.")
        print("\nTo run this analysis:")
        print("1. Download loan data from Kaggle (e.g., Lending Club)")
        print("2. Place CSV file in 'data/loan_data.csv'")
        print("3. Run this script again")
        return
    
    # Step 2: EDA
    visualize_eda(data)
    
    # Step 3: Preprocess
    X_train, X_test, y_train, y_test, feature_names, data = clean_and_preprocess_data(data)
    
    if X_train is None:
        return
    
    # Step 4: Train models
    models, predictions = train_models(X_train, X_test, y_train, y_test)
    
    # Step 5: Evaluate
    results_df = evaluate_models(models, predictions, y_test)
    
    # Step 6: Visualizations
    plot_roc_curves(predictions, y_test)
    plot_precision_recall_curves(predictions, y_test)
    
    # Step 7: Feature importance
    plot_feature_importance(models['XGBoost'], feature_names)
    
    # Summary
    print("\n" + "="*80)
    print("MODEL COMPARISON SUMMARY")
    print("="*80)
    print(results_df.to_string(index=False))
    print("="*80)
    
    print("\n📊 Generated Files:")
    print("  - eda_analysis.png")
    print("  - roc_curves.png")
    print("  - precision_recall_curves.png")
    print("  - feature_importance.png")
    print("\n✅ Credit risk modeling completed successfully!")


if __name__ == "__main__":
    main()
