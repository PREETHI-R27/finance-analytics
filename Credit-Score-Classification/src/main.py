import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import os

def run_credit_scoring():
    print("Running Credit Score Classification...")
    os.makedirs('images', exist_ok=True)
    os.makedirs('data', exist_ok=True)
    
    # Generate data
    n = 1000
    data = {
        'Income': np.random.randint(20000, 200000, n),
        'Age': np.random.randint(18, 80, n),
        'LoanAmount': np.random.randint(1000, 50000, n),
        'CreditHistory': np.random.choice([0, 1], n, p=[0.2, 0.8]),
        'ExistingDebts': np.random.randint(0, 10, n)
    }
    df = pd.DataFrame(data)
    # Simple logic for credit score
    df['Score'] = np.where((df['CreditHistory'] == 1) & (df['Income'] > 50000), 'Good', 
                          np.where(df['Income'] < 30000, 'Poor', 'Average'))
    df.to_csv('data/credit_data.csv', index=False)

    X = df.drop('Score', axis=1)
    y = df['Score']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    clf = GradientBoostingClassifier()
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=clf.classes_, yticklabels=clf.classes_)
    plt.title('Credit Score Confusion Matrix')
    plt.savefig('images/confusion_matrix.png')
    plt.close()

    print("Success: Credit model trained and evaluated.")

if __name__ == "__main__":
    run_credit_scoring()
