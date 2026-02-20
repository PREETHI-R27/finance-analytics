import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
import os

def run_real_estate():
    print("Training Real Estate Price Prediction Model...")
    os.makedirs('images', exist_ok=True)
    os.makedirs('data', exist_ok=True)
    
    # Generate data
    n = 1000
    data = {
        'SqFt': np.random.randint(500, 5000, n),
        'Bedrooms': np.random.randint(1, 6, n),
        'Bathrooms': np.random.randint(1, 4, n),
        'YearBuilt': np.random.randint(1950, 2023, n),
        'LocationRating': np.random.randint(1, 11, n)
    }
    df = pd.DataFrame(data)
    df['Price'] = (df['SqFt'] * 150) + (df['Bedrooms'] * 20000) + (df['LocationRating'] * 50000) + np.random.normal(0, 10000, n)
    df.to_csv('data/housing_data.csv', index=False)

    X = df.drop('Price', axis=1)
    y = df['Price']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestRegressor(n_estimators=100)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    # Visualization: Prediction vs Actual
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, preds, alpha=0.5)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
    plt.title('Actual vs Predicted House Prices')
    plt.savefig('images/prediction_accuracy.png')
    plt.close()

    print(f"Success: Model RMSE: {np.sqrt(mean_squared_error(y_test, preds)):.2f}")

if __name__ == "__main__":
    run_real_estate()
