import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

def setup():
    for f in ['data', 'images']: os.makedirs(f, exist_ok=True)

def run_analysis():
    print("Analyzing Global AI Job Market Trends...")
    # Generating synthetic data
    years = range(2020, 2027)
    roles = ['Data Scientist', 'ML Engineer', 'Data Analyst', 'AI Architect', 'BI Developer']
    data = []
    for yr in years:
        for role in roles:
            avg_salary = 80000 + (yr-2020)*15000 + np.random.randint(-5000, 5000)
            demand = 100 + (yr-2020)*40 + np.random.randint(-10, 10)
            data.append([yr, role, avg_salary, demand])
    
    df = pd.DataFrame(data, columns=['Year', 'Role', 'Avg_Salary', 'Demand_Index'])
    df.to_csv('data/ai_job_market.csv', index=False)

    # Visualization
    plt.figure(figsize=(12, 6))
    sns.lineplot(data=df, x='Year', y='Avg_Salary', hue='Role', marker='o')
    plt.title('Salary Trends in AI (2020-2026)')
    plt.ylabel('Salary ($)')
    plt.savefig('images/salary_trends.png')
    plt.close()
    print("Success: AI Job Market Analysis saved to images/")

if __name__ == "__main__":
    setup()
    run_analysis()
