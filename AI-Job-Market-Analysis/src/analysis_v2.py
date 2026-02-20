import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

def run_job_analysis():
    os.makedirs('../data', exist_ok=True)
    os.makedirs('../images', exist_ok=True)
    
    # Simulate Kaggle-like dataset (2020-2026)
    np.random.seed(42)
    n = 3000
    roles = ['Data Scientist', 'ML Engineer', 'Data Analyst', 'AI Architect', 'BI Developer', 'Research Scientist']
    exp_levels = ['Entry', 'Mid', 'Senior', 'Executive']
    locations = ['United States', 'India', 'Germany', 'United Kingdom', 'Canada', 'Remote']
    
    data = []
    for i in range(n):
        role = np.random.choice(roles)
        exp = np.random.choice(exp_levels)
        loc = np.random.choice(locations)
        year = np.random.choice([2020, 2021, 2022, 2023, 2024, 2025, 2026])
        
        base = 60000
        if role == 'AI Architect': base += 40000
        if role == 'ML Engineer': base += 25000
        if exp == 'Senior': base *= 1.5
        if exp == 'Executive': base *= 2.2
        if loc == 'United States': base *= 1.4
        
        # Trend over years
        base *= (1 + (year - 2020) * 0.08)
        salary = base + np.random.normal(0, 10000)
        
        data.append([year, role, exp, loc, salary])
        
    df = pd.DataFrame(data, columns=['Year', 'Role', 'Experience', 'Location', 'Salary'])
    df.to_csv('../data/ai_job_market_v2.csv', index=False)

    plt.style.use('dark_background')
    
    # 1. Salary Distribution by Role
    plt.figure(figsize=(12, 6))
    sns.boxplot(data=df, x='Role', y='Salary', palette='viridis')
    plt.xticks(rotation=45)
    plt.title('Global AI Salary Distribution by Role (2020-2026)', fontsize=15, fontweight='bold')
    plt.tight_layout()
    plt.savefig('../images/v1_salary_by_role.png')
    plt.close()

    # 2. Avg Salary Growth over Time
    plt.figure(figsize=(12, 6))
    sns.lineplot(data=df, x='Year', y='Salary', hue='Experience', marker='o', linewidth=2.5)
    plt.title('AI Salary Projection by Experience (2020-2026)', fontsize=15, fontweight='bold')
    plt.grid(alpha=0.2)
    plt.savefig('../images/v2_salary_trend.png')
    plt.close()

    # 3. Role Demand Share
    plt.figure(figsize=(10, 8))
    df['Role'].value_counts().plot.pie(autopct='%1.1f%%', cmap='Spectral')
    plt.title('Market Demand Share per Role', fontsize=15, fontweight='bold')
    plt.ylabel('')
    plt.savefig('../images/v3_role_demand.png')
    plt.close()

    # 4. Salary Heatmap: Exp vs Location
    pivot = df.pivot_table(index='Location', columns='Experience', values='Salary', aggfunc='mean')
    plt.figure(figsize=(10, 6))
    sns.heatmap(pivot, annot=True, fmt=".0f", cmap='magma')
    plt.title('Avg Salary: Experience Level vs Location', fontsize=15, fontweight='bold')
    plt.savefig('../images/v4_salary_heatmap.png')
    plt.close()

    # 5. Remote vs In-Office Gap
    df['is_remote'] = df['Location'] == 'Remote'
    plt.figure(figsize=(10, 6))
    sns.barplot(data=df, x='Year', y='Salary', hue='is_remote', palette='coolwarm')
    plt.title('Salary Comparison: Remote vs On-Site Growth', fontsize=15, fontweight='bold')
    plt.savefig('../images/v5_remote_gap.png')
    plt.close()

    print("Success: 5 quality visualizations generated for AI Job Market.")

if __name__ == "__main__":
    run_job_analysis()
