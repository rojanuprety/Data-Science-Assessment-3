#Assignment3 -Investigation A and B
# Bats VS Rats
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import chi2_contingency, ttest_ind
import statsmodels.api as sm
import os

# Loading data-sets
print("Current working Directory:", os.getcwd())

try:
    df1= pd.read_csv("dataset1.csv")
    df2=pd.read_csv("dataset2.csv")
    print("Datasets loaded successfully! \n")
except FileNotFoundError as e:
    print(f"Error:{e}")
    print("Ensure both 'dataset1.csv' and 'dataset2.csv' are in the same directory.")
    exit()
    
# Initial Overview of data

print("Dataset1 shape", df1.shape)
print("Dataset2 shape", df2.shape)
print(df1.head(), "\n")
print(df2.head(), "\n")

# Descriptive Analysis - Investigation A

print("=== Investigation A: Do bats perceive rats as predators? === \n")

# Distribution of risk- taking and reward 
risk_counts = df1['risk'].value_counts(normalize=True)
reward_counts=df1['reward'].value_counts(normalize=True)

print("Risk-taking proportion:\n", risk_counts)
print("\nReward proportion:\n", reward_counts)

# Count plots Risk-Taking vs Avoidance
sns.countplot(x='risk', data=df1,palette="Set2")
plt.title("Risk_taking vs Avoidance")
plt.xlabel("Risk (0=Avoid, 1=Risk-taking)")
plt.ylabel("Count")
plt.show()

sns.countplot(x='reward', data=df1,palette="Set1")
plt.title("Reward vs No Reward")
plt.xlabel("Reward (0=No, 1=Yes)")
plt.ylabel("Count")
plt.show()

# Relationship between risk and reward

ct=pd.crosstab(df1['risk'],df1['reward'])
print("\n Risk vs Reward Table: \n",ct)

chi2, p, dof, expected = chi2_contingency(ct)
print(f"\n Chi-Square: {chi2:.3f},p-value:{p:.4f}")
if p<0.05:
    print(" Significant relationship: reward depends on risk behaviour.")
else:
    print(" No significant relationship detected.")
    
# Stacked bar plot
(ct.div(ct.sum(1), axis=0)).plot(kind="bar", stacked=True)
plt.title("Proportion of Rewards within Risk Strategies")
plt.xlabel("Risk (0=Avoid, 1=Risk-taking)")
plt.ylabel("Proportion")
plt.show()

# Investigation B: SEASONAL EFFECTS
print("\n=== Investigation B: Do behaviours change with season? ===\n")

# Average risk-taking by season
season_risk = df1.groupby('season')['risk'].mean()
print("Average risk-taking by season:\n", season_risk, "\n")

season_risk.plot(kind='bar', color='orange')
plt.title("Average Risk-taking by Season")
plt.xlabel("Season (0=Winter, 1=Spring)")
plt.ylabel("Mean Risk-taking")
plt.show()

# T-test comparing seasonal risk-taking
winter = df1[df1['season'] == 0]['risk']
spring = df1[df1['season'] == 1]['risk']

t_stat, p_val = ttest_ind(winter, spring, equal_var=False)
print(f"T-test for risk-taking (Winter vs Spring): t={t_stat:.3f}, p={p_val:.4f}")
if p_val < 0.05:
    print("Significant difference: risk-taking behaviour changes by season.")
else:
    print(" No significant seasonal difference detected.")

sns.boxplot(x='season', y='risk', data=df1, palette='coolwarm')
plt.title("Risk-taking Distribution by Season")
plt.xlabel("Season (0=Winter, 1=Spring)")
plt.ylabel("Risk-taking (0/1)")
plt.show()

# Rats vs Bats activity on dataset 2
sns.scatterplot(x='rat_arrival_number', y='bat_landing_number', hue='food_availability', data=df2, palette='viridis')
plt.title("Rat Arrivals vs Bat Landings (coloured by Food Availability)")
plt.xlabel("Rat Arrival Number")
plt.ylabel("Bat Landing Number")
plt.show()

# Correlation check
corr_matrix = df2[['rat_arrival_number', 'bat_landing_number', 'food_availability']].corr()
print("\nCorrelation Matrix:\n", corr_matrix, "\n")

sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
plt.title("Correlation Heatmap")
plt.show()

# Regression analysis - Predicting Bat Activity

print("Regression Analysis: Predicting bat landing from rat arrivals and food availability \n")

X = df2[['rat_arrival_number', 'food_availability']]
y = df2['bat_landing_number']
X = sm.add_constant(X)
model = sm.OLS(y, X).fit()
print(model.summary())

print("\n=== INTERPRETATION GUIDE ===")
print("""
- If 'rat_arrival_number' coefficient is negative and significant → more rats = fewer bat landings → avoidance due to perceived predation.
- If 'food_availability' coefficient is positive → more food = more bat activity → reduced avoidance.
- Seasonal t-test significance supports seasonal behavioural change (Investigation B).
""")
print("\n Analysis complete. Use these outputs and plots to prepare your group report.")

