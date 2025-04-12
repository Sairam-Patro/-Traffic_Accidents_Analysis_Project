import pandas as pd
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Setting seaborn style
sns.set(style="whitegrid")
plt.rcParams['figure.figsize'] = (10, 6)

# Load the dataset
df = pd.read_csv("D:/sem4/INT375/Project/traffic_accidents.csv")
df.head(10)

# Display basic information
print("Dataset Info:\n")
print(df.info())
print("\nFirst 5 rows:\n")
print(df.head())

# Check for missing values
print("\nMissing values:\n")
print(df.isnull().sum())

##Outliers Detection
numeric_cols = ['injuries_total','injuries_fatal','injuries_incapacitating','injuries_reported_not_evident','injuries_no_indication','crash_hour','crash_day_of_week','crash_month']
outliers = []
for col in numeric_cols:
    q1 = df[col].quantile(0.25)
    q3 = df[col].quantile(0.75)
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    outliers.extend(df[(df[col] < lower_bound) | (df[col] > upper_bound)].index)
outliers

# ------------------------------
# Objective 1: Most Common Accident Cause
# ------------------------------
cause_counts = df['prim_contributory_cause'].value_counts().head(10)
sns.barplot(x=cause_counts.values, y=cause_counts.index, hue=cause_counts.index, dodge=False, palette='viridis', legend=False)

plt.title('Top 10 Most Common Accident Causes')
plt.xlabel('Number of Accidents')
plt.ylabel('Primary Cause')
for i, v in enumerate(cause_counts.values):
    plt.text(v + 10, i, str(v), color='black', va='center')
plt.tight_layout()
plt.show()

# ------------------------------
# Objective 2: Crashes in Different Weather
# ------------------------------
weather_crashes = df['weather_condition'].value_counts()
sns.barplot(x=weather_crashes.values, y=weather_crashes.index, hue=weather_crashes.index, dodge=False, palette='coolwarm', legend=False)

plt.title('Crashes by Weather Condition')
plt.xlabel('Number of Crashes')
plt.ylabel('Weather')
for i, v in enumerate(weather_crashes.values):
    plt.text(v + 10, i, str(v), color='black', va='center')
plt.tight_layout()
plt.show()

# ------------------------------
# Objective 3: Injuries by Lighting Condition
# ------------------------------
injury_lighting = df.groupby('lighting_condition')['most_severe_injury'].value_counts().unstack().fillna(0)
injury_lighting.plot(kind='bar', stacked=True, colormap='Set2')
plt.title('Injury Severity by Lighting Condition')
plt.xlabel('Lighting Condition')
plt.ylabel('Count')
plt.tight_layout()
plt.show()

# ------------------------------
# Objective 4: Crash Type Frequency (Top 10 crash types)
# ------------------------------
crash_type_counts = df['first_crash_type'].value_counts().head(10)

sns.barplot(
    x=crash_type_counts.values,
    y=crash_type_counts.index,
    hue=crash_type_counts.index,
    dodge=False,
    palette='magma',
    legend=False
)
plt.title('Top 10 Crash Types')
plt.xlabel('Frequency')
plt.ylabel('Crash Type')

# Add value labels
for i, v in enumerate(crash_type_counts.values):
    plt.text(v + 10, i, str(v), va='center')

plt.tight_layout()
plt.show()

# ------------------------------
# Objective 5: Damage Comparison
# ------------------------------
damage_counts = df['damage'].value_counts()
sns.barplot(x=damage_counts.values, y=damage_counts.index, hue=damage_counts.index, palette='flare', dodge=False, legend=False)

plt.title('Vehicle Damage Comparison')
plt.xlabel('Number of Crashes')
plt.ylabel('Damage Level')

for i, v in enumerate(damage_counts.values):
    plt.text(v + 10, i, str(v), color='black', va='center')
plt.tight_layout()
plt.show()

# Select only numerical columns
numeric_df = df.select_dtypes(include=['float64', 'int64'])

# Calculate the correlation matrix
correlation_matrix = numeric_df.corr()

# Set the figure size and plot the heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
plt.title("Heatmap of Numerical Features Correlation")
plt.show()