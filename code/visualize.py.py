import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


df = pd.read_csv("loan_data.csv")


print(df.head())
print(df.info())
print(df.describe())


plt.figure(figsize=(6,4))
sns.countplot(x='loan_status', data=df)
plt.title("Loan Approval Count")
plt.xlabel("Loan Status (0 = Reject, 1 = Approved)")
plt.ylabel("Count")
plt.show()


num_cols = ['person_age', 'person_income', 'loan_amnt', 'loan_int_rate', 'credit_score']
plt.figure(figsize=(12,10))
for i, col in enumerate(num_cols, 1):
    plt.subplot(3,2,i)
    sns.histplot(df[col], kde=True)
    plt.title(f"Distribution of {col}")
plt.tight_layout()
plt.show()

plt.figure(figsize=(6,4))
sns.boxplot(x='loan_status', y='credit_score', data=df)
plt.title("Credit Score by Loan Outcome")
plt.show()

plt.figure(figsize=(10,8))
sns.heatmap(df[num_cols + ['loan_status']].corr(), annot=True, cmap='coolwarm')
plt.title("Correlation Matrix")
plt.show()


cat_cols = ['person_gender', 'person_education', 'person_home_ownership', 'loan_intent', 'previous_loan_defaults_on_file']
plt.figure(figsize=(12,10))
for i, col in enumerate(cat_cols, 1):
    plt.subplot(3,2,i)
    sns.countplot(x=col, hue='loan_status', data=df)
    plt.xticks(rotation=45)
    plt.title(f"{col} vs Loan Status")
plt.tight_layout()
plt.show()
