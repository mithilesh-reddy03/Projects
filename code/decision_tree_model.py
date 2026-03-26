# 1. Import Libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder

# 2. Load Dataset
df = pd.read_csv("loan_data.csv")

# 3. Handle Missing Values
df.fillna(df.median(numeric_only=True), inplace=True)

# For categorical columns, fill with mode
for col in df.select_dtypes(include='object'):
    df[col].fillna(df[col].mode()[0], inplace=True)

# 4. Encode Categorical Columns
le = LabelEncoder()
for col in df.select_dtypes(include='object'):
    df[col] = le.fit_transform(df[col])

# 5. Split Features and Target
X = df.drop("loan_status", axis=1)
y = df["loan_status"]

# 6. Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 7. Create Decision Tree Model
model = DecisionTreeClassifier(
    max_depth=5,
    random_state=42
)

# 8. Train the Model
model.fit(X_train, y_train)

# 9. Make Predictions
y_pred = model.predict(X_test)

# 10. Evaluate Model
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
