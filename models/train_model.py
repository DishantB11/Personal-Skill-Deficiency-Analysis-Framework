import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Load dataset
df = pd.read_csv("../data/career_dataset.csv")

# Separate features and label
X = df.drop("Role", axis=1)
y = df["Role"]

# Encode role labels
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y_encoded, test_size=0.2, random_state=42
)

# -------------------------
# Decision Tree
# -------------------------
dt = DecisionTreeClassifier()
dt.fit(X_train, y_train)
dt_preds = dt.predict(X_test)
dt_acc = accuracy_score(y_test, dt_preds)

# -------------------------
# Random Forest
# -------------------------
rf = RandomForestClassifier(n_estimators=100)
rf.fit(X_train, y_train)
rf_preds = rf.predict(X_test)
rf_acc = accuracy_score(y_test, rf_preds)

# Print Results
print("Decision Tree Accuracy:", dt_acc)
print("Random Forest Accuracy:", rf_acc)

print("\nClassification Report (Random Forest):")
print(classification_report(y_test, rf_preds))

print("\nConfusion Matrix (Random Forest):")
print(confusion_matrix(y_test, rf_preds))

# Save best model
best_model = rf if rf_acc > dt_acc else dt

joblib.dump(best_model, "../saved_models/best_model.pkl")
joblib.dump(scaler, "../saved_models/scaler.pkl")
joblib.dump(label_encoder, "../saved_models/label_encoder.pkl")

print("\nBest model saved successfully.")
