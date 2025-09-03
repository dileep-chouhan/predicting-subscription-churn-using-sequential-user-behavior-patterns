import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
# --- 1. Synthetic Data Generation ---
np.random.seed(42) # for reproducibility
# Simulate user behavior sequences (simplified example)
num_users = 500
sequence_length = 5
features = ['logins', 'purchases', 'support_tickets', 'engagement_score']
data = []
for i in range(num_users):
    sequence = np.random.randint(0, 10, size=(sequence_length, len(features)))
    churned = 1 if np.mean(sequence[-2:, 1]) < 2 else 0  # Simulate churn based on recent purchases
    data.append(np.concatenate(([churned], sequence.flatten())))
df = pd.DataFrame(data, columns=['churned'] + [f'{f}_{t}' for f in features for t in range(sequence_length)])
# --- 2. Data Preparation ---
X = df.drop('churned', axis=1)
y = df['churned']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# --- 3. Model Training ---
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)
# --- 4. Model Evaluation ---
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)
print(f"Model Accuracy: {accuracy:.2f}")
print("Classification Report:\n", report)
# --- 5. Visualization (Feature Importance) ---
feature_importances = model.feature_importances_
feature_importance_df = pd.DataFrame({'Feature': X.columns, 'Importance': feature_importances})
feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)
plt.figure(figsize=(10, 6))
sns.barplot(x='Importance', y='Feature', data=feature_importance_df)
plt.title('Feature Importance')
plt.xlabel('Importance')
plt.ylabel('Feature')
plt.tight_layout()
output_filename = 'feature_importance.png'
plt.savefig(output_filename)
print(f"Plot saved to {output_filename}")