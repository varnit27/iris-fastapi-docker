from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
import joblib

# 1. Load data
iris = load_iris()
X, y = iris.data, iris.target

# 2. Train model
model = RandomForestClassifier(n_estimators=10, random_state=42)
model.fit(X, y)

# 3. Save the trained model
joblib.dump(model, "model.pkl")
print("Model saved as model.pkl")