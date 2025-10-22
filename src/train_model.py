import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
import joblib
import os

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data", "processed_data.xlsx")
MODEL_PATH = os.path.join(BASE_DIR, "models", "crop_yield_model.pkl")

# Load data
df = pd.read_excel(DATA_PATH, engine="openpyxl")

# Target
TARGET_COLUMN = "Cost (₹/ha)"
X = df.drop(columns=[TARGET_COLUMN])
y = df[TARGET_COLUMN]

# Identify categorical and numerical features
categorical_features = X.select_dtypes(include=["object"]).columns.tolist()
numerical_features = X.select_dtypes(include=["int64", "float64"]).columns.tolist()

# Ensure categorical columns are string type
for col in categorical_features:
    X[col] = X[col].astype(str).str.strip()

# Fill missing numerical values
for col in numerical_features:
    X[col] = X[col].fillna(X[col].median())

# --------------------
# Create pipeline with SimpleImputer + OrdinalEncoder for categoricals
categorical_pipeline = make_pipeline(
    SimpleImputer(strategy='constant', fill_value='Unknown'),
    OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
)

preprocessor = ColumnTransformer(
    transformers=[
        ("cat", categorical_pipeline, categorical_features),
        ("num", "passthrough", numerical_features),
    ]
)

model_pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("regressor", RandomForestRegressor(n_estimators=200, random_state=42)),
])

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model_pipeline.fit(X_train, y_train)

# Evaluate
score = model_pipeline.score(X_test, y_test)
print(f"✅ Model R² Score: {score:.3f}")

# Save model
os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
joblib.dump(model_pipeline, MODEL_PATH)
print(f"✅ Model saved to: {MODEL_PATH}")
