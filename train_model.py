import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import mean_squared_error, accuracy_score
import joblib

# ==================== 1. Data Collection ====================
df = pd.read_csv("student_mat.csv", sep=';')
print("Data Loaded Successfully. Shape:", df.shape)

# ==================== 2. Data Preprocessing ====================
# Create binary label for classification
df['pass_fail'] = df['G3'].apply(lambda x: 1 if x >= 10 else 0)

# Features to keep
features = ['sex', 'G1', 'G2', 'absences']
X = df[features]
y_reg = df['G3']
y_clf = df['pass_fail']

# Label encode 'sex'
le_sex = LabelEncoder()
X['sex'] = le_sex.fit_transform(X['sex'])

# Save label encoder
label_encoders = {'sex': le_sex}
joblib.dump(label_encoders, "label_encoders.pkl")

# Scale numeric features
numeric_cols = ['G1', 'G2', 'absences']
scaler = StandardScaler()
X[numeric_cols] = scaler.fit_transform(X[numeric_cols])

# Save scaler
joblib.dump(scaler, "scaler.pkl")

# ==================== 3. EDA ====================
print("\n--- Dataset Info ---")
print(df.info())

print("\n--- Statistical Summary ---")
print(df.describe())

print("\n--- Missing Values ---")
print(df.isnull().sum())

# ==================== 4. Model Building ====================
# Split data
X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(X, y_reg, test_size=0.2, random_state=42)
X_train_clf, X_test_clf, y_train_clf, y_test_clf = train_test_split(X, y_clf, test_size=0.2, random_state=42)

# Train models
reg_model = LinearRegression()
reg_model.fit(X_train_reg, y_train_reg)

# Train classification model performance
print(f"PERFORMACE OF REGRESSION MODEL: {reg_model.score(X_train_reg, y_train_reg)}")

clf_model = LogisticRegression()
clf_model.fit(X_train_clf, y_train_clf)

# Train classification model performance
print(f"PERFORMACE OF CLASSIFICATION MODEL: {clf_model.score(X_train_clf, y_train_clf)}")

# Evaluate
print("Regression MSE:", mean_squared_error(y_test_reg, reg_model.predict(X_test_reg)))
print("Classification Accuracy:", accuracy_score(y_test_clf, clf_model.predict(X_test_clf)))

# Save models
joblib.dump(reg_model, "regression_model.pkl")
joblib.dump(clf_model, "classification_model.pkl")
