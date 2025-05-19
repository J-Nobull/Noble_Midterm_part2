import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import joblib

df = pd.read_csv("StudentsPerformance.csv")
df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")
df = df.rename(columns={
    'math_score': 'math',
    'reading_score': 'reading',
    'writing_score': 'writing'
})
df = df[['math', 'reading', 'writing', 'race/ethnicity']]
X = df[['math', 'reading', 'writing']]
y = df['race/ethnicity']
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.25, random_state=42)
model = LogisticRegression(multi_class='multinomial', solver='lbfgs', max_iter=1000)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"Accuracy: {acc:.4f}")
joblib.dump(model, 'model.pkl')
joblib.dump(scaler, 'scaler.pkl')
