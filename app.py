from flask import Flask, request, render_template
import joblib
import numpy as np

app = Flask(__name__)
model = joblib.load('model.pkl')
scaler = joblib.load('scaler.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        math = int(request.form['math'])
        reading = int(request.form['reading'])
        writing = int(request.form['writing'])

        features = np.array([[math, reading, writing]])
        features_scaled = scaler.transform(features)
        pred = model.predict(features_scaled)[0]
        return render_template('index.html', prediction=f'Predicted Race/Ethnicity: {pred}')
    except Exception as e:
        return f'Error: {str(e)}'

if __name__ == '__main__':
    app.run(debug=True)
