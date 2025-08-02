from flask import Flask, render_template, request, redirect, url_for
import pandas as pd
import joblib
import warnings

warnings.filterwarnings("ignore")

app = Flask(__name__)

try:
    model_pipeline = joblib.load('med_insurance_svr_pipeline.joblib')
except FileNotFoundError:
    model_pipeline = None

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if model_pipeline is None:
        return "Error: Model not loaded. Check the console for details.", 500

    form_data = request.form.to_dict()

    try:
        numeric_fields = ['age', 'bmi', 'children']
        for field in numeric_fields:
            form_data[field] = float(form_data[field])

        input_df = pd.DataFrame([form_data])
        prediction = model_pipeline.predict(input_df)[0]
        prediction_text = f"${prediction:.2f}"
        return redirect(url_for('result', prediction_text=prediction_text))

    except ValueError:
        return "Error: Please Enter valid input", 400
    except Exception as e:
        return f"An internal error occurred: {e}"

@app.route('/result')
def result():
    prediction_text = request.args.get('prediction_text', 'No prediction made.')
    return render_template('result.html', prediction_text=prediction_text)

if __name__ == '__main__':
    app.run(debug=True)
