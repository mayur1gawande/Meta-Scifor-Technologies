from flask import Flask, render_template, request, redirect, url_for
import pandas as pd
import joblib
import warnings
warnings.filterwarnings("ignore")

app = Flask(__name__)

try:
    model_pipeline = joblib.load('loan_prediction_pipeline.joblib')
except FileNotFoundError:
    print("Error: 'loan_prediction_pipeline.joblib' not found.")
    model_pipeline = None

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if model_pipeline is None:
        return "Error: Model not loaded. Please check the console for details."

    form_data = request.form.to_dict()

    try:
        form_data['ApplicantIncome'] = float(form_data['ApplicantIncome'])
        form_data['CoapplicantIncome'] = float(form_data['CoapplicantIncome'])
        form_data['LoanAmount'] = float(form_data['LoanAmount'])
        form_data['Loan_Amount_Term'] = float(form_data['Loan_Amount_Term'])
        form_data['Credit_History'] = float(form_data['Credit_History'])
    except ValueError:
        return "Error: Please enter valid Details."

    input_df = pd.DataFrame([form_data])
    prediction = model_pipeline.predict(input_df)[0]
    if prediction==1:
        text = 'Loan Approved (Y)'
    else:
        text = 'Loan Rejected (N)'
    return redirect(url_for('result', prediction=text))

@app.route('/result')
def result():
    prediction = request.args.get('prediction')
    is_approved = 'Approved' in prediction
    return render_template('result.html', prediction_text=prediction, is_approved=is_approved)

if __name__ == "__main__":
    app.run(debug=False)
