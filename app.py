from flask import Flask, render_template, request
import pickle
import pandas as pd

app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))
@app.route('/')
def Home():
    return render_template('index.html')

@app.route("/predict", methods=['GET','POST'])
def predict():
    if request.method == 'POST':
        gender = request.form['gender']
        if(gender == 'Male'):
            gender = 0
        else:
            gender = 1
        married = request.form['married']
        if (married == 'Single'):
            married = 0
        else:
            married = 1
        dependent = int(request.form['dependent'])
        education = request.form['education']
        if (education == 'Not Graduate'):
            education = 0
        else:
            education = 1
        employment = request.form['self_employed']
        if (employment == 'Not Self Employed'):
            employment = 0
        else:
            employment = 1
        applicant_income = request.form['ApplicantIncome']
        coappplicant_income = request.form['CoapplicantIncome']
        loan_amount = request.form['loan_amount']
        loan_term = request.form['loan_term']
        credit_history = request.form['credit_history']
        property_area = request.form['property_area']
        if property_area == 'Rural':
            property_area = 0
        elif property_area == 'Semi-Urban':
            property_area = 1
        else:
            property_area = 2
        df = pd.DataFrame([[gender,married,dependent,education,employment,applicant_income,coappplicant_income,loan_amount,loan_term,credit_history,property_area]])
        prediction = model.predict(df)
        if prediction == 0:
            return render_template('index.html', prediction_text="Sorry!!! Your Loan is REJECTED")
        else:
            return render_template('index.html', prediction_text="Congratulation!!! Your Loan is APPROVED")
    else:
        return render_template('index.html')

if __name__ == "__main__":
    app.run(debug=True)

