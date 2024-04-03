import joblib
from flask import Flask, render_template, request
import sklearn

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('Home_1.html')

@app.route('/Predict')
def prediction():
    return render_template('Index.html')

@app.route('/form', methods=["POST"])
def brain():
    # Extracting data from the form
    Nitrogen = float(request.form['Nitrogen'])
    Phosphorus = float(request.form['Phosphorus'])
    Potassium = float(request.form['Potassium'])
    Temperature = float(request.form['Temperature'])
    Humidity = float(request.form['Humidity'])
    Ph = float(request.form['ph'])
    Rainfall = float(request.form['Rainfall'])

    # Constructing input array
    values = [Nitrogen, Phosphorus, Potassium, Temperature, Humidity, Ph, Rainfall]

    # Loading the trained model
    model = joblib.load('Pickle_RL_Model.pkl')  # Corrected file extension to .pkl

    # Making prediction
    acc = model.predict([values])

    # Returning prediction
    return render_template('prediction.html', prediction=str(acc))

if __name__ == '__main__':
    app.run(debug=True)
