from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

# Load the model

with open('model.pkl', 'rb') as file:
    model = pickle.load(file)

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/submit', methods=['POST'])
def submit():
    if request.method == 'POST':
        sepal_length = float(request.form['sepal_length'])
        sepal_width = float(request.form['sepal_width'])
        petal_length = float(request.form['petal_length'])
        petal_width = float(request.form['petal_width'])
        
        # Prepare the feature vector for prediction
        features = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
        
        # Make the prediction
        prediction = model.predict(features)
        result = prediction[0]
        
        return render_template('submit.html', result=result)

if __name__ == '__main__':
    app.run(debug=True)
