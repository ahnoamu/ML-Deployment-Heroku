import numpy as np
from flask import Flask, request, jsonify, render_template, url_for
import pickle

app = Flask(__name__)
model = pickle.load(open('reg_model.pkl', 'rb'))

@app.route('/')  #Decorator to tell flask what URL should trigger the web functions
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    int_features = [int(x) for x in request.form.values()]  #forming list of submitted values
    final_features = [np.array(int_features)]   #converting the list to numpy array
    prediction = model.predict(final_features)  #predicting

    output = round(prediction[0], 2)

    return render_template('index.html', prediction_text='Global videogame sale should be $ {}'.format(output))

#driver function
if __name__ == "__main__":
    app.run(debug=True)
