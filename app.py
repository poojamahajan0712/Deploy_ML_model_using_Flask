import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle
#import os

##os.chdir("D:/Pooja/Learning/Deployment_using_flask")

#render_template - helps to redirect to the home page 

app = Flask(__name__)  #initialise the flask app

#app = Flask(__name__, template_folder='D:/Pooja/Learning/Deployment_using_flask/templates')

model = pickle.load(open('model.pkl', 'rb')) #opening pickle file in read mode

#in flask we need to use @app to create any number of URIs
@app.route('/')       #root node direct to index.html
def home():
    return render_template('index.html')

#@app.route('/predict') -> will create API ->localhost address/predict
@app.route('/predict',methods=['POST'])
def predict():
    # for rendering results on html gui
    
    int_features = [int(x) for x in request.form.values()] # since it is post request so form.values 
    final_features = [np.array(int_features)]
    prediction = model.predict(final_features)

    output = round(prediction[0], 2)
    # prediction_text in html page should be replaced with below value of prediction_text
    
    return render_template('index.html', prediction_text='Sales should be $ {}'.format(output))
#
#@app.route('/results',methods=['POST'])
#def results():
#
#    data = request.get_json(force=True)
#    prediction = model.predict([np.array(list(data.values()))])
#
#    output = prediction[0]
#    return jsonify(output)

if __name__ == "__main__":
    app.run(debug=True)