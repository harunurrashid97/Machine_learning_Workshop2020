from flask import Flask,request, url_for, redirect, render_template
import numpy as np
import pickle

app = Flask(__name__)
model = pickle.load(open('knn_model.pkl','rb'))


@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    #x = {1: 'apple', 2: 'mandarin', 3: 'orange', 4: 'lemon'}
    int_features = [float(x) for x in request.form.values()]
    final_features = [np.array(int_features)]
    prediction = model.predict(final_features)

    output = round(prediction[0], 2)

    return render_template('result.html',prediction_text = output)

if __name__ == "__main__":
    app.run(debug=True)
