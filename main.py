#import libraries
import numpy as np
from flask import Flask, render_template, request
import pickle  # Initialize the flask App
app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))

# default page of our web-app


@app.route('/')
def home():
    return render_template('index.html')

# To use the predict button in our web-app


@app.route('/', methods=['POST'])
def predict():
    # For rendering results on HTML GUI
    str_features = [str(X) for X in request.form.values()]
    int_features = []

    for i in str_features:
        if i != "":
            int_features.append(float(i))
        elif i == "":
            int_features.append(float(i+"0"))

    final_features = [np.array(int_features)]
    prediction = model.predict(final_features)

    if 0 in int_features:
        return render_template('index.html', prediction_text='Please Fill in All Fields')
    else:
        return render_template('index.html', prediction_text='Rp {}'.format("%.0f" % prediction))


if __name__ == "__main__":
    app.run(debug=True)
