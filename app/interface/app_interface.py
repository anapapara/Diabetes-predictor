import numpy as np
import pandas as pd
from flask import Flask, request, render_template
import pickle
import random as r
import os 

app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html',prediction_text=None, error_message=None)

@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    features = [x for x in request.form.values()]

    error_message = None
    final_features =[] 
    try:
        final_features.append(float(features[0]))
        final_features.append(float(features[1]))
        final_features.append(float(features[2]))
        #walk diff
        match features[4]:
            case 'yes':
                final_features.append(1.0)
            case 'no':
                final_features.append(2.0)
            case 'dn':
                final_features.append(7.0)
            case _:
                error_message = "Please select an option for Walk difficulty."

        #depression
        match features[3]:
            case 'yes':
                final_features.append(1.0)
            case 'no':
                final_features.append(2.0)
            case 'dn':
                final_features.append(7.0)
            case _:
                error_message = "Please select an option for Depressive disorder."

    except ValueError:
        error_message = "Invalid numeric values"

    if float(features[0]) <= 0:
        error_message = "Weight must have positive value."
    elif float(features[1]) <= 0:
        error_message = "Height must have positive value."
    elif float(features[2]) < 0 or float(features[2]) > 24:
        error_message = "Sleep Time must be between 0 and 24 hours."

    if error_message:
        return render_template('index.html', error_message=error_message)

    print("the featrures are:", final_features)
    final_features = np.array(final_features).reshape(1, -1)
    prediction = model.predict(final_features)
    print(f"the prediction is: {prediction}")

    if prediction==1.0:
        pred="DIABETIC, you should see a doctor!"
    elif prediction==2.0:
        pred="NON-DIABETIC, congrats! :)"
    else:
        pred="not clear"

    return render_template('index.html', prediction_text=f"{pred}")
    



if __name__ == "__main__":
    # app.run(debug=True)
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)