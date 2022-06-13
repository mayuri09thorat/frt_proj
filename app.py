# importing the necessary dependencies
from flask import Flask, render_template, request
from flask_cors import CORS,cross_origin
import pickle

app = Flask(__name__) # initializing a flask app

@app.route('/',methods=['GET'])  # route to display the home page
@cross_origin()
def homePage():
    return render_template("index.html",static_url_path='/static')

@app.route('/predict',methods=['POST']) # route to show the predictions in a web UI
@cross_origin()
def predict():
        try:
            #  reading the inputs given by the user
            Pregnancies=int(request.form['Pregnancies'])
            Glucose = float(request.form['Glucose'])
            BloodPressure = float(request.form['BloodPressure'])
            SkinThickness = float(request.form['SkinThickness'])
            Insulin = float(request.form['Insulin'])
            BMI = float(request.form['BMI'])
            DiabetesPedigreeFunction = request.form['DiabetesPedigreeFunction']
            Age = int(request.form['Age'])
            filename = 'model.pkl'
            loaded_model = pickle.load(open(filename,'rb')) # loading the model file from the storage
            # predictions using the loaded model file
            prediction=loaded_model.predict_proba([[Pregnancies,Glucose,BloodPressure,SkinThickness,Insulin,BMI,DiabetesPedigreeFunction,Age]])
            # showing the prediction results in a UI
            return render_template('predict.html',prediction=round(100*prediction[0][1]))
        except Exception as e:
            return('The Exception message is: ',e)
            #return 'something is wrong'

if __name__ == "__main__":
    app.run(debug=True) # running the app