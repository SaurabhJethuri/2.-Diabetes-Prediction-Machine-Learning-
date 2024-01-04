from flask import Flask, render_template, request
from flask_pymongo import PyMongo
import numpy as np
import pickle


app = Flask(__name__)
app.config["MONGO_URI"] = "mongodb://localhost:27017/diabetesDB"
mongo = PyMongo(app)
# Load the trained model
loaded_model = pickle.load(open("trained_model.sav", 'rb'))


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/', methods=['POST'])
def prediction():
        username = request.form['username']
        pregnancies = float(request.form['pregnancies'])
        glucose = float(request.form['glucose'])
        blood_pressure = float(request.form['bloodpressure'])
        skin_thickness = float(request.form['skinthickness'])
        insulin = float(request.form['insulin'])
        age = float(request.form['age'])
        bmi = float(request.form['bmi'])
        diabetes_pedigree_function = float(request.form['diabetespedigreefunction'])
        name=username
        data=[pregnancies,glucose,blood_pressure,skin_thickness,insulin,age,bmi,diabetes_pedigree_function]
        input_data_as_numpy_array = np.asarray(data)

        # Reshape the array as we are predicting for one instance
        input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)
        prediction = loaded_model.predict(input_data_reshaped)
        if prediction[0] == 0:
            mongo.db.userData.insert_one({"name":name,"report":'That person does not have diabetes'})
            result = "That person does not have diabetes"
        else:
           mongo.db.userData.insert_one({"name":name,"report":'That person has diabetes'})
           result = "That person has diabetes"
 
        return render_template('index.html',res=result)
     

if __name__ == '__main__':
    app.run(debug=True)



