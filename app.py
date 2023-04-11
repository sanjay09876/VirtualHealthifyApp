from flask import Flask, render_template, url_for, flash, redirect
import joblib
from flask import request
import numpy as np
import pickle

from predict_disease import predict_disease

app = Flask(__name__, template_folder='templates')

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/kidney")
def kidney():
    return render_template("kidney.html")

def ValuePredictor1(to_predict_list, size):
    to_predict = np.array(to_predict_list).reshape(1,size)
    if(size==7):
        loaded_model = joblib.load(r'/Users/sanjaykumar/Downloads/Health-App-main/Kidney_API/kidney_model.pkl')
        result = loaded_model.predict(to_predict)
    return result[0]

@app.route('/predictkidney', methods = ["POST"])
def predictkidney():
    if request.method == "POST":
        to_predict_list = request.form.to_dict()
        to_predict_list = list(to_predict_list.values())
        to_predict_list = list(map(float, to_predict_list))
         #diabetes
        if(len(to_predict_list)==7):
            result = ValuePredictor1(to_predict_list,7)
    
    if(int(result)==1):
        prediction = "You might have the possibility of kidney disease! Sorry. Please consult the doctor immediately"
    else:
        prediction = "No need to fear. You have no symptoms for the disease"
    return(render_template("result.html", prediction_text=prediction)) 

@app.route("/cancer")
def cancer():
    return render_template("cancer.html")

def ValuePredictor2(to_predict_list, size):
    to_predict = np.array(to_predict_list).reshape(1,size)
    if(size==5):
        loaded_model = joblib.load(r'/Users/sanjaykumar/Downloads/Health-App-main/Breast_Cancer API/cancer_model.pkl')
        result = loaded_model.predict(to_predict)
    return result[0]

@app.route('/predictcancer', methods = ["POST"])
def predictcancer():
    if request.method == "POST":
        to_predict_list = request.form.to_dict()
        to_predict_list = list(to_predict_list.values())
        to_predict_list = list(map(float, to_predict_list))
         #cancer
        if(len(to_predict_list)==5):
            result = ValuePredictor2(to_predict_list,5)
    
    if(int(result)==1):
        prediction = "You might have the possibility of cancer! Sorry. Please consult the doctor immediately"
    else:
        prediction = "No need to fear. You have no symptoms for the disease"
    return(render_template("result.html", prediction_text=prediction))  

@app.route("/liver")
def liver():
    return render_template("liver.html")

def ValuePredictor3(to_predict_list, size):
    to_predict = np.array(to_predict_list).reshape(1,size)
    if(size==7):
        loaded_model = joblib.load(r'/Users/sanjaykumar/Downloads/Health-App-main/Liver_API/liver_model.pkl')
        result = loaded_model.predict(to_predict)
    return result[0]

@app.route('/predictliver', methods = ["POST"])
def predictliver():
    if request.method == "POST":
        to_predict_list = request.form.to_dict()
        to_predict_list = list(to_predict_list.values())
        to_predict_list = list(map(float, to_predict_list))
         #liver
        if(len(to_predict_list)==7):
            result = ValuePredictor3(to_predict_list,7)
    
    if(int(result)==1):
        prediction = "You might have the possibility of liver disease! Sorry. Please consult the doctor immediately"
    else:
        prediction = "No need to fear. You have no symptoms for the disease"
    return(render_template("result.html", prediction_text=prediction))      

@app.route("/diabetes")
def diabetes():
    return render_template("diabetes.html")

def ValuePredictor4(to_predict_list, size):
    to_predict = np.array(to_predict_list).reshape(1,size)
    if(size==6):
        loaded_model = joblib.load(r'/Users/sanjaykumar/Downloads/Health-App-main/Diabetes_API/diabetes_model.pkl')
        result = loaded_model.predict(to_predict)
    return result[0]    

@app.route('/predictdiabetes', methods = ["POST"])
def predictdiabetes():
    if request.method == "POST":
        to_predict_list = request.form.to_dict()
        to_predict_list = list(to_predict_list.values())
        to_predict_list = list(map(float, to_predict_list))
         #diabetes
        if(len(to_predict_list)==6):
            result = ValuePredictor4(to_predict_list,6)
    
    if(int(result)==1):
        prediction = "You might have the possibility of diabetes! Sorry. Please consult the doctor immediately"
    else:
        prediction = "No need to fear. You have no symptoms for the disease"
    return(render_template("result.html", prediction_text=prediction))       

@app.route("/heart")
def heart():
    return render_template("heart.html")

def ValuePredictor5(to_predict_list, size):
    to_predict = np.array(to_predict_list).reshape(1,size)
    if(size==7):
        loaded_model = joblib.load(r'/Users/sanjaykumar/Downloads/Health-App-main/Heart_API/heart_model.pkl')
        result1 = loaded_model.predict(to_predict)
    return result1[0]

@app.route('/predictheart', methods = ["POST"])
def predictheart():
    if request.method == "POST":
        to_predict_list = request.form.to_dict()
        to_predict_list = list(to_predict_list.values())
        to_predict_list = list(map(float, to_predict_list))
         #diabetes
        if(len(to_predict_list)==7):
            result = ValuePredictor5(to_predict_list,7)
    
    if(int(result)==1):
        prediction = "You might have the possibility of heart disease! Sorry. Please consult the doctor immediately"
    else:
        prediction = "No need to fear. You have no symptoms for the disease"
    return(render_template("result.html", prediction_text=prediction))    

with open('/Users/sanjaykumar/Downloads/Health-App-main/Welness Hub API/symptoms.plk', 'rb') as f:
    symptoms = pickle.load(f)

with open('/Users/sanjaykumar/Downloads/Health-App-main/Welness Hub API/RandomForestClassifier.plk', 'rb') as f:
    rfc = pickle.load(f)

with open('/Users/sanjaykumar/Downloads/Health-App-main/Welness Hub API/precaution_dict.plk', 'rb') as f:
    precaution_dict = pickle.load(f)


@app.route('/wellness')
def wellness():
    return render_template('wellness.html', symptoms=symptoms)


@app.route('/predictwellness', methods=['POST'])
def predictwellness():
    user_input = list(request.form.values())

    # Remove empty strings if any
    while '' in user_input:
        user_input.remove('')

    return render_template('predict.html',  prediction=predict_disease(user_input, rfc), precaution=precaution_dict)


if __name__ == "__main__":
    from waitress import serve
    serve(app, host="0.0.0.0", port=8080)
