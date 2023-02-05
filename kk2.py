from flask import Flask,render_template,request,redirect
from flask_cors import CORS,cross_origin
import pickle
import pandas as pd
import numpy as np
import streamlit as st

app=Flask(__name__)
cors=CORS(app)
model=pickle.load(open('deployment_with_tuned_Bayesian_Ridge_1.pkl','rb'))
df=pd.read_csv('sample_test_data.csv')




@app.route('/',methods=['GET','POST'])
def index():
    car_models=sorted(df['name'].unique())
    km_driven = st.selectbox('km_driven',df['km_driven'].unique())
    fuel= st.selectbox('fuel',['Petrol' ,'Diesel' ,'CNG' ,'LPG', 'Electric'])
    seller_type= st.selectbox('seller_type',['Individual' ,'Dealer' ,'Trustmark Dealer'])
    transmission= st.selectbox('transmission',['Manual' ,'Automatic'])
    owner= st.selectbox('owner',['First Owner', 'Second Owner' ,'Third Owner', 'Fourth & Above Owner'])
    no_year= st.selectbox('no_year',df['no_year'].unique())
    car_models.insert(0,'name')
    return render_template('index.html',car_models=name,km_driven=km_driven, years=year,fuel_types=fuel,seller_type=seller_type,transmission=transmission,owner=owner)


@app.route('/predict',methods=['POST'])
@cross_origin()
def predict():

    
    car_model=request.form.get('name')
    year=request.form.get('year')
    fuel_type=request.form.get('fuel')
    driven=request.form.get('km_driven')
    seller_type=request.form.get('seller_type')
    transmission=request.form.get('transmission')
    owner=request.form.get('owner')

    prediction=model.predict(pd.DataFrame(columns=['name', 'seller_type', 'year', 'km_driven', 'fuel','transmission','owner'],
                              data=np.array([car_models,km_driven	,fuel	,seller_type,	transmission,	owner,	no_year]).reshape(1, 7)))
    print(prediction)

    return str(np.round(prediction[0],2))



if __name__=='__main__':
    app.run()





#streamlit run kk2.py

    

