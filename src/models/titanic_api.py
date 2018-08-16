
from flask import Flask,request
import pickle
import numpy as np
import pandas as pd
import os
import json

app=Flask(__name__)


#load modal and scaler files
model_path=os.path.join(os.path.pardir,'models')
model_file_path=os.path.join(model_path,'lr_model.pkl')
scaler_file_path=os.path.join(model_path,'lr_scalar.pkl')

#open files in read mode
model_file_pickle=open(model_file_path,'rb')
scaler_file_pickle=open(scaler_file_path,'rb')

scaler=pickle.load(model_file_pickle)
model=pickle.load(scaler_file_pickle)

model_file_pickle.close()
scaler_file_pickle.close()

columns=['Age', 'Fare', 'FamilySize', 'IsMother', 'Ismale', 'Deck_A',\
       'Deck_B', 'Deck_C', 'Deck_D', 'Deck_E', 'Deck_F', 'Deck_G', 'Deck_Z',\
       'Pclass_1', 'Pclass_2', 'Pclass_3', 'Title_Lady', 'Title_Master',\
       'Title_Miss', 'Title_Mr', 'Title_Mrs', 'Title_Officer', 'Title_Sir',\
       'Fare_Bin_very_low', 'Fare_Bin_low', 'Fare_Bin_high',\
       'Fare_Bin_very_high', 'Embarked_C', 'Embarked_Q', 'Embarked_S',\
       'AgeState_Adult', 'AgeState_Child']

@app.route('\api',methods=['POST'])

def make_prediction():
    #read json and convert into json string
    data=json.dumps(request.get_json(force=True))
    #create panda dataframe
    df=pd.read_json(data)
    #extract passengerIds
    passenger_ids=df['PassengerId'].ravel()
    #actual survived values
    actuals=df['Survived'].ravel()
    #extract required columns and convert to matrix
    X=df[columns].as_matrix.astype('float')
    #transform the input
    X_scaled=scaler.transform(X)
    #make predictions
    predictions=model.predict(X_scaled)
    #create response dataframe
    df_response=pd.DataFrame({'PassengerId':passenger_ids,'Predicted':predictions,'Actuals':actuals})
    #return json
    return df_response.to_json()

if __name__=='__main__':
    #host flask app at port 10001
    app.run(port=10001,debug=True)