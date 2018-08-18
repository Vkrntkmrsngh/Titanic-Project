
from flask import Flask,request
import pickle
import numpy as np
import pandas as pd
import os
import json

app=Flask(__name__)


#load modal and scaler files
model_path=os.path.join(os.path.pardir,os.path.pardir,'models')
model_file_path=os.path.join(model_path,'model1.pkl')
scaler_file_path=os.path.join(model_path,'scaler1.pkl')

# model = pickle.load(f)

#with open(scaler_file_path, 'rb') as f:
#    scaler = pickle.load(f)
model=pickle.load(open(model_file_path,'rb'))
   

columns=[u'Age', u'Fare', u'FamilySize', u'IsMother', u'Ismale', u'Deck_A',\
       u'Deck_B', u'Deck_C', u'Deck_D', u'Deck_E', u'Deck_F', u'Deck_G', u'Deck_Z',\
       u'Pclass_1', u'Pclass_2', u'Pclass_3', u'Title_Lady', u'Title_Master',\
       u'Title_Miss', u'Title_Mr', u'Title_Mrs', u'Title_Officer', u'Title_Sir',\
       u'Fare_Bin_very_low', u'Fare_Bin_low', u'Fare_Bin_high',\
       u'Fare_Bin_very_high', u'Embarked_C', u'Embarked_Q', u'Embarked_S',\
       u'AgeState_Adult', u'AgeState_Child']

@app.route('/api',methods=['POST'])

def make_prediction():
    #read json and convert into json string
    data1=json.dumps(request.get_json(True))
    #create panda dataframe
    df=pd.read_json(data1)
    #extract passengerIds
    passenger_ids=df['PassengerId'].ravel()
    #actual survived values
    actuals=df['Survived'].ravel()
    #extract required columns and convert to matrix
    X=df[columns].as_matrix().astype('float')
    #transform the input
    #X_scaled=scaler.transform(X)
    #make predictions
    predictions=model.predict(X)
    #create response dataframe
    df_response=pd.DataFrame({'PassengerId':passenger_ids,'Predicted':predictions,'Actuals':actuals})
    #return json
    return df_response.to_json()

if __name__=='__main__':
    #host flask app at port 10001
    app.run(port=10002,debug=True)