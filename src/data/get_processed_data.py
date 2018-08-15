import numpy as np
import pandas as pd
import os

def read_data():
    #set the path of the raw data
    raw_data_path=os.path.join(os.path.pardir,'data','raw')
    train_data_path=os.path.join(raw_data_path,'train.csv')
    test_data_path=os.path.join(raw_data_path,'test.csv')

    #read data file with all default calumns
    train_df=pd.read_csv(train_data_path,index_col='PassengerId')
    test_df=pd.read_csv(test_data_path,index_col='PassengerId')

    test_df['Survived']=-999 #default value
    df=pd.concat((train_df,test_df),axis=0)
    return df

def process_data(df):
    #using the method chaining concept
    return(df
           #create title first
           .assign(Title=lambda x:x.Name.map(GetTitle))
           #working with missing values
           .pipe(fill_missing_values)
           .assign(Fare_Bin=lambda x: pd.qcut(df.Fare,4,labels=['very_low','low','high','very_high']))
           #create AgeState
           .assign(AgeState=lambda x:np.where(x.Age>=18,'Adult','Child'))
           .assign(FamilySize=lambda x:x.Parch+x.SibSp+1)
           .assign(IsMother=lambda x:np.where(((x.Sex=='female') & (x.Parch>0) & (x.Age>=18) & (x.Title!='Miss')),1,0))
           #create deck feature
           .assign(Cabin=lambda x:np.where(x.Cabin=='T',np.nan,x.Cabin))
           .assign(Deck=lambda x: x.Cabin.map(GetDeck))
           #feature encoding
           .assign(Ismale=lambda x:np.where(x.Sex=='male',1,0))
           .pipe(pd.get_dummies,columns=['Deck','Pclass','Title','Fare_Bin','Embarked','AgeState'])
           #drop useless columns
           .drop(['Cabin','Name','Ticket','Parch','SibSp','Sex'],axis=1)
           #reorder columns
           .pipe(reorder_columns)
            )

def reorder_columns(df):
    columns=[column for column in df.columns if column!='Survived']
    columns=['Survived']+columns
    df=df[columns]
    return df

#extract first caracter of cabin
def GetDeck(Cabin):
    return np.where(pd.notnull(Cabin),str(Cabin)[0],'Z')


def fill_missing_values(df):
    #Embarked
    df.Embarked.fillna('C',inplace=True)
    #Fare
    median_fare=df[(df.Pclass==3) & (df.Embarked=='S')]['Fare'].median()
    df.Fare.fillna(median_fare,inplace=True)
    #Age
    title_age_median=df.groupby('Title').Age.transform('median') 
    df.Age.fillna(title_age_median,inplace=True)   
    return df
                                                        
                                                        
#function to extract title from name
def GetTitle(name):
    title_group={  'mr' :'Mr'
                 , 'mrs':'Mrs'
                 , 'miss':'Miss'
                 , 'master':'Master'
                 , 'don':'Sir'
                 , 'rev':'Sir'
                 , 'dr':'Officer'
                 , 'mme':'Mrs'
                 , 'ms':'Officer'
                 , 'major':'Officer'
                 , 'lady':'Lady'
                 , 'sir':'Sir'
                 , 'mlle':'Miss'
                 , 'col':'Officer'
                 , 'capt':'Officer'
                 , 'the countess':'Lady'
                 , 'jonkheer':'Sir'
                 , 'dona':'Lady'
                 }
    first_name_with_title=name.split(',')[1]
    title=first_name_with_title.split('.')[0]
    title=title.strip().lower()
    return title_group[title]

def write_data(df):
    processed_data_path=os.path.join(os.path.pardir,'data','processed')
    write_train_path=os.path.join(processed_data_path,'train.csv')
    write_test_path=os.path.join(processed_data_path,'test.csv')

    #train data
    df.loc[df.Survived!=-999].to_csv(write_train_path)
    #test data
    columns=[column for column in df.columns if column!='Survived']
    df.loc[df.Survived==-999,columns].to_csv(write_test_path)

if __name__=='__main__':
    df=read_data()
    df=process_data(df)
    write_data(df)