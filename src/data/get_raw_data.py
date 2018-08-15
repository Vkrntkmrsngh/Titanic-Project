
import os
from dotenv import find_dotenv,load_dotenv
from requests import session
import logging

#payload for login to kaggle

payload={'action':'Login',
         'username':os.getenv("KAGGLE_USERNAME"),
         'password':os.getenv("KAGGLE_PASSWORD")
        }

#extract data file

def extract_data(url,file_path):
    '''
    extract data from source website
    '''
    #setup session
    with session() as c:
        c.post('https://www.kaggle.com/account/login',data=payload)
        #open file to write
        with open(file_path,'w') as handle:
            response=c.get(url,stream=True)
            for block in response.iter_content(1024):
                handle.write(str(block))

def main(project_dir):
    '''
    main method
    '''
    #get logger
    logger=logging.getLogger(__name__)
    logger.info('getting raw data')
    logger.info(project_dir)
    
    #urls
    train_url='https://www.kaggle.com/c/titanic/download/train.csv'
    test_url='https://www.kaggle.com/c/titanic/download/test.csv'

    #file path
    raw_data_path=os.path.join(project_dir,'data','raw')
    train_data_path=os.path.join(raw_data_path,'train.csv')
    test_data_path=os.path.join(raw_data_path,'test.csv')
    logger.info(train_data_path)
    logger.info(test_data_path)
    logger.info('passed path')
    extract_data(train_url,train_data_path)
    extract_data(test_url,test_data_path)
    logger.info('downloaded raw train and test data')
    
if __name__=='__main__':
    #getting root dir
    project_dir=os.path.join(os.path.dirname(__file__),os.pardir,os.pardir)
    
    
    #setup logger
    log_fmt='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO,format=log_fmt)
    
    #to find env file in upward direction untill it got found
    dotenv_path=find_dotenv()
    #to load path into load_dotenv
    load_dotenv(dotenv_path)

    #call the main
    main(project_dir)
    