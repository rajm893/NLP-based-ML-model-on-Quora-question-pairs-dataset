import uvicorn
from fastapi import FastAPI 
import pickle
from feature_metadata import FeatureMeta
from features import *

app = FastAPI()
model = pickle.load(open("./model/xgb_model1.pkl", "rb"))

@app.get('/')
def index():
    return "Question pair similarity"   

@app.post('/qp_similarity')
def predict_score(data:FeatureMeta):
    '''
        Predict if the two questions are duplicate of each other    
        1 : is a duplicate question
        0 : not a duplicate question
    '''
    log = logging.getLogger("qp_similarity.predict_score")
    data = data.dict()
    test_data = pd.DataFrame({'question1': [data['question1']], 'question2': [data['question1']]})
    res = app_predict(test_data, model)    
    log.info("predicted result")
    output = {'is_duplicate': str(res[0])}
    return output


if __name__ == "__main__":
    uvicorn.run(app, host='0.0.0.0', port=8212)