import re, pickle, datetime
import pandas as pd
from nltk.corpus import stopwords
# import distance
import nltk
from nltk.stem import PorterStemmer
from bs4 import BeautifulSoup
nltk.download('stopwords')
from fuzzywuzzy import fuzz
from sklearn.model_selection import train_test_split
from sklearn.metrics import log_loss
import os
import logging
import xgboost as xgb
import numpy as np

if not os.path.exists("model"):
    os.makedirs("model")
if not os.path.exists("logs"):
    os.makedirs("logs")
if not os.path.exists("outputs"):
    os.makedirs("outputs")

logs = "logs/"
log = logging.getLogger("qp_similarity.features")

def get_train_data():
    return pd.read_csv("inputs/train.csv")

def get_test_data():
    return pd.read_csv("inputs/test.csv")

SAFE_DIV = 0.0001 
STOP_WORDS = stopwords.words("english")

def preprocess(x):
    x = str(x).lower()
    x = x.replace(",000,000", "m").replace(",000", "k").replace("′", "'").replace("’", "'")\
                           .replace("won't", "will not").replace("cannot", "can not").replace("can't", "can not")\
                           .replace("n't", " not").replace("what's", "what is").replace("it's", "it is")\
                           .replace("'ve", " have").replace("i'm", "i am").replace("'re", " are")\
                           .replace("he's", "he is").replace("she's", "she is").replace("'s", " own")\
                           .replace("%", " percent ").replace("₹", " rupee ").replace("$", " dollar ")\
                           .replace("€", " euro ").replace("'ll", " will")
    x = re.sub(r"([0-9]+)000000", r"\1m", x)
    x = re.sub(r"([0-9]+)000", r"\1k", x)  
    porter = PorterStemmer()
    pattern = re.compile('\W')
    if type(x) == type(''):
        x = re.sub(pattern, ' ', x)
    if type(x) == type(''):
        x = porter.stem(x)
        example1 = BeautifulSoup(x)
        x = example1.get_text()
    return x
    
def normalized_word_Common(row):
    w1 = set(map(lambda word: word.lower().strip(), str(row['question1']).split(" ")))
    w2 = set(map(lambda word: word.lower().strip(), str(row['question2']).split(" ")))    
    return 1.0 * len(w1 & w2)

def normalized_word_Total(row):
    w1 = set(map(lambda word: word.lower().strip(), str(row['question1']).split(" ")))
    w2 = set(map(lambda word: word.lower().strip(), str(row['question2']).split(" ")))    
    return 1.0 * (len(w1) + len(w2))

def normalized_word_share(row):
    w1 = set(map(lambda word: word.lower().strip(), str(row['question1']).split(" ")))
    w2 = set(map(lambda word: word.lower().strip(), str(row['question2']).split(" ")))    
    return 1.0 * len(w1 & w2)/(len(w1) + len(w2))


def get_token_features(q1, q2):
    token_features = [0.0]*10
    # Converting the Sentence into Tokens: 
    q1_tokens = q1.split()
    q2_tokens = q2.split()
    if len(q1_tokens) == 0 or len(q2_tokens) == 0:
        return token_features
    # Get the non-stopwords in Questions
    q1_words = set([word for word in q1_tokens if word not in STOP_WORDS])
    q2_words = set([word for word in q2_tokens if word not in STOP_WORDS])
    
    #Get the stopwords in Questions
    q1_stops = set([word for word in q1_tokens if word in STOP_WORDS])
    q2_stops = set([word for word in q2_tokens if word in STOP_WORDS])
    
    # Get the common non-stopwords from Question pair
    common_word_count = len(q1_words.intersection(q2_words))
    
    # Get the common stopwords from Question pair
    common_stop_count = len(q1_stops.intersection(q2_stops))
    
    # Get the common Tokens from Question pair
    common_token_count = len(set(q1_tokens).intersection(set(q2_tokens)))
    
    token_features[0] = common_word_count / (min(len(q1_words), len(q2_words)) + SAFE_DIV)
    token_features[1] = common_word_count / (max(len(q1_words), len(q2_words)) + SAFE_DIV)
    token_features[2] = common_stop_count / (min(len(q1_stops), len(q2_stops)) + SAFE_DIV)
    token_features[3] = common_stop_count / (max(len(q1_stops), len(q2_stops)) + SAFE_DIV)
    token_features[4] = common_token_count / (min(len(q1_tokens), len(q2_tokens)) + SAFE_DIV)
    token_features[5] = common_token_count / (max(len(q1_tokens), len(q2_tokens)) + SAFE_DIV)
    
    # Last word of both question is same or not
    token_features[6] = int(q1_tokens[-1] == q2_tokens[-1])
    
    # First word of both question is same or not
    token_features[7] = int(q1_tokens[0] == q2_tokens[0])
    
    token_features[8] = abs(len(q1_tokens) - len(q2_tokens))
    
    #Average Token Length of both Questions
    token_features[9] = (len(q1_tokens) + len(q2_tokens))/2
    return token_features

def extract_features(qp_df):
    # preprocessing each question
    log = logging.getLogger("qp_similarity.features.extract_features")
    log.info("Extracting features")
    qp_df["question1"] = qp_df["question1"].fillna("").apply(preprocess)
    qp_df["question2"] = qp_df["question2"].fillna("").apply(preprocess)
    qp_df['q1len']         = qp_df['question1'].str.len() 
    qp_df['q2len']         = qp_df['question2'].str.len()
    qp_df['q1_n_words']    = qp_df['question1'].apply(lambda row: len(str(row).split(" ")))
    qp_df['q2_n_words']    = qp_df['question2'].apply(lambda row: len(str(row).split(" ")))
    qp_df['word_Common']   = qp_df.apply(normalized_word_Common, axis=1)
    qp_df['word_Total']    = qp_df.apply(normalized_word_Total, axis=1)
    qp_df['word_share']    = qp_df.apply(normalized_word_share, axis=1)
    log.info("Computing token features...")
    token_features = qp_df.apply(lambda x: get_token_features(x["question1"],
                                                         x["question2"]), axis=1)
    qp_df["cwc_min"]       = list(map(lambda x: x[0], token_features))
    qp_df["cwc_max"]       = list(map(lambda x: x[1], token_features))
    qp_df["csc_min"]       = list(map(lambda x: x[2], token_features))
    qp_df["csc_max"]       = list(map(lambda x: x[3], token_features))
    qp_df["ctc_min"]       = list(map(lambda x: x[4], token_features))
    qp_df["ctc_max"]       = list(map(lambda x: x[5], token_features))
    qp_df["last_word_eq"]  = list(map(lambda x: x[6], token_features))
    qp_df["first_word_eq"] = list(map(lambda x: x[7], token_features))
    qp_df["abs_len_diff"]  = list(map(lambda x: x[8], token_features))
    qp_df["mean_len"]      = list(map(lambda x: x[9], token_features))
    log.info("Computing fuzzy features")
    qp_df["token_set_ratio"]       = qp_df.apply(lambda x: fuzz.token_set_ratio(
                                                    x["question1"], x["question2"]), axis=1)
    qp_df["token_sort_ratio"]      = qp_df.apply(lambda x: fuzz.token_sort_ratio(
                                                    x["question1"], x["question2"]), axis=1)
    qp_df["fuzz_ratio"]            = qp_df.apply(lambda x: fuzz.QRatio(
                                                    x["question1"], x["question2"]), axis=1)
    qp_df["fuzz_partial_ratio"]    = qp_df.apply(lambda x: fuzz.partial_ratio(
                                                    x["question1"], x["question2"]), axis=1)
    return qp_df

def train(df):
    log = logging.getLogger("qp_similarity.features.train")    
    qp_df = extract_features(df)
    y_true = qp_df['is_duplicate']
    qp_df.drop(['id','qid1', 'qid2', 'question1', 'question2','is_duplicate'],  
                                                            axis=1, inplace=True)
    X_train, X_test, y_train, y_test = train_test_split(qp_df, y_true, 
                                       stratify=y_true, test_size=0.3,random_state=43)    
    log.info("X_train shape: {}, X_test shape:{}".format(X_train.shape, X_test.shape))
    clf = xgb.XGBClassifier(max_depth=3,learning_rate=0.02,n_estimators=400,n_jobs=-1)
    log.info("Training")
    clf.fit(X_train,y_train)    
    pickle.dump(clf,open("model/xgb_model1.pkl","wb"))
    predict_y = clf.predict_proba(X_test)
    log.info("The log loss is:",log_loss(y_test, predict_y, eps=1e-15))

def predict(df, clf):
    qp_df = df[['question1', 'question2']]
    qp_df = extract_features(qp_df)    
    qp_df.drop(['question1', 'question2'], axis=1, inplace=True)
    predict_y = clf.predict_proba(qp_df)
    predicted_y =np.argmax(predict_y,axis=1)
    df['duplicate_pred'] = predicted_y
    return df

def app_predict(df, clf):
    log = logging.getLogger("qp_similarity.features.app_predict")        
    qp_df = df[['question1', 'question2']]
    qp_df = extract_features(qp_df)  
    qp_df.drop(['question1', 'question2'], axis=1, inplace=True)
    log.info("Predict is_duplicate")
    predict_y = clf.predict_proba(qp_df)
    predicted_y =np.argmax(predict_y,axis=1)
    return predicted_y
  