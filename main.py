from features import *

# create log and define it's name and format
log =logging.getLogger("qp_similarity")
log.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# create the logging file handler
fh = logging.FileHandler(logs + datetime.datetime.today().strftime('%m%d%y') + ".txt")
fh.setFormatter(formatter)
log.addHandler(fh)
log.info("Program started")

if __name__ == "__main__":
    log = logging.getLogger("qp_similarity.main")
    if os.path.exists("model/xgb_model1.pkl"):
        test_data = get_test_data()
        model = pickle.load(open('model/xgb_model1.pkl', 'rb'))
        log.info("Model loaded..")
        df = predict(test_data, model)
        df.to_csv("outputs/test_prediction.csv", index=False)
    else:
        train_data = get_train_data()
        train(train_data)
 