from includes import *
from utilities import Utilities as ut
class Predictor:
    def __init__(self, path):
        self.finalised_model = self.get_saved_model_data()
        self.outpath = path
        self.y_pred_1 = pd.DataFrame()
        self.y_pred_5 = pd.DataFrame()

    def get_saved_model_data(self):
        y_cols = ['1-day-predict', '5-day-predict']
        finalised_model = {}
        for i in y_cols:
            filename = "./model/final_feat_imp.pickle"
            with open(filename, 'rb') as handle:
                finalised_model = pickle.load(handle)
        return finalised_model

    def predict_validation_data(self, X_val, X_cols,model_name):
        """ This function will call the predict model for  validation data"""
        y_cols = ['1-day-predict', '5-day-predict']
        y_pred = {}
        for i in y_cols:
            #X_cols = self.finalised_model[i]['imp_feat']
            X = X_val[X_cols]
            # if validation:
            y = pd.DataFrame(X_val[i])
            y = y.to_numpy()
            print("X--Shape", X.shape)
            print("y--Shape", y.shape)
            #model_name = self.finalised_model[i]['model'].item()
            filename = model_name + i + ".sav"
            path = "./model/" + filename
            print("path",path)
            loaded_model = pickle.load(open(path, 'rb'))
            index = i+"model"
            print("Indexname",index)
            y_pred[index] = loaded_model.predict(X)
            print('\n=====Validation Data ==========================: \n')
            print('\n=====Confusion Matrix==========: \n')
            print(confusion_matrix(y, y_pred[index]))
            print('\n==== Classification Report=====: \n')
            print(classification_report(y, y_pred[index], target_names=['False', 'True'], digits=4))

            X_val[index] = pd.DataFrame(y_pred[index], columns=[index])
        path_prefix = "./" + self.outpath + "/"
        filename = path_prefix + "X_val" + '.tsv'
        X_val.to_csv(filename, sep='\t')

    def predict_n_fill_test_data(self, X_test,X_cols,model):
        # y_cols = ['1-day-predict', '5-day-predict']
        print("predict_n_fill_test_data")
        y_cols = ['1-day-predict', '5-day-predict']
        y_pred = {}
        for i in y_cols:
            #X_cols = self.finalised_model[i]['imp_feat']
            X = X_test[X_cols]
            print("X--Shape", X.shape)
            #model_name = self.finalised_model[i]['model'].item()
            model_name = model
            filename = model_name + i + ".sav"
            path = "./model/" + filename
            loaded_model = pickle.load(open(path, 'rb'))
            loaded_model.verbose = False
            y_pred[i] = loaded_model.predict(X)

        X_test.drop(columns=['1-day-predict', '5-day-predict'], axis=1, inplace=True)
        df1 = pd.DataFrame(data=y_pred['1-day-predict'], columns=['1-day-predict'], index=X_test.index.copy())
        df_out = pd.merge(X_test, df1, how='left', left_index=True, right_index=True)
        df2 = pd.DataFrame(data=y_pred['5-day-predict'], columns=['5-day-predict'], index=X_test.index.copy())
        df_out = pd.merge(df_out, df2, how='left', left_index=True, right_index=True)
        self.y_pred_1 = df1
        self.y_pred_5 = df2
        final_df = df_out[['datetime', 'site_id', 'mlid', 'rlf', '1-day-predict', '5-day-predict']]
        path_prefix = "./" + self.outpath + "/" + self.outpath.split('_', 2)[1]
        filepath = path_prefix + "_predicts.tsv"
        rlf_pred = ut.read_data_to_df(filepath)
        rlf_pred.drop(columns=['1-day-predict', '5-day-predict'], axis=1, inplace=True)
        rlf_pred['datetime'] = pd.to_datetime(rlf_pred['datetime'])
        rld_pred_final = pd.merge(rlf_pred, final_df, on=["datetime", "site_id", "mlid"], how='left')
        outpath = path_prefix + "_predicts_updated.tsv"
        rld_pred_final.to_csv(outpath, sep='\t')
