from includes import *

class Model:
    def __init__(self):
        self.model = [0] * 5
        # distances df
        self.dict_of_metricsdf = {}
        self.dict_of_metrics_meandf = {}
        self.bootstrap_df = ""
        self.results_long = ""
        self.finalised_model = {}
        self.feature_imp = {}

    def train(self, final_df, X_cols):
        """  This function will train the model """


        imp_feature_df_final = final_df
        y_cols = ['1-day-predict', '5-day-predict']
        for i in y_cols:
            X = imp_feature_df_final[X_cols]
            y = pd.DataFrame(imp_feature_df_final[i])
            print("X--Shape", X.shape)
            print("y--Shape", y.shape)

            X, y = self.rover_sample_data(X, y)
            print("After ROS")
            print("X--Shape", X.shape)
            print("y--Shape", y.shape)
            X_train, X_test, y_train, y_test = self.split_train_test(X, y)
            key_name = i
            print("key_name==>", key_name)
            self.dict_of_metricsdf[key_name] = self.run_models(X_train, y_train, X_test, y_test)
            self.evaluate_model(key_name,X_test)
            #self.finalise_model_impfeat(key_name,X_train, y_train, X_test, y_test,X_cols)
            self.save_feat_imp()

    def save_feat_imp(self):
        filename = "./model/final_feat_imp.pickle"
        with open(filename, 'wb') as handle:
            pickle.dump(self.finalised_model, handle, protocol=pickle.HIGHEST_PROTOCOL)

    def evaluate_model(self, key_name,X_test):
        metrics = list(set(self.dict_of_metricsdf[key_name].columns))
        metrics_df = self.dict_of_metricsdf[key_name]
        self.dict_of_metrics_meandf[key_name] = \
            metrics_df.groupby(['model'])[metrics].agg([np.std, np.mean]).sort_values(
                by=[('test_accuracy', 'mean')], ascending=False)
        self.dict_of_metrics_meandf[key_name].reset_index(inplace=True)
        print("Model with best test accuracy is ", self.dict_of_metrics_meandf[key_name].loc[0,'model'])
        self.finalised_model[key_name] = self.dict_of_metrics_meandf[key_name].loc[0,'model'].astype('str')

        feat_imp = self.dict_of_metricsdf[key_name]['estimator'][0].feature_importances_
        print("feat_imp::",key_name, feat_imp)
        features_cols = X_test.columns
        self.feature_imp[key_name] = pd.DataFrame({'features': features_cols,
                                    'importance_percent': feat_imp * 100}).sort_values(
            by='importance_percent', ascending=False).reset_index(drop=True)
        feature_imp = self.feature_imp[key_name]
        # drop the features that are not present in the training data
        list_imp_feat = list(feature_imp['features'])
        feat_drop = ['wind_dir_max','wind_speed_max']
        for feat in feat_drop:
            if feat in list_imp_feat:
                list_imp_feat.remove(feat)
        imp_feat = list_imp_feat[:40]
        #if feature_imp['features']
        #imp_feat = list(feature_imp['features'].head(40))

        model_name = self.finalised_model[key_name].item()
        self.finalised_model[key_name] = {"model": self.finalised_model[key_name],
                                                  "imp_feat": imp_feat}
        metrics_df.drop("estimator",axis=1,inplace=True)
        self.create_resultsdf_for_analysis(metrics_df)

        self.plot_time_metrics(key_name)
        self.plot_perf_metrics(key_name)


    def finalise_model_impfeat(self,key_name, X_train, y_train, X_test, y_test,X_cols):
        print("finalise_model")
        #X_cols = self.finalised_model[key_name]['imp_feat']
        model_name = self.finalised_model[key_name]['model'].item()
        filename = "./model/" + model_name + key_name + '.sav'
        y_cols = ['1-day-predict', '5-day-predict']
        X_test = X_test[X_cols]
        X_train = X_train[X_cols]
        if model_name == "RF":
            model = RandomForestClassifier(n_estimators=150,
                                          #max_features=40,
                                          oob_score=True,
                                          verbose=1,
                                          min_samples_split=10,
                                          max_depth=75)
        elif model_name == "XGB":
            model = XGBClassifier(max_depth=9,
                                  subsample=0.9,
                                  min_child_weight=2,
                                  colsample_bytree=0.7,
                                  n_estimators=100,
                                  learning_rate=0.08,
                                  n_jobs=-1)

        clf = model.fit(X_train, y_train)
        clf.verbose = False
        print("Model", filename)
        pickle.dump(clf, open(filename, 'wb'))  # save model file for future use
        pred = clf.predict(X_test)  # Predicting the test data
        print('\nConfusion Matrix: \n')
        print(confusion_matrix(y_test, pred))
        print('\n Classification Report: \n')
        print(classification_report(y_test, pred, target_names=['False', 'True'], digits=4))




    def create_resultsdf_for_analysis(self,metrics_df):
        bootstraps = []
        for model in list(set(metrics_df.model.values)):
            model_df = metrics_df.loc[metrics_df.model == model]
            bootstrap = model_df.sample(n=30, replace=True)
            bootstraps.append(bootstrap)
        self.bootstrap_df = pd.concat(bootstraps, ignore_index=True)
        self.results_long = pd.melt(self.bootstrap_df, id_vars=['model'], var_name='metrics', value_name='values')

    def plot_time_metrics(self,key_name):
        ## TIME METRICS
        time_metrics = ['fit_time', 'score_time']  # fit time metrics
        results_long_fit = self.results_long.loc[self.results_long['metrics'].isin(time_metrics)]  # df with fit data
        results_long_fit = results_long_fit.sort_values(by='values')
        plt.figure(figsize=(20, 12))
        sns.set(font_scale=2.5)
        g = sns.boxplot(x="model", y="values", hue="metrics", data=results_long_fit, palette="Set3")
        plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
        plt.title('Comparison of Model by Fit and Score Time')
        parent_path = os.getcwd()
        plots_path = parent_path + "./plots"
        isdir = os.path.isdir(plots_path)
        path = os.path.join(parent_path, "plots")
        if isdir:
            print("Directory exists")
        else:
            print("Creating plots directory")
            os.mkdir(path)

        file_name = "\\" + key_name + "benchmark_models_time.png"
        filepath = path + file_name
        print(filepath)
        plt.savefig(filepath, dpi=300)

    def plot_perf_metrics(self, key_name):
        ## PERFORMANCE METRICS
        time_metrics = ['fit_time', 'score_time']  # fit time metrics
        results_long_nofit = self.results_long.loc[~self.results_long['metrics'].isin(time_metrics)]  # get df without fit data
        results_long_nofit = results_long_nofit.sort_values(by='values')
        plt.figure(figsize=(40, 30))
        sns.set(font_scale=2.5)
        g = sns.boxplot(x="model", y="values", hue="metrics", data=results_long_nofit, palette="Set3")
        plt.legend(bbox_to_anchor=(1.05, 1), loc=3, borderaxespad=0.)
        plt.title('Comparison of Model by Classification Metric')
        parent_path = os.getcwd()
        plots_path = parent_path + "./plots"
        isdir = os.path.isdir(plots_path)
        path = os.path.join(parent_path, "plots")
        if isdir:
            print("Directory exists")
        else:
            print("Creating plots directory")
            os.mkdir(path)
        file_name = "\\" + key_name + "benchmark_models_perf.png"
        filepath = path + file_name
        plt.savefig(filepath, dpi=300)

    def run_models(self, X_train: pd.DataFrame, y_train: pd.DataFrame, X_test: pd.DataFrame,
                 y_test: pd.DataFrame):
        '''
        Lightweight script to test many models and find winners
        :param X_train: training split
        :param y_train: training target vector
        :param X_test: test split
        :param y_test: test target vector
        :return: DataFrame of predictions
        '''
        dfs = []
        models = [
            ('RF', RandomForestClassifier(n_estimators=150,
                                          #max_features=40,
                                          oob_score=True,
                                          verbose=1,
                                          min_samples_split=10,
                                          max_depth=75)),
            ('ADA', AdaBoostClassifier()),
            ('XGB', XGBClassifier(max_depth=9,
                                  subsample=0.9,
                                  min_child_weight=2,
                                  colsample_bytree=0.7,
                                  n_estimators=100,
                                  learning_rate=0.08,
                                  n_jobs=-1)),
            ('ET',ExtraTreesClassifier())

        ]
        results = []
        names = []
        scoring = ['accuracy', 'precision_weighted', 'recall_weighted', 'f1_weighted', 'roc_auc']
        target_names = ['False', 'True']
        ext = '.sav'

        for name, model in models:
            kfold = model_selection.KFold(n_splits=5, shuffle=True, random_state=90210)
            cv_results = model_selection.cross_validate(model, X_train, y_train, cv=kfold, scoring=scoring,return_estimator = True)
            #clf.verbose = False
            clf = model.fit(X_train, y_train)
            clf.verbose = False
            file = y_train.columns[-1]
            filename = "./model/" + name + file + ext
            print("Model", filename)
            pickle.dump(clf, open(filename, 'wb'))  # save model file for future use
            y_pred = clf.predict(X_test)
            print("y_pred.shape", y_pred.shape)

            print(name)
            print('\nConfusion Matrix: \n')
            print(confusion_matrix(y_test, y_pred))
            print(classification_report(y_test, y_pred, target_names=target_names))
            results.append(cv_results)
            names.append(name)
            this_df = pd.DataFrame(cv_results)
            this_df['model'] = name
            dfs.append(this_df)
        final = pd.concat(dfs, ignore_index=True)
        return final

    def rover_sample_data(self, X, y):
        """ This function does random over sampling of the data """
        oversampler = RandomOverSampler()
        X, y = oversampler.fit_resample(X, y)
        return X, y

    def split_train_test(self, X, y):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=100)
        return X_train, X_test, y_train, y_test


    def xgboost_model_train(self, X_train, y_train):
        xgb = XGBClassifier(max_depth=9, subsample=0.9, objective='multi:softmax', num_class=3, min_child_weight=2,
                            colsample_bytree=0.7,
                            n_estimators=1000,
                            learning_rate=0.08,
                            n_jobs=-1)
        xgb.fit(X_train, y_train, verbose=True)
        file = y_train.columns[-1]
        ext = '.sav'
        filename = file + ext
        pickle.dump(xgb, open(filename, 'wb'))  # save model file for future use
        return filename


    def xgboost_model_predict(self,X_test, y_test, filename):
        model = pickle.load(open(filename, 'rb'))
        xgb_pred = model.predict(X_test)  # Predicting the test data
        ac = accuracy_score(y_test, xgb_pred) * 100  # calculating accuracy of predicted data
        # print("\n \n Prediction for RLF_"+i+" is: \n")
        print("Accuracy is: ", ac)
        # print('\nConfusion Matrix: \n')
        # print(confusion_matrix(y_test, xgb_pred))
        # print('\n Classification Report: \n')
        # print(classification_report(y_test, xgb_pred, target_names=['Flase', 'True'], digits=4))
        return xgb_pred

