from includes import *
from utilities import Utilities as ut
from distances import Distances

class rl_kpi_data(object):
    def __init__(self,path):
        prefix = "./" + path
        filepath = prefix + "/rl-kpis.tsv"
        print(filepath)
        self.rl_kpis_df = self.read_df(filepath)
        print("rl_kpi_data.raw_df::shape", self.rl_kpis_df.shape)
        filepath = prefix + "/rl-sites.tsv"
        self.rl_kpis_sites_df = self.merge_rl_sites_data(filepath)
        print("rl_kpis_sites_df::shape", self.rl_kpis_sites_df.shape)

        self.imputed_rl_kpis_df = self.rl_kpis_sites_df

    def get_rl_kpis_df(self):
        return self.rl_kpis_df

    def read_df(self,fPathName):
        df = ut.read_data_to_df(fPathName)
        df.drop(["Unnamed: 0"], axis=1, inplace=True)
        return df

    def merge_rl_sites_data(self,filepath):
        rl_sites_df = ut.read_data_to_df(filepath)
        rl_sites_df.drop(["Unnamed: 0"], axis=1, inplace=True)
        return pd.merge(self.rl_kpis_df, rl_sites_df, on='site_id', how='inner')

    def add_target_labels(self,train):
        """
            1. Prepare Labels
            2. Prepare target days (prediction days)
            3. Join dataset to get RLF colunms for the target days
            4. Finalize labels for 1-day and 5-day predictions
        :return:
        """
        df_labels = self.rl_kpis_sites_df[["datetime", "site_id", "mlid"]]

        #  Prepare columns for the following days. We will join data with these columns to find RLF
        prediction_interval = 5
        for i in range(prediction_interval):
            df_labels[f"T+{i + 1}"] = df_labels["datetime"] + pd.DateOffset(days=i + 1)
        rl_kpis_view = self.rl_kpis_sites_df[["datetime", "site_id", "mlid", "rlf"]]
        # Merge the RLf_1 to RLF_5
        for i in range(prediction_interval):
            target_day_column_name = f"T+{i + 1}"

            df_labels = df_labels.merge(rl_kpis_view,
                                        how="left",
                                        left_on=("site_id", "mlid", target_day_column_name),
                                        right_on=("site_id", "mlid", "datetime"),
                                        suffixes=("", "_y")
                                        )
            df_labels.rename(columns={"rlf": f"{target_day_column_name}_rlf"}, inplace=True)
        df_labels.drop(columns=["datetime_y"], inplace=True)
        # 1 day predict is equal to T+1 rlf
        df_labels["1-day-predict"] = df_labels["T+1_rlf"]

        # Interval predict (5-day predict) is based on T+1, T+2, T+3, T+4 and T+5
        following_days_rlf_columns = [f"T+{i + 1}_rlf" for i in range(prediction_interval)]
        df_labels["x-day-predict"] = df_labels[following_days_rlf_columns].isnull().apply(lambda x: all(x), axis=1)
        df_labels = df_labels[df_labels["x-day-predict"] == False]

        df_labels["5-day-predict"] = df_labels[following_days_rlf_columns].any(axis=1)
        df_labels = df_labels[["datetime", "site_id", "mlid", "1-day-predict", "5-day-predict"]]

        print(f"df_labels.shape: {df_labels.shape}")
        print(f"df_labels 1-day rlf sum: {df_labels['1-day-predict'].sum()}")
        print(f"df_labels 5-day rlf sum: {df_labels['5-day-predict'].sum()}")
        # Now join labels with rl-kpis
        self.rl_kpis_sites_df = self.rl_kpis_sites_df.merge(df_labels,
                                                  how="left",
                                                  on=["datetime", "site_id", "mlid"])
        # fill false for the nan entries
        self.rl_kpis_sites_df.loc[self.rl_kpis_sites_df['1-day-predict'].isna(), '1-day-predict'] = False
        self.rl_kpis_sites_df.loc[self.rl_kpis_sites_df['5-day-predict'].isna(), '5-day-predict'] = False




    def impute_rl_kpis_for_test_data(self):

        self.encode_cat_cols()
        list_feat_to_impute = ut.getColslistneedingImputation(self.rl_kpis_sites_df)
        if "freq_band" in list_feat_to_impute:
            print("impute_rl_kpis:: Imputing Freq_band")
            df_only_blanks = self.rl_kpis_sites_df[self.rl_kpis_sites_df.freq_band.isna()]
            self.freq_predict(df_only_blanks)
        if "capacity" in list_feat_to_impute:
            print("impute_rl_kpis:: Imputing capacity")
            self.capacity_impute()

    def encode_cat_cols(self):
        cat_cols = ['type', 'tip', 'card_type', 'adaptive_modulation', 'modulation','clutter_class']
        for i in cat_cols:
            self.rl_kpis_sites_df[i] = LabelEncoder().fit_transform(self.rl_kpis_sites_df[i])
        if self.rl_kpis_sites_df['mw_connection_no'].dtype != 'int64':
            self.rl_kpis_sites_df['mw_connection_no'] = self.rl_kpis_sites_df['mw_connection_no'].str.replace(',', '')


    def impute_rl_kpis(self):
        self.freq_impute()
        self.capacity_impute()

    def freq_impute_preprocess(self, df_no_blanks, x_cols, cat_cols):
        X, y = df_no_blanks[x_cols], df_no_blanks['freq_band']
        y = pd.DataFrame(y).astype(str)

        freq_band_map = {0: "f1", 1: "f2", 2: "f3", 3: "f4", 4: "f5"}
        freq_band_map_rev = {"f1": 0, "f2": 1, "f3": 2, "f4": 3, "f5": 4}

        #    clutter_class_map = {0: "OPEN LAND", 1: "DENSE TREE", 2: "OPEN IN URBAN", 3: "AVERAGE-MEDIUM URBAN",
        #                         4: "HIGH-SPARSE URBAN", 5: "LOW-MEDIUM URBAN", 6: "AVERAGE-DENSE URBAN",
        #                         7: "SPARSE TREE", 8: "HIGH-DENSE URBAN", 9: "LOW-DENSE URBAN", 10: "INDUSTRIAL & COMMERCIAL",
        #                         11: "LOW-SPARSE URBAN", 12: "VERYHIGH-DENSE BLOCK BUILDINGS", 13: "AVERAGE-SPARSE URBAN",
        #                         14: "VERYHIGH-SPARSE BLOCK BUILDINGS", 15: "INLAND WATER", 16: "BUILTUP-VILLAGE",
        #                         17: "HIGH-ISOLATED-BUILDINGS", 18: "HIGH-MEDIUM URBAN", 19: "VERYHIGH-MEDIUM BLOCK BUILDINGS",
        #                         20: "GREEN HOUSE"}
        #    clutter_class_map_rev = {"OPEN LAND": 0, "DENSE TREE": 1, "OPEN IN URBAN": 2, "AVERAGE-MEDIUM URBAN": 3,
        #                         "HIGH-SPARSE URBAN": 4, "LOW-MEDIUM URBAN": 5, "AVERAGE-DENSE URBAN": 6,
        #                         "SPARSE TREE": 7, "HIGH-DENSE URBAN": 8, "LOW-DENSE URBAN": 9, "INDUSTRIAL & COMMERCIAL": 10,
        #                         "LOW-SPARSE URBAN": 11, "VERYHIGH-DENSE BLOCK BUILDINGS": 12, "AVERAGE-SPARSE URBAN": 13,
        #                         "VERYHIGH-SPARSE BLOCK BUILDINGS": 14, "INLAND WATER": 15, "BUILTUP-VILLAGE": 16,
        #                         "HIGH-ISOLATED-BUILDINGS": 17, "HIGH-MEDIUM URBAN": 18, "VERYHIGH-MEDIUM BLOCK BUILDINGS": 19,
        #                         "GREEN HOUSE": 20}

        for i in cat_cols:
            X[i] = LabelEncoder().fit_transform(X[i])

        X['mw_connection_no'] = X['mw_connection_no'].str.replace(',', '')
        # X['clutter_class'] = X['clutter_class'].map(clutter_class_map_rev)
        y['freq_band'] = y['freq_band'].map(freq_band_map_rev)

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=100, stratify=y)

        return X_train, X_test, y_train, y_test

    def freq_impute_model(self, X_train, X_test, y_train, y_test):
        rf = RandomForestClassifier()
        start = time.time()
        print("freq_band model training started...")
        rf.fit(X_train, y_train)
        stop = time.time()
        print("freq_band model training completed in ", stop - start, " seconds")
        rf_pred = rf.predict(X_test)  # Predicting the test data
        print("freq_band model calculating accuracy...")
        ac = accuracy_score(y_test, rf_pred) * 100  # calculating accuracy of predicted data
        print("LR-Classifier Multi-class Set-Accuracy is ", ac)
        print("freq_band model export in-progress...")
        filename = 'freq_impute.sav'
        pickle.dump(rf, open(filename, 'wb'))  # save model file for future use
        print("freq_band model exported as freq_impute.sav...")
        return rf

    def freq_predict(self,df_only_blanks):
        freq_band_map = {0: "f1", 1: "f2", 2: "f3", 3: "f4", 4: "f5"}
        freq_band_map_rev = {"f1": 0, "f2": 1, "f3": 2, "f4": 3, "f5": 4}
        x_cols = ['type', 'tip', 'mw_connection_no', 'card_type', 'adaptive_modulation',
                  'severaly_error_second', 'error_second', 'unavail_second', 'avail_time',
                  'bbe', 'rxlevmax', 'capacity', 'modulation']
        y_cols = 'freq_band'
        X, y = df_only_blanks[x_cols], df_only_blanks['freq_band']
        y = pd.DataFrame(y).astype(str)

        y['freq_band'] = LabelEncoder().fit_transform(y['freq_band'])
        print("freq_band model inference started...")

        loaded_model = pickle.load(open("./model/freq_impute.sav", 'rb'))
        #result = loaded_model.score(X_test, Y_test)
        y = pd.DataFrame(loaded_model.predict(X))

        y.rename(columns={0: 'freq_band'}, inplace=True)
        y['freq_band'] = y['freq_band'].map(freq_band_map)
        y.set_index(df_only_blanks.index, inplace=True)
        df_only_blanks['freq_band'] = y['freq_band']

        df_only_blanks_idx = df_only_blanks.index

        #print("freq_band missing value imputation started...")
        start = time.time()
        df = self.rl_kpis_sites_df
        for idx in df_only_blanks_idx:
            df.loc[idx, 'freq_band'] = df_only_blanks.loc[
                idx, 'freq_band']  # iterate through index to impute the missing capacity
        stop = time.time()
        print("freq_band missing value imputation completed in ", stop - start, "seconds.")
        self.imputed_rl_kpis_df = df



    def freq_impute_predict(self, df_only_blanks, rf, x_cols, cat_cols):

        freq_band_map = {0: "f1", 1: "f2", 2: "f3", 3: "f4", 4: "f5"}
        freq_band_map_rev = {"f1": 0, "f2": 1, "f3": 2, "f4": 3, "f5": 4}

        X, y = df_only_blanks[x_cols], df_only_blanks['freq_band']
        y = pd.DataFrame(y).astype(str)

        for i in cat_cols:
            X[i] = LabelEncoder().fit_transform(X[i])

        X['mw_connection_no'] = X['mw_connection_no'].str.replace(',', '')
        y['freq_band'] = LabelEncoder().fit_transform(y['freq_band'])

        print("freq_band model inference started...")
        y = pd.DataFrame(rf.predict(X))

        # X['clutter_class'] = X['clutter_class'].map(clutter_class_map)

        y.rename(columns={0: 'freq_band'}, inplace=True)
        y['freq_band'] = y['freq_band'].map(freq_band_map)
        y.set_index(df_only_blanks.index, inplace=True)
        df_only_blanks['freq_band'] = y['freq_band']

        return df_only_blanks



    def freq_impute(self):

        df = self.rl_kpis_sites_df

        x_cols = ['type', 'tip', 'mw_connection_no', 'card_type', 'adaptive_modulation',
                  'severaly_error_second', 'error_second', 'unavail_second', 'avail_time',
                  'bbe', 'rxlevmax', 'capacity', 'modulation']

        y_cols = 'freq_band'

        cat_cols = ['type', 'tip', 'card_type', 'adaptive_modulation', 'modulation']

        df_no_blanks = df[['type', 'tip', 'mw_connection_no', 'card_type', 'adaptive_modulation',
                           'severaly_error_second', 'error_second', 'unavail_second', 'avail_time',
                           'bbe', 'rxlevmax', 'capacity', 'modulation', 'freq_band']].dropna(axis=0,
                                                                                             subset=['freq_band'])

        df_no_blanks.dropna(axis=0, subset=['capacity'], inplace=True)

        df_only_blanks = df[df.freq_band.isna()]

        print("freq_band impute data preprocess started...")
        X_train, X_test, y_train, y_test = self.freq_impute_preprocess(df_no_blanks, x_cols,
                                                                  cat_cols)  # call function to preprocess
        print("freq_band impute data preprocess complete.")
        rf = self.freq_impute_model(X_train, X_test, y_train, y_test)  # call function to create estimator
        df_pred = self.freq_impute_predict(df_only_blanks, rf, x_cols, cat_cols)  # call function to predict
        print("freq_band model inference complete and missing value populated.")

        df_only_blanks_idx = df_only_blanks.index

        print("freq_band missing value imputation started...")
        start = time.time()
        for idx in df_only_blanks_idx:
            df.loc[idx, 'freq_band'] = df_pred.loc[ idx, 'freq_band']  # iterate through index to impute the missing capacity
        stop = time.time()
        print("freq_band missing value imputation completed in ", stop - start, "seconds.")
        self.imputed_rl_kpis_df = df

    def capacity_impute_preprocess(self,df_no_blanks, x_cols, cat_cols):
        X, y = df_no_blanks[x_cols], df_no_blanks['capacity']
        y = pd.DataFrame(y)

        for i in cat_cols:
            X[i] = LabelEncoder().fit_transform(X[i])

        freq_band_map = {0: "f1", 1: "f2", 2: "f3", 3: "f4", 4: "f5"}
        freq_band_map_rev = {"f1": 0, "f2": 1, "f3": 2, "f4": 3, "f5": 4}

        X['freq_band'] = X['freq_band'].map(freq_band_map_rev)
        X['mw_connection_no'] = X['mw_connection_no'].str.replace(',', '')

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=100)

        return X_train, X_test, y_train, y_test

    def capacity_impute_model(self, X_train, X_test, y_train, y_test):
        rf = RandomForestClassifier()
        start = time.time()
        print("capacity impute model training started...")
        rf.fit(X_train, y_train)
        stop = time.time()
        print("capacity impute model training completed in ", stop - start, " seconds")
        rf_pred = rf.predict(X_test)  # Predicting the test data
        print("capacity impute model calculating accuracy...")
        ac = accuracy_score(y_test, rf_pred) * 100  # calculating accuracy of predicted data
        print("LR-Classifier Multi-class Set-Accuracy is ", ac)
        print("capacity impute model export in-progress...")
        filename = 'capacity_impute.sav'
        pickle.dump(rf, open(filename, 'wb'))  # save model file for future use
        print("capacity impute model exported as freq_impute.sav...")
        return rf

    def capacity_impute_predict(self, df_only_blanks, rf, x_cols, cat_cols):
        X, y = df_only_blanks[x_cols], df_only_blanks['capacity']
        y = pd.DataFrame(y)

        for i in cat_cols:
            X[i] = LabelEncoder().fit_transform(X[i])

        freq_band_map = {0: "f1", 1: "f2", 2: "f3", 3: "f4", 4: "f5"}
        freq_band_map_rev = {"f1": 0, "f2": 1, "f3": 2, "f4": 3, "f5": 4}

        X['freq_band'] = X['freq_band'].map(freq_band_map_rev)
        X['mw_connection_no'] = X['mw_connection_no'].str.replace(',', '')

        print("freq_band model inference started...")
        y = pd.DataFrame(rf.predict(X))
        y.rename(columns={0: 'capacity'}, inplace=True)
        y.set_index(df_only_blanks.index, inplace=True)
        df_only_blanks['capacity'] = y['capacity']

        return df_only_blanks

    def capacity_impute(self):
        df = self.imputed_rl_kpis_df
        x_cols = ['type', 'tip', 'mw_connection_no', 'card_type', 'adaptive_modulation',
                  'severaly_error_second', 'error_second', 'unavail_second', 'avail_time',
                  'bbe', 'rxlevmax', 'modulation', 'freq_band']

        cat_cols = ['type', 'tip', 'card_type', 'adaptive_modulation', 'modulation']

        df_no_blanks = df[['type', 'tip', 'mw_connection_no', 'card_type', 'adaptive_modulation',
                           'severaly_error_second', 'error_second', 'unavail_second', 'avail_time',
                           'bbe', 'rxlevmax', 'capacity', 'modulation', 'freq_band']].dropna(axis=0,
                                                                                             subset=['capacity'])
        df_only_blanks = df[df.capacity.isna()]

        print("capacity impute data preprocess started...")
        X_train, X_test, y_train, y_test = self.capacity_impute_preprocess(df_no_blanks, x_cols,
                                                                      cat_cols)  # call function to preprocess
        print("capacity impute data preprocess complete.")
        rf = self.capacity_impute_model(X_train, X_test, y_train, y_test)  # call function to create estimator
        df_pred = self.capacity_impute_predict(df_only_blanks, rf, x_cols, cat_cols)  # call function to predict
        print("capacity impute model inference complete and missing value populated.")

        null_idx = df.index[df['capacity'].isnull()]

        print("capacity missing value imputation started...")
        start = time.time()
        for idx in null_idx:
            df.loc[idx, 'capacity'] = df_pred.loc[
                idx, 'capacity']  # iterate through index to impute the missing capacity
        stop = time.time()
        print("capacity missing value imputation completed in ", stop - start, "seconds.")

        df.to_csv("./Output/rl_kpi_met_real_forecast_data_1.tsv", sep='\t')  # save the csv file for backup

        self.imputed_rl_kpis_df = df

