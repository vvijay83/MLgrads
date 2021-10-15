from includes import *
from utilities import Utilities as ut
from distances import Distances
from sklearn.preprocessing import LabelEncoder
from fancyimpute import KNN
class met_forecast_data(object):
    def __init__(self, path):
        filepath = "./" + path + "/met-forecast.tsv"
        print("Distances", filepath)
        self.raw_df = ut.read_data_to_df(filepath)
        self.raw_df.drop(['Unnamed: 0'], axis=1, inplace=True)
        print("met_forecast_data::shape", self.raw_df.shape)
        self.missing_val = self.raw_df.isna().sum()

        if ut.check_if_missing_val_imputation_needed(self.raw_df):
            print("Missing Value found... Imputation required for Met-real data")
        else:
            print("Missing Value Imputation not required for Met-real data")
        self.list_stations = self.raw_df.station_no.unique()
        self.met_forecast_df = self.raw_df
        self.dt = Distances(path)
        self.st_most_missing_val = ""
        self.met_forecast_resample_df = self.met_forecast_df
        self.imputed_forecast_df = self.met_forecast_resample_df

    def get_missing_val_station_list(self):
        """ Return the list of stations having missing values in descending order """
        ##Find the station_no with the most missing values
        list_station = self.raw_df[self.raw_df.humidity_max_day1.isna()]['station_no'].value_counts()
        self.list_station_missing_val = \
            list(dict(sorted(list_station.items(), key=operator.itemgetter(1), reverse=True)))
        print('Dictionary in descending order by value : ', self.list_station_missing_val)
        self.st_most_missing_val = self.list_station_missing_val[0]
        return self.list_station_missing_val

    def find_nrst_st_present_in_metforecast(self, list_dist: list , list_forecast_stations: list):
        """ Return the list of nearest stations having missing values in descending order """
        list_nrst_st = []
        for x in list_dist:
            if x in list_forecast_stations:
                list_nrst_st.append(x)
        return list_nrst_st

    def check_humidity_is_non_null(self, df2: pd.DataFrame, index: int):
        if (ut.isNaN(df2.loc[index, f'humidity_max_day1']) & ut.isNaN(df2.loc[index, f'humidity_max_day2'])
                & ut.isNaN(df2.loc[index, f'humidity_max_day3']) & ut.isNaN(df2.loc[index, f'humidity_max_day4']) &
                ut.isNaN(df2.loc[index, f'humidity_max_day5'])):
            return False
        else:
            return True

    def impute_forecast_cols(self, met_forecast_df: pd.DataFrame, df2: pd.DataFrame, null_df_idx: list,
                             date_matching_idx: list):
        list_index_imputed = []
        for index in null_df_idx:
            for index_1 in date_matching_idx:
                if (met_forecast_df.loc[index, 'datetime'] == df2.loc[index_1, 'datetime']):
                    if self.check_humidity_is_non_null(df2, index_1):
                        for x in range(5):
                            met_forecast_df.loc[index, f'humidity_max_day{x + 1}'] = df2.loc[
                                index_1, f'humidity_max_day{x + 1}']
                            met_forecast_df.loc[index, f'humidity_min_day{x + 1}'] = df2.loc[
                                index_1, f'humidity_min_day{x + 1}']
                            met_forecast_df.loc[index, f'wind_dir_day{x + 1}'] = df2.loc[
                                index_1, f'wind_dir_day{x + 1}']
                            met_forecast_df.loc[index, f'wind_speed_day{x + 1}'] = df2.loc[
                                index_1, f'wind_speed_day{x + 1}']
                            if index not in list_index_imputed:
                                list_index_imputed.append(index)
            if index in list_index_imputed:
                null_df_idx.remove(index)

        print("No of Indices imputed", len(list_index_imputed))
        return  met_forecast_df

    def impute_val_from_nearest_station (self):
        """ Impute the forecast entries by filling in the values from their nearest stations for the same date
         Assumption being nearest station will have similar weather conditions """
        distances_df = self.dt._df
        self.st_most_missing_val = self.get_missing_val_station_list()[0]
        met_forecast_stations = list(set(self.met_forecast_df['station_no']))
        temp_df = distances_df[[self.st_most_missing_val, 'Unnamed: 0']].sort_values(by=[self.st_most_missing_val])
        list_nrst_st = list(temp_df[temp_df['Unnamed: 0'].str.startswith('WS', na=False)]['Unnamed: 0'])
        list_nrst_st.remove(self.st_most_missing_val)
        list_nearest_st = self.find_nrst_st_present_in_metforecast(list_nrst_st, met_forecast_stations)
        print("list_nearest_st",list_nearest_st)
        df1 = self.met_forecast_df[self.met_forecast_df['station_no'] == self.st_most_missing_val][
            ['station_no', 'datetime', 'humidity_max_day1']].copy(deep=False)
        null_df = df1[df1['humidity_max_day1'].isnull()]
        null_df_idx = list(df1.index[df1['humidity_max_day1'].isnull()])
        self.met_forecast_df.drop('report_time', axis=1, inplace=True)
        print("Iterate through the nearest station list")
        for station in list_nearest_st:
            print("station::======>", station)
            df2 = self.met_forecast_df[self.met_forecast_df['station_no'] == station].copy(deep=False)
            date_matching_idx = df2['datetime'].isin(null_df['datetime']).index
            if len(date_matching_idx) > 0:
                self.met_forecast_df = self.impute_forecast_cols(self.met_forecast_df, df2, null_df_idx, date_matching_idx)
        return self.met_forecast_df

    def impute_wd_wsd_from_prev_day(self):
        """ Impute the weather day and wind speed day from previous days value 
         """

        wsd5_null_idx = self.met_forecast_df.index[self.met_forecast_df['wind_speed_day5'].isnull()]
        for index in wsd5_null_idx:
            self.met_forecast_df.loc[index, 'wind_speed_day5'] = self.met_forecast_df.loc[index, 'wind_speed_day4']
        wd5_null_idx = self.met_forecast_df.index[self.met_forecast_df['weather_day5'].isnull()]
        for index in wd5_null_idx:
            self.met_forecast_df.loc[index, 'weather_day5'] = self.met_forecast_df.loc[index, 'weather_day4']

    def find_prevIdx(self, met_forecast_resample_df: pd.DataFrame, dayntime, station_no):
        full_station_idx = met_forecast_resample_df.index[met_forecast_resample_df['station_no'] == station_no]
        idx_ = met_forecast_resample_df.index[met_forecast_resample_df['humidity_max_day1'].isnull() &
                                              (met_forecast_resample_df['station_no'] == station_no)]
        previous_date = dayntime - datetime.timedelta(days=1)
        prev_idx = met_forecast_resample_df.index[(met_forecast_resample_df['datetime'] == previous_date) & (
                    met_forecast_resample_df['station_no'] == station_no)]
        #print(prev_idx)
        if len(prev_idx) > 1:
            return prev_idx[1]
        else:
            return prev_idx[0]

    def find_nextIdx(self, met_forecast_resample_df: pd.DataFrame, dayntime, station_no):
        full_station_idx = met_forecast_resample_df.index[met_forecast_resample_df['station_no'] == station_no]
        idx_ = met_forecast_resample_df.index[met_forecast_resample_df['humidity_max_day1'].isnull() &
                                              (met_forecast_resample_df['station_no'] == station_no)]
        Next_Date = dayntime + datetime.timedelta(days=1)
        next_idx = met_forecast_resample_df.index[(met_forecast_resample_df['datetime'] == Next_Date) & (
                    met_forecast_resample_df['station_no'] == station_no)]
        if (len(next_idx) > 1):
            return next_idx[1]
        else:
            return next_idx[0]

    def resample_impute_num_cols_from_prev_next_index(self, met_forecast_df: pd.DataFrame):
        "Impute the weather forecast  day entries from previous day's entries if present  "
        met_forecast_numerical_cols = ['station_no', 'datetime',
                                       'temp_max_day1', 'temp_min_day1', 'humidity_max_day1', 'humidity_min_day1',
                                       'wind_dir_day1', 'wind_speed_day1',
                                       'temp_max_day2', 'temp_min_day2', 'humidity_max_day2', 'humidity_min_day2',
                                       'wind_dir_day2', 'wind_speed_day2',
                                       'temp_max_day3', 'temp_min_day3', 'humidity_max_day3', 'humidity_min_day3',
                                       'wind_dir_day3', 'wind_speed_day3',
                                       'temp_max_day4', 'temp_min_day4', 'humidity_max_day4', 'humidity_min_day4',
                                       'wind_dir_day4', 'wind_speed_day4',
                                       'temp_max_day5', 'temp_min_day5', 'humidity_max_day5', 'humidity_min_day5',
                                       'wind_dir_day5', 'wind_speed_day5']
        #resample and impute
        met_forecast_resample_df = met_forecast_df[met_forecast_numerical_cols]
        met_forecast_resample_df = met_forecast_resample_df.groupby('station_no').resample('D',
                                                                            on='datetime').mean().reset_index()
        null_idx = met_forecast_resample_df.index[met_forecast_resample_df['humidity_max_day1'].isnull()]
        null_day5_idx = met_forecast_resample_df.index[met_forecast_resample_df['humidity_max_day5'].isnull()]

        for indx in null_idx:
            prevIdx = self.find_prevIdx(met_forecast_resample_df, met_forecast_resample_df.loc[indx, 'datetime'],
                                   met_forecast_resample_df.loc[indx, 'station_no'])
            met_forecast_resample_df.loc[indx, 'humidity_max_day1'] = met_forecast_resample_df.loc[
                prevIdx, 'humidity_max_day2']
            met_forecast_resample_df.loc[indx, 'humidity_max_day2'] = met_forecast_resample_df.loc[
                prevIdx, 'humidity_max_day3']
            met_forecast_resample_df.loc[indx, 'humidity_max_day3'] = met_forecast_resample_df.loc[
                prevIdx, 'humidity_max_day4']
            met_forecast_resample_df.loc[indx, 'humidity_max_day4'] = met_forecast_resample_df.loc[
                prevIdx, 'humidity_max_day5']

            met_forecast_resample_df.loc[indx, 'humidity_min_day1'] = met_forecast_resample_df.loc[
                prevIdx, 'humidity_min_day2']
            met_forecast_resample_df.loc[indx, 'humidity_min_day2'] = met_forecast_resample_df.loc[
                prevIdx, 'humidity_min_day3']
            met_forecast_resample_df.loc[indx, 'humidity_min_day3'] = met_forecast_resample_df.loc[
                prevIdx, 'humidity_min_day4']
            met_forecast_resample_df.loc[indx, 'humidity_min_day4'] = met_forecast_resample_df.loc[
                prevIdx, 'humidity_min_day5']

            met_forecast_resample_df.loc[indx, 'wind_dir_day1'] = met_forecast_resample_df.loc[prevIdx, 'wind_dir_day2']
            met_forecast_resample_df.loc[indx, 'wind_dir_day2'] = met_forecast_resample_df.loc[prevIdx, 'wind_dir_day3']
            met_forecast_resample_df.loc[indx, 'wind_dir_day3'] = met_forecast_resample_df.loc[prevIdx, 'wind_dir_day4']
            met_forecast_resample_df.loc[indx, 'wind_dir_day4'] = met_forecast_resample_df.loc[prevIdx, 'wind_dir_day5']

            met_forecast_resample_df.loc[indx, 'wind_speed_day1'] = met_forecast_resample_df.loc[
                prevIdx, 'wind_speed_day2']
            met_forecast_resample_df.loc[indx, 'wind_speed_day2'] = met_forecast_resample_df.loc[
                prevIdx, 'wind_speed_day3']
            met_forecast_resample_df.loc[indx, 'wind_speed_day3'] = met_forecast_resample_df.loc[
                prevIdx, 'wind_speed_day4']
            met_forecast_resample_df.loc[indx, 'wind_speed_day4'] = met_forecast_resample_df.loc[
                prevIdx, 'wind_speed_day5']

            met_forecast_resample_df.loc[indx, 'temp_max_day1'] = met_forecast_resample_df.loc[prevIdx, 'temp_max_day2']
            met_forecast_resample_df.loc[indx, 'temp_max_day2'] = met_forecast_resample_df.loc[prevIdx, 'temp_max_day3']
            met_forecast_resample_df.loc[indx, 'temp_max_day3'] = met_forecast_resample_df.loc[prevIdx, 'temp_max_day4']
            met_forecast_resample_df.loc[indx, 'temp_max_day4'] = met_forecast_resample_df.loc[prevIdx, 'temp_max_day5']

            met_forecast_resample_df.loc[indx, 'temp_min_day1'] = met_forecast_resample_df.loc[prevIdx, 'temp_min_day2']
            met_forecast_resample_df.loc[indx, 'temp_min_day2'] = met_forecast_resample_df.loc[prevIdx, 'temp_min_day3']
            met_forecast_resample_df.loc[indx, 'temp_min_day3'] = met_forecast_resample_df.loc[prevIdx, 'temp_min_day4']
            met_forecast_resample_df.loc[indx, 'temp_min_day4'] = met_forecast_resample_df.loc[prevIdx, 'temp_min_day5']

            # for x in range(4):
            #     met_forecast_resample_df.loc[indx, f'humidity_max_day{x + 1}'] = met_forecast_resample_df.loc[
            #         prevIdx, f'humidity_max_day{x + 2}']
            #     met_forecast_resample_df.loc[indx, f'humidity_min_day{x + 1}'] = met_forecast_resample_df.loc[
            #         prevIdx, f'humidity_min_day{x + 1}']
            #     met_forecast_resample_df.loc[indx, f'wind_dir_day{x + 1}'] = met_forecast_resample_df.loc[
            #         prevIdx, f'wind_dir_day{x + 1}']
            #     met_forecast_resample_df.loc[indx, f'wind_speed_day{x + 1}'] = met_forecast_resample_df.loc[
            #         prevIdx, f'wind_speed_day{x + 1}']
            #     met_forecast_resample_df.loc[indx, f'temp_max_day{x + 1}'] = met_forecast_resample_df.loc[
            #         prevIdx, f'temp_max_day{x + 1}']
            #     met_forecast_resample_df.loc[indx, f'temp_min_day{x + 1}'] = met_forecast_resample_df.loc[
            #         prevIdx, f'temp_min_day{x + 1}']
        for indx in null_day5_idx:
            prevIdx = self.find_nextIdx(met_forecast_resample_df, met_forecast_resample_df.loc[indx, 'datetime'],
                                        met_forecast_resample_df.loc[indx, 'station_no'])
            met_forecast_resample_df.loc[indx, 'humidity_max_day5'] = met_forecast_resample_df.loc[
                prevIdx, 'humidity_max_day4']
            met_forecast_resample_df.loc[indx, 'humidity_min_day4'] = met_forecast_resample_df.loc[
                prevIdx, 'humidity_max_day3']

            met_forecast_resample_df.loc[indx, 'humidity_min_day5'] = met_forecast_resample_df.loc[
                prevIdx, 'humidity_min_day4']
            met_forecast_resample_df.loc[indx, 'humidity_min_day4'] = met_forecast_resample_df.loc[
                prevIdx, 'humidity_min_day3']

            met_forecast_resample_df.loc[indx, 'wind_dir_day5'] = met_forecast_resample_df.loc[prevIdx, 'wind_dir_day4']
            met_forecast_resample_df.loc[indx, 'wind_dir_day4'] = met_forecast_resample_df.loc[prevIdx, 'wind_dir_day3']

            met_forecast_resample_df.loc[indx, 'wind_speed_day5'] = met_forecast_resample_df.loc[
                prevIdx, 'wind_speed_day4']
            met_forecast_resample_df.loc[indx, 'wind_speed_day4'] = met_forecast_resample_df.loc[
                prevIdx, 'wind_speed_day3']

            met_forecast_resample_df.loc[indx, 'temp_max_day5'] = met_forecast_resample_df.loc[prevIdx, 'temp_max_day4']
            met_forecast_resample_df.loc[indx, 'temp_max_day4'] = met_forecast_resample_df.loc[prevIdx, 'temp_max_day3']

            met_forecast_resample_df.loc[indx, 'temp_min_day5'] = met_forecast_resample_df.loc[prevIdx, 'temp_min_day4']
            met_forecast_resample_df.loc[indx, 'temp_min_day4'] = met_forecast_resample_df.loc[prevIdx, 'temp_min_day3']
        #     nextIdx = self.find_nextIdx(met_forecast_resample_df, met_forecast_resample_df.loc[indx, 'datetime'],
        #                                 met_forecast_resample_df.loc[indx, 'station_no'])
        #     for x in range(5,3,-1):
        #         met_forecast_resample_df.loc[indx, f'humidity_max_day{x }'] = met_forecast_resample_df.loc[
        #             nextIdx, f'humidity_max_day{x - 1}']
        #         met_forecast_resample_df.loc[indx, f'humidity_min_day{x }'] = met_forecast_resample_df.loc[
        #             nextIdx, f'humidity_min_day{x -1 }']
        #         met_forecast_resample_df.loc[indx, f'wind_dir_day{x}'] = met_forecast_resample_df.loc[
        #             nextIdx, f'wind_dir_day{x - 1}']
        #         met_forecast_resample_df.loc[indx, f'wind_speed_day{x}'] = met_forecast_resample_df.loc[
        #             nextIdx, f'wind_speed_day{x - 1}']
        #         met_forecast_resample_df.loc[indx, f'temp_max_day{x}'] = met_forecast_resample_df.loc[
        #             nextIdx, f'temp_max_day{x - 1}']
        #         met_forecast_resample_df.loc[indx, f'temp_min_day{x}'] = met_forecast_resample_df.loc[
        #             nextIdx, f'temp_min_day{x - 1}']
       # self.met_forecast_df = met_forecast_resample_df
        return met_forecast_resample_df

    def impute_using_fancy_impute(self, met_forecast_resample_df: pd.DataFrame ):
        met_forecast_resample_df['sno'] = np.arange(len(met_forecast_resample_df))
        test_df = met_forecast_resample_df.copy(deep=False)
        test_df['station_no'] = LabelEncoder().fit_transform(test_df['station_no'])
        test_df.drop(['datetime'], axis=1, inplace=True)
        knn_imputer = KNN(k=3)
        cols = ['station_no', 'temp_max_day1', 'temp_min_day1',
                'humidity_max_day1', 'humidity_min_day1', 'wind_dir_day1',
                'wind_speed_day1', 'temp_max_day2', 'temp_min_day2',
                'humidity_max_day2', 'humidity_min_day2', 'wind_dir_day2',
                'wind_speed_day2', 'temp_max_day3', 'temp_min_day3',
                'humidity_max_day3', 'humidity_min_day3', 'wind_dir_day3',
                'wind_speed_day3', 'temp_max_day4', 'temp_min_day4',
                'humidity_max_day4', 'humidity_min_day4', 'wind_dir_day4',
                'wind_speed_day4', 'temp_max_day5', 'temp_min_day5',
                'humidity_max_day5', 'humidity_min_day5', 'wind_dir_day5',
                'wind_speed_day5', 'sno']
        Xtrans = knn_imputer.fit_transform(test_df)
        df = pd.DataFrame(Xtrans, columns=cols)
        df.to_csv('imputed_df2_knn_stationlb.tsv', sep='\t')
        df.drop('station_no', axis=1, inplace=True)
        df = df.round()
        left = met_forecast_resample_df[['station_no', 'datetime', 'sno']]
        new_df = pd.merge(left, df, on='sno')
        new_df.to_csv("merged_updated_forecast_df.tsv", sep='\t')
        new_df.drop_duplicates(inplace=True)
        return new_df

    def merge_imputed_cat_num_df(self, imputed_num_df: pd.DataFrame):
        weatherday_df = self.met_forecast_df[
            ['datetime', 'station_no', 'weather_day1', 'weather_day2', 'weather_day3', 'weather_day4', 'weather_day5']]
        weatherday_df.drop_duplicates(inplace=True)
        imputed_num_df.drop_duplicates(inplace=True)
        imputed_merged_forecast_df = pd.merge(imputed_num_df, weatherday_df, on=['datetime', 'station_no'], how='left')
        imputed_merged_forecast_df.to_csv("imputed_merged_forecast_df.tsv", sep='\t')
        self.met_forecast_df = imputed_merged_forecast_df
        self.imputed_forecast_df = self.met_forecast_df
    def encode_weather_day(self):
        weather_day_map = {"hot day": 0,
                           "clear sky": 0,
                           "scattered clouds": 1,
                           "few clouds": 1,
                           "overcast clouds": 1,
                           "misty": 2,
                           "foggy": 2,
                           "windy": 3,
                           "light rain": 4,
                           "light rain showers": 4,
                           "light intensity shower rain": 4,
                           "rain": 5,
                           "heavy rain": 5,
                           "heavy rain showers": 5,
                           "thunderstorm with heavy rain": 6,
                           "heavy thunderstorm with rain showers": 6,
                           "light snow": 7,
                           "snow": 7,
                           "sleet": 7}
        for x in range(5):
            self.met_forecast_df[f'weather_day{x + 1}'] = self.met_forecast_df[f'weather_day{x + 1}'].map(weather_day_map)
        cols = ['weather_day1', 'weather_day2', 'weather_day3', 'weather_day4', 'weather_day5']
        self.met_forecast_df[cols].to_csv("met_forecast_df_wd.tsv",sep='\t')



    def weatherday_impute_preprocess(self, df_no_blanks, i, X_cols, y_cols):
        X, y = df_no_blanks[X_cols], df_no_blanks[y_cols]
        y = pd.DataFrame(y).astype(str)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.10, random_state=100)
        return X_train, X_test, y_train, y_test

    def weatherday_impute_model(self, X_train, X_test, y_train, y_test, i):
        rf = RandomForestClassifier()
        start = time.time()
        rf.fit(X_train, y_train)  # training the model
        stop = time.time()
        rf_pred = rf.predict(X_test)  # Predicting the test data
        ac = accuracy_score(y_test, rf_pred) * 100  # calculating accuracy of predicted data
        print("LR-Classifier Multi-class Set-Accuracy is ", ac)
        filename = 'weatherday' + i + '_impute.sav'
        pickle.dump(rf, open(filename, 'wb'))  # save model file for future use
        return rf

    def weatherday_impute_predict(self, df_blanks, rf, i, X_cols, y_cols):
        X, y = df_blanks[X_cols], df_blanks[y_cols]
        y = pd.DataFrame(y).astype(str)
        index_list = y.index
        y = pd.DataFrame(rf.predict(X))  # Predicting the test data
        y.rename(columns={0: y_cols[0]}, inplace=True)
        y.set_index(index_list, inplace=True)
        return y[y_cols[0]]

    def weatherday_impute(self, merged_forecast_df):
        weather_days = ['1', '2', '3', '4', '5']
        new_df = pd.DataFrame()

        for i in weather_days:
            cols = ['weather_day' + i, 'temp_max_day' + i, 'temp_min_day' + i, 'humidity_max_day' + i,
                    'humidity_min_day' + i,
                    'wind_dir_day' + i, 'wind_speed_day' + i]
            X_cols = ['temp_max_day' + i, 'temp_min_day' + i, 'humidity_max_day' + i, 'humidity_min_day' + i,
                      'wind_dir_day' + i, 'wind_speed_day' + i]
            y_cols = ['weather_day' + i]

            df_blanks = merged_forecast_df[merged_forecast_df[y_cols[0]].isna()]
            df_no_blanks = merged_forecast_df[cols].dropna(axis=0, subset=[y_cols[0]])

            X_train, X_test, y_train, y_test = self.weatherday_impute_preprocess(df_no_blanks, str(i), X_cols, y_cols)

            rf = self.weatherday_impute_model(X_train, X_test, y_train, y_test, str(i))

            pred_values = self.weatherday_impute_predict(df_blanks, rf, str(i), X_cols, y_cols)

            new_df = pd.concat([new_df, pred_values], axis=1)
            new_df = new_df.astype('float')
        imputed_merged_forecast_df_idx = merged_forecast_df.index[merged_forecast_df['weather_day1'].isnull()]
        weather_day_list = ['weather_day1', 'weather_day2', 'weather_day3', 'weather_day4', 'weather_day5']

        for idx in imputed_merged_forecast_df_idx:
            for weather_day in weather_day_list:
                merged_forecast_df.loc[idx, weather_day] = new_df.loc[idx, weather_day]

        self.imputed_forecast_df = merged_forecast_df
    def prep_sort_df_for_merge(self):

        sorted_forecast_df = self.imputed_forecast_df
        sorted_forecast_df.drop('sno', axis=1, inplace=True)
        sorted_forecast_df.rename(columns={'station_no': 'met-forecast-station_no'}, inplace=True)
        sorted_forecast_df = sorted_forecast_df.sort_values(by=['met-forecast-station_no', 'datetime'])
        sorted_forecast_df.drop_duplicates(inplace=True)
        sorted_forecast_df['datetime-station_no'] = sorted_forecast_df['met-forecast-station_no'] + "-" + \
                                                    sorted_forecast_df['datetime'].astype('str')
        sorted_forecast_df.drop(['met-forecast-station_no', 'datetime'], axis=1, inplace=True)
        # shift column 'Name' to first position
        first_column = sorted_forecast_df.pop('datetime-station_no')
        # insert column using insert(position,column_name,
        # first_column) function
        sorted_forecast_df.insert(0, 'datetime-station_no', first_column)
        self.imputed_forecast_df = sorted_forecast_df.groupby('datetime-station_no').mean().reset_index()
        self.imputed_forecast_df.to_csv("./Output/sorted_imputed_forecast_df.tsv", sep='\t')

    def impute_met_forecast(self):
        self.impute_wd_wsd_from_prev_day()
        met_forecast_df = self.impute_val_from_nearest_station()
        met_forecast_resample_df = self.resample_impute_num_cols_from_prev_next_index(self.met_forecast_df)
        met_forecast_resample_df = self.impute_using_fancy_impute(met_forecast_resample_df)
        self.merge_imputed_cat_num_df(met_forecast_resample_df)
        self.encode_weather_day()
        self.weatherday_impute(self.met_forecast_df)
        print("Prepare  met-forecast to merge with train data df")
        self.prep_sort_df_for_merge()


    def impute_met_forecast_for_test_data(self,list_feat_to_impute):
        self.impute_wd_wsd_from_prev_day()
        met_forecast_df = self.impute_val_from_nearest_station()
        met_forecast_resample_df = self.resample_impute_num_cols_from_prev_next_index(self.met_forecast_df)
        met_forecast_resample_df = self.impute_using_fancy_impute(met_forecast_resample_df)
        self.merge_imputed_cat_num_df(met_forecast_resample_df)
        list_wday = ['weather_day1', 'weather_day2', 'weather_day3', 'weather_day4', 'weather_day5']
        self.encode_weather_day()
        need_imputation = any(item in list_wday for item in list_feat_to_impute)
        print("Check if still imputation is required ")
        list_feat_to_impute = ut.getColslistneedingImputation(self.met_forecast_df)
        if len(list_feat_to_impute) == 0:
            need_imputation = False
        if need_imputation:
            print("Weather day needs imputation")
            self.weatherday_impute(self.met_forecast_df)
        else:
            print("Weather day no imputation required")
            self.imputed_forecast_df = self.met_forecast_df
        self.prep_sort_df_for_merge()



