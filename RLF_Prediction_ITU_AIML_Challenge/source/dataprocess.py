#*********************
### @MLgrads Team
######################
from includes import *
from distances import Distances
from met_forecast import met_forecast_data
from met_real import met_real_data
from rl_kpi import rl_kpi_data
from utilities import Utilities as ut
from scipy.stats import chi2
class TrainData:
    def __init__(self):
        self.train_data = ""
        # distances df
        self.dt = Distances("train")
        self.dtrldf = self.dt.rl_df
        # get the rl kpis data merged with rl sites
        rlkpi = rl_kpi_data("train")
        self.rlkpi = rlkpi
        self.rl_kpis_df = rlkpi.rl_kpis_sites_df
        self.metfcast = met_forecast_data("train")
        self.met_forecast_df = self.metfcast.met_forecast_df
        self.metreal = met_real_data("train")
        self.met_real_resample_df = self.metreal.resampled_df
        self.imp_feat_train_df = ""

    def add_met_real_forecast_station_col_to_rlkpis(self):
        """Add met-real-station_no & met-forecast-station_no to rl_kpis_df data """
        rl_kpi_siteId_df = pd.DataFrame(self.rlkpi.imputed_rl_kpis_df.site_id.unique(), columns=['site_id'])

        # Assign the nearest station from met-real to the site Id
        rl_kpi_siteId_df['met-real-station_no'] = \
            rl_kpi_siteId_df['site_id'].apply(lambda x: self.dt.find_nearest_station_no(x,self.dtrldf))

        #rl_kpi_met_data = pd.merge(self.rl_kpis_df, rl_kpi_siteId_df, on='site_id', how='inner')
        met_forecast_dist_df = self.dtrldf[self.dtrldf['Unnamed: 0'].isin(self.metfcast.list_stations)]
        met_forecast_dist_df.reset_index(drop=True, inplace=True)

        #Assign the nearest station from met-forecast to the site Id
        rl_kpi_siteId_df['met-forecast-station_no'] = \
            rl_kpi_siteId_df['site_id'].apply(lambda x: self.dt.find_nearest_station_no(x, met_forecast_dist_df))
        # Add met-real-station_no & met-forecast-station_no in rl_kpis_df
        rl_kpi_met_data = pd.merge(self.rlkpi.imputed_rl_kpis_df, rl_kpi_siteId_df, on='site_id', how='inner')
        self.train_data = rl_kpi_met_data

    def merge_met_real_sampled_df_to_rlkpis(self):
        print("train_data::shape", self.train_data.shape)
        self.met_real_resample_df = self.metreal.process_met_real_data()
        rl_kpi_met_real_data = pd.merge(self.train_data, self.met_real_resample_df,
                                        left_on=['met-real-station_no', 'datetime'],
                                        right_on=['station_no', 'datetime'], how='left')
        first_column = rl_kpi_met_real_data.pop('met-forecast-station_no')
        rl_kpi_met_real_data.insert(0, 'met-forecast-station_no', first_column)
        sorted_rl_kpi_met_real_data_df = rl_kpi_met_real_data.sort_values(by=['met-forecast-station_no', 'datetime'])
        sorted_rl_kpi_met_real_data_df.reset_index(drop=True, inplace=True)
        sorted_rl_kpi_met_real_data_df['datetime-station_no'] = sorted_rl_kpi_met_real_data_df[
                                                                    'met-forecast-station_no'] + "-" + \
                                                                sorted_rl_kpi_met_real_data_df['datetime'].astype('str')
        sorted_rl_kpi_met_real_data_df.drop(['met-forecast-station_no', 'datetime'], axis=1, inplace=True)
        # shift column 'Name' to first position
        first_column = sorted_rl_kpi_met_real_data_df.pop('datetime-station_no')

        # insert column using insert(position,column_name,
        # first_column) function
        sorted_rl_kpi_met_real_data_df.insert(0, 'datetime-station_no', first_column)
        self.train_data = sorted_rl_kpi_met_real_data_df

    def merge_met_forecast_to_sorted_rl_kpi_met_real_data_df(self):
        self.train_data = pd.merge(self.train_data, self.metfcast.imputed_forecast_df,
                                                   on=['datetime-station_no'], indicator=True, how='inner')
        self.train_data.drop_duplicates(inplace=True)
        self.train_data.drop('_merge', axis=1, inplace=True)
        self.train_data.to_csv("rl_kpi_met_real_forecast_data_1.tsv", sep='\t')

    def prepare_train_data(self):
        """ 1 Add 'met-real-station_no' & met-forecast-station_no to rl_kpis_df so that met-forecast
            and met-real data can be merged easily
            2 merge met-real resample data
            3 merge met-forecast data
        """
        ## Impute rlkpis
        print("Imputing rlKPI df")
        self.rlkpi.add_target_labels(1)
        self.rlkpi.impute_rl_kpis()

        print("Add 'met-real-station_no' & met-forecast-station_no to rl_kpis_df")
        self.add_met_real_forecast_station_col_to_rlkpis()
        print("Merge 'met-real-sampled df to rl kps ")
        self.merge_met_real_sampled_df_to_rlkpis()

        ## Imputations for met-forecast
        print("Impute met-forecast")
        met_forecast_obj = self.metfcast
        met_forecast_obj.impute_met_forecast()

        #Merge met forecast data to earlier merged data
        print("Merge Train data with imputed forecast df")
        self.train_data = pd.merge(self.train_data,
                                                   met_forecast_obj.imputed_forecast_df,
                                                   on=['datetime-station_no'], indicator=True, how='inner')
        print("Check any imputation needed", self.train_data.isna().sum().sum())
        self.train_data.drop(['_merge'], axis=1, inplace=True)
        self.perform_data_under_sampling(self.train_data)

    def encode_normalise_features(self, df, all_cols, num_cols, target_col, cat_cols_encode):
        imp_feature_df = df[all_cols]

        for i in target_col:
            #imp_feature_df[i] = imp_feature_df[i].astype(bool)
            imp_feature_df[i] = LabelEncoder().fit_transform(imp_feature_df[i])

        for i in cat_cols_encode:
            imp_feature_df[i] = LabelEncoder().fit_transform(imp_feature_df[i])

        imp_feature_df['mw_connection_no'] = imp_feature_df['mw_connection_no'].str.replace(',', '').astype(str).astype(
            int)
        imp_feature_df[num_cols] = self.normalize_num_features(imp_feature_df, num_cols)
        return imp_feature_df

    def normalize_num_features(self, num_df, num_cols):
        scaler = MinMaxScaler().fit(num_df[num_cols])
        num_df[num_cols] = scaler.transform(num_df[num_cols])
        return num_df[num_cols]

    def usage():
        process = psutil.Process(os.getpid())
        return process.memory_info()[0] / float(2 ** 20)

    def mahalanobis(self, x=None, data=None, cov=None):
        """Compute the Mahalanobis Distance between each row of x and the data
        x    : vector or matrix of data with, say, p columns.
        data : ndarray of the distribution from which Mahalanobis distance of each observation of x is to be computed.
        cov  : covariance matrix (p x p) of the distribution. If None, will be computed from data.
        """
        x_minus_mu = x - np.mean(data)
        if not cov:
            cov = np.cov(data.values.T)
        inv_covmat = sp.linalg.inv(cov)
        left_term = np.dot(x_minus_mu, inv_covmat)
        mahal = np.dot(left_term, x_minus_mu.T)
        return mahal.diagonal()

    def fill_mahalanaobis_score(self, imp_feature_df_false):

        imp_feature_df_false_index = list(imp_feature_df_false.index)

        batch_size = 40000
        result = []
        for batch_number, batch_df in imp_feature_df_false.groupby(np.arange(len(imp_feature_df_false)) // batch_size):
            df_x = batch_df[['type', 'tip', 'card_type', 'adaptive_modulation', 'freq_band',
                             'capacity', 'clutter_class', 'modulation', 'weather_day1',
                             'weather_day2', 'weather_day3', 'weather_day4', 'weather_day5',
                             'mw_connection_no', 'severaly_error_second',
                             'error_second', 'unavail_second', 'avail_time', 'bbe', 'rxlevmax', 'groundheight', 'temp',
                             'temp_max',
                             'temp_min', 'wind_dir', 'wind_speed', 'wind_speed_max', 'humidity',
                             'precipitation', 'precipitation_coeff', 'temp_max_day1',
                             'temp_min_day1', 'humidity_max_day1', 'humidity_min_day1',
                             'wind_dir_day1', 'wind_speed_day1', 'temp_max_day2', 'temp_min_day2',
                             'humidity_max_day2', 'humidity_min_day2', 'wind_dir_day2',
                             'wind_speed_day2', 'temp_max_day3', 'temp_min_day3',
                             'humidity_max_day3', 'humidity_min_day3', 'wind_dir_day3',
                             'wind_speed_day3', 'temp_max_day4', 'temp_min_day4',
                             'humidity_max_day4', 'humidity_min_day4', 'wind_dir_day4',
                             'wind_speed_day4', 'temp_max_day5', 'temp_min_day5',
                             'humidity_max_day5', 'humidity_min_day5', 'wind_dir_day5',
                             'wind_speed_day5']]

            df_x['mahala'] = self.mahalanobis(x=df_x,
                                         data=batch_df[['type', 'tip', 'card_type', 'adaptive_modulation', 'freq_band',
                                                        'capacity', 'clutter_class', 'modulation', 'weather_day1',
                                                        'weather_day2', 'weather_day3', 'weather_day4', 'weather_day5',
                                                        'mw_connection_no', 'severaly_error_second',
                                                        'error_second', 'unavail_second', 'avail_time', 'bbe',
                                                        'rxlevmax', 'groundheight', 'temp', 'temp_max',
                                                        'temp_min', 'wind_dir', 'wind_speed', 'wind_speed_max',
                                                        'humidity',
                                                        'precipitation', 'precipitation_coeff', 'temp_max_day1',
                                                        'temp_min_day1', 'humidity_max_day1', 'humidity_min_day1',
                                                        'wind_dir_day1', 'wind_speed_day1', 'temp_max_day2',
                                                        'temp_min_day2',
                                                        'humidity_max_day2', 'humidity_min_day2', 'wind_dir_day2',
                                                        'wind_speed_day2', 'temp_max_day3', 'temp_min_day3',
                                                        'humidity_max_day3', 'humidity_min_day3', 'wind_dir_day3',
                                                        'wind_speed_day3', 'temp_max_day4', 'temp_min_day4',
                                                        'humidity_max_day4', 'humidity_min_day4', 'wind_dir_day4',
                                                        'wind_speed_day4', 'temp_max_day5', 'temp_min_day5',
                                                        'humidity_max_day5', 'humidity_min_day5', 'wind_dir_day5',
                                                        'wind_speed_day5']])

            result.append(pd.Series(df_x['mahala']))

            del batch_df, df_x

            print("Batch ", batch_number, ": completed")
            print("length of result: ", len(result))

            flat_list = [item for sublist in result for item in sublist]
            flat_list = pd.DataFrame(flat_list)
            imp_feature_df_false['mahala'] = flat_list[0]
            imp_feature_df_false['p-value'] = 1 - chi2.cdf(imp_feature_df_false['mahala'], 65)
            return imp_feature_df_false

    def perform_data_under_sampling(self, feature_df):

        cat_cols = ['type', 'tip', 'card_type', 'adaptive_modulation',
                    'freq_band', 'capacity', 'clutter_class', 'modulation', 'weather_day1', 'weather_day2',
                    'weather_day3',
                    'weather_day4', 'weather_day5']

        cat_cols_encode = ['type', 'tip', 'card_type', 'adaptive_modulation',
                           'freq_band', 'capacity', 'modulation', 'clutter_class','rlf']

        num_cols = ['mw_connection_no', 'severaly_error_second',
                    'error_second', 'unavail_second', 'avail_time', 'bbe', 'rxlevmax', 'groundheight',
                    'temp', 'temp_max', 'temp_min',
                    'wind_dir', 'wind_speed', 'wind_speed_max', 'humidity', 'precipitation',
                    'precipitation_coeff', 'temp_max_day1', 'temp_min_day1',
                    'humidity_max_day1', 'humidity_min_day1', 'wind_dir_day1',
                    'wind_speed_day1', 'temp_max_day2', 'temp_min_day2',
                    'humidity_max_day2', 'humidity_min_day2', 'wind_dir_day2',
                    'wind_speed_day2', 'temp_max_day3', 'temp_min_day3',
                    'humidity_max_day3', 'humidity_min_day3', 'wind_dir_day3',
                    'wind_speed_day3', 'temp_max_day4', 'temp_min_day4',
                    'humidity_max_day4', 'humidity_min_day4', 'wind_dir_day4',
                    'wind_speed_day4', 'temp_max_day5', 'temp_min_day5',
                    'humidity_max_day5', 'humidity_min_day5', 'wind_dir_day5',
                    'wind_speed_day5']

        target_col = ['1-day-predict','5-day-predict']

        all_cols = ['type', 'tip', 'card_type', 'adaptive_modulation',
                    'freq_band', 'capacity', 'clutter_class', 'modulation', 'weather_day1', 'weather_day2',
                    'weather_day3', 'weather_day4', 'weather_day5', 'mw_connection_no', 'severaly_error_second',
                    'error_second', 'unavail_second', 'avail_time', 'bbe',
                    'rxlevmax', 'groundheight', 'temp', 'temp_max', 'temp_min', 'wind_dir',
                    'wind_speed', 'wind_speed_max', 'humidity', 'precipitation',
                    'precipitation_coeff', 'temp_max_day1', 'temp_min_day1',
                    'humidity_max_day1', 'humidity_min_day1', 'wind_dir_day1',
                    'wind_speed_day1', 'temp_max_day2', 'temp_min_day2',
                    'humidity_max_day2', 'humidity_min_day2', 'wind_dir_day2',
                    'wind_speed_day2', 'temp_max_day3', 'temp_min_day3',
                    'humidity_max_day3', 'humidity_min_day3', 'wind_dir_day3',
                    'wind_speed_day3', 'temp_max_day4', 'temp_min_day4',
                    'humidity_max_day4', 'humidity_min_day4', 'wind_dir_day4',
                    'wind_speed_day4', 'temp_max_day5', 'temp_min_day5',
                    'humidity_max_day5', 'humidity_min_day5', 'wind_dir_day5',
                    'wind_speed_day5', 'rlf', '1-day-predict', '5-day-predict']

        # step 1 important feature pre-process:
        print("imp feature preprocess started...")
        processed_feat_df = self.encode_normalise_features(feature_df, all_cols, num_cols, target_col,
                                                cat_cols_encode)  # call function important feature preprocess
        print("imp feature preprocess complete.")
        # step 2 split the dataset true/false:
        processed_feat_df_false = processed_feat_df[(processed_feat_df['rlf'] == False) & (processed_feat_df['1-day-predict'] == False) & (
                    processed_feat_df['5-day-predict'] == False) ]
        processed_feat_df_true = processed_feat_df[
            (processed_feat_df['rlf'] == True) | (processed_feat_df['1-day-predict'] == True) | (processed_feat_df['5-day-predict'] == True)]
        # step 3 perform mahalanobis
        print("imp feature mahalanobis score calculation started...")
        processed_feat_df_false = self.fill_mahalanaobis_score(processed_feat_df_false)
        print("imp feature mahalanobis score calculation complete.")
        # step 4 outlier analysis and trim dataset
        processed_feat_df_false_10000 = processed_feat_df_false.sort_values('mahala', ascending=True).head(20000)
        processed_feat_df_false_10000.drop(['mahala', 'p-value'], axis=1, inplace=True)
        processed_feat_df_final = pd.concat([processed_feat_df_true, processed_feat_df_false_10000])
        self.imp_feat_train_df = processed_feat_df_final
        #self.train_data = imp_feature_df_final
