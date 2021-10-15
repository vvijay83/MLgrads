from includes import *
from distances import Distances
from met_forecast import met_forecast_data
from met_real import met_real_data
from rl_kpi import rl_kpi_data
from utilities import Utilities as ut
from scipy.stats import chi2

class Testdata:
    def __init__(self, path):
        self.test_data = ""
        self.dt = Distances(path)
        self.dtrldf = self.dt.rl_df
        # get the rl kpis data merged with rl sites
        rlkpi = rl_kpi_data(path)
        self.rlkpi = rlkpi
        self.rl_kpis_df = rlkpi.rl_kpis_sites_df
        self.metfcast = met_forecast_data(path)
        self.met_forecast_df = self.metfcast.met_forecast_df
        self.metreal = met_real_data(path)
        self.met_real_resample_df = self.metreal.resampled_df
        self.imp_feat_test_df = ""
        self.basepath = path
        self.out_path = self.create_output_dir()
        self.X_test = ""
        self.X_val = ""

    def create_output_dir(self):
        out_path = self.basepath + "/output"
        isdir = os.path.isdir(out_path)
        path = os.path.join(self.basepath, "output")
        if isdir:
            print("Directory exists",path)
        else:
            print("Creating Output directory",path)
            os.mkdir(path)
        return path

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
        self.test_data = rl_kpi_met_data

        self.test_data.to_csv("./Output/test_data_met_rfcast.tsv", sep='\t')

    def merge_met_real_sampled_df_to_rlkpis(self):
        print("test_data::shape", self.test_data.shape)
        self.met_real_resample_df = self.metreal.process_met_real_data()
        rl_kpi_met_real_data = pd.merge(self.test_data, self.met_real_resample_df,
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
        self.test_data = sorted_rl_kpi_met_real_data_df
        self.test_data.to_csv("./Output/test_data_met_rfcast_dt.tsv", sep='\t')

    def prepare_test_data(self):
        ## Impute rlkpis
        print("Imputing rlKPI df")
        self.rlkpi.add_target_labels(0)
        self.rlkpi.impute_rl_kpis_for_test_data()

        print("Add 'met-real-station_no' & met-forecast-station_no to rl_kpis_df")
        self.add_met_real_forecast_station_col_to_rlkpis()
        print("Merge 'met-real-sampled df to rl kps ")
        self.merge_met_real_sampled_df_to_rlkpis()

        ## Imputations for met-forecast
        print("Impute met-forecast")
        met_forecast_obj = self.metfcast
        list_feat_to_impute = ut.getColslistneedingImputation(self.metfcast.met_forecast_df)
        if len(list_feat_to_impute) != 0:
            met_forecast_obj.impute_met_forecast_for_test_data(list_feat_to_impute)

        # Merge met forecast data to earlier merged data
        print("Merge Test data with imputed forecast df")
        self.test_data = pd.merge(self.test_data,
                                   met_forecast_obj.imputed_forecast_df,
                                   on=['datetime-station_no'], indicator=True, how='inner')
        print("Check any imputation needed", self.test_data.isna().sum().sum())
        self.test_data.drop(['_merge'], axis=1, inplace=True)
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
        cat_cols_encode = ['type', 'tip', 'card_type', 'adaptive_modulation',
                           'freq_band', 'capacity', 'modulation', 'clutter_class','rlf']
        target_col = ['1-day-predict', '5-day-predict']



        for i in cat_cols_encode:
            self.test_data[i] = LabelEncoder().fit_transform(self.test_data[i])
        self.test_data[num_cols] = ut.normalize_num_features(self.test_data[num_cols], num_cols)
        # scaler = MinMaxScaler().fit(self.test_data[num_cols])
        # self.test_data[num_cols] = scaler.transform(self.test_data[num_cols])
        # drop wind_dir_max and	wind_speed_max columns as test data has empty values
        self.test_data.drop(['wind_dir_max','wind_speed_max'], axis =1, inplace=True)
        self.test_data['datetime-station_no'] = self.test_data['datetime-station_no'].astype('str')
        self.test_data['datetime'] = self.test_data['datetime-station_no'].apply(lambda a: a.split("-", 1)[1])
        self.test_data['datetime'] = pd.to_datetime(self.test_data['datetime'])
        self.test_data['station-no'] = self.test_data['datetime-station_no'].apply(lambda a: a.split("-", 1)[0])
        test_dateTime = self.test_data['datetime'].iloc[-1]
        filepath = self.out_path + '/test_data.tsv'
        self.test_data.to_csv(filepath, sep='\t')
        self.X_test = self.test_data[self.test_data.datetime == test_dateTime]
        self.X_val = self.test_data[self.test_data.datetime != test_dateTime]
        self.X_val = self.X_val[ self.X_val['1-day-predict'].notna() & self.X_val['1-day-predict'].notna()]

        for i in target_col:
            self.X_val[i] = LabelEncoder().fit_transform(self.X_val[i])
