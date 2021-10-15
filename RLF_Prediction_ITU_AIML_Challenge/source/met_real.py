
from includes import *
from utilities import Utilities as ut

class met_real_data(object):
    def __init__(self,path ):
        filepath = "./" + path + "/met-real.tsv"
        print("Distances", filepath)
        self.rawdata = ut.read_data_to_df(filepath)
        print("met_real_rawdata::shape",self.rawdata.shape)
        self.resampled_df = self.rawdata

    def resample_interpolate(self, in_df: pd.DataFrame):
        in_df = in_df.groupby('station_no').resample('D', on='datetime').mean().reset_index()
        print("met_real_resample_df::Shape After ", in_df.shape)
        # interpolate and drop unnecessary columns
        in_df.interpolate(method="quadratic", inplace=True)
        in_df.drop(['Unnamed: 0', 'measured_hour'], axis=1, inplace=True)
        print("met_real...in_df:shape after interpolate", in_df.shape)
        ##round the decimals to 2 positions and update the original dataframe
        in_df_num_df = in_df[['temp', 'temp_max', 'temp_min', 'wind_dir', 'wind_speed',
                              'wind_speed_max', 'humidity', 'precipitation', 'precipitation_coeff']]
        in_df_num_df = round(in_df_num_df, 2)
        in_df.update(in_df_num_df)
        return in_df

    def process_met_real_data(self):
        # Read data into a dataframe
        #met_real_df = ut.read_data_to_df(fPathName)
        #print("Met-Real-Shape", met_real_df.shape)
        ### Resampling the data to have more entries on date time
        self.resampled_df = self.resample_interpolate(self.rawdata)
        if ut.check_if_missing_val_imputation_needed(self.resampled_df):
            print("Missing Value Imputation not required for Met-real data")
        else:
            print("Missing Value Imputation required for Met-real data")
        return self.resampled_df

