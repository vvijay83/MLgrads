from utilities import Utilities as ut
import pandas as pd

class Distances:
    def get_rldf(self):
        rldf = self._df[~self._df['Unnamed: 0'].str.contains("RL_")]
        rldf = rldf[rldf.columns.drop(list(rldf.filter(regex='WS_')))]
        return rldf

    def __init__(self,path ):
        filepath = "./" + path + "/distances.tsv"
        print("Distances", filepath)
        self._df = ut.read_data_to_df(filepath)
        print("distances::shape",self._df.shape)
        self.rl_df = self.get_rldf()
        print("rl_df distances::shape", self.rl_df.shape)

    def find_nearest_station_no(self, site_id, distdf: pd.DataFrame):
         station_no = distdf.iloc[distdf.sort_values(by=site_id).index.values[0], 0]
         return station_no

