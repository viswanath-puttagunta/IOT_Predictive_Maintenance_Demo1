"""
Module for Data Wrangling and plotting convenience functions
"""
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap

cmap_bold = ListedColormap(['#00FF00','#FF0000'])

class GetFrames:
    """
    Class for Data Wrangling and plotting convenience functions
    """
    def __init__(self, csvpath, sfeature, ldays=-10, lday_strict=True):
        """
        Initializes all the dataframes for given feature
        csvpath: path of csv file
        sfeature: Feature that is selected
        ldays: Last N days to consider for aggregations
        lday_strict:
          True: Log range and max also for last N days
          False: Log range and max for all date ranges

        Class attributes that get generated
        self.df: Original dataframe (with date format conversions and column renaming)
        self.failed_devs: Dataframe of failed devices
        self.failed_dev_hist: Dataframe with failed device history
        self.good_devs: Dataframe of good devices
        self.good_devs_hist: Dataframe with good devices history
        self.all_devs: Dataframe with all devices with tagging (no historical data) (device, fail=1/pass=0)
        self.df_sfeature: Derived features from selected feature history
          device: Device id
          axl_range: Natural log range for last N days(ldays) (for all date range if lday_strict = False)
          axd_mean: Mean of Differential(percentage change) of natural log for last N days(ldays)
          axd_std: Standard Deviation of Differential(percentage change) of natural log for last N days(ldays)
          axd_max: Max of Differential(percentage change) of natural log for last N days(ldays) (for all date range if lday_strict = False)
        """
        self.sfeature = sfeature
        self.ldays = ldays
        df = pd.read_csv(csvpath)
        scols = scols = ["date", "device", sfeature, "failure"]
        df.loc[:,'date'] = pd.to_datetime(df['date'])
        df.columns = ['date', 'device', 'failure', 'a1', 'a2','a3','a4','a5','a6','a7','a8','a9']
        fcols = ['a1', 'a2','a3','a4','a5','a6','a7','a8','a9']
        df = df[scols]
        self.df = df

        failed_devs = pd.DataFrame(df[df['failure'] == 1].device.unique())
        failed_devs.columns = ["device"]
        self.failed_devs = failed_devs
        self.failed_devs_hist = pd.merge(df, failed_devs, on=["device"])
        
        good_devs = pd.DataFrame(list(set(df.device.unique()) - set(failed_devs["device"])))
        good_devs.columns = ["device"]
        self.good_devs = good_devs
        self.good_devs_hist = pd.merge(df, good_devs, on=["device"])

        good_devs["failure"] = 0
        failed_devs["failure"] = 1
        all_devs = pd.concat([good_devs, failed_devs], axis=0)
        all_devs.set_index("device", inplace=True)
        self.all_devs = all_devs

        self.df_sfeature = self.logDifferentialDf(lday_strict)

    def logDifferentialDf(self, strict=True):
        """
        Output is Dataframe with below columns
          device: Device id
          axl_range: Natural log range for last N days(ldays) (for all date range if lday_strict = False)
          axd_mean: Mean of Differential(percentage change) of natural log for last N days(ldays)
          axd_std: Standard Deviation of Differential(percentage change) of natural log for last N days(ldays)
          axd_max: Max of Differential(percentage change) of natural log for last N days(ldays) (for all date range if lday_strict = False)
        Inputs
          strict
            True: All readings are for self.ldays
            False: lrange,max is for all days. std, mean is for last ldays
        """
        dd = dict()
        pref = self.sfeature
        ttdf = self.df.copy()
        ttdf.set_index("date", inplace=True)
        ttdf.loc[:,pref + "_t"] = ttdf[pref].map(lambda x: np.log(x + 0.1))

        for dev in ttdf["device"].unique():
            if strict == True:
                ktdf = ttdf[ttdf["device"] == dev][pref + "_t"].ix[self.ldays:]
                dd[dev] = dict()
                dd[dev][pref + "l_range"] = ktdf.max() - ktdf.min()
                kktdf = ktdf.pct_change()
                dd[dev][pref + "d_mean"] = kktdf.mean()
                dd[dev][pref + "d_max"] = kktdf.max()
                dd[dev][pref + "d_std"] = kktdf.std()
            else:
                ktdf = ttdf[ttdf["device"] == dev][pref + "_t"]
                dd[dev] = dict()
                dd[dev][pref + "l_range"] = ktdf.max() - ktdf.min()
                kktdf = ktdf.pct_change()
                dd[dev][pref + "d_max"] = kktdf.max()

                kktdf = kktdf[self.ldays:]
                dd[dev][pref + "d_mean"] = kktdf.mean()
                dd[dev][pref + "d_std"] = kktdf.std()

        return pd.DataFrame(dd).transpose().join(self.all_devs)

    def plot_history(self, devname, plot_type = "ldiff"):
        """
        Plot history of specified device
        plot_type:
          "ldiff": Standard values
          "lroll": Rolling means and rolling std deviations (for a1 feature)
        """
        tdf = self.df
        fdf = tdf[tdf["device"] == devname]
        fdf.set_index("date", inplace=True)
        feature = self.sfeature
        fdf.loc[:,feature + "_log"] = np.log(fdf[feature] + 0.1)
        fdf.loc[:,feature + "_ldiff"] = fdf[feature + "_log"].diff()
        if plot_type != "ldiff":
            fdf.loc[:,feature + "_lroll_mean"] = pd.rolling_mean(fdf[feature + "_log"], 10)
            fdf.loc[:,feature + "_lroll_std"] = pd.rolling_std(fdf[feature + "_log"], 10)
            fig, axs = plt.subplots(1,4)
        else:
            fig, axs = plt.subplots(1,2)
        fdf.plot(y=[feature+"_ldiff", feature+"_log"], figsize=(10,2), ax=axs[0])
        fdf.plot(y=[feature], figsize=(10,2), ax=axs[1])
        
        if plot_type != "ldiff":
            fdf.plot(y=[feature + "_lroll_mean"], figsize=(10,2), ax=axs[2])
            fdf.plot(y=[feature + "_lroll_std"], figsize=(10,2), ax=axs[3])

    def plot_sample_history(self, dev_list, sample_cnt, plot_type = "ldiff"):
        """
        Take same of devices from dev_list and plot historical data
        """
        if sample_cnt == 0:
            sample_devs = dev_list
        else:
            sample_devs = dev_list.sample(sample_cnt)
        for device in sample_devs:
            self.plot_history(device, plot_type)

if __name__ == "__main__":
    gf = GetFrames("../data/device_failure.csv", "a2")
    print gf.df.head()

    print "Failed Devices"
    print gf.failed_devs.head()
    print gf.failed_devs_hist.head()

    print "Good Devices"
    print gf.good_devs.head()
    print gf.good_devs_hist.head()

    print "All Devs"
    print gf.all_devs[gf.all_devs["failure"] == 1].head()
    print gf.all_devs[gf.all_devs["failure"] == 0].head()
    
    print "Logdifferential Df"
    print gf.df_sfeature.head()


        


