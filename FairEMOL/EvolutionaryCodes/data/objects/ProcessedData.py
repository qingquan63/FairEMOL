import pandas as pd

TAGS = ["original", "numerical", "numerical-binsensitive", "categorical-binsensitive", "numerical-for-NN", "original_info"]
TRAINING_PERCENT = 2.0 / 3.0


class ProcessedData:
    def __init__(self, data_obj):
        self.data = data_obj
        self.dfs = dict((k, pd.read_csv(self.data.get_filename(k)))
                        for k in TAGS)
        self.splits = dict((k, []) for k in TAGS)
        self.has_splits = False
        self.has_orig_data = False
        self.orig_data = None

    def get_processed_filename(self, tag):
        return self.data.get_filename(tag)

    def get_dataframe(self, tag):
        return self.dfs[tag]

    def create_train_test_splits(self, num=0):
        return self.dfs

    def get_sensitive_values(self, tag):
        """
        Returns a dictionary mapping sensitive attributes in the data to a list of all possible
        sensitive values that appear.
        """
        if tag == "numerical-for-NN":
            TAG_info = "original_info"
            df = pd.read_csv(self.data.get_filename(TAG_info))
            # df = self.get_dataframe(tag)
            all_sens = self.data.get_sensitive_attributes_with_joint()
            sensdict = {}
            for sens in all_sens:
                 sensdict[sens] = list(set(df[sens].values.tolist()))
            self.orig_data = df
            self.has_orig_data = True
            return sensdict
        else:
            df = self.get_dataframe(tag)
            all_sens = self.data.get_sensitive_attributes_with_joint()
            sensdict = {}
            for sens in all_sens:
                sensdict[sens] = list(set(df[sens].values.tolist()))
            return sensdict

    def get_orig_data(self):
        if self.has_orig_data:
            return self.orig_data

        return pd.read_csv(self.data.get_filename("original_info"))

