from FairEMOL.EvolutionaryCodes.data.objects.Data import Data

class Bank(Data):
    # 参考https://github.com/Trusted-AI/AIF360/blob/master/aif360/datasets/bank_dataset.py
    def __init__(self):
        Data.__init__(self)
        self.dataset_name = 'bank-additional-full'
        self.class_attr = 'y'
        self.positive_class_val = 'yes'
        self.negative_class_val = 'no'

        self.sensitive_attrs = ['age']
        self.privileged_class_names = ['adult']
        self.categorical_features = ['job', 'marital', 'education', 'default',
                                     'housing', 'loan', 'contact', 'month', 'day_of_week',
                                     'poutcome']
        self.features_to_keep = ['age', 'job', 'marital', 'education', 'default',
                                 'housing', 'loan', 'contact', 'month', 'day_of_week',
                                 'poutcome', 'duration', 'campaign', 'pdays', 'previous', 'emp.var.rate',
                                'cons.price.idx', 'cons.conf.idx', 'euribor3m', 'nr.employed', 'y']
        self.missing_val_indicators = ["unknown"]

    def data_specific_processing(self, dataframe):
        old = dataframe['age'] >= 25
        dataframe.loc[old, 'age'] = 'adult'
        young = dataframe['age'] != 'adult'
        dataframe.loc[young, 'age'] = 'youth'
        return dataframe
