from FairEMOL.EvolutionaryCodes.data.objects.Data import Data

# https://archive.ics.uci.edu/ml/datasets/default+of+credit+card+clients
class Default(Data):
    def __init__(self):
        Data.__init__(self)
        self.dataset_name = 'default'
        self.class_attr = 'default_payment_next_month'
        self.positive_class_val = '1'
        self.negative_class_val = '0'
        self.sensitive_attrs = ['SEX']
        self.privileged_class_names = ['1']
        self.categorical_features = ['EDUCATION', 'MARRIAGE']
        self.features_to_keep = ['LIMIT_BAL', 'SEX', 'EDUCATION', 'MARRIAGE', 'AGE',
                                 'PAY_0', 'PAY_2', 'PAY_3', 'PAY_4', 'PAY_5', 'PAY_6',
                                 'BILL_AMT1', 'BILL_AMT2', 'BILL_AMT3', 'BILL_AMT4', 'BILL_AMT5', 'BILL_AMT6',
                                 'PAY_AMT1', 'PAY_AMT2', 'PAY_AMT3', 'PAY_AMT4', 'PAY_AMT5', 'PAY_AMT6',
                                 'default_payment_next_month']
        self.missing_val_indicators = ['?']

    def data_specific_processing(self, dataframe):
        return dataframe
