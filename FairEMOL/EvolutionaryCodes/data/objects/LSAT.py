from FairEMOL.EvolutionaryCodes.data.objects.Data import Data


class LSAT(Data):
    def __init__(self):
        Data.__init__(self)
        self.dataset_name = 'LSAT'
        self.class_attr = 'first_pf'
        self.positive_class_val = '1'
        self.negative_class_val = '0'
        self.sensitive_attrs = ['race', 'sex']
        self.privileged_class_names = ['White', 'male']
        self.categorical_features = ['region_first']
        self.features_to_keep = ['race', 'sex', 'LSAT', 'UGPA', 'region_first', 'ZFYA', 'sander_index', 'first_pf']
        self.missing_val_indicators = ['?']

    def data_specific_processing(self, dataframe):
        old = dataframe['sex'] == 1
        dataframe.loc[old, 'sex'] = 'famale'
        young = dataframe['sex'] != 'famale'
        dataframe.loc[young, 'sex'] = 'male'
        return dataframe
