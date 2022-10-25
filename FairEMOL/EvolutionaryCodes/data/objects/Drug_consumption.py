from FairEMOL.EvolutionaryCodes.data.objects.Data import Data


# https://archive.ics.uci.edu/ml/datasets/Drug+consumption+%28quantified%29
class Drug_consumption(Data):
    def __init__(self):
        Data.__init__(self)
        self.dataset_name = 'drug_consumption'
        self.class_attr = 'Mushrooms'
        self.positive_class_val = 'None_user'
        self.negative_class_val = 'User'
        self.sensitive_attrs = ['Gender']
        self.privileged_class_names = ['female']
        self.categorical_features = ['Country', 'Ethnicity', 'Impulsive', 'SS']
        self.features_to_keep = ['Age', 'Gender', 'Education', 'Country', 'Ethnicity',
                                'Nscore', 'Escore', 'Oscore', 'Ascore', 'Cscore',
                                'Impulsive', 'SS', 'Mushrooms']
        self.missing_val_indicators = ['?']

    def data_specific_processing(self, dataframe):
        None_user = dataframe['Mushrooms'] == 'CL0'
        dataframe.loc[None_user, 'Mushrooms'] = 'None_user'
        None_user = dataframe['Mushrooms'] == 'CL1'
        dataframe.loc[None_user, 'Mushrooms'] = 'None_user'
        User = dataframe['Mushrooms'] != 'None_user'
        dataframe.loc[User, 'Mushrooms'] = 'User'
        return dataframe



