from FairEMOL.EvolutionaryCodes.data.objects.Data import Data


class Dutch(Data):
    def __init__(self):
        Data.__init__(self)
        self.dataset_name = 'dutch'
        self.class_attr = 'occupation'
        self.positive_class_val = '2_1'  # not sure
        self.negative_class_val = '5_4_9'  # not sure
        self.sensitive_attrs = ['sex']
        self.privileged_class_names = ['1']
        self.categorical_features = ['household_position', 'prev_residence_place',
                                 'citizenship', 'country_birth', 'edu_level', 'economic_status', 'cur_eco_activity',
                                 'Marital_status']
        self.features_to_keep = ['sex', 'age', 'household_position', 'household_size', 'prev_residence_place',
                                 'citizenship', 'country_birth', 'edu_level', 'economic_status', 'cur_eco_activity',
                                 'Marital_status', 'occupation']
        self.missing_val_indicators = ['?']
        self.preserve_data = {'sex': {'1', '2'}}

    def data_specific_processing(self, dataframe):
        return dataframe
