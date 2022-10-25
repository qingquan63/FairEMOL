from FairEMOL.EvolutionaryCodes.data.objects.Data import Data


# https://archive.ics.uci.edu/ml/datasets/Heart+failure+clinical+records
class Heart_failure(Data):
    def __init__(self):
        Data.__init__(self)
        self.dataset_name = 'heart_failure'
        self.class_attr = 'DEATH_EVENT'
        self.positive_class_val = 0
        self.negative_class_val = 1
        self.sensitive_attrs = ['sex']
        self.privileged_class_names = ['1']
        self.categorical_features = ['anaemia', 'diabetes', 'ejection_fraction', 'high_blood_pressure', 'smoking']
        self.features_to_keep = ['age', 'anaemia', 'creatinine_phosphokinase', 'diabetes', 'ejection_fraction',
                                'high_blood_pressure', 'platelets', 'serum_creatinine', 'serum_sodium', 'sex',
                                'smoking', 'time', 'DEATH_EVENT']
        self.missing_val_indicators = ['?']

    def data_specific_processing(self, dataframe):
        return dataframe



