from FairEMOL.EvolutionaryCodes.data.objects.Data import Data


# https://www.kaggle.com/manishkc06/patient-treatment-classification
# Sadikin, Mujiono (2020), “EHR Dataset for Patient Treatment Classification”, Mendeley Data, V1, doi: 10.17632/7kv3rctx7m.1
class Patient_treatment(Data):
    def __init__(self):
        Data.__init__(self)
        self.dataset_name = 'patient_treatment'
        self.class_attr = 'SOURCE'
        self.positive_class_val = 1
        self.negative_class_val = 0
        self.sensitive_attrs = ['SEX']
        self.privileged_class_names = ['M']
        self.categorical_features = []
        self.features_to_keep = ['HAEMATOCRIT', 'HAEMOGLOBINS', 'ERYTHROCYTE', 'LEUCOCYTE', 'THROMBOCYTE',
                                'MCH', 'MCHC', 'MCV', 'AGE', 'SEX', 'SOURCE']
        self.missing_val_indicators = ['?']

    def data_specific_processing(self, dataframe):
        return dataframe



