from FairEMOL.EvolutionaryCodes.data.objects.Data import Data


# https://archive.ics.uci.edu/ml/datasets/Early+stage+diabetes+risk+prediction+dataset
class Diabetes(Data):
    def __init__(self):
        Data.__init__(self)
        self.dataset_name = 'diabetes'
        self.class_attr = 'class'
        self.positive_class_val = 'Positive'
        self.negative_class_val = 'Negative'
        self.sensitive_attrs = ['Gender']
        self.privileged_class_names = ['Female']
        self.categorical_features = ['Polyuria', 'Polyuria', 'Polydipsia', 'sudden_weight_loss', 'weakness',
                                     'Polyphagia', 'Genital_thrush', 'visual_blurring', 'Itching', 'Irritability',
                                     'delayed_healing', 'partial_paresis', 'muscle_stiffness', 'Alopecia', 'Obesity']
        self.features_to_keep = ['Age', 'Gender', 'Polyuria', 'Polydipsia', 'sudden_weight_loss', 'weakness',
                                 'Polyphagia', 'Genital_thrush', 'visual_blurring', 'Itching', 'Irritability',
                                 'delayed_healing', 'partial_paresis', 'muscle_stiffness', 'Alopecia', 'Obesity', 'class']
        self.missing_val_indicators = ['?']

    def data_specific_processing(self, dataframe):
        return dataframe



