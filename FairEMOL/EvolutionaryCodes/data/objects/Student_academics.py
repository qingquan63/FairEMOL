from FairEMOL.EvolutionaryCodes.data.objects.Data import Data


# https://archive.ics.uci.edu/ml/datasets/Student+Academics+Performance
class Student_academics(Data):
    def __init__(self):
        Data.__init__(self)
        self.dataset_name = 'student_academics'
        self.class_attr = 'atd'
        self.positive_class_val = 'Good'
        self.negative_class_val = 'other'
        self.sensitive_attrs = ['gender']
        self.privileged_class_names = ['F']
        self.categorical_features = ['tnp', 'twp', 'iap', 'arr', 'as', 'fmi', 'fs', 'fq',
                                     'fo', 'nf', 'sh', 'ss', 'me', 'tt']
        self.features_to_keep = ['tnp', 'twp', 'iap', 'arr', 'as', 'fmi', 'fs', 'fq', 'fo',
                                 'nf', 'sh', 'ss', 'me', 'tt', 'atd', 'gender']
        self.missing_val_indicators = ['?']

    def data_specific_processing(self, dataframe):
        User = dataframe['atd'] == 'Average'
        dataframe.loc[User, 'atd'] = 'other'
        User = dataframe['atd'] == 'Poor'
        dataframe.loc[User, 'atd'] = 'other'
        return dataframe



