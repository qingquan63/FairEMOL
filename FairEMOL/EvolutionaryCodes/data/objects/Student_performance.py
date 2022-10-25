from FairEMOL.EvolutionaryCodes.data.objects.Data import Data


# https://archive.ics.uci.edu/ml/datasets/Student+Performance+on+an+entrance+examination
class Student_performance(Data):
    # categorical_features 中没有sensitive attrs 和 class_attr
    def __init__(self):
        Data.__init__(self)
        self.dataset_name = 'student_performance'
        self.class_attr = 'Performance'
        self.positive_class_val = 'Excellent'
        self.negative_class_val = 'Other'
        self.sensitive_attrs = ['Gender']
        self.privileged_class_names = ['female']
        self.categorical_features = ['Caste', 'Coaching', 'Class_ten_education', 'time',
                                'twelve_education', 'medium', 'Class_X_Percentage', 'Class_XII_Percentage', 'Father_occupation',
                                'Mother_occupation']
        self.features_to_keep = ['Gender', 'Caste', 'Coaching', 'Class_ten_education', 'time',
                                'twelve_education', 'medium', 'Class_X_Percentage', 'Class_XII_Percentage', 'Father_occupation',
                                'Mother_occupation', 'Performance']
        self.missing_val_indicators = ['?']

    def data_specific_processing(self, dataframe):
        None_user = dataframe['Performance'] == 'Vg'
        dataframe.loc[None_user, 'Performance'] = 'Excellent'
        User = dataframe['Performance'] != 'Excellent'
        dataframe.loc[User, 'Performance'] = 'Other'
        return dataframe



