from FairEMOL.EvolutionaryCodes.data.objects.Data import Data
import numpy as np


class Student_mat(Data):
    def __init__(self):
        Data.__init__(self)
        self.dataset_name = 'student_mat'
        self.class_attr = 'G3'
        self.positive_class_val = 'pass'
        self.negative_class_val = 'fail'
        self.sensitive_attrs = ['age', 'sex']
        self.privileged_class_names = ['F', 'M']
        self.categorical_features = ['school', 'address', 'famsize', 'Pstatus', 'Medu', 'Fedu', 'Mjob',
                                 'Fjob', 'reason', 'guardian', 'traveltime', 'studytime', 'failures', 'schoolsup',
                                 'famsup', 'paid', 'activities', 'nursery', 'higher', 'internet', 'romantic', 'famrel',
                                 'freetime', 'goout', 'Dalc', 'Walc', 'health', 'absences']
        self.features_to_keep = ['school', 'sex', 'age', 'address', 'famsize', 'Pstatus', 'Medu', 'Fedu', 'Mjob',
                                 'Fjob', 'reason', 'guardian', 'traveltime', 'studytime', 'failures', 'schoolsup',
                                 'famsup', 'paid', 'activities', 'nursery', 'higher', 'internet', 'romantic', 'famrel',
                                 'freetime', 'goout', 'Dalc', 'Walc', 'health', 'absences', 'G1', 'G2', 'G3']
        self.missing_val_indicators = ['?']
        self.preserve_data = {'age': {15, 16, 17, 18}}

    def data_specific_processing(self, dataframe):
        core_pass = dataframe['G3'] >= 10
        dataframe.loc[core_pass, 'G3'] = 'pass'
        score_faile = dataframe['G3'] != 'pass'
        dataframe.loc[score_faile, 'G3'] = 'fail'

        for attribution in self.preserve_data:
            delete_classes = self.preserve_data[attribution]
            index_deleting = np.where(~dataframe[attribution].isin(delete_classes))
            dataframe = dataframe.drop(index=np.array(index_deleting[0]))
        dataframe.reset_index(inplace=True, drop=True)
        return dataframe

