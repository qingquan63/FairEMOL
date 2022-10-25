from FairEMOL.EvolutionaryCodes.data.objects.Data import Data
import numpy as np


class PropublicaRecidivism(Data):
    def __init__(self):
        Data.__init__(self)
        self.dataset_name = 'propublica-recidivism'
        self.class_attr = 'two_year_recid'
        self.positive_class_val = 1
        self.negative_class_val = 0
        self.sensitive_attrs = ['sex', 'race']
        self.privileged_class_names = ['Male', 'Caucasian']
        self.categorical_features = ['age_cat', 'c_charge_degree', 'c_charge_desc']
        self.features_to_keep = ["sex", "age", "age_cat", "race", "juv_fel_count", "juv_misd_count",
                                 "juv_other_count", "priors_count", "c_charge_degree",
                                 "c_charge_desc", "decile_score", "score_text", "two_year_recid",
                                 "days_b_screening_arrest", "is_recid"]
        self.missing_val_indicators = []
        self.dele_data = {'race': {'Asian', 'Native American', 'Hispanic', 'Other'}}

    def data_specific_processing(self, dataframe):
        # Filter as done here:
        # https://github.com/propublica/compas-analysis/blob/master/Compas%20Analysis.ipynb
        dataframe = dataframe[(dataframe.days_b_screening_arrest <= 30) &
                              (dataframe.days_b_screening_arrest >= -30) &
                              (dataframe.is_recid != -1) &
                              (dataframe.c_charge_degree != '0') &
                              (dataframe.score_text != 'N/A')]
        dataframe = dataframe.drop(columns=['days_b_screening_arrest', 'is_recid',
                                              'decile_score', 'score_text'])

        dataframe.reset_index(inplace=True, drop=True)
        for attribution in self.dele_data:
            delete_classes = self.dele_data[attribution]
            index_deleting = np.where(dataframe[attribution].isin(delete_classes))
            dataframe = dataframe.drop(index=np.array(index_deleting[0]))
        dataframe.reset_index(inplace=True, drop=True)
        return dataframe
