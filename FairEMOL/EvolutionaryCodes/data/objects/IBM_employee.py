from FairEMOL.EvolutionaryCodes.data.objects.Data import Data


# https://www.kaggle.com/pavansubhasht/ibm-hr-analytics-attrition-dataset/data
# https://rstudio-pubs-static.s3.amazonaws.com/371220_a5b20d73372c4e7ba94dee6fc6921354.html
# https://arxiv.org/pdf/2012.01286.pdf
class IBM_employee(Data):
    def __init__(self):
        Data.__init__(self)
        self.dataset_name = 'IBM_employee'
        self.class_attr = 'Attrition'
        self.positive_class_val = 'Yes'
        self.negative_class_val = 'No'
        self.sensitive_attrs = ['Gender']
        self.privileged_class_names = ['Female']
        self.categorical_features = ['BusinessTravel', 'Department',
                                'Education', 'EducationField',
                                'EnvironmentSatisfaction', 'JobInvolvement', 'JobLevel',
                                 'JobRole', 'JobSatisfaction', 'MaritalStatus',
                                 'Over18', 'OverTime', 'PerformanceRating',
                                 'RelationshipSatisfaction', 'StandardHours', 'StockOptionLevel',
                                 'WorkLifeBalance']

        self.features_to_keep = ['Age', 'Attrition', 'BusinessTravel', 'DailyRate', 'Department',
                                'DistanceFromHome', 'Education', 'EducationField', 'EmployeeNumber',
                                'EnvironmentSatisfaction', 'Gender', 'HourlyRate', 'JobInvolvement', 'JobLevel',
                                 'JobRole', 'JobSatisfaction', 'MaritalStatus', 'MonthlyIncome', 'MonthlyRate',
                                 'NumCompaniesWorked', 'Over18', 'OverTime', 'PercentSalaryHike', 'PerformanceRating',
                                 'RelationshipSatisfaction', 'StandardHours', 'StockOptionLevel', 'TotalWorkingYears',
                                 'TrainingTimesLastYear', 'WorkLifeBalance', 'YearsAtCompany', 'YearsInCurrentRole',
                                 'YearsSinceLastPromotion', 'YearsWithCurrManager']
        self.missing_val_indicators = ['?']

    def data_specific_processing(self, dataframe):
        return dataframe



