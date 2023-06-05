import csv
import database
import re
import pandas as pd
import numpy as np
from datetime import datetime

class DataAnalyser:
    def __init__(self, data: pd.DataFrame):
        self.data = data
        self.depressed_reg = re.compile(database.depression_regex)
        self.stressed_reg = re.compile(database.stress_regex)
        self.anxiety_reg = re.compile(database.anxiety_regex)
        self.keep_same_cols = ["AgeOfStopSmoking", 'LDL_Cholesterol',
                          'HDL_Cholesterol', 'SmokingPacksYears',
                          'DurationModerateActivity',
                          'ModeratePhysicalActivity_NumDaysWeek',
                          'VigorousPhysicalActivity_NumDaysWeek',
                          'AgeLastEpisodeOfDepression', 'DBloodPressure',
                               'SBloodPressure', 'WalkingActivity_NumDaysWeek', 'SleepDuration',
                               'BMI', 'AgeAtDeath']
        self.not_appears_in_new_data_cols = ["rs429358", "rs7412", "BirthYear", "BiologicalSex",
                                  'eid', 'AgeRecruitment', 'ICD10_Dates', 'Alzheimer_Date',
                                  'Mild Cognitive_Diag', 'Mild Cognitive_Date',
                                  'Gingiv_Diag', 'Gingiv_Date', 'ICD10_Diag', 'Alcohol_Date',
                                'Medication']
        self.cols_to_divide = {"CardioInsulinMedications": ('CardioMedications','InsulinMedications'),
                               'MedsCholesterolHypertensionDiabetes': ('MedsCardiovascular', 'MedsInsulin'),
                               "LeisureSocialActivities": ('MentalActivities', 'SocialActivities')}


    def get_depression_meds(self):
        return self.data['Medication'].apply(lambda s: bool(self.depressed_reg.search(s)))

    def get_stress_meds(self):
        return self.data['Medication'].apply(lambda s: bool(self.stressed_reg.search(str(s))))

    def get_anxiety_meds(self):
        return self.data['Medication'].apply(lambda s: bool(self.anxiety_reg.search(s)))

    def get_depression_ICD10(self):
        return self.data.ICD10_Diags.apply(lambda s: bool(database.depression_ICD10s_reg.search(s)))

    def get_anxiety_ICD10(self):
        return self.data.ICD10_Diags.apply(lambda s: bool(database.anxiety_ICD10s_reg.search(s)))

    def get_alchohol_ICD10(self):
        return self.data.ICD10_Diags.apply(lambda s: bool(database.alcohol_ICD10s_reg.search(s)))

    def get_alzheimer_ICD10(self):
        return self.data.ICD10_Diags.apply(lambda s: bool(database.alzheimer_ICD10s_reg.search(s)))

    def get_gingivitis_ICD10(self):
        return self.data.ICD10_Diags.apply(lambda s: bool(database.gingiv_ICD10s_reg.search(s)))

    def get_mild_cognitive_ICD10(self):
        return self.data.ICD10_Diags.apply(lambda s: bool(database.mild_cognitive_ICD10s_reg.search(s)))

    def get_cardiovascular_ICD10(self):
        return self.data.ICD10_Diags.apply(lambda s: bool(database.cardiovascular_ICD10s_reg.search(s)))

    def get_diabetes_ICD10(self):
        return self.data.ICD10_Diags.apply(lambda s: bool(database.diabetes_ICD10s_reg.search(s)))

    def get_smoking_ICD10(self):
        return self.data.ICD10_Diags.apply(lambda s: bool(database.smoking_ICD10s_reg.search(s)))

    def get_cholesterol_ICD10(self):
        return self.data.ICD10_Diags.apply(lambda s: bool(database.cholesterol_ICD10s_reg.search(s)))

    def get_column_data(self, original_name,  new_column_name):
        if "_Diag" in original_name:
            return self.data[original_name].apply(lambda s: 1 if s != "Prefer not to answer" else 0)
        if original_name in self.keep_same_cols:
            self.data[original_name].fillna("nan")
            return self.data[original_name].apply(lambda s: self.get_keep_cols(s))
        if original_name == "APOE_alles":
            return self.data[original_name].apply(lambda s: 1 if s != "E4, E4" else 0)
        # column is a categorical column
        return self.data[original_name].apply(lambda s: self.get_column_line_data(new_column_name, s))


    def get_keep_cols(self, s):
        if type(s) == str and "|" in s:
            s = float(s.split("|")[-1])
        return int(float(s) if s not in ["nan", "Unable to walk", "Prefer not to answer", "Do not know"] \
                   else np.nan)

    def get_column_line_data(self, column_name, line):
        if database.columns[column_name] is None:
            return line
        options_dict = database.columns[column_name]
        if "|" in line:
            return self.find_max_value(line, options_dict)
        for options, value in options_dict.items():
            if line in options:
                return value

    def get_alzheimers_age_of_diagnosis(self):
        return self.data.apply(lambda row: self.calculate_age_at_diagnosis(row['BirthYear'], row['Alzheimer_Date']),
                               axis=1)

    def calculate_age_at_diagnosis(self, birth_year, alzheimer_date):
        date_format = "%d/%m/%Y"
        if not alzheimer_date:
            return None

        try:
            # Convert birth year to datetime object
            birth_date = datetime.strptime(f"01/01/{birth_year}", date_format)

            # Convert Alzheimer's diagnosed date to datetime object
            diagnosed_date = datetime.strptime(alzheimer_date,
                                               date_format)

            # Calculate the age at diagnosis
            age_at_diagnosis = (diagnosed_date - birth_date).days / 365.25

            return age_at_diagnosis

        except ValueError:
            return None

    def get_diagnosed_since_recruitment(self):
        age_diagnosed = self.get_alzheimers_age_of_diagnosis()
        return age_diagnosed - self.data['AgeRecruitment']
        #self.apply(lambda row: row['AgeDiagnosed'] - row['AgeRecruitment'], axis=1)

    def find_max_value(self, row, categories_dict):
        # Generator expression to iterate over stripped elements
        elements = (element.strip() for element in row.split('|'))
        # Generator expression to get valid values
        valid_values = (categories_dict.get(element) for element in elements if
                        element in categories_dict)
        # Find the maximal value with default=None if no valid values found
        max_value = max(valid_values, default=0)
        return max_value




