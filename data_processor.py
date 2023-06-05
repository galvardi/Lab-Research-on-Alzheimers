import data_analyser
import pandas as pd
import re

import database


class DataProcessor:
    def __init__(self, data : pd.DataFrame, data_analyser : data_analyser.DataAnalyser):
        self.data = data
        self.new_data = pd.DataFrame()
        self.data_analyser = data_analyser

    def process_medications(self):
        depressed_med_col = self.data_analyser.get_depression_meds()
        self.new_data['depress_medication'] = depressed_med_col.astype(int)
        stress_med_col = self.data_analyser.get_stress_meds()
        self.new_data['stress_medication'] = stress_med_col.astype(int)
        anxiety_med_col = self.data_analyser.get_anxiety_meds()
        self.new_data['anxiety_medication'] = anxiety_med_col.astype(int)

    def process_ICD10(self):
        depression_ICD10_col = self.data_analyser.get_depression_ICD10()
        self.new_data['depression_ICD10'] = depression_ICD10_col.astype(int)
        anxiety_ICD10_col = self.data_analyser.get_anxiety_ICD10()
        self.new_data['anxiety_ICD10'] = anxiety_ICD10_col.astype(int)
        alchohol_ICD10_col = self.data_analyser.get_alchohol_ICD10()
        self.new_data['alchohol_ICD10'] = alchohol_ICD10_col.astype(int)
        alzheimer_ICD10_col = self.data_analyser.get_alzheimer_ICD10()
        self.new_data['alzheimer_ICD10'] = alzheimer_ICD10_col.astype(int)
        gingivitis_ICD10_col = self.data_analyser.get_gingivitis_ICD10()
        self.new_data['gingivitis_ICD10'] = gingivitis_ICD10_col.astype(int)
        diabetes_ICD10_col = self.data_analyser.get_diabetes_ICD10()
        self.new_data['diabetes_ICD10'] = diabetes_ICD10_col.astype(int)
        cardiovascular_ICD10_col = self.data_analyser.get_cardiovascular_ICD10()
        self.new_data['cardiovascular_ICD10'] = cardiovascular_ICD10_col.astype(int)
        smoking_ICD10_col = self.data_analyser.get_smoking_ICD10()
        self.new_data['smoking_ICD10'] = smoking_ICD10_col.astype(int)
        cholesterol_ICD10_col = self.data_analyser.get_cholesterol_ICD10()
        self.new_data['cholesterol_ICD10'] = cholesterol_ICD10_col.astype(int)

    def process_data(self):
        self.process_medications()
        self.process_ICD10()
        for col in self.data:
            if col in self.data_analyser.not_appears_in_new_data_cols:
                continue
            if col in self.data_analyser.cols_to_divide.keys():
                self.generate_divided_column(col)
                continue
            self.add_column_to_new_data(col)
        self.add_years_from_rec_to_diagnosis()
        self.clean_nan_values()

    def clean_nan_values(self):
        for col in self.new_data:
            if col in database.zero_cols:
                self.new_data[col] = self.new_data[col].fillna(0)
            if col in database.means_cols:
                self.new_data[col] = self.new_data[col].fillna(self.new_data[col].mean())
    def add_years_from_rec_to_diagnosis(self):
        years_from_rec_to_diagnosis = self.data_analyser.get_diagnosed_since_recruitment()
        self.new_data['years_from_rec_to_diagnosis'] = years_from_rec_to_diagnosis
    def add_column_to_new_data(self, col):
        generated_new_col = self.data_analyser.get_column_data(col, col)
        if generated_new_col is not None:
            self.new_data[col] = generated_new_col

    def generate_divided_column(self, col):
        cols_num_to_divide_to = len(self.data_analyser.cols_to_divide[col])
        cols_names_to_divide_to = self.data_analyser.cols_to_divide[col]
        for i in range(cols_num_to_divide_to):
            generated_new_col = self.data_analyser.get_column_data(col, cols_names_to_divide_to[i])
            if generated_new_col is not None:
                self.new_data[cols_names_to_divide_to[i]] = generated_new_col

# reg = re.compile(rf'^R03\b|\|R03\b')
# "isLowBP = df.ICD10_Diags.apply(lambda s: bool(reg.search(s)))"
# "isLowBP.sum()
#####