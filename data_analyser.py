import csv
import re
import pandas as pd
from datetime import datetime

class DataAnalyser:
    def __init__(self, data: pd.DataFrame):
        self.data = data
        self.depressed_reg = re.compile(depression_regex)
        self.stressed_reg = re.compile(stress_regex)
        self.anxiety_reg = re.compile(anxiety_regex)

    def get_depression_meds(self):
        return self.data['Medication'].apply(lambda s: bool(self.depressed_reg.search(s)))

    def get_stress_meds(self):
        return self.data['Medication'].apply(lambda s: bool(self.stressed_reg.search(str(s))))

    def get_anxiety_meds(self):
        return self.data['Medication'].apply(lambda s: bool(self.anxiety_reg.search(s)))

    def get_depression_ICD10(self):
        return self.data.ICD10_Diags.apply(lambda s: bool(depression_ICD10s_reg.search(s)))

    def get_anxiety_ICD10(self):
        return self.data.ICD10_Diags.apply(lambda s: bool(anxiety_ICD10s_reg.search(s)))

    def get_alchohol_ICD10(self):
        return self.data.ICD10_Diags.apply(lambda s: bool(alcohol_ICD10s_reg.search(s)))

    def get_alzheimer_ICD10(self):
        return self.data.ICD10_Diags.apply(lambda s: bool(alzheimer_ICD10s_reg.search(s)))

    def get_gingivitis_ICD10(self):
        return self.data.ICD10_Diags.apply(lambda s: bool(gingiv_ICD10s_reg.search(s)))

    def get_mild_cognitive_ICD10(self):
        return self.data.ICD10_Diags.apply(lambda s: bool(mild_cognitive_ICD10s_reg.search(s)))

    def get_cardiovascular_ICD10(self):
        return self.data.ICD10_Diags.apply(lambda s: bool(cardiovascular_ICD10s_reg.search(s)))

    def get_diabetes_ICD10(self):
        return self.data.ICD10_Diags.apply(lambda s: bool(diabetes_ICD10s_reg.search(s)))

    def get_smoking_ICD10(self):
        return self.data.ICD10_Diags.apply(lambda s: bool(smoking_ICD10s_reg.search(s)))

    def get_cholesterol_ICD10(self):
        return self.data.ICD10_Diags.apply(lambda s: bool(cholesterol_ICD10s_reg.search(s)))

    def get_column_data(self, column_name):
        if "_Diag" in column_name or column_name == "AgeOfStopSmoking" or \
                column_name == 'LDL_Cholesterol' or column_name == 'HDL_Cholesterol' or \
                column_name == 'SmokingPacksYears' or column_name == 'DurationModerateActivity':
            return self.data[column_name].apply(lambda s: 1 if s != "Prefer not to answer" else 0)
        if "_Date" in column_name or column_name == "APOE_alles" or \
                column_name == "BirthYear" or column_name == "BiologicalSex":
            return None
        # column is not a categorical column
        if column_name == "rs429358" or column_name == "rs7412" or \
        columns[column_name] is None:
            return self.data[column_name]
        # column is a categorical column
        return self.data[column_name].apply(lambda s: self.get_column_line_data(column_name, s))

    def get_column_line_data(self, column_name, line):
        if "|" in line:
            line = line.split("|")[0]
        options_dict = columns[column_name]
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


column_to_id = {
    # 'BirthYear': 34,
    # 'BiologicalSex': 31,
    'Medication': 41202,
    'CardiovascularDiagnosis': 6150,
    'Diabetes diagnosed': 2443,
    'AgeOfStopSmoking': 22507,
    'Qualifications': 6138,
    'DurationHeavyDIY': 2634,
    'DurationModerateActivity': 894,
    'AlcoholFrequency': 1558,
    'ProfessionalInformedAnxiety': 20428,
    'ProfessionalInformedDepression': 20448,
    'FrequencyFriendsFamilyVisits': 1031,
    'LeisureSocialActivities': 6160,
    'DentalProblems': 6149,
    'PlaysComputer': 2237,
    'HearingAid': 3393,
    'LDL_Cholesterol': 23405,
    'HDL_Cholesterol': 23406,
    'Insomnia': 1200,
    'WalkingActivity_NumDaysWeek': 864,
    'SmokingStatus': 20116,
    'SmokingPacksYears': 20161,
    'SleepDuration': 1160,
    'BMI': 21001,
    'BipolarAndMajorDepressionStat': 20162,
    'HearingDifficulties': 2247,
    'ICD10_Diags': 41202, # delt in the class
    'ICD10_Dates': 41202, # delt in the class
    'Alzheimer_Diag': 41202, # delt in the class
    'Alzheimer_Date': 41202, # delt in the class
    'Mild Cognitive_Diag': 41202,
    'Mild Cognitive_Date': 41202,
    'Gingiv_Diag': 41202,
    'Gingiv_Date': 41202,
    'Alcohol_Diag': 41202,
    'Alcohol_Date': 41202,
    'rs429358': 1220,
    'rs7412': 1220,
    # 'APOE_alles': 1220
}

depression_medications = [
    'fluoxetine',
    'sertraline',
    'escitalopram',
    'paroxetine',
    'citalopram',
    'venlafaxine',
    'duloxetine',
    'desvenlafaxine',
    'bupropion',
    'mirtazapine',
    'trazodone',
    'amitriptyline',
    'nortriptyline',
    'imipramine',
    'clomipramine',
    'phenelzine',
    'tranylcypromine',
    'isocarboxazid'
]
stress_medications = [
    'sertraline',
    'escitalopram',
    'fluoxetine',
    'paroxetine',
    'venlafaxine',
    'duloxetine',
    'alprazolam',
    'diazepam',
    'lorazepam',
    'clonazepam',
    'buspirone',
    'propranolol',
    'atenolol',
    'metoprolol',
    'hydroxyzine',
    'pregabalin',
    'amitriptyline',
    'nortriptyline',
    'imipramine',
    'gabapentin',
    'quetiapine',
    'olanzapine',
    'risperidone'
]
Anxiety_medications = [

    'escitalopram',
    'sertraline',
    'paroxetine',
    'fluoxetine',
    'venlafaxine',
    'duloxetine',
    'alprazolam',
    'diazepam',
    'lorazepam',
    'clonazepam',
    'buspirone',
    'propranolol',
    'atenolol',
    'metoprolol',
    'hydroxyzine',
    'pregabalin',
    'amitriptyline',
    'imipramine',
    'nortriptyline',
    'gabapentin',
    'diphenhydramine',
    'hydroxyzine'
]

depression_regex = re.compile(r'\b(?:' + '|'.join(depression_medications) + r')\b')
stress_regex = re.compile(r'\b(?:' + '|'.join(stress_medications) + r')\b')
anxiety_regex = re.compile(r'\b(?:' + '|'.join(Anxiety_medications) + r')\b')

depression_ICD10s = ['F32.*', 'F33.*']
anxiety_ICD10s = ['F40.*', 'F41.*']
alcohol_ICD10s = ['F10.0', 'F10.1', 'F10.2']
alzheimer_ICD10s = ['G30.*']
cardiovascular_ICD10s = ['I10', 'R03']
diabetes_ICD10s = ['E10', 'E11', 'E12', 'E13', 'E14']
gingiv_ICD10s = ['K05.*']
mild_Cognitive_ICD10s = ['F06.*']
smoking_ICD10s = ['F17.*']
cholesterol_ICD10s = ['E78.0']


depression_ICD10s_reg = re.compile(r'\b(?:' + '|'.join(depression_ICD10s) + r')\b')
anxiety_ICD10s_reg = re.compile(r'\b(?:' + '|'.join(anxiety_ICD10s) + r')\b')
alcohol_ICD10s_reg = re.compile(r'\b(?:' + '|'.join(alcohol_ICD10s) + r')\b')
alzheimer_ICD10s_reg = re.compile(r'\b(?:' + '|'.join(alzheimer_ICD10s) + r')\b')
cardiovascular_ICD10s_reg = re.compile(r'\b(?:' + '|'.join(cardiovascular_ICD10s) + r')\b')
diabetes_ICD10s_reg = re.compile(r'\b(?:' + '|'.join(diabetes_ICD10s) + r')\b')
gingiv_ICD10s_reg = re.compile(r'\b(?:' + '|'.join(gingiv_ICD10s) + r')\b')
mild_cognitive_ICD10s_reg = re.compile(r'\b(?:' + '|'.join(mild_Cognitive_ICD10s) + r')\b')
smoking_ICD10s_reg = re.compile(r'\b(?:' + '|'.join(smoking_ICD10s) + r')\b')
cholesterol_ICD10s_reg = re.compile(r'\b(?:' + '|'.join(cholesterol_ICD10s) + r')\b')


columns = {
    # todo ICD10s and 20003
    # 'Depression':
    # 'FromICD10_41270': {'F32.*', 'F33.*'},
    'BipolarAndMajorDepressionStat': {
        ('Bipolar I Disorder', 'Bipolar II Disorder', 'Prefer not to answer'): 0,
        ('No Bipolar or Depression'): 0,
        ('Single Probable major depression episode'): 1,
        ('Probable Recurrent major depression (moderate)'): 2,
        ('Probable Recurrent major depression (severe)'): 3},

    'ProfessionalInformedDepression': {
        ('Prefer not to answer', 'Do not Know'): 0,
        ('Yes'): 1,
        ('No'): 0},

    # 'Anxiety':
    # 'FromICD10': ['F40.*', 'F41.*'],
    'ProfessionalInformedAnxiety': {
        ('Prefer not to answer', 'Do not Know'): 0,
        ('Yes'): 1,
        ('No'): 0},

    # 'Sleeplessness':
    'SleepDuration': None,
    'Insomnia': {('Prefer not to answer'): -1,
                 ('Usually'): 1,
                 ('Sometimes'): 2,
                 ('Never/rarely'): 3},

    # 'Smoking':
    'AgeOfStopSmoking': None,
    'SmokingStatus': {('Prefer not to answer'): -1,
                      ('Never'): 0,
                      ('Previous'): 1,
                      ('Current'): 2},
    'SmokingPacksYears': None,

    # 'Alcohol':
    # 'FromICD10': ['F10.0', 'F10.1', 'F10.2'],
    'AlcoholFrequency': {('Prefer not to answer'): -1,
                         ('Never', 'Special occasions only',
                          'One to three times a month'): 0,
                         ('Once or twice a week',
                          'Three or four times a week'): 1,
                         ('Daily or almost daily'): 2},

    # 'Cardiovascular':
    # 'FromICD10': ['I10', 'R03'],
    # '6177_Medication': {['Blood pressure medication']: 1,
    #                     ['Cholesterol lowering medication', 'Insulin',
    #                      'None of the above', 'Do not know',
    #                      'Prefer not to answer']: 0},
    # '6153_Medication': {['Blood pressure medication']: 1,
    #                     ['Cholesterol lowering medication', 'Insulin',
    #                      'Hormone replacement therapy',
    #                      'Oral contraceptive pill or minipill',
    #                      'None of the above', 'Do not know',
    #                      'Prefer not to answer']: 0},
    # '4179': None,  # todo get from data
    # '4080': None,  # todo get from data
    'CardiovascularDiagnosis': {('Prefer not to answer'): -1,
                                ('None of the above'): 0,
                                ('Heart attack', 'Angina', 'Stroke'): 1,
                                ('High blood pressure'): 2},

    # 'BMI':
    'BMI': None,

    # 'ActivityPhysical':   # todo consider divide according to median.
    'DurationModerateActivity': None,  # todo change
    # '914_Vigorous': None,  # todo get from data
    'WalkingActivity_NumDaysWeek': None,
    # considering merging with moderate

    # 'ActivitySocial':
    'LeisureSocialActivities': {
        ('Prefer not to answer', 'None of the above'): 0,
        ('Sports club or gym', 'Pub or social club',
         'Religious group',
         'Adult education class',
         'Other group activity'): 1},
    'FrequencyFriendsFamilyVisits': {('Prefer not to answer', 'Do not know',
                                      'No friends/family outside household',
                                      'Never or almost never',
                                      'Once every few months'): 0,
                                     ('About once a month',
                                      'About once a week',
                                      '2-4 times a week', 'Almost daily'): 1},

    # 'ActivityMental':
    'LeisureSocialActivities': {('Prefer not to answer', 'None of the above',
                                 'Sports club or gym', 'Pub or social club',
                                 'Religious group',
                                 'Other group activity'): 0,
                                ('Adult education class'): 1},
    'PlaysComputer': {('Prefer not to answer'): 0,
                      ('Never/rarely'): 0,
                      ('Sometimes', 'Often'): 1},
    'DurationHeavyDIY': {('Prefer not to answer', 'Do not know'): 0,
                         ('Less than 15 minutes', 'Between 15 and 30 minutes',
                          'Between 30 minutes and 1 hour'): 1,
                         ('Between 1 and 1.5 hours', 'Between 1.5 and 2 hours',
                          'Between 2 and 3 hours', 'Over 3 hours'): 2},

    # 'Sex':
    # 'BiologicalSex': None,

    # 'Age':
    # 'BirthYear': None,

    # 'Education':
    'Qualifications': {
        ('Prefer not to answer', 'Do not know', 'None of the above'): 0,
        ('O levels/GCSEs or equivalent',
         'A levels/AS levels or equivalent',
         'NVQ or HND or HNC or equivalent',
         'College or University degree',
         'CSEs or equivalent',
         'Other professional qualifications eg: nursing, teaching'): 1},

    # 'Stress':  # todo add 1 when one of the options exists
    # '6145_Stress': {['Prefer not to answer']: -1,  # todo get from data, meanwhile from ICD10
    #                 ['None of the above']: 0,
    #                 [' Serious illness, injury or assault to yourself',
    #                  ' Serious illness, injury or assault of a close relative'
    #                  'Death of a close relative',
    #                  'Death of a spouse or partner',
    #                  'Martital separation/divorce']: 1},

    # 'Diabetes':
    # '6177_Medication': {['Insulin']: 1,
    #                     ['Cholesterol lowering medication',
    #                      'Blood pressure medication',
    #                      'None of the above', 'Do not know',
    #                      'Prefer not to answer']: 0},
    # '6153_Medication': {['Insulin']: 1,
    #                     ['Blood pressure medication',
    #                      'Cholesterol lowering medication',
    #                      'Hormone replacement therapy',
    #                      'Oral contraceptive pill or minipill',
    #                      'None of the above', 'Do not know',
    #                      'Prefer not to answer']: 0},
    'Diabetes diagnosed': {('Prefer not to answer'): -1,
                           ('No', 'Do not know'): 0,
                           ('Yes'): 1},
    # 'FromICD10': ['E10', 'E11', 'E12', 'E13', 'E14'],

    # 'Cholesterol':
    # '6177_Medication': {['Cholesterol lowering medication']: 1,
    #                     ['Insulin', 'Blood pressure medication',
    #                      'None of the above', 'Do not know',
    #                      'Prefer not to answer']: 0},
    # '6153_Medication': {['Cholesterol lowering medication']: 1,
    #                     ['Blood pressure medication', 'Insulin',
    #                      'Hormone replacement therapy',
    #                      'Oral contraceptive pill or minipill',
    #                      'None of the above', 'Do not know',
    #                      'Prefer not to answer']: 0},
    # '26037_Cholesterol': None, # todo get from data
    'HDL_Cholesterol': None,
    'LDL_Cholesterol': None,
    # 'FromICD10': ['E78.0', 'Normald'],

    # 'APOE_E4E4':
    #     'Positive': ['Y', 'y', 'Yes', 'yes']

    # 'VZV':
    # todo get from data
    # '23052_VZV': {['False']: 0,
    #               ['True']: 1},

    # 'Gums':
    'DentalProblems': {('Prefer not to answer'): -1,
                       ('None of the above', 'Loose teeth', 'Toothache',
                        'Dentures'): 0,
                       ('Bleeding gums', 'Mouth ulcers', 'Painful gums'): 1,},

   # 'Hearing':
   'HearingAid': {('Prefer not to answer'): 0,
                  ('No'): 0,
                  ('Yes'): 1},

    'HearingDifficulties': {('Prefer not to answer'): 0,
                            ('No', 'I am completely deaf', 'Do not know'): 0,
                            ('Yes'): 1},

}

