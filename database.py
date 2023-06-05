import re
import numpy as np

column_to_id = {
    # 'eid': 0,
    # 'BirthYear': 34,
    # 'BiologicalSex': 31,
    'Medication': 41202,
    # 'AgeRecruitment': 21022,
    'CardiovascularDiagnosis': 6150,
    'CardioInsulinMedications': 6177,
    'Diabetes diagnosed': 2443,
    'AgeOfStopSmoking': 22507,
    'Qualifications': 6138,
    'AgeLastEpisodeOfDepression': 20434,
    'DurationOfWorstDepression': 20438,
    'DurationHeavyDIY': 2634,
    'DurationModerateActivity': 894,
    'AlcoholFrequency': 1558,
    'ProfessionalInformedAnxiety': 20428,
    'IllnessInjuriesStressLast2Years': 6145,
    'ProfessionalInformedDepression': 20448,
    'FrequencyFriendsFamilyVisits': 1031,
    'LeisureSocialActivities': 6160,
    'DBloodPressure': 4079,
    'SBloodPressure': 4080,
    'MedsCholesterolHypertensionDiabetes': 6153,
    'DentalProblems': 6149,
    'PlaysComputer': 2237,
    'HearingAid': 3393,
    'LDL_Cholesterol': 23405,
    'HDL_Cholesterol': 23406,
    'Insomnia': 1200,
    'WalkingActivity_NumDaysWeek': 864,
    'ModeratePhysicalActivity_NumDaysWeek': 884,
    'VigorousPhysicalActivity_NumDaysWeek': 904,
    'SmokingStatus': 20116,
    'SmokingPacksYears': 20161,
    'SleepDuration': 1160,
    'BMI': 21001,
    'AgeAtDeath': 40007,
    'BipolarAndMajorDepressionStat': 20162,
    'SeenGpForNervesAnxietyTensionDepression': 2090,
    'SeenShrinkForNervesAnxietyTensionDepression': 2100,
    'HearingDifficulties': 2247,
    # 'ICD10_Diags': 41202,
    # 'ICD10_Dates': 41202,
    'Alzheimer_Diag': 41202,
    # 'Alzheimer_Date': 41202,
    # 'Mild Cognitive_Diag': 41202,
    # 'Mild Cognitive_Date': 41202,
    # 'Gingiv_Diag': 41202,
    # 'Gingiv_Date': 41202,
    'Alcohol_Diag': 41202,
    # 'Alcohol_Date': 41202,
    # 'rs429358': 1220,
    # 'rs7412': 1220,
    'APOE_alles': 1220  # remove E4 E4 from the data
}

depression_medications = [
    'fluoxetine', 'sertraline', 'escitalopram', 'paroxetine', 'citalopram',
    'venlafaxine', 'duloxetine', 'desvenlafaxine', 'bupropion', 'mirtazapine',
    'trazodone', 'amitriptyline', 'nortriptyline', 'imipramine',
    'clomipramine',
    'phenelzine', 'tranylcypromine', 'isocarboxazid'
]
stress_medications = [
    'sertraline', 'escitalopram', 'fluoxetine', 'paroxetine', 'venlafaxine',
    'duloxetine', 'alprazolam', 'diazepam', 'lorazepam', 'clonazepam',
    'buspirone', 'propranolol', 'atenolol', 'metoprolol', 'hydroxyzine',
    'pregabalin', 'amitriptyline', 'nortriptyline', 'imipramine', 'gabapentin',
    'quetiapine', 'olanzapine', 'risperidone'
]
Anxiety_medications = [
    'escitalopram', 'sertraline', 'paroxetine', 'fluoxetine', 'venlafaxine',
    'duloxetine', 'alprazolam', 'diazepam', 'lorazepam', 'clonazepam',
    'buspirone', 'propranolol', 'atenolol', 'metoprolol', 'hydroxyzine',
    'pregabalin', 'amitriptyline', 'imipramine', 'nortriptyline', 'gabapentin',
    'diphenhydramine', 'hydroxyzine'
]

depression_regex = re.compile(
    r'\b(?:' + '|'.join(depression_medications) + r')\b')
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

depression_ICD10s_reg = re.compile(
    r'\b(?:' + '|'.join(depression_ICD10s) + r')\b')
anxiety_ICD10s_reg = re.compile(r'\b(?:' + '|'.join(anxiety_ICD10s) + r')\b')
alcohol_ICD10s_reg = re.compile(r'\b(?:' + '|'.join(alcohol_ICD10s) + r')\b')
alzheimer_ICD10s_reg = re.compile(
    r'\b(?:' + '|'.join(alzheimer_ICD10s) + r')\b')
cardiovascular_ICD10s_reg = re.compile(
    r'\b(?:' + '|'.join(cardiovascular_ICD10s) + r')\b')
diabetes_ICD10s_reg = re.compile(r'\b(?:' + '|'.join(diabetes_ICD10s) + r')\b')
gingiv_ICD10s_reg = re.compile(r'\b(?:' + '|'.join(gingiv_ICD10s) + r')\b')
mild_cognitive_ICD10s_reg = re.compile(
    r'\b(?:' + '|'.join(mild_Cognitive_ICD10s) + r')\b')
smoking_ICD10s_reg = re.compile(r'\b(?:' + '|'.join(smoking_ICD10s) + r')\b')
cholesterol_ICD10s_reg = re.compile(
    r'\b(?:' + '|'.join(cholesterol_ICD10s) + r')\b')

columns = {
    # Depression:
    'BipolarAndMajorDepressionStat': {
        ('Bipolar I Disorder', 'Bipolar II Disorder',
         'Prefer not to answer'): np.nan,
        ('No Bipolar or Depression'): 0,
        ('Single Probable major depression episode'): 1,
        ('Probable Recurrent major depression (moderate)'): 2,
        ('Probable Recurrent major depression (severe)'): 3},

    'ProfessionalInformedDepression': {
        ('Prefer not to answer', 'Do not Know'): np.nan,
        ('Yes'): 1,
        ('No'): 0},

    'DurationOfWorstDepression': {('Prefer not to answer'): np.nan,
                                  ('Less than a month',
                                   'Between one and three months',
                                   'Over three months, but less than six months'): 1,
                                  (
                                      'Over six months, but less than 12 months',
                                      'One to two years',
                                      'Over two years'): 2},

    'AgeLastEpisodeOfDepression': None,

    # Anxiety:
    'ProfessionalInformedAnxiety': {
        ('Prefer not to answer', 'Do not Know'): np.nan,
        ('Yes'): 1,
        ('No'): 0},

    # Sleeplessness:
    'SleepDuration': None,
    'Insomnia': {('Prefer not to answer'): np.nan,
                 ('Usually'): 1,
                 ('Sometimes'): 2,
                 ('Never/rarely'): 3},

    # Smoking:
    'AgeOfStopSmoking': None,
    'SmokingStatus': {('Prefer not to answer'): np.nan,
                      ('Never'): 0,
                      ('Previous'): 1,
                      ('Current'): 2},
    'SmokingPacksYears': None,

    # Alcohol:
    'AlcoholFrequency': {('Prefer not to answer'): np.nan,
                         ('Never', 'Special occasions only',
                          'One to three times a month'): 0,
                         ('Once or twice a week',
                          'Three or four times a week'): 1,
                         ('Daily or almost daily'): 2},
    # Cardiovascular:
    'CardioMedications': {('Prefer not to answer', 'Do not know'): np.nan,
                          ('Blood pressure medication'): 1,
                          ('Cholesterol lowering medication', 'Insulin',
                           'None of the above'): 0},
    'MedsCardiovascular': {('Prefer not to answer', 'Do not know'): np.nan,
                           ('Blood pressure medication'): 1,
                           ('Cholesterol lowering medication', 'Insulin',
                            'Hormone replacement therapy',
                            'Oral contraceptive pill or minipill',
                            'None of the above', 'Do not know'): 0},
    'DBloodPressure': None,
    'SBloodPressure': None,
    'CardiovascularDiagnosis': {('Prefer not to answer'): np.nan,
                                ('None of the above'): 0,
                                ('Heart attack', 'Angina', 'Stroke'): 1,
                                ('High blood pressure'): 2},

    # BMI:
    'BMI': None,

    # ActivityPhysical:
    'DurationModerateActivity': None,
    'ModeratePhysicalActivity_NumDaysWeek': None,
    'VigorousPhysicalActivity_NumDaysWeek': None,
    'WalkingActivity_NumDaysWeek': None,

    # ActivitySocial:
    'SocialActivities': {
        ('Prefer not to answer'): np.nan,
        ('None of the above'): 0,
        ('Sports club or gym', 'Pub or social club',
         'Religious group',
         'Adult education class',
         'Other group activity'): 1},
    'FrequencyFriendsFamilyVisits': {
        ('Prefer not to answer', 'Do not know'): np.nan,
        ('No friends/family outside household',
         'Never or almost never',
         'Once every few months'): 0,
        ('About once a month',
         'About once a week',
         '2-4 times a week', 'Almost daily'): 1},

    # ActivityMental:
    'MentalActivities': {
        ('Prefer not to answer', 'Do not know'): np.nan,
        ('None of the above',
         'Sports club or gym', 'Pub or social club',
         'Religious group',
         'Other group activity'): 0,
        ('Adult education class'): 1},
    'PlaysComputer': {('Prefer not to answer'): np.nan,
                      ('Never/rarely'): 0,
                      ('Sometimes', 'Often'): 1},
    'DurationHeavyDIY': {('Prefer not to answer', 'Do not know'): np.nan,
                         ('Less than 15 minutes',
                          'Between 15 and 30 minutes',
                          'Between 30 minutes and 1 hour'): 1,
                         ('Between 1 and 1.5 hours',
                          'Between 1.5 and 2 hours',
                          'Between 2 and 3 hours', 'Over 3 hours'): 2},

    # Sex:
    'BiologicalSex': None,

    # Age:
    'BirthYear': None,
    'AgeAtDeath': None,

    # Education:
    'Qualifications': {
        ('Prefer not to answer', 'Do not know'): np.nan,
        ('None of the above'): 0,
        ('O levels/GCSEs or equivalent',
         'A levels/AS levels or equivalent',
         'NVQ or HND or HNC or equivalent',
         'College or University degree',
         'CSEs or equivalent',
         'Other professional qualifications eg: nursing, teaching'): 1},

    # Stress:
    'IllnessInjuriesStressLast2Years': {('Prefer not to answer'): np.nan,
                                        ('None of the above'): 0,
                                        (
                                        ' Serious illness, injury or assault to yourself',
                                        ' Serious illness, injury or assault of a close relative'
                                        'Death of a close relative',
                                        'Death of a spouse or partner',
                                        'Martital separation/divorce'): 1},

    # Diabetes:
    'InsulinMedications': {('Prefer not to answer', 'Do not know'): np.nan,
                           ('Insulin'): 1,
                           ('Cholesterol lowering medication',
                            'Blood pressure medication',
                            'None of the above'): 0},
    'MedsInsulin': {('Prefer not to answer', 'Do not know'): np.nan,
                    ('Insulin'): 1,
                    ('Blood pressure medication',
                     'Cholesterol lowering medication',
                     'Hormone replacement therapy',
                     'Oral contraceptive pill or minipill',
                     'None of the above'): 0},
    'Diabetes diagnosed': {('Prefer not to answer'): np.nan,
                           ('No', 'Do not know'): 0,
                           ('Yes'): 1},

    # Cholesterol:
    'CardioInsulinMedications': { ('Prefer not to answer', 'Do not know'): np.nan,
                                ('Cholesterol lowering medication'): 1,
                                 ('Insulin', 'Blood pressure medication',
                                  'None of the above'): 0},

    'HDL_Cholesterol': None,
    'LDL_Cholesterol': None,

    # APOE_E4E4:
    'rs429358': None,
    'rs7412': None,

    # Gums:
    'DentalProblems': {('Prefer not to answer'): np.nan,
                       ('None of the above', 'Loose teeth', 'Toothache',
                        'Dentures'): 0,
                       ('Bleeding gums', 'Mouth ulcers',
                        'Painful gums'): 1, },

    # Hearing:
    'HearingAid': {('Prefer not to answer'): np.nan,
                   ('No'): 0,
                   ('Yes'): 1},

    'HearingDifficulties': {('Prefer not to answer', 'Do not know'): np.nan,
                            ('No', 'I am completely deaf'): 0,
                            ('Yes'): 1},

    # Therapies:
    'SeenGpForNervesAnxietyTensionDepression': {
        ('Prefer not to answer', 'Do not know'): np.nan,
        ('No'): 0,
        ('Yes'): 1},

    'SeenShrinkForNervesAnxietyTensionDepression': {
        ('Prefer not to answer', 'Do not know'): np.nan,
        ('No'): 0,
        ('Yes'): 1},
}

# df = df.dropna(subset=['AgeRecruitment', 'BirthYear'])
#     fillna = {col:-99 for col in df.columns if col.startswith('Age')}
#     fillna.update( {col:'' for col in df.columns if col.endswith('Date')} )
#     fillna.update( dict(ICD10_Diags='', DBloodPressure='', DBloodPressure='') )
#     df.fillna(fillna, inplace=True)
