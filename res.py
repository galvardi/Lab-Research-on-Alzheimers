import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

COLS_DICT = {
0: 'depress_medication',
1: 'stress_medication',
2: 'anxiety_medication',
3: 'depression_ICD10',
4: 'anxiety_ICD10',
5: 'gingivitis_ICD10',
6: 'diabetes_ICD10',
7: 'cardiovascular_ICD10',
8: 'smoking_ICD10',
9: 'cholesterol_ICD10',
10: 'CardiovascularDiagnosis',
11: 'CardioMedications',
12: 'InsulinMedications',
13: 'CholesterolMedications',
14: 'Diabetes diagnosed',
15: 'AgeOfStopSmoking',
16: 'AgeLastEpisodeOfDepression',
17: 'DurationOfWorstDepression',
18: 'Qualifications',
19: 'AlcoholFrequency',
20: 'ProfessionalInformedAnxiety',
21: 'ProfessionalInformedDepression',
22: 'IllnessInjuriesStressLast2Years',
23: 'FrequencyFriendsFamilyVisits',
24: 'MentalActivities',
25: 'SocialActivities',
26: 'DentalProblems',
27: 'DBloodPressure',
28: 'SBloodPressure',
29: 'PlaysComputer',
30: 'HearingAid',
31: 'LDL_Cholesterol',
32: 'HDL_Cholesterol',
33: 'Insomnia',
34: 'VigorousPhysicalActivity_NumDaysWeek',
35: 'SmokingStatus',
36: 'SmokingPacksYears',
37: 'SleepDuration',
38: 'BMI',
39: 'BipolarAndMajorDepressionStat',
40: 'SeenGpForNervesAnxietyTensionDepression',
41: 'SeenShrinkForNervesAnxietyTensionDepression',
42: 'HearingDifficulties',
43: 'Alcohol_Diag',
44: 'APOE_alles',
45: 'Moderate_Activity',
}

def for_a_single_feature(feat_name, df):
    cols2idxs = {k: str(i) for i, k in COLS_DICT.items()}
    attr = feat_name
    k = cols2idxs[attr]
    indxsT = df.loc[df[k] == 1].index.values
    indxsF = df.loc[df[k] == 0].index.values
    dfX0 = pd.read_csv(
        r"C:\Users\tomer\PycharmProjects\Lab-NN-for-Alzheimers\Lab-NN-For-Alzheimers\saves\DataLabels\X_test4.csv")
    valsT = dfX0.loc[indxsT][k].values
    valsF = dfX0.loc[indxsF][k].values

    X = [valsT, valsF]

    # Grouped violin plot
    fig, ax = plt.subplots()
    ax.violinplot(X)
    # add x-tick labels
    plt.xticks([1, 2], ['Chose', 'NOT Chose'])
    # add a title
    plt.title('Violin plot for ' + attr)
    plt.legend(loc='upper left')
    plt.show()

    print(np.percentile(valsF[valsF > 0], 80))
    print(np.percentile(valsT, 20))


def plot_single_category(name):
    group_cols = [colors[d[i][1]] for i in d.keys() if d[i][1] in [name]]
    names = [i[:10] for i in d.keys() if d[i][1] in [name]]
    plt.bar(range(len(names)), [d[i][0] for i in d.keys() if d[i][1] in [name]],
            color=group_cols)
    plt.xticks(range(len(names)), names, rotation=70)
    # add a title
    plt.title('Sample Visualization for ' + name)
    plt.show()

def plot_all_categories():
    # group_cols = [colors[d[i][1]] for i in d.keys()]
    # names = [i[:10] for i in d.keys()]
    # plt.bar(range(len(names)), [d[i][0] for i in d.keys()],
    #         color=group_cols)
    # plt.xticks(range(len(names)), names, rotation=80)
    # # add a title
    # plt.title('Sample Visualization for all categories')
    # plt.show()

    mtls = [d[i][0] for i in d.keys() if d[i][1] == 'mtl']
    mtbs = [d[i][0] for i in d.keys() if d[i][1] == 'mtb']
    lss = [d[i][0] for i in d.keys() if d[i][1] == 'ls']
    mtl_mean = np.max(mtls).astype(int)
    mtb_mean = np.max(mtbs).astype(int)
    ls_mean = np.max(lss).astype(int)
    plt.bar(range(3), [ls_mean, mtb_mean,mtl_mean],
            color=['red', 'blue', 'green'])
    plt.xticks(range(3), ['lifestyle', 'metabolic', 'mental'])
    plt.title('Sample Visualization for all categories')
    plt.show()


if __name__ == '__main__':
    gates = []
    for i in range(10):
        gates.append(pd.read_csv(fr"C:\Users\tomer\PycharmProjects\Lab-NN-for-Alzheimers\Lab-NN-For-Alzheimers\saves\csvGatesLabels\gates_test{i}.csv").to_numpy()[:,1:])
        gates[0] += gates[i]
    gates[0] = gates[0] / 10
    avg_gate = gates[0]


    d = {
    'depression_ICD10': [96, 'mtl'],
    "SmokingStatus": [70,'ls'],
    'SmokingPacksYears': [57, 'ls'],
    'LDL_Cholesterol': [52, 'mtb'],
    'DurationOfWorstDepression': [51, 'mtl'],
    'anxiety_ICD10': [43, 'mtl'],
    'Moderate_Activity': [33, 'ls'],
    'IllnessInjuriesStressLast2Years': [30, 'mtl'],
    'Alcohol_Diag': [30, 'ls'],
    'AlcoholFrequency': [29, 'ls'],
    'cardiovascular_ICD10': [28, 'mtb'],
    'DBloodPressure': [26, 'mtb'],
    "BMI": [23, 'ls'],
    'APOE_alles': [20, 'mtb'],
    'CardiovascularDiagnosis': [19, 'mtb'],
    'SBloodPressure': [19, 'mtb'],
    'Insomnia': [19, 'mtl'],
    'VigorousPhysicalActivity_NumDaysWeek': [19, 'ls'],
    'HDL_Cholesterol': [18, 'mtb'],
    'FrequencyFriendsFamilyVisits': [14, 'ls'],
    'SeenShrinkForNervesAnxietyTensionDepression': [14, 'mtl'],
    'stress_medication': [12, 'mtl'],
    'diabetes_ICD10': [10, 'mtb'],
    'CardioMedications': [10, 'mtb'],
    'AgeofStopSmoking': [10, 'ls'],
    'AgeLastEpisodeOfDepression': [10, 'mtl'],
    'PlaysComputer': [10, 'ls'],
    'BipolarAndMajorDepressionStat': [10, 'mtl'],
    }

    colors = {"ls":'red', 'mtl':'green', 'mtb':'blue'}
    plot_single_category('ls')
    # plot_single_category('mtl')
    # plot_single_category('mtb')
    plot_all_categories()
    df0 = pd.read_csv(
        "C:/Users/tomer/PycharmProjects/Lab-NN-for-Alzheimers/Lab-NN-For-Alzheimers/saves/csvGatesLabels/gates_test4.csv")
    # for_a_single_feature('SmokingPacksYears', df0)

    # DBloodPressure = test1
    # HDL_Cholesterol = test0, test7, test5
    # LDL_Cholesterol = test0, test1, test4, test6,
    # cardiovascular_ICD10 = test3, test7
    # DurationOfWorstDepression = test2



    # dfx1 = pd.read_csv(
    #     "C:/Users/tomer/PycharmProjects/Lab-NN-for-Alzheimers/Lab-NN-For-Alzheimers/saves/csvGatesLabels/gates_test1.csv")
    # feat_sum1 = np.sum(dfx1.to_numpy(), axis=0, keepdims=True)
    # feat_sum1 /= dfx1.shape[0]
    # feat_sum1 = feat_sum1.flatten()[1:]
    # dict1 = {COLS_DICT[i]: feat_sum1[i] for i in range(46) if feat_sum1[i] > 0}
    # sorted_dict = sorted(dict.items(), key=lambda x: -x[1])
    # sorted_dict1 = sorted(dict1.items(), key=lambda x: -x[1])