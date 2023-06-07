import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from tqdm import trange

from model import Model
from model import convertToOneHot, DataSet_meta
from sklearn.model_selection import train_test_split
from sklearn.metrics import jaccard_score
import optuna

#data
import data_processor
import data_analyser
import pandas as pd

# label_col = 37
# label_col = 40

SMALL = "saves/small_data.csv"
BIG = "saves/temp.csv"
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
# 43: 'Alzheimer_Diag',
43: 'Alcohol_Diag',
44: 'APOE_alles',
45: 'Moderate_Activity',
}

def see_res(path):
    old = np.load
    np.load = lambda *a, **k: old(*a, allow_pickle=True, **k)
    k = np.load(path)
    # print(k)
    np.load = old
    return k


def get_data(use_full):
    # saved = False
    saved = True
    features = []
    if not saved: # if you want to load parsed dataset from disk
        if use_full:
            data = pd.read_csv(BIG).fillna("nan")
        else:
            data = pd.read_csv(SMALL).fillna("nan")
        analyser = data_analyser.DataAnalyser(data)
        processor = data_processor.DataProcessor(data, analyser)
        processor.process_data()
        features = processor.new_data.columns
        label_col = processor.new_data.columns.get_loc("Alzheimer_Diag")
        dataset = processor.new_data.to_numpy()
        if use_full:
            labels = dataset[:, label_col]
            dataset = np.delete(dataset, label_col, axis=1)
            np.save("saves/dataset_big.npy", dataset)
            np.save("saves/labels_big.npy", labels)
            print("saved full database")
        else:
            labels = dataset[:, label_col]
            dataset = np.delete(dataset, label_col, axis=1)
            np.save("saves/dataset.npy", dataset)
            np.save("saves/labels.npy", labels)
            print("saved small database")
    else:
        np_load_old = np.load
        # modify the default parameters of np.load
        np.load = lambda *a, **k: np_load_old(*a, allow_pickle=True, **k)
        if use_full:
            dataset = np.load("saves/dataset_big.npy")
            labels = np.load("saves/labels_big.npy")
        else:
            dataset = np.load("saves/dataset.npy")
            labels = np.load("saves/labels.npy")
        np.load = np_load_old
    return dataset, labels, features



def train_model(): # for training without trials
    global model
    model = Model(**model_params)
    train_acces, train_losses, val_acces, val_losses = model.train(
                                            dataset=dataset,**training_params)

    # print(f"train_acces - {train_acces[-1]}, train_losses - "
    #       f"{train_losses[-1]}, "
    #       f"val_acces - {val_acces[-1]}, "
    #       f"val_losses - {val_losses[-1]}")
    print(f"train loss {train_losses[-1]}")


    print("Fin training")




def lstg_objective(trial):
    global model

    training_params['lr'] = trial.suggest_loguniform('learning_rate', 0.05, 0.1)
    training_params["num_epoch"] = trial.suggest_categorical('num_epoch',[2000, 5000, 7000, 9000])

    model = Model(**model_params)
    train_acces, train_losses, val_acces, val_losses = model.train(
        dataset=dataset,
        **training_params

        )

    alpha_mat_valid = model.get_prob_alpha(X_valid)
    print("In trial:---------------------")
    print(
        "union feat: {}".format(sum(np.sum(alpha_mat_valid > 0, axis=0) > 0)))
    print("median feat: {}".format(
        np.median(np.sum(alpha_mat_valid > 0, axis=1))))

    loss = val_losses[-1]

    return loss

def callback(study, trial):
    global best_model
    if study.best_trial == trial:
        best_model = model
        print("inside cal back!!!")


def adjust_labels_for_model(labels_init):
    return np.c_[np.ones(labels_init.shape) - labels_init,labels_init]

def return_labels(labels):
    return np.zeros(labels.shape[0]) + labels[:,1]

def predict(data):
    global model
    return model.test(data)
def uiget_gates(data):
    global model
    return model.get_prob_alpha(data)

def unison_shuffled_copies(a, b):
    assert len(a) == len(b)
    p = np.random.permutation(len(a))
    return a[p],b[p]


def split_data(dataset_init, labels, label_split, validation):
    data_pos, data_neg = dataset_init[:label_split,:], dataset_init[
                                                       label_split:,:]
    lab_pos, lab_neg = labels[:label_split, :], labels[label_split:, :]
    X_train_pos, X_test_pos, Y_train_pos, Y_test_pos = train_test_split(
        data_pos,
        lab_pos,
        test_size=test_portion,
        train_size=train_portion)

    X_train_neg, X_test_neg, Y_train_neg, Y_test_neg = train_test_split(
        data_neg,
        lab_neg,
        test_size=test_portion,
        train_size=train_portion)


    if validation:
        X_valid_pos, X_test_pos, Y_valid_pos, Y_test_pos = train_test_split(
            X_test_pos,
            Y_test_pos,
            test_size=0.5,
            train_size=0.5)
        X_valid_neg, X_test_neg, Y_valid_neg, Y_test_neg = train_test_split(
            X_test_neg,
            Y_test_neg,
            test_size=0.5,
            train_size=0.5)
        X_valid = np.vstack([X_valid_neg, X_valid_pos])
        Y_valid = np.vstack([Y_valid_neg, Y_valid_pos])
        X_valid , Y_valid = unison_shuffled_copies(X_valid, Y_valid)


    X_train = np.vstack([X_train_neg, X_train_pos])
    X_test = np.vstack([X_test_neg, X_test_pos])
    Y_train = np.vstack([Y_train_neg, Y_train_pos])
    Y_test = np.vstack([Y_test_neg, Y_test_pos])
    X_train, Y_train = unison_shuffled_copies(X_train, Y_train)
    X_test, Y_test = unison_shuffled_copies(X_test, Y_test)
    if not validation:
        X_valid = X_test
        Y_valid = Y_test
    return X_train, Y_train, X_test, Y_test, X_valid , Y_valid



def visulize_gates(trainTest='train'):
    print(f'Gates for {trainTest}:')
    gates = []
    for i in range(reps):
        gates.append(np.load(f"saves/gates_{trainTest}{i}.npy"))
        gates[0] += gates[i]
        gates[0] /= gates[0].shape[0]
    gates[0] = gates[0] / (reps*2)
    avg_gate = gates[0]

    corr = (pd.DataFrame(avg_gate).corr()*1e2).fillna(0).astype(int)
    corr = corr.values[np.tril_indices(avg_gate.shape[1], -1)]
    print('Correlation values', corr.mean())
    print(corr)

    sbjSum = avg_gate.sum(1)
    print(f'Num of sbjs with 0 feats: {(sbjSum==0).sum()}')
    feat_sum = np.sum(avg_gate, axis=0, keepdims=True)
    # k = see_res("saves/features.npy")
    # k = np.delete(k, 44)
    feat_sum = feat_sum.flatten()
    # dict = {k[i]:feat_sum[i] for i in range(47) if feat_sum[i]>0}
    # print(dict)
    dict = {COLS_DICT[i]: feat_sum[i] for i in range(46) if feat_sum[i] > 0}
    dict = {k:int(v*1e2) for k,v in dict.items()}
    feats = sorted(dict.items(), key=lambda x: -x[1])
    print(feats)
    fig, ax = plt.subplots()
    ax.bar(np.arange(46), feat_sum.flatten())
    plt.xlabel("feature idx")
    plt.ylabel("percentage chosen")
    plt.title(f"results for {trainTest}")
    plt.show()
    np.save(f"saves/abg_gates_{trainTest}.npy",avg_gate)
    k = jaccard_score(return_labels(Y_test), predictions)
    print(f"for test_set - jaccard score - {k}")


if __name__ == '__main__':
    #___init___
    use_full_dataset = True #True means full dataset # False means small
    use_vaildation = False
    use_optuna = False  # must use validation
    train = True  # train model
    train_portion = 0.8
    test_portion = 1 - train_portion
    reps = 10


    dataset_init, labels_init, features = get_data(use_full_dataset)
    label_split = np.where(labels_init == 0)[0][0]
    labels = adjust_labels_for_model(labels_init)
    for i in trange(reps):
        print(f"run {i}")
        X_train, Y_train, X_test, Y_test, X_valid , Y_valid = split_data(dataset_init, labels,
                                                       label_split, use_vaildation)
        pd.DataFrame(X_train).to_csv(f"saves/DataLabels/X_train{i}.csv")
        pd.DataFrame(Y_train).to_csv(f"saves/DataLabels/Y_train{i}.csv")
        pd.DataFrame(X_test).to_csv(f"saves/DataLabels/X_test{i}.csv")
        pd.DataFrame(Y_test).to_csv(f"saves/DataLabels/Y_test{i}.csv")
        pd.DataFrame(X_valid).to_csv(f"saves/DataLabels/X_valid{i}.csv")
        pd.DataFrame(Y_valid).to_csv(f"saves/DataLabels/Y_valid{i}.csv")


        # organize data for model


        dataset = DataSet_meta(
            **{'_data': X_train, '_labels': Y_train,
               '_meta': Y_train,
               '_valid_data': X_valid, '_valid_labels': Y_valid,
               '_valid_meta': Y_valid,
               '_test_data': X_test, '_test_labels': Y_test,
               '_test_meta': Y_test, })

        # model_params = {'input_node': X_train.shape[1],
        #                 'hidden_layers_node': [500, 100, 1],
        #                 'output_node': 2,  # classification
        #                 'feature_selection': True,
        #                 'gating_net_hidden_layers_node': [100],
        #                 'display_step': 1000,
        #                 'activation_gating': 'tanh',
        #                 'activation_pred': 'l_relu',
        #                 'lam': 1, 'gamma1': 0.1}

        model_params = {'input_node': X_train.shape[1],
                        'hidden_layers_node': [100, 50, 30],
                        'output_node': 2, #classification
                        'feature_selection': True,
                        'gating_net_hidden_layers_node': [100],
                        'display_step': 1000,
                        'activation_gating': 'tanh',
                        'activation_pred': 'l_relu',
                        'lam': .25,'gamma1':50, 'gamma2':100}

        training_params = {'batch_size': X_train.shape[0]}



        # using trials optima :
        model = None

        # __optona__
        if use_optuna:
            best_model = None
            study = optuna.create_study(pruner=None)
            # originally 20 trials
            study.optimize(lstg_objective, n_trials=3, callbacks=[callback])

        # not using optima
        else:
            model = Model(**model_params)
            training_params = ({**training_params, 'lr':
                0.05418309743636845, 'num_epoch': 2000})  # from optima
            if train:
                train_model()

        print("train -------- acc")
        a, l = model.evaluate(X_train,Y_train, Y_train, None)

        predictions = predict(X_test)
        # patient = X_test[:2,:]
        gates_test = uiget_gates(X_test)
        gates_train = uiget_gates(X_train)
        # np.save("saves/features.npy",features)
        np.save(f"saves/gates_test{i}.npy",gates_test)
        pd.DataFrame(gates_test).to_csv(f"saves/csvGatesLabels/gates_test{i}.csv")
        np.save(f"saves/gates_train{i}.npy",gates_train)
        pd.DataFrame(gates_test).to_csv(f"saves/csvGatesLabels/gates_train{i}.csv")
        np.save("saves/gates_labels.npy",return_labels(Y_test))
        pd.DataFrame(return_labels(Y_test)).to_csv(f"saves/csvGatesLabels/gates_labels{i}.csv")
        np.save("saves/preds.npy",predictions)
        pd.DataFrame(predictions).to_csv(f"saves/csvGatesLabels/preds{i}.csv")
    visulize_gates('train')
    visulize_gates('test')
    print()

