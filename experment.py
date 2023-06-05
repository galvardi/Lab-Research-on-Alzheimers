import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from model import Model
from model import convertToOneHot, DataSet_meta
from sklearn.model_selection import train_test_split
import optuna


#data
import data_processor
import data_analyser
import pandas as pd

LABEL_COL_SMALL = 37
LABEL_COL_BIG = 40

SMALL = "saves/temp_small.csv"
BIG = "saves/temp_tomer.csv"

def get_data(use_full):
    # saved = False
    saved = True
    if not saved: # if you want to load parsed dataset from disk
        if use_full:
            data = pd.read_csv(BIG).fillna('Prefer not to answer')
        else:
            data = pd.read_csv(SMALL).fillna('Prefer not to answer')
        analyser = data_analyser.DataAnalyser(data)
        processor = data_processor.DataProcessor(data, analyser)
        processor.process_data()
        processor.new_data = processor.new_data.dropna() #remove nan
        non_numeric_columns = processor.new_data.select_dtypes(exclude=[
            pd.np.number]).columns.tolist() #todo fix them
        processor.new_data = processor.new_data.drop(non_numeric_columns,
                                                     axis=1)

        #tring to extract all alzhimers people

        # for i, col in enumerate(processor.new_data):
        #     print(i, col)
        #
        # size = len(processor.new_data.Alzheimer_Diag)
        # sick = np.arange(size)[processor.new_data['Alzheimer_Diag'] == 1]
        # alzhmiers_only = processor.new_data.loc[sick, :]
        # dataset_sick = alzhmiers_only.to_numpy()

        dataset = processor.new_data.to_numpy()
        if use_full:
            labels = dataset[:, LABEL_COL_BIG]
            dataset = np.delete(dataset, LABEL_COL_BIG, axis=1)
            np.save("saves/dataset_big.npy", dataset)
            np.save("saves/labels_big.npy", labels)
        else:
            labels = dataset[:, LABEL_COL_SMALL]
            dataset = np.delete(dataset, LABEL_COL_SMALL, axis=1)
            np.save("saves/dataset.npy", dataset)
            np.save("saves/labels.npy", labels)
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
    return dataset, labels



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

    training_params['lr'] = trial.suggest_loguniform('learning_rate', 0.01,
                                                     0.1)
    training_params["num_epoch"] = trial.suggest_categorical('num_epoch',
                                                             [2000, 3000,
                                                              5000, 7000])

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

def get_gates(data):
    global model
    return model.get_prob_alpha(data)


if __name__ == '__main__':
    #___init___
    use_full_dataset = False #True means full dataset # False means small
    use_vaildation = False
    train_portion = 0.7
    test_portion = 0.3

    dataset_init, labels_init = get_data(use_full_dataset)
    labels = adjust_labels_for_model(labels_init)
    X_valid = np.empty(0)
    Y_valid = np.empty(0)
    X_train, X_test, Y_train, Y_test = train_test_split(dataset_init,
                                                        labels,
                                                        test_size=test_portion, train_size=train_portion)
    if use_vaildation:
        X_test, X_valid = np.array_split(X_test,2,axis=0)
        Y_test, Y_valid = np.array_split(Y_test,2,axis=0)

    np.random.seed(10)

    # organize data for model


    dataset = DataSet_meta(
        **{'_data': X_train, '_labels': Y_train,
           '_meta': Y_train,
           '_valid_data': X_valid, '_valid_labels': Y_valid,
           '_valid_meta': Y_valid,
           '_test_data': X_test, '_test_labels': Y_test,
           '_test_meta': Y_test, })

    model_params = {'input_node': X_train.shape[1],
                    'hidden_layers_node': [500, 100, 1],
                    'output_node': 2, #classification
                    'feature_selection': True,
                    'gating_net_hidden_layers_node': [100],
                    'display_step': 1000,
                    'activation_gating': 'tanh',
                    'activation_pred': 'l_relu',
                    'lam': 1,'gamma1': 0.1}

    training_params = {'batch_size': X_train.shape[0]}



    # using trials optima :
    model = None


    # saved_model = True
    model = Model(**model_params)
    # model.load("saves/checkpoint") #load doesn't work

    # best_model = None
    # study = optuna.create_study(pruner=None)
    # # originally 20 trials
    # study.optimize(lstg_objective, n_trials=2, callbacks=[callback])

    # not using optima
    training_params = ({**training_params, 'lr':
        0.07512140104607376, 'num_epoch': 5000})  # from optima
    train_model()
    # model.save(1, "saves/")
    # patient, patient_lab = dataset.test_data[0], dataset.test_labels[0]
    # model.sess.run()
    predictions = model.test(X_test)
    # gates = get_gates(patient)
    print()

