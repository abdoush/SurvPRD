from lifelines import CoxPHFitter
from sksurv.ensemble import RandomSurvivalForest
import pandas as pd
import numpy as np
from Data.dataset import SimPHData, CMPASS
import time
from sklearn.model_selection import train_test_split
from tensorflow.keras.optimizers import Adam
import tensorflow.keras.backend as K
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras import layers
from tensorflow import keras
import random
import os
import matplotlib.pyplot as plt
import seaborn as sns


class Model:
    def __init__(self):
        self.model = None

    def fit(self, train):
        pass

    def score(self, test):
        pass


class CPH(Model):
    def fit(self, train):
        x_train, y_train, e_train = train
        xy_train_df = pd.DataFrame(x_train)
        xy_train_df['T'] = y_train
        xy_train_df['E'] = e_train

        self.model = CoxPHFitter(penalizer=0.1).fit(xy_train_df, 'T', 'E')

    def score(self, test):
        if self.model is None:
            print('Model not fitted')
            return 0.5
        x_test, y_test, e_test = test
        xy_test_df = pd.DataFrame(x_test)
        xy_test_df['T'] = y_test
        xy_test_df['E'] = e_test

        cindex = self.model.score(xy_test_df, scoring_method='concordance_index')
        return cindex

    def score_ci(self, test):
        if self.model is None:
            print('Model not fitted')
            return 0.5
        x_test, y_test, e_test = test
        xy_test_df = pd.DataFrame(x_test)
        xy_test_df['T'] = y_test
        xy_test_df['E'] = e_test

        cindex = self.model.score(xy_test_df, scoring_method='concordance_index')
        return cindex


class Experiment:
    def __init__(self, censoring_p_orig=0.5, correctly_labeled_p=0.5, test_p=0.5):
        self.censoring_p_orig = censoring_p_orig
        self.test_p = test_p
        self.correctly_labeled_p = correctly_labeled_p

    def train(self, x, t, e):
        #t_diff = self.ds.t - t  # (for the wrongly labeled) difference between the true event time and the applied censoring time (the model is trained on the censored time but wrongly labeled as events)
        mdl = CPH()
        #mdl.fit(train=(self.x, t, e))
        mdl.fit(train=(x, t, e))
        return mdl

    def score(self, mdl, x, t, e):
        #ci = mdl.score(test=(self.x, t, e))
        ci = mdl.score(test=(x, t, e))
        return ci

    def score_ci(self, mdl, x, t, e):
        #ci = mdl.score(test=(self.x, t, e))
        ci = mdl.score_ci(test=(x, t, e))
        return ci

    def train_some_e_events_part_only(self, some_e_train):
        full_some_e_train = self.e_train.copy()
        full_some_e_train[full_some_e_train == 1] = np.array(some_e_train)
        some_model = self.train(self.x_train, self.new_known_time_train, full_some_e_train)
        return some_model

    def score_some_e_events_part_only(self, some_model, some_e_test):
        full_some_e_test = self.e_test.copy()
        full_some_e_test[full_some_e_test == 1] = np.array(some_e_test)
        some_ci = self.score(some_model, self.x_test, self.new_known_time_test, full_some_e_test)
        return some_ci

    def train_score_some_e_events_part_only(self, some_e_train): # train on some_e_train and score in the correct test
        some_mdl = self.train_some_e_events_part_only(some_e_train)
        some_ci_test = self.score(some_mdl, self.x_test, self.new_known_time_test, self.new_unknown_true_e_test)
        return some_ci_test

    def score_some_e_events_part_only_on_correct_model(self, some_e_test):
        full_some_e_test = self.e_test.copy()
        full_some_e_test[full_some_e_test == 1] = np.array(some_e_test)
        some_ci = self.score(self.correct_model, self.x_test, self.new_known_time_test, full_some_e_test)
        return some_ci

    def update_latest_model(self):
        if self.latest_task == 'train_find_in_test':
            # full_some_e_train = self.e_train.copy()
            # full_some_e_train[full_some_e_train == 1] = np.array(some_e)
            # self.latest_e_train = full_some_e_train
            self.latest_model = self.train(self.x_train, self.new_known_time_train, self.latest_e_train)
        else: #'test_find_in_train'
            # full_some_e_test = self.e_test.copy()
            # full_some_e_test[full_some_e_test == 1] = np.array(some_e)
            # self.latest_e_test = full_some_e_test
            self.latest_model = self.train(self.x_test, self.new_known_time_test, self.latest_e_test)

    def score_some_e_events_part_only_on_latest_model(self, some_e):
        if self.latest_task == 'train_find_in_test':
            full_some_e_test = self.e_test.copy()
            full_some_e_test[full_some_e_test == 1] = np.array(some_e)
            some_ci = self.score(self.latest_model, self.x_test, self.new_known_time_test, full_some_e_test)
        else: # 'test_find_in_train'
            full_some_e_train = self.e_train.copy()
            full_some_e_train[full_some_e_train == 1] = np.array(some_e)
            some_ci = self.score(self.latest_model, self.x_train, self.new_known_time_train, full_some_e_train)
        return some_ci

    def score_some_e_events_part_only_on_latest_model_ci(self, some_e):
        if self.latest_task == 'train_find_in_test':
            full_some_e_test = self.e_test.copy()
            full_some_e_test[full_some_e_test == 1] = np.array(some_e)
            some_ci = self.score_ci(self.latest_model, self.x_test, self.new_known_time_test, full_some_e_test)
        else: # 'test_find_in_train'
            full_some_e_train = self.e_train.copy()
            full_some_e_train[full_some_e_train == 1] = np.array(some_e)
            some_ci = self.score_ci(self.latest_model, self.x_train, self.new_known_time_train, full_some_e_train)
        return some_ci

    def update_e_and_flip_task(self, best_e):
        if self.latest_task == 'train_find_in_test':
            full_some_e_test = self.e_test.copy()
            full_some_e_test[full_some_e_test == 1] = np.array(best_e)
            #print('best test', full_some_e_test)
            self.latest_e_test = full_some_e_test.copy()

            self.latest_task = 'test_find_in_train'
        else: # 'test_find_in_train'
            full_some_e_train = self.e_train.copy()
            full_some_e_train[full_some_e_train == 1] = np.array(best_e)
            #print('best train', full_some_e_train)
            self.latest_e_train = full_some_e_train.copy()

            self.latest_task = 'train_find_in_test'


class CPHExperimentSim(Experiment):
    def __init__(self, n_samples, dims_hazard_ratios=[1, 2, 3, 4], baseline_hazard=0.1, censoring_p_orig=0.5,
                 random_state_seed=20, correctly_labeled_p=0.5, test_p=0.5):
        super().__init__(censoring_p_orig, correctly_labeled_p, test_p)
        self.censoring_p_orig = censoring_p_orig
        self.test_p = test_p
        self.correctly_labeled_p = correctly_labeled_p

        self.ds = SimPHData(n_samples=n_samples, dims_hazard_ratios=dims_hazard_ratios, baseline_hazard=baseline_hazard,
                            percentage_cens=censoring_p_orig, random_state_seed=random_state_seed)
        self.x, self.e, self.new_known_time, self.new_unknown_true_e = self.ds.get_wrongly_labeled_data(
            wrongly_labeled_p=(1 - correctly_labeled_p))

        (self.x_train, self.x_test,
         self.e_train, self.e_test,
         self.new_known_time_train, self.new_known_time_test,
         self.new_unknown_true_e_train, self.new_unknown_true_e_test) = train_test_split(self.x, self.e,
                                                                                         self.new_known_time, self.new_unknown_true_e,
                                                                                         test_size = test_p,
                                                                                         stratify=self.e,
                                                                                         random_state = 42)

        self.latest_e_train, self.latest_e_test = self.new_unknown_true_e_train.copy(), self.new_unknown_true_e_test.copy()

        np.random.shuffle(self.latest_e_train)
        np.random.shuffle(self.latest_e_test)

        self.correct_model = self.train(self.x_train, self.new_known_time_train, self.new_unknown_true_e_train)  # the model trained on the correct labels (just for reference)
        self.correct_score = self.score(self.correct_model, self.x_test, self.new_known_time_test, self.new_unknown_true_e_test)

        self.correct_model_test = self.train(self.x_test, self.new_known_time_test, self.new_unknown_true_e_test)  # the model trained on the correct labels (just for reference)
        self.correct_score_test = self.score(self.correct_model_test, self.x_train, self.new_known_time_train, self.new_unknown_true_e_train)

        self.latest_model = None
        self.latest_task = 'train_find_in_test' # or 'test_find_in_train'
        self.best_train_hist = []
        self.best_test_hist = []

    def train(self, x, t, e):
        mdl = CPH()
        mdl.fit(train=(x, t, e))
        return mdl


class CPHExperimentCMPASS(Experiment):
    def __init__(self, censoring_p_orig=0.5, correctly_labeled_p=0.5, test_p=0.5):
        super().__init__(censoring_p_orig, correctly_labeled_p, test_p)

        self.ds = CMPASS(dataset_file_path='data/CMPASS_Censored.csv')


        self.x, self.e, self.new_known_time, self.new_unknown_true_e = self.ds.get_wrongly_labeled_data(
            wrongly_labeled_p=(1 - correctly_labeled_p))

        rnd = np.random.RandomState(seed=20)
        ids = rnd.choice(range(1, 100), 50, replace=False)
        df1 = pd.DataFrame()
        df1['unit_nr'] = self.ds.unit_nr
        df1['E'] = self.e

        df1 = df1[['unit_nr', 'E']].groupby('unit_nr').first().reset_index()
        ids_train, ids_test = train_test_split(df1['unit_nr'].values, test_size=0.5, stratify=df1['E'].values,
                                               random_state=42)
        self.x_train = self.x[np.isin(self.ds.unit_nr,ids_train)]
        self.x_test = self.x[np.isin(self.ds.unit_nr,ids_test)]
        self.e_train = self.e[np.isin(self.ds.unit_nr,ids_train)]
        self.e_test = self.e[np.isin(self.ds.unit_nr,ids_test)]
        self.new_known_time_train = self.new_known_time[np.isin(self.ds.unit_nr,ids_train)]
        self.new_known_time_test = self.new_known_time[np.isin(self.ds.unit_nr,ids_test)]
        self.new_unknown_true_e_train = self.new_unknown_true_e[np.isin(self.ds.unit_nr,ids_train)]
        self.new_unknown_true_e_test = self.new_unknown_true_e[np.isin(self.ds.unit_nr,ids_test)]

        print(self.e_train.sum(), self.e_test.sum())

        self.latest_e_train, self.latest_e_test = self.new_unknown_true_e_train.copy(), self.new_unknown_true_e_test.copy()

        np.random.shuffle(self.latest_e_train)
        np.random.shuffle(self.latest_e_test)

        self.correct_model = self.train(self.x_train, self.new_known_time_train, self.new_unknown_true_e_train)  # the model trained on the correct labels (just for reference)
        self.correct_score = self.score(self.correct_model, self.x_test, self.new_known_time_test, self.new_unknown_true_e_test)

        self.correct_model_test = self.train(self.x_test, self.new_known_time_test, self.new_unknown_true_e_test)  # the model trained on the correct labels (just for reference)
        self.correct_score_test = self.score(self.correct_model_test, self.x_train, self.new_known_time_train, self.new_unknown_true_e_train)

        self.latest_model = None
        self.latest_task = 'train_find_in_test' # or 'test_find_in_train'
        self.best_train_hist = []
        self.best_test_hist = []


    def train(self, x, t, e):
        mdl = CPH()
        mdl.fit(train=(x, t, e))
        return mdl


class ExpOneBitChange:
    def __init__(self, n_samples, dims_hazard_ratios=[1, 2, 3, 4], baseline_hazard=0.1, censoring_p_orig=0.5,
                 random_state_seed=20, correctly_labeled_p=0.5, verbose=False):
        self.verbose = verbose
        self.censoring_p_orig = censoring_p_orig
        self.results_df = pd.DataFrame(
            columns=['censoring_p_orig', 'exp_id', 'flipped_e_counter', 'flipped_e_idx', 'is_wrong_event_label',
                     'base_ci', 'new_ci', 'y_original', 'y_censored', 'y_diff', 'diff_from_base_ci'])
        self.ds = SimPHData(n_samples=n_samples, dims_hazard_ratios=dims_hazard_ratios, baseline_hazard=baseline_hazard,
                            percentage_cens=censoring_p_orig, random_state_seed=random_state_seed)
        self.x, self.e, self.new_known_time, self.new_unknown_true_e = self.ds.get_wrongly_labeled_data(
            wrongly_labeled_p=(1 - correctly_labeled_p))
        self.model, self.base_ci = self.train(self.new_known_time,
                                              self.e)  # the model is trained on the erraronous labels (e not the true_e)

    def train(self, t, e):
        self.t_diff = self.ds.t - t  # (for the wrongly labeled) difference between the true event time and the applied censoring time (the model is trained on the censored time but wrongly labeled as events)
        mdl = CPH()
        mdl.fit(train=(self.x, t, e))
        ci = mdl.score(test=(self.x, t, e))
        return mdl, ci

    def score_some_e(self, some_e):
        ci = self.model.score(test=(self.x, self.new_known_time, some_e))
        return ci

    def score_some_e_events_part_only(self, some_e):
        e = self.e.copy()
        e[e == 1] = np.array(some_e)
        ci = self.model.score(test=(self.x, self.new_known_time, e))
        return ci

    @staticmethod
    def flip_e(e, i):
        new_e = e.copy()
        idxs = np.where(e == 1)[0]
        new_e[idxs[i]] = 0
        return new_e

    def run(self, exp_id=0):
        start = time.time()
        # ae = np.array(self.true_e)
        idxs = np.where(self.e == 1)[0]
        n = len(idxs)
        if self.verbose:
            print(f'Number Changes: {n}')
        for i, idx in enumerate(idxs):
            # print('index', i)
            some_e = self.flip_e(self.e, i)
            is_error_label = int(self.new_unknown_true_e[idx] == 0)
            some_ci = self.score_some_e(some_e)
            y_diffi = self.t_diff[idx]
            y_original = self.ds.t[idx]  # self.original_y_train[idx]
            y_censored = self.new_known_time[idx]  # self.trained_on_y_train[idx]
            self.results_df.loc[len(self.results_df)] = {'censoring_p_orig': self.censoring_p_orig, 'exp_id': exp_id,
                                                         'flipped_e_counter': i, 'flipped_e_idx': idx,
                                                         'is_wrong_event_label': is_error_label,
                                                         'base_ci': self.base_ci, 'new_ci': some_ci,
                                                         'y_original': y_original, 'y_censored': y_censored,
                                                         'y_diff': y_diffi,
                                                         'diff_from_base_ci': (some_ci - self.base_ci)}
            # display(self.results_df)
            print('Bits% {:.4f}'.format(((i+1) / n)), end='\r', flush=True)
        if self.verbose:
            print('')
            print(f'Computation Time: {time.time() - start} s')


class ExpTrueVsRandom:
    def __init__(self, dataset_class, dataset_filepath, test_fract=0.3, percentage_cens=0.5):
        self.dataset_class = dataset_class
        self.dataset_filepath = dataset_filepath
        # self.percentage_cens = percentage_cens

        self.test_fract = test_fract
        ds = dataset_class(dataset_file_path=dataset_filepath, test_fract=test_fract, percentage_cens=percentage_cens)
        self.percentage_cens = ds.percentage_cens
        self.events_number = int(ds.df['E'].sum())
        self.exp_df = ds.df.copy().drop(columns=['New_E', 'Some_E'])

    def run(self, repetitions=10):
        self.repetitions = repetitions

    def plot_true_vs_random_box(self):
        for i in range(9):
            fig, ax = plt.subplots(1, 3, figsize=(15, 3))
            # p_rel = ttest_rel(ctrains_true_ps[i], ctrains_some_ps[i]).pvalue
            # p_ind = ttest_ind(ctrains_true_ps[i], ctrains_some_ps[i]).pvalue
            plt.title('Correctly Labeled Events: {:.1f}'.format(((i + 1) * 0.1)))
            ax[0].boxplot([np.array(self.ctrains_true_ps[i]), np.array(self.ctrains_some_ps[i])]);
            ax[0].set_xticklabels(['True', 'Random'])
            ax[0].set_title('Train')
            ax[1].boxplot([np.array(self.cvals_true_ps[i]), np.array(self.cvals_some_ps[i])]);
            ax[1].set_xticklabels(['True', 'Random'])
            ax[1].set_title('Val')
            ax[2].boxplot([np.array(self.accuracies[i]), np.array(self.accuracies_true[i])])
            ax[2].set_xticklabels(['Accuracy', '1s Accuracy'])

    def plot_ones_acc_vs_cindex(self):
        pass

    def plot(self):
        self.plot_true_vs_random_box()
        self.plot_ones_acc_vs_cindex()


class ExpMultiTrues(ExpTrueVsRandom):
    def run(self, repetitions=10):
        super().run(repetitions)
        self.ctrains_true_ps, self.cvals_true_ps = [], []
        self.ctrains_some_ps, self.cvals_some_ps = [], []
        self.accuracies, self.accuracies_true = [], []

        for p in range(1, 10):
            events_p = 0.1 * p  # percentage of correctly labeled events
            base_e = [1 if i <= int(self.events_number * events_p) else 0 for i in range(self.events_number)]

            ctrains_true, cvals_true = [], []
            ctrains_some, cvals_some = [], []
            accs = []
            accs_true = []

            for i in range(self.repetitions):
                new_e = base_e.copy()
                random.shuffle(new_e)
                some_e = base_e.copy()
                # random.seed(20)
                random.shuffle(some_e)
                ds = self.dataset_class(dataset_file_path=self.dataset_filepath, test_fract=self.test_fract,
                                        percentage_cens=self.percentage_cens, new_e=new_e, some_e=some_e)
                self.exp_df[f'New_E{i}'] = ds.df['New_E']
                self.exp_df[f'Some_E{i}'] = ds.df['Some_E']
                acc = ds.df[ds.df['E'] == 1][ds.df['New_E'] == ds.df['Some_E']].shape[0] / ds.df['E'].sum()
                acc_true = ds.df.loc[(ds.df['New_E'] == 1), 'Some_E'].mean()
                accs.append(acc)
                accs_true.append(acc_true)
                (
                    x_train, ye_train, y_train, e_train, new_ye_train, new_y_train, new_e_train, some_ye_train,
                    some_e_train,
                    x_val, ye_val, y_val, e_val, new_ye_val, new_y_val, new_e_val, some_ye_val, some_e_val
                ) = ds.get_train_val_from_splits_new(val_id=1)

                mdl = CPH()
                # fit and test on the true e
                mdl.fit(train=(x_train, new_y_train, new_e_train))
                ctrain_true = mdl.score(test=(x_train, new_y_train, new_e_train))
                cval_true = mdl.score(test=(x_val, new_y_val, new_e_val))
                ctrains_true.append(ctrain_true), cvals_true.append(cval_true)

                #                 ctrain_true, cval_true = mdl.score(train=(x_train, new_y_train, new_e_train), test=(x_val, new_y_val, new_e_val))
                #                 ctrains_true.append(ctrain_true), cvals_true.append(cval_true)

                # fit and test on random e
                mdl.fit(train=(x_train, new_y_train, some_e_train))
                ctrain_some = mdl.score(test=(x_train, new_y_train, some_e_train))
                cval_some = mdl.score(test=(x_val, new_y_val, some_e_val))
                ctrains_some.append(ctrain_some), cvals_some.append(cval_some)

            #                 ctrain_some, cval_some = mdl.score(train=(x_train, new_y_train, some_e_train), test=(x_val, new_y_val, some_e_val))
            #                 ctrains_some.append(ctrain_some), cvals_some.append(cval_some)

            self.ctrains_true_ps.append(ctrains_true), self.cvals_true_ps.append(cvals_true)
            self.ctrains_some_ps.append(ctrains_some), self.cvals_some_ps.append(cvals_some)
            self.accuracies.append(accs)
            self.accuracies_true.append(accs_true)

    def plot_ones_acc_vs_cindex(self):
        fig, ax = plt.subplots(1, 2, figsize=(15, 3))
        ax[0].set_title('Train')
        ax[0].scatter(np.array(self.accuracies_true).reshape(-1), np.array(self.ctrains_true_ps).reshape(-1),
                      label='True Labels', alpha=0.2)
        ax[0].scatter(np.array(self.accuracies_true).reshape(-1), np.array(self.ctrains_some_ps).reshape(-1),
                      label='Random Labels', alpha=0.2)
        # ax[0].scatter(np.array(self.accuracies_true[5:]).reshape(-1), np.array(self.ctrains_some_ps[5:]).reshape(-1), label='High True 1s', alpha=0.2)
        # ax[0].scatter(np.array(range(1, len(self.ctrains_true_ps)+1))*0.1, self.ctrains_true_ps, marker='+', s=100, c='k')
        ax[0].set_xlabel('E Accuracy')
        ax[0].set_ylabel('C-Index')
        ax[0].legend()

        ax[1].set_title('Val')
        ax[1].scatter(np.array(self.accuracies_true).reshape(-1), np.array(self.cvals_true_ps).reshape(-1), alpha=0.2,
                      label='True Labels')
        ax[1].scatter(np.array(self.accuracies_true).reshape(-1), np.array(self.cvals_some_ps).reshape(-1), alpha=0.2,
                      label='Random Labels')
        # ax[1].scatter(np.array(self.accuracies_true[5:]).reshape(-1), np.array(self.cvals_some_ps[5:]).reshape(-1), label='High True 1s', alpha=0.2)

        # ax[1].scatter(np.array(range(1, len(self.cvals_true_ps)+1))*0.1, self.cvals_true_ps, marker='+', s=100, c='k')
        ax[1].set_xlabel('E Accuracy')
        ax[1].set_ylabel('C-Index')
        ax[1].legend()

    def plot_true_vs_random(self):
        self.cdiff_train_ps = []
        self.cdiff_val_ps = []
        self.inc_prob_train_ps = []
        self.inc_prob_val_ps = []
        for i in range(9):
            cdiff_train = np.array(self.ctrains_true_ps[i]) - np.array(self.ctrains_some_ps[i])
            cdiff_val = np.array(self.cvals_true_ps[i]) - np.array(self.cvals_some_ps[i])
            inc_prob_train, inc_prob_val = (cdiff_train > 0).mean(), (cdiff_val > 0).mean()
            self.cdiff_train_ps.append(cdiff_train)
            self.cdiff_val_ps.append(cdiff_val)
            self.inc_prob_train_ps.append(inc_prob_train)
            self.inc_prob_val_ps.append(inc_prob_val)

            fig, ax = plt.subplots(1, 3, figsize=(15, 3))
            # 'Correctly Labeled Events: {:.1f}'.format(((i+1)*0.1))
            fig.suptitle(f'Censoring:{self.percentage_cens:.1f}, Correctly labeled event {((i + 1) * 0.1):.1f}')
            ax[0].set_title('Train C-Index')
            ax[0].plot(self.ctrains_true_ps[i], label='True Events')
            ax[0].plot(self.ctrains_some_ps[i], label='Random Events')
            ax[0].set_xlabel('Iterations')
            ax[0].set_ylabel('CI')
            ax[0].legend()

            ax[1].set_title('Val C-Index')
            ax[1].plot(self.cvals_true_ps[i], label='True Events')
            ax[1].plot(self.cvals_some_ps[i], label='Random Events')
            ax[1].set_xlabel('Iterations')
            ax[1].set_ylabel('CI')
            ax[1].legend()

            ax[2].set_title('P(true>random), Train: {:.2f}, Val: {:.2f}'.format(inc_prob_train, inc_prob_val))
            ax[2].plot(cdiff_train, label='Train C-Index Diff', c='k')
            ax[2].plot(cdiff_val, label='Val C-Index Diff', c='gray')
            plt.axhline(0, c='C2', ls=':')
            ax[2].set_xlabel('Iterations')
            ax[2].set_ylabel('CI diff')
            ax[2].legend()
            fig.tight_layout()

    def plot(self):
        super().plot()
        self.plot_true_vs_random()


class PlottingExp:
    @staticmethod
    def boxplot(df, x, y, x_labels_col, title):
        ax = sns.boxplot(x=x, y=y, data=df);
        x_vals = df.groupby(x)[x_labels_col].mean()
        str_x_vals = ['{:.1f}'.format(k) for k in x_vals]
        plt.xticks(range(0, len(x_vals)), str_x_vals, rotation=90);
        ax.set_xticklabels(ax.get_xticklabels(), rotation=90);
        ax.set_xlabel(x_labels_col)
        ax.set_title(title)

    @staticmethod
    def boxplot_exp(exp_df):
        #     plt.figure()
        #     boxplot(exp_df, x='changed_e_p', y='true_ci_train', x_labels_col='ones_acc', title='Train')
        #     plt.figure()
        #     boxplot(exp_df, x='changed_e_p', y='true_ci_val', x_labels_col='ones_acc', title='Val')

        plt.figure()
        PlottingExp.boxplot(exp_df, x='changed_e_p', y='changed_ci_train', x_labels_col='ones_acc', title='Train')
        plt.axhline(exp_df['true_ci_train'].values[0], ls=':', label='True Labels CI')
        plt.figure()
        PlottingExp.boxplot(exp_df, x='changed_e_p', y='changed_ci_val', x_labels_col='ones_acc', title='Val')
        plt.axhline(exp_df['true_ci_val'].values[0], ls=':', label='True Labels CI')

    @staticmethod
    def linesplot(df, x, y, x_labels_col, label, title):
        # ax = sns.boxplot(x=x, y=y, data=df);
        ms = df.groupby(x)[y].mean().values
        xx = range(len(ms))
        # print(ms)
        stds = df.groupby(x)[y].std().values
        # print(stds)
        plt.plot(xx, ms, label=label)
        plt.fill_between(xx, (ms - stds), (ms + stds), alpha=.1)  # color='b',
        x_vals = df.groupby(x)[x_labels_col].mean()
        str_x_vals = ['{:.1f}'.format(k) for k in x_vals]
        plt.xticks(range(0, len(x_vals)), str_x_vals, rotation=90);
        # plt.xticklabels(ax.get_xticklabels(),rotation=90);
        plt.xlabel(x_labels_col)
        plt.title(title)
        plt.legend()

    @staticmethod
    def lineplot_exp(exp_df, common_title):
        fig = plt.figure()
        PlottingExp.linesplot(exp_df, x='changed_e_p', y='true_ci_train', x_labels_col='ones_acc', label='true', title='Train')
        PlottingExp.linesplot(exp_df, x='changed_e_p', y='changed_ci_train', x_labels_col='ones_acc', label='changed',
                  title='Train')
        fig.suptitle(common_title)

        fig = plt.figure()
        PlottingExp.linesplot(exp_df, x='changed_e_p', y='true_ci_val', x_labels_col='ones_acc', label='true', title='Val')
        PlottingExp.linesplot(exp_df, x='changed_e_p', y='changed_ci_val', x_labels_col='ones_acc', label='changed', title='Val')
        fig.suptitle(common_title)
