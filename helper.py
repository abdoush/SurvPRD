import numpy as np
import pandas as pd
import logging


def configure_logger(logdir, name):
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s:%(message)s')
    file_handler = logging.FileHandler(logdir + '/' + name + '.log')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    return logger


class Results:
    @staticmethod
    def save(g, exp_dir, correct_labels_p, censoring_p_orig=0.50):
        np.save(f'results/{exp_dir}/g_exp_best_train_hist_{censoring_p_orig}_{correct_labels_p}.npy',
                np.array(g.exp.best_train_hist))
        np.save(f'results/{exp_dir}/g_exp_best_test_hist_{censoring_p_orig}_{correct_labels_p}.npy',
                np.array(g.exp.best_test_hist))
        np.save(f'results/{exp_dir}/g_accs_test_{censoring_p_orig}_{correct_labels_p}.npy', np.array(g.accs_test))
        np.save(f'results/{exp_dir}/g_accs_train_{censoring_p_orig}_{correct_labels_p}.npy', np.array(g.accs_train))

        np.save(f'results/{exp_dir}/g_ones_accs_test_{censoring_p_orig}_{correct_labels_p}.npy',
                np.array(g.ones_accs_test))
        np.save(f'results/{exp_dir}/g_ones_accs_train_{censoring_p_orig}_{correct_labels_p}.npy',
                np.array(g.ones_accs_train))

        np.save(f'results/{exp_dir}/g_exp_new_unknown_true_e_test_{censoring_p_orig}_{correct_labels_p}.npy',
                g.exp.new_unknown_true_e_test)
        np.save(f'results/{exp_dir}/g_exp_new_unknown_true_e_train_{censoring_p_orig}_{correct_labels_p}.npy',
                g.exp.new_unknown_true_e_train)

        np.save(f'results/{exp_dir}/g_exp_x_test_{censoring_p_orig}_{correct_labels_p}.npy', g.exp.x_test)
        np.save(f'results/{exp_dir}/g_exp_x_train_{censoring_p_orig}_{correct_labels_p}.npy', g.exp.x_train)
        np.save(f'results/{exp_dir}/g_exp_e_test_{censoring_p_orig}_{correct_labels_p}.npy', g.exp.e_test)
        np.save(f'results/{exp_dir}/g_exp_e_train_{censoring_p_orig}_{correct_labels_p}.npy', g.exp.e_train)
        np.save(f'results/{exp_dir}/g_exp_new_known_time_test_{censoring_p_orig}_{correct_labels_p}.npy',
                g.exp.new_known_time_test)
        np.save(f'results/{exp_dir}/g_exp_new_known_time_train_{censoring_p_orig}_{correct_labels_p}.npy',
                g.exp.new_known_time_train)

        np.save(f'results/{exp_dir}/g_exp_e_test{censoring_p_orig}_{correct_labels_p}.npy', g.exp.e_test)
        np.save(f'results/{exp_dir}/g_exp_e_train_{censoring_p_orig}_{correct_labels_p}.npy', g.exp.e_train)

        for i in range(len(g.log_dfs)):
            g.log_dfs[i].to_csv(f'results/{exp_dir}/dfs/log_df_{i}.csv', index=False)

    def load(self, exp_dir, correct_labels_p, max_iters, censoring_p_orig=0.50):
        self.g_exp_best_train_hist = np.load(f'results/{exp_dir}/g_exp_best_train_hist_{censoring_p_orig}_{correct_labels_p}.npy')
        self.g_exp_best_test_hist = np.load(f'results/{exp_dir}/g_exp_best_test_hist_{censoring_p_orig}_{correct_labels_p}.npy')
        self.g_accs_test = np.load(f'results/{exp_dir}/g_accs_test_{censoring_p_orig}_{correct_labels_p}.npy')
        self.g_accs_train = np.load(f'results/{exp_dir}/g_accs_train_{censoring_p_orig}_{correct_labels_p}.npy')

        self.g_ones_accs_test = np.load(f'results/{exp_dir}/g_ones_accs_test_{censoring_p_orig}_{correct_labels_p}.npy')
        self.g_ones_accs_train = np.load(f'results/{exp_dir}/g_ones_accs_train_{censoring_p_orig}_{correct_labels_p}.npy')

        self.g_exp_new_unknown_true_e_test = np.load(f'results/{exp_dir}/g_exp_new_unknown_true_e_test_{censoring_p_orig}_{correct_labels_p}.npy')
        self.g_exp_new_unknown_true_e_train = np.load(f'results/{exp_dir}/g_exp_new_unknown_true_e_train_{censoring_p_orig}_{correct_labels_p}.npy')

        self.g_exp_x_test = np.load(f'results/{exp_dir}/g_exp_x_test_{censoring_p_orig}_{correct_labels_p}.npy')
        self.g_exp_x_train = np.load(f'results/{exp_dir}/g_exp_x_train_{censoring_p_orig}_{correct_labels_p}.npy')
        self.g_exp_e_test = np.load(f'results/{exp_dir}/g_exp_e_test_{censoring_p_orig}_{correct_labels_p}.npy')
        self.g_exp_e_train = np.load(f'results/{exp_dir}/g_exp_e_train_{censoring_p_orig}_{correct_labels_p}.npy')
        self.g_exp_new_known_time_test = np.load(f'results/{exp_dir}/g_exp_new_known_time_test_{censoring_p_orig}_{correct_labels_p}.npy')
        self.g_exp_new_known_time_train = np.load(f'results/{exp_dir}/g_exp_new_known_time_train_{censoring_p_orig}_{correct_labels_p}.npy')

        self.g_exp_e_test = np.load(f'results/{exp_dir}/g_exp_e_test{censoring_p_orig}_{correct_labels_p}.npy')
        self.g_exp_e_train = np.load(f'results/{exp_dir}/g_exp_e_train_{censoring_p_orig}_{correct_labels_p}.npy')

        self.g_log_dfs = []
        for i in range(max_iters):
            df_temp = pd.read_csv(f'results/{exp_dir}/dfs/log_df_{i}.csv')
            self.g_log_dfs.append(df_temp)
