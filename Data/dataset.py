import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


class Dataset:
    def __init__(self, dataset_file_path=None, number_of_splits=5, test_fract=0,
                 drop_percentage=0, events_only=True, drop_feature=None, random_state_seed=20,
                 random_seed=20, some_e=None, new_e=None, verbose=False):
        self.dataset_file_path = dataset_file_path
        self.number_of_splits = number_of_splits
        self.drop_percentage = drop_percentage
        self.events_only = events_only
        self.drop_feature = drop_feature
        self.some_e = some_e
        self.new_e = new_e
        self.df = self._load_data()
        self.percentage_cens = (1-self.df['E'].mean())
        self.rnd = np.random.RandomState(seed=random_state_seed)
        self.f = self.rnd.uniform(0.1, 0.9, self.df['E'].shape[0])
        self.df['New_E'] = self.df['E']
        self.df['New_T'] = self.df['T']
        self.df['Some_E'] = self.df['E']
        self.fill_some_e(some_e)
        self.apply_new_e(new_e)
        self.rest_df, self.test_df = self._get_test_split(fract=test_fract, seed=random_seed)
        self.n_splits = self._get_n_splits(seed=random_seed)
        if verbose:
            self.print_dataset_summery()

    def fill_some_e(self, e):
        if e is not None:
            self.df.loc[self.df['E']==1, 'Some_E'] = e

    def apply_new_e(self, e):
        if e is not None:
            self.df.loc[self.df['E']==1, 'New_E'] = e
            #n_faked_censored = self.df[(self.df['E'] == 1) & (self.df['New_E'] == 0)].shape[0]
            cond = (self.df['E']==1) & (self.df['New_E']==0)


            self.df.loc[cond, 'New_T'] = self.df.loc[cond, 'T'] - self.f[cond] * self.df.loc[cond, 'T']


    def get_dataset_name(self):
        pass

    def _preprocess_x(self, x_df):
        pass

    def _preprocess_y(self, y_df, normalizing_val=None):
        pass

    def _preprocess_e(self, e_df):
        pass

    def _fill_missing_values(self, x_train_df, x_val_df=None, x_test_df=None, x_tune_df=None):
        pass

    def _load_data(self):
        pass

    def get_x_dim(self):
        return self.df.shape[1]-5

    def _scale_x(self, x_train_df, x_val_df=None, x_test_df=None, x_tune_df=None):
        pass

    def print_dataset_summery(self):
        s = 'Dataset Description =======================\n'
        s += 'Dataset Name: {}\n'.format(self.get_dataset_name())
        s += 'Dataset Shape: {}\n'.format(self.df.shape)
        s += 'Events: %.2f %%\n' % (self.df['E'].sum()*100 / len(self.df))
        s += 'NaN Values: %.2f %%\n' % (self.df.isnull().sum().sum()*100 / self.df.size)
        s += f'Size and Events % in splits: '
        for split in self.n_splits:
            s += '({}, {:.2f}%), '.format((split.shape[0]), (split["E"].mean()*100))
        s += '\n'
        if self.test_df is not None:
            s += '-------------------------------------------\n'
            s += 'Hold-out Testset % of Data: {:.2f}%\n'.format((self.test_df.shape[0] * 100 / self.df.shape[0]))
            s += 'Hold-out Testset Size and Events %: ({:}, {:.2f}%) \n'.format(self.test_df.shape[0], (self.test_df["E"].mean()*100))
        s += '===========================================\n'
        print(s)
        return s

    @staticmethod
    def max_transform(df, cols, powr):
        df_transformed = df.copy()
        for col in cols:
            df_transformed[col] = ((df_transformed[col]) / df_transformed[col].max()) ** powr
        return df_transformed

    @staticmethod
    def log_transform(df, cols):
        df_transformed = df.copy()
        for col in cols:
            df_transformed[col] = np.abs(np.log(df_transformed[col] + 1e-8))
        return df_transformed

    @staticmethod
    def power_transform(df, cols, powr):
        df_transformed = df.copy()
        for col in cols:
            df_transformed[col] = df_transformed[col] ** powr
        return df_transformed

    def _get_test_split(self, fract=0.4, seed=20):
        if fract == 0:
            return self.df, None
        rest_df, test_df = train_test_split(self.df, test_size=fract, random_state=seed, shuffle=True, stratify=self.df['E'])
        return rest_df, test_df

    def _get_n_splits(self, seed=20):
        k = self.number_of_splits
        train_df = self.rest_df
        df_splits = []
        for i in range(k, 1, -1):
            train_df, test_df = train_test_split(train_df, test_size=(1 / i), random_state=seed, shuffle=True,
                                                 stratify=train_df['E'])
            df_splits.append(test_df)
            if i == 2:
                df_splits.append(train_df)
        return df_splits


# New Fake Censoring ===================================================================================================

    def get_train(self):
        df_splits_temp = self.n_splits.copy()
        train_df_splits = [df_splits_temp[i] for i in range(len(df_splits_temp))]
        train_df = pd.concat(train_df_splits)

        x_train_df, y_train_df, e_train_df, new_y_train_df, new_e_train_df = self._split_columns_new(train_df)

        (self.x_train_df, self.y_train_df, self.e_train_df,
         self.new_y_train_df, self.new_e_train_df) = (x_train_df, y_train_df, e_train_df,
                                                                            new_y_train_df, new_e_train_df)

        self._fill_missing_values(x_train_df)

        x_train = self._preprocess_x(x_train_df)

        x_train = self._scale_x(x_train)

        y_train = self._preprocess_y(y_train_df)
        new_y_train = self._preprocess_y(new_y_train_df)

        e_train = self._preprocess_e(e_train_df)

        # new
        new_e_train = self._preprocess_e(new_e_train_df)

        ye_train = np.array(list(zip(y_train, e_train)))
        new_ye_train = np.array(list(zip(new_y_train, new_e_train)))

        return (
            x_train, y_train, e_train, new_y_train, new_e_train
        )

    def get_train_val_from_splits_new(self, val_id):
        df_splits_temp = self.n_splits.copy()
        val_df = df_splits_temp[val_id]
        train_df_splits = [df_splits_temp[i] for i in range(len(df_splits_temp)) if i not in [val_id]]
        train_df = pd.concat(train_df_splits)

        x_train_df, y_train_df, e_train_df, new_y_train_df, new_e_train_df, some_e_train_df = self._split_columns_new(train_df)
        x_val_df, y_val_df, e_val_df, new_y_val_df, new_e_val_df, some_e_val_df = self._split_columns_new(val_df)

        self._fill_missing_values(x_train_df, x_val_df)

        x_train, x_val = self._preprocess_x(x_train_df), self._preprocess_x(x_val_df)

        x_train, x_val = self._scale_x(x_train, x_val)

        y_train, y_val = self._preprocess_y(y_train_df), self._preprocess_y(y_val_df)
        new_y_train, new_y_val = self._preprocess_y(new_y_train_df), self._preprocess_y(new_y_val_df)

        e_train, e_val = self._preprocess_e(e_train_df), self._preprocess_e(e_val_df)

        # new
        new_e_train, new_e_val = self._preprocess_e(new_e_train_df), self._preprocess_e(new_e_val_df)
        some_e_train, some_e_val = self._preprocess_e(some_e_train_df), self._preprocess_e(some_e_val_df)

        ye_train, ye_val = np.array(list(zip(y_train, e_train))), np.array(list(zip(y_val, e_val)))
        new_ye_train, new_ye_val = np.array(list(zip(new_y_train, new_e_train))), np.array(
            list(zip(new_y_val, new_e_val)))
        some_ye_train, some_ye_val = np.array(list(zip(new_y_train, some_e_train))), np.array(
            list(zip(new_y_val, some_e_val)))

        return (
            x_train, ye_train, y_train, e_train, new_ye_train, new_y_train, new_e_train, some_ye_train, some_e_train,
            x_val, ye_val, y_val, e_val, new_ye_val, new_y_val, new_e_val, some_ye_val, some_e_val
        )

    def get_train_val_test_from_splits_new(self, val_id, test_id):
        df_splits_temp = self.n_splits.copy()
        val_df = df_splits_temp[val_id]
        test_df = df_splits_temp[test_id]
        train_df_splits = [df_splits_temp[i] for i in range(len(df_splits_temp)) if i not in [val_id, test_id]]
        train_df = pd.concat(train_df_splits)

        x_train_df, y_train_df, e_train_df, new_y_train_df, new_e_train_df, some_e_train_df = self._split_columns_new(train_df)
        x_val_df, y_val_df, e_val_df, new_y_val_df, new_e_val_df, some_e_val_df = self._split_columns_new(val_df)
        x_test_df, y_test_df, e_test_df, new_y_test_df, new_e_test_df, some_e_test_df = self._split_columns_new(test_df)

        self._fill_missing_values(x_train_df, x_val_df, x_test_df)

        x_train, x_val, x_test = self._preprocess_x(x_train_df), \
                                 self._preprocess_x(x_val_df), \
                                 self._preprocess_x(x_test_df)

        x_train, x_val, x_test = self._scale_x(x_train, x_val, x_test)

        y_normalizing_val = y_train_df.max()

        y_train, y_val, y_test = self._preprocess_y(y_train_df, normalizing_val=y_normalizing_val), \
                                 self._preprocess_y(y_val_df, normalizing_val=y_normalizing_val), \
                                 self._preprocess_y(y_test_df, normalizing_val=y_normalizing_val)

        new_y_normalizing_val = new_y_train_df.max()

        new_y_train, new_y_val, new_y_test = self._preprocess_y(new_y_train_df, normalizing_val=new_y_normalizing_val), \
                                             self._preprocess_y(new_y_val_df, normalizing_val=new_y_normalizing_val), \
                                             self._preprocess_y(new_y_test_df, normalizing_val=new_y_normalizing_val)

        e_train, e_val, e_test = self._preprocess_e(e_train_df), \
                                 self._preprocess_e(e_val_df), \
                                 self._preprocess_e(e_test_df)

        new_e_train, new_e_val, new_e_test = self._preprocess_e(new_e_train_df), \
                                             self._preprocess_e(new_e_val_df), \
                                             self._preprocess_e(new_e_test_df)

        some_e_train, some_e_val, some_e_test = self._preprocess_e(some_e_train_df), \
                                                self._preprocess_e(some_e_val_df), \
                                                self._preprocess_e(some_e_test_df)

        ye_train, ye_val, ye_test = np.array(list(zip(y_train, e_train))), \
                                    np.array(list(zip(y_val, e_val))), \
                                    np.array(list(zip(y_test, e_test)))

        new_ye_train, new_ye_val, new_ye_test = np.array(list(zip(new_y_train, new_e_train))), \
                                                np.array(list(zip(new_y_val, new_e_val))), \
                                                np.array(list(zip(new_y_test, new_e_test)))

        some_ye_train, some_ye_val, some_ye_test = np.array(list(zip(new_y_train, some_e_train))), \
                                                   np.array(list(zip(new_y_val, some_e_val))), \
                                                   np.array(list(zip(new_y_test, some_e_test)))

        return (x_train, ye_train, y_train, e_train, new_ye_train, new_y_train, new_e_train, some_ye_train, some_e_train,
                x_val, ye_val, y_val, e_val, new_ye_val, new_y_val, new_e_val, some_ye_val, some_e_val,
                x_test, ye_test, y_test, e_test, new_ye_test, new_y_test, new_e_test, some_ye_test, some_e_test
                )

    @staticmethod
    def _split_columns_new(df):
        y_df = df['T']
        e_df = df['E']
        new_y_df = df['New_T']
        new_e_df = df['New_E']
        some_e_df = df['Some_E']

        x_df = df.drop(['T', 'E', 'New_T', 'New_E', 'Some_E'], axis=1)
        return x_df, y_df, e_df, new_y_df, new_e_df, some_e_df

    def get_wrongly_labeled_data(self, wrongly_labeled_p, random_state_seed=20):
        rnd = np.random.RandomState(seed=random_state_seed)

        x = self.df.drop(columns=['T', 'E']).values
        e = self.df['E'].values
        t = self.df['T'].values

        new_unknown_true_e = e.copy()
        num_wrong_e = int(wrongly_labeled_p * new_unknown_true_e.sum())
        num_correct_e = int(new_unknown_true_e.sum() - num_wrong_e)
        ee = np.concatenate([np.ones(num_correct_e), np.zeros(num_wrong_e)])
        rnd.shuffle(ee)
        # print('before')
        # print(new_unknown_true_e.sum())
        new_unknown_true_e[new_unknown_true_e == 1] = ee
        # print('after')
        # print(new_unknown_true_e.sum())
        f = rnd.uniform(0.1, 0.9, len(e))
        c = t - (f * t)

        new_known_time = np.where(new_unknown_true_e, t, c)

        return x, e, new_known_time, new_unknown_true_e

# End New Fake Censoring ===============================================================================================


class CMPASS(Dataset):
    def _load_data(self):
        df = pd.read_csv(self.dataset_file_path)
        print(df.shape)
        df = df.groupby('unit_nr').apply(lambda df: df.sample(n=20, random_state=20))
        df = df[~df['unit_nr'].isin([100, 7])]
        self.unit_nr = df['unit_nr'].values
        df.drop(columns=['unit_nr'], inplace=True)

        ohdf = df
        return ohdf

    def get_dataset_name(self):
        return 'CMPASS'

    def _preprocess_x(self, x_df):
        return x_df

    def _preprocess_y(self, y_df, normalizing_val=None):
        if normalizing_val is None:
            normalizing_val = y_df.max()
        return ((y_df / normalizing_val).to_numpy() ** 0.5).astype('float32')

    def _preprocess_e(self, e_df):
        return e_df.to_numpy().astype('float32')

    def _scale_x(self, x_train_df, x_val_df, x_test_df=None, x_tune_df=None):
        scaler = StandardScaler().fit(x_train_df)
        x_train = scaler.transform(x_train_df)
        x_val = scaler.transform(x_val_df)
        if (x_tune_df is not None) & (x_test_df is not None):
            x_test = scaler.transform(x_test_df)
            x_tune = scaler.transform(x_tune_df)
            return x_train, x_val, x_test, x_tune
        elif x_test_df is not None:
            x_test = scaler.transform(x_test_df)
            return x_train, x_val, x_test
        else:
            return x_train, x_val



# Simulated data -------------------------------------------------------------------------------------------


class SimPHDataset(Dataset):
    def __init__(self, n_samples=1000, dims_hazard_ratios=[1, 2, 3, 4], baseline_hazard=0.1, percentage_cens=0.5, random_state_seed=20,
                 dataset_file_path=None, number_of_splits=5, test_fract=0, random_seed=20, some_e=None, new_e=None, verbose=False):
        self.n_samples = n_samples
        self.dims_hazard_ratios = dims_hazard_ratios
        self.baseline_hazard = baseline_hazard
        self.percentage_cens = percentage_cens
        self.random_state_seed = random_state_seed
        self.dataset_file_path = dataset_file_path
        super().__init__(dataset_file_path=self.dataset_file_path, number_of_splits=number_of_splits, test_fract=test_fract, random_seed=random_seed, some_e=some_e, new_e=new_e, verbose=verbose)

    def _load_data(self):
        self.simds = SimPHData(n_samples=self.n_samples, dims_hazard_ratios=self.dims_hazard_ratios, baseline_hazard=self.baseline_hazard,
                            percentage_cens=self.percentage_cens, random_state_seed=self.random_state_seed)

        ohdf = pd.DataFrame(self.simds.x, columns=[f'x{i}' for i in range(self.simds.x.shape[1])])
        ohdf['T'] = self.simds.t
        ohdf['E'] = self.simds.e
        return ohdf

    def get_dataset_name(self):
        return 'SimPHDataset'

    def _preprocess_x(self, x_df):
        return x_df

    def _preprocess_y(self, y_df, normalizing_val=None):
        if normalizing_val is None:
            normalizing_val = y_df.max()
        return ((y_df / normalizing_val).to_numpy() ** 0.5).astype('float32')

    def _preprocess_e(self, e_df):
        return e_df.to_numpy().astype('float32')

    def _scale_x(self, x_train_df, x_val_df=None, x_test_df=None, x_tune_df=None):
        scaler = StandardScaler().fit(x_train_df)
        x_train = scaler.transform(x_train_df)
        if (x_tune_df is not None) & (x_test_df is not None) & (x_val_df is not None):
            x_val = scaler.transform(x_val_df)
            x_test = scaler.transform(x_test_df)
            x_tune = scaler.transform(x_tune_df)
            return x_train, x_val, x_test, x_tune
        elif (x_test_df is not None) & (x_val_df is not None):
            x_val = scaler.transform(x_val_df)
            x_test = scaler.transform(x_test_df)
            return x_train, x_val, x_test
        elif x_val_df is not None:
            x_val = scaler.transform(x_val_df)
            return x_train, x_val
        else:
            return x_train


class SimPHData:
    def __init__(self, n_samples, dims_hazard_ratios, baseline_hazard, percentage_cens, random_state_seed=20):
        self.n_samples = n_samples
        self.hazard_ratios = dims_hazard_ratios
        self.n_dims = len(dims_hazard_ratios)
        self.percentage_cens = percentage_cens
        self.baseline_hazard = baseline_hazard

        self.rnd = np.random.RandomState(seed=random_state_seed)
        self.x, self.t, self.e, self.true_t, self.c, self.frac = self._generate_survival_data(n_samples=self.n_samples,
                                                                          n_dims=self.n_dims,
                                                                          hazard_ratios=self.hazard_ratios,
                                                                          baseline_hazard=self.baseline_hazard,
                                                                          percentage_cens=self.percentage_cens,
                                                                          rnd=self.rnd)

    @staticmethod
    def _generate_marker(n_samples, n_dims, hazard_ratios, baseline_hazard, rnd):
        # create synthetic risk score
        x = rnd.randn(n_samples, n_dims)

        # create linear model
        hazard_ratios = np.array(hazard_ratios)
        logits = np.dot(x, np.log(hazard_ratios))

        # draw actual survival times from exponential distribution,
        # refer to Bender et al. (2005), https://doi.org/10.1002/sim.2059
        u = rnd.uniform(size=n_samples)
        event_time = -np.log(u) / (baseline_hazard * np.exp(logits))

        # compute the actual concordance in the absence of censoring
        x = np.squeeze(x)
        return x, event_time  # , actual[0]

    def _generate_survival_data(self, n_samples, n_dims, hazard_ratios, baseline_hazard, percentage_cens, rnd):
        x, true_t = self._generate_marker(n_samples, n_dims, hazard_ratios, baseline_hazard, rnd)
        # print('(1 - percentage_cens):',(1 - percentage_cens))
        number_events = int(round((1 - round(percentage_cens, 5)) * n_samples, 5))
        # print('number_events', number_events)
        event = np.concatenate([np.ones(number_events), np.zeros(n_samples - number_events)])
        rnd.shuffle(event)
        f = rnd.uniform(0.1, 0.9, n_samples)
        c = true_t - (f * true_t)
        censored_time = np.where(event, true_t, c)
        #et = Surv.from_arrays(event=event, time=censored_time)
        return x, censored_time, event, true_t, c, f

    def get_wrongly_labeled_data(self, wrongly_labeled_p, random_state_seed=20):
        rnd = np.random.RandomState(seed=random_state_seed)
        new_unknown_true_e = self.e.copy()
        num_wrong_e = int(wrongly_labeled_p * new_unknown_true_e.sum())
        num_correct_e = int(new_unknown_true_e.sum() - num_wrong_e)
        ee = np.concatenate([np.ones(num_correct_e), np.zeros(num_wrong_e)])
        rnd.shuffle(ee)
        # print('before')
        # print(new_unknown_true_e.sum())
        new_unknown_true_e[new_unknown_true_e == 1] = ee
        # print('after')
        # print(new_unknown_true_e.sum())
        new_known_time = np.where(new_unknown_true_e, self.t, self.c)

        return self.x, self.e, new_known_time, new_unknown_true_e

