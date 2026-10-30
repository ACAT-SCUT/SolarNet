import os
import numpy as np
import pandas as pd
import os
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from utils.timefeatures import time_features
from utils.solar import (
    build_solar_pack,
    build_solar_pack_v3,
    minute_bucket_divisor,
    should_encode_minute,
)
import warnings

warnings.filterwarnings('ignore')


class Dataset_Custom(Dataset):
    _cache = {}

    def __init__(self, root_path, flag='train', size=None,
                 features='S', data_path='ETTh1.csv',
                 target='OT', scale=True, timeenc=0, freq='h', cycle=None,
                 lat=40.0, lon=-83.0, timezone_name='UTC', use_solar_features=True,
                 solar_feature_version=1, train_ratio=0.6): # 新增 lat, lon 默认ohio州坐标
        # size [seq_len, label_len, pred_len]
        if size == None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
            
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq
        self.cycle = cycle
        
        # 物理坐标
        self.lat = lat
        self.lon = lon
        self.timezone_name = timezone_name
        self.use_solar_features = use_solar_features
        self.solar_feature_version = solar_feature_version
        self.train_ratio = float(train_ratio)
        if not 0 < self.train_ratio <= 0.6:
            raise ValueError('train_ratio must be in the interval (0, 0.6]')

        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        cache_key = (
            os.path.join(self.root_path, self.data_path),
            self.features,
            self.target,
            self.scale,
            self.timeenc,
            self.freq,
            self.cycle,
            self.lat,
            self.lon,
            self.timezone_name,
            self.use_solar_features,
            self.solar_feature_version,
            self.seq_len,
            self.train_ratio,
        )

        if cache_key not in self._cache:
            scaler = StandardScaler()
            df_raw = pd.read_csv(os.path.join(self.root_path, self.data_path))

            cols = list(df_raw.columns)
            cols.remove('date')
            df_raw = df_raw[['date'] + cols]

            if self.features == 'M' or self.features == 'MS':
                cols_data = df_raw.columns[1:]
                df_data = df_raw[cols_data]
            elif self.features == 'S':
                cols_data = df_raw.columns[-1:]
                df_data = df_raw[cols_data]

            num_train = int(len(df_raw) * self.train_ratio)
            train_data = df_data[:num_train]
            if self.scale:
                scaler.fit(train_data.values)
                data = scaler.transform(df_data.values).astype(np.float32)
            else:
                data = df_data.values.astype(np.float32)

            full_indices = (np.arange(len(df_raw), dtype=np.int64) % self.cycle)
            if self.use_solar_features:
                solar_pack_builder = (
                    build_solar_pack_v3
                    if self.solar_feature_version == 3
                    else build_solar_pack
                )
                solar_pack = solar_pack_builder(
                    df_raw['date'],
                    self.lat,
                    self.lon,
                    self.cycle,
                    timezone_name=self.timezone_name
                )
            else:
                solar_feature_count = 9 if self.solar_feature_version == 3 else 4
                solar_pack = np.zeros((len(df_raw), solar_feature_count), dtype=np.float32)
                solar_pack[:, 0] = full_indices.astype(np.float32)

            df_stamp = pd.DataFrame({'date': pd.to_datetime(df_raw['date'])})
            if self.timeenc == 0:
                df_stamp['month'] = df_stamp['date'].dt.month
                df_stamp['day'] = df_stamp['date'].dt.day
                df_stamp['weekday'] = df_stamp['date'].dt.weekday
                df_stamp['hour'] = df_stamp['date'].dt.hour
                if should_encode_minute(self.freq):
                    minute_scale = minute_bucket_divisor(self.freq)
                    df_stamp['minute'] = df_stamp['date'].dt.minute // minute_scale
                data_stamp = df_stamp.drop(columns=['date']).to_numpy(dtype=np.float32)
            else:
                data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=self.freq).transpose(1, 0).astype(np.float32)

            self._cache[cache_key] = {
                'scaler': scaler,
                'data': data,
                'data_stamp': data_stamp,
                'solar_pack': solar_pack.astype(np.float32),
                'full_indices': full_indices,
                'length': len(df_raw),
            }

        cached = self._cache[cache_key]
        self.scaler = cached['scaler']

        # 计算切分点
        original_num_train = int(cached['length'] * 0.6)
        num_train = int(cached['length'] * self.train_ratio)
        num_test = int(cached['length'] * 0.2)
        num_vali = cached['length'] - original_num_train - num_test
        # Keep the original validation/test boundaries fixed. Only the train
        # endpoint and scaler fitting data change with train_ratio.
        border1s = [0, original_num_train - self.seq_len, cached['length'] - num_test - self.seq_len]
        border2s = [num_train, original_num_train + num_vali, cached['length']]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]
        self.full_indices = cached['full_indices'][border1:border2]
        self.solar_pack = cached['solar_pack'][border1:border2]
        self.data_x = cached['data'][border1:border2]
        self.data_y = cached['data'][border1:border2]
        self.data_stamp = cached['data_stamp'][border1:border2]

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        # --- 核心修改：返回打包好的 Cycle Info ---
        # 我们需要 Past (seq_len) 和 Future (pred_len) 的所有物理信息
        # 对应的索引区间：
        # Past: [s_begin : s_end]
        # Future: [s_end : s_end + self.pred_len] (注意：不需要 label_len 那段重复的，只看未来)
        
        pack_x = self.solar_pack[s_begin:s_end]                # [seq_len, 4]
        pack_y = self.solar_pack[s_end : s_end + self.pred_len] # [pred_len, 4]
        
        # 拼接在一起返回，满足 "One Variable" 的格式要求
        # 形状: [seq_len + pred_len, 4]
        solar_pack = np.concatenate([pack_x, pack_y], axis=0)

        cycle_index = torch.tensor(self.full_indices[s_end], dtype=torch.long)

        return seq_x, seq_y, seq_x_mark, seq_y_mark, cycle_index, solar_pack

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)

## TODO add cycle
class Dataset_Pred(Dataset):
    def __init__(self, root_path, flag='pred', size=None,
                 features='S', data_path='ETTh1.csv',
                 target='OT', scale=True, inverse=False, timeenc=0, freq='15min', cols=None):
        # size [seq_len, label_len, pred_len]
        # info
        if size == None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        assert flag in ['pred']

        self.features = features
        self.target = target
        self.scale = scale
        self.inverse = inverse
        self.timeenc = timeenc
        self.freq = freq
        self.cols = cols
        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path))
        '''
        df_raw.columns: ['date', ...(other features), target feature]
        '''
        if self.cols:
            cols = self.cols.copy()
            cols.remove(self.target)
        else:
            cols = list(df_raw.columns)
            cols.remove(self.target)
            cols.remove('date')
        df_raw = df_raw[['date'] + cols + [self.target]]
        border1 = len(df_raw) - self.seq_len
        border2 = len(df_raw)

        if self.features == 'M' or self.features == 'MS':
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features == 'S':
            df_data = df_raw[[self.target]]

        if self.scale:
            self.scaler.fit(df_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        tmp_stamp = df_raw[['date']][border1:border2]
        tmp_stamp['date'] = pd.to_datetime(tmp_stamp.date)
        pred_dates = pd.date_range(tmp_stamp.date.values[-1], periods=self.pred_len + 1, freq=self.freq)

        df_stamp = pd.DataFrame(columns=['date'])
        df_stamp.date = list(tmp_stamp.date.values) + list(pred_dates[1:])
        if self.timeenc == 0:
            df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
            df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
            df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
            df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
            df_stamp['minute'] = df_stamp.date.apply(lambda row: row.minute, 1)
            df_stamp['minute'] = df_stamp.minute.map(lambda x: x // 15)
            data_stamp = df_stamp.drop(['date'], 1).values
        elif self.timeenc == 1:
            data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=self.freq)
            data_stamp = data_stamp.transpose(1, 0)

        self.data_x = data[border1:border2]
        if self.inverse:
            self.data_y = df_data.values[border1:border2]
        else:
            self.data_y = data[border1:border2]
        self.data_stamp = data_stamp

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        if self.inverse:
            seq_y = self.data_x[r_begin:r_begin + self.label_len]
        else:
            seq_y = self.data_y[r_begin:r_begin + self.label_len]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return len(self.data_x) - self.seq_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)


class Dataset_Solar(Dataset):
    def __init__(self, root_path, flag='train', size=None,
                 features='S', data_path='ETTh1.csv',
                 target='OT', scale=True, timeenc=0, freq='h', seasonal_patterns=None, cycle=None):
        # size [seq_len, label_len, pred_len]
        # info
        self.seq_len = size[0]
        self.label_len = size[1]
        self.pred_len = size[2]
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq
        self.cycle = cycle

        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = []
        with open(os.path.join(self.root_path, self.data_path), "r", encoding='utf-8') as f:
            for line in f.readlines():
                line = line.strip('\n').split(',')
                data_line = np.stack([float(i) for i in line])
                df_raw.append(data_line)
        df_raw = np.stack(df_raw, 0)
        df_raw = pd.DataFrame(df_raw)

        num_train = int(len(df_raw) * 0.7)
        num_test = int(len(df_raw) * 0.2)
        num_valid = int(len(df_raw) * 0.1)
        border1s = [0, num_train - self.seq_len, len(df_raw) - num_test - self.seq_len]
        border2s = [num_train, num_train + num_valid, len(df_raw)]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        df_data = df_raw.values

        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data)
            data = self.scaler.transform(df_data)
        else:
            data = df_data

        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]

        # add cycle
        self.cycle_index = (np.arange(len(data)) % self.cycle)[border1:border2]

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = torch.zeros((seq_x.shape[0], 1))
        seq_y_mark = torch.zeros((seq_x.shape[0], 1))

        cycle_index = torch.tensor(self.cycle_index[s_end])

        return seq_x, seq_y, seq_x_mark, seq_y_mark, cycle_index

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)


class Dataset_PEMS(Dataset):
    def __init__(self, root_path, flag='train', size=None,
                 features='S', data_path='ETTh1.csv',
                 target='OT', scale=True, timeenc=0, freq='h', cycle=None):
        # size [seq_len, label_len, pred_len]
        # info
        self.seq_len = size[0]
        self.label_len = size[1]
        self.pred_len = size[2]
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq
        self.cycle = cycle

        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        data_file = os.path.join(self.root_path, self.data_path)
        data = np.load(data_file, allow_pickle=True)
        data = data['data'][:, :, 0]

        num_train = int(len(data) * 0.6)
        num_test = int(len(data) * 0.2)
        num_valid = int(len(data) * 0.2)
        border1s = [0, num_train - self.seq_len, len(data) - num_test - self.seq_len]
        border2s = [num_train, num_train + num_valid, len(data)]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        if self.scale:
            train_data = data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data)
            data = self.scaler.transform(data)

        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]

        # add cycle
        self.cycle_index = (np.arange(len(data)) % self.cycle)[border1:border2]

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = torch.zeros((seq_x.shape[0], 1))
        seq_y_mark = torch.zeros((seq_x.shape[0], 1))

        cycle_index = torch.tensor(self.cycle_index[s_end])

        return seq_x, seq_y, seq_x_mark, seq_y_mark, cycle_index

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)
