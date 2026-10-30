from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from models import Transformer, DLinear, PatchTST, SparseTSF, CycleNet, iTransformer, TQNet, GRU, LSTM, SolarNet, SolarNetV2, SolarNetV3, SolarNetV4, SolarNetV5, SolarNetV6, Autoformer, MPFGL
from models.Linear import Model as Linear
from models.SolarNetLinear import Model as SolarNetLinear
from models.SolarNetGRU import Model as SolarNetGRU
from models.SolarNetDLinear import Model as SolarNetDLinear
from models.SolarNetPatchTST import Model as SolarNetPatchTST
from importlib import import_module
from utils.tools import EarlyStopping, adjust_learning_rate, visual, test_params_flop
from utils.metrics import StreamingMetricAccumulator
from utils.experiment import append_results_row, build_run_metadata, ensure_dir, flatten_results_row, write_json

import numpy as np
import torch
import torch.nn as nn
from torch import optim
from torch.optim import lr_scheduler

import os
import time

import warnings
import matplotlib.pyplot as plt
import numpy as np

warnings.filterwarnings('ignore')


def _to_scalar(value):
    array = np.asarray(value)
    if array.ndim == 0:
        return float(array)
    return float(array.mean())


def _inverse_transform_batch(dataset, values, features, scaler_dim):
    original_shape = values.shape
    flattened = values.reshape(-1, original_shape[-1])
    if features == 'MS':
        padded = np.zeros((flattened.shape[0], scaler_dim), dtype=np.float32)
        padded[:, -1] = flattened[:, -1]
        transformed = dataset.inverse_transform(padded)[:, -1:]
    else:
        transformed = dataset.inverse_transform(flattened)
    return transformed.astype(np.float32).reshape(original_shape)


class Exp_Main(Exp_Basic):
    def __init__(self, args):
        super(Exp_Main, self).__init__(args)
        self.non_blocking = bool(getattr(args, 'pin_memory', 0) and args.use_gpu)

    def _build_model(self):
        model_dict = {
            'Transformer': Transformer,
            'DLinear': DLinear,
            'PatchTST': PatchTST,
            'SparseTSF': SparseTSF,
            'CycleNet': CycleNet,
            'iTransformer': iTransformer,
            'TQNet': TQNet,
            'GRU': GRU,
            'LSTM': LSTM,
            'SolarNet': SolarNet,
            'SolarNetV2': SolarNetV2,
            'SolarNetV3': SolarNetV3,
            'SolarNetV4': SolarNetV4,
            'SolarNetV5': SolarNetV5,
            'SolarNetV6': SolarNetV6,
            'Autoformer': Autoformer,
            'MPFGL': MPFGL,
            'Linear': type('_Module', (), {'Model': Linear}),
            'SolarNetLinear': type('_Module', (), {'Model': SolarNetLinear}),
            'SolarNetGRU': type('_Module', (), {'Model': SolarNetGRU}),
            'SolarNetDLinear': type('_Module', (), {'Model': SolarNetDLinear}),
            'SolarNetPatchTST': type('_Module', (), {'Model': SolarNetPatchTST}),
        }
        if self.args.model in {'SegRNN', 'DUET', 'TimeBase'}:
            model_dict[self.args.model] = import_module('models.' + self.args.model)
            if self.args.model in {'DUET', 'TimeBase'} and self.args.use_multi_gpu:
                raise ValueError(f'{self.args.model} auxiliary loss requires one GPU per experiment')
        model = model_dict[self.args.model].Model(self.args).float()

        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader

    def _select_optimizer(self):
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim

    def _select_criterion(self):
        # criterion = nn.MSELoss()
        criterion = nn.L1Loss()
        return criterion

    def _add_auxiliary_loss(self, loss, training=False):
        if self.args.model not in {'DUET', 'TimeBase'}:
            return loss
        model = self.model.module if isinstance(self.model, nn.DataParallel) else self.model
        if self.args.model == 'TimeBase':
            if not training or not self.args.use_orthogonal:
                return loss
            if model.auxiliary_loss is None:
                raise RuntimeError('TimeBase did not expose its orthogonal loss')
            return loss + self.args.orthogonal_weight * model.auxiliary_loss
        if model.auxiliary_loss is None:
            raise RuntimeError('DUET did not expose its router auxiliary loss')
        return loss + model.auxiliary_loss

    def vali(self, vali_data, vali_loader, criterion):
        total_loss = []
        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark, batch_cycle, batch_solar) in enumerate(vali_loader):
                batch_x = batch_x.float().to(self.device, non_blocking=self.non_blocking)
                batch_y = batch_y.float()

                batch_x_mark = batch_x_mark.float().to(self.device, non_blocking=self.non_blocking)
                batch_y_mark = batch_y_mark.float().to(self.device, non_blocking=self.non_blocking)
                batch_cycle = batch_cycle.int().to(self.device, non_blocking=self.non_blocking)
                batch_solar = batch_solar.float().to(self.device, non_blocking=self.non_blocking)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device, non_blocking=self.non_blocking)
                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if any(substr in self.args.model for substr in {'CycleNet', 'TQNet'}):
                            outputs = self.model(batch_x, batch_cycle)
                        elif any(substr in self.args.model for substr in {'SolarNet'}):
                            outputs = self.model(batch_x, batch_solar)
                        elif any(substr in self.args.model for substr in
                                 {'Linear', 'iTransformer', 'GRU', 'PatchTST', 'LSTM', 'SparseTSF', 'MPFGL', 'SegRNN', 'DUET', 'TimeBase'}):
                            outputs = self.model(batch_x)
                        else:
                            if self.args.output_attention:
                                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                            else:
                                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    if any(substr in self.args.model for substr in {'CycleNet', 'TQNet'}):
                        outputs = self.model(batch_x, batch_cycle)
                    elif any(substr in self.args.model for substr in {'SolarNet'}):
                        outputs = self.model(batch_x, batch_solar)
                    elif any(substr in self.args.model for substr in {'Linear', 'iTransformer', 'GRU', 'PatchTST', 'LSTM', 'SparseTSF', 'MPFGL', 'SegRNN', 'DUET', 'TimeBase'}):
                        outputs = self.model(batch_x)
                    else:
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device, non_blocking=self.non_blocking)

                loss = self._add_auxiliary_loss(criterion(outputs, batch_y))
                total_loss.append(loss.detach().cpu())
        total_loss = np.average(total_loss)
        self.model.train()
        return total_loss

    def train(self, setting):
        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')
        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        time_now = time.time()

        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        model_optim = self._select_optimizer()
        criterion = self._select_criterion()

        if self.args.use_amp:
            scaler = torch.cuda.amp.GradScaler()

        scheduler = lr_scheduler.OneCycleLR(optimizer=model_optim,
                                            steps_per_epoch=train_steps,
                                            pct_start=self.args.pct_start,
                                            epochs=self.args.train_epochs,
                                            max_lr=self.args.learning_rate)

        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []

            self.model.train()
            epoch_time = time.time()
            # max_memory = 0
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark, batch_cycle, batch_solar) in enumerate(train_loader):
                iter_count += 1
                model_optim.zero_grad()
                batch_x = batch_x.float().to(self.device, non_blocking=self.non_blocking)

                batch_y = batch_y.float().to(self.device, non_blocking=self.non_blocking)
                batch_x_mark = batch_x_mark.float().to(self.device, non_blocking=self.non_blocking)
                batch_y_mark = batch_y_mark.float().to(self.device, non_blocking=self.non_blocking)
                batch_cycle = batch_cycle.int().to(self.device, non_blocking=self.non_blocking)
                batch_solar = batch_solar.float().to(self.device, non_blocking=self.non_blocking)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device, non_blocking=self.non_blocking)

                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if any(substr in self.args.model for substr in {'CycleNet', 'TQNet'}):
                            outputs = self.model(batch_x, batch_cycle)
                        elif any(substr in self.args.model for substr in {'SolarNet'}):
                            outputs = self.model(batch_x, batch_solar)
                        elif any(substr in self.args.model for substr in
                                 {'Linear', 'iTransformer', 'GRU', 'PatchTST', 'LSTM', 'SparseTSF', 'MPFGL', 'SegRNN', 'DUET', 'TimeBase'}):
                            outputs = self.model(batch_x)
                        else:
                            if self.args.output_attention:
                                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                            else:
                                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                        f_dim = -1 if self.args.features == 'MS' else 0
                        outputs = outputs[:, -self.args.pred_len:, f_dim:]
                        batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device, non_blocking=self.non_blocking)
                        loss = self._add_auxiliary_loss(criterion(outputs, batch_y), training=True)
                        train_loss.append(loss.item())
                else:
                    if any(substr in self.args.model for substr in {'CycleNet', 'TQNet'}):
                        outputs = self.model(batch_x, batch_cycle)
                    elif any(substr in self.args.model for substr in {'SolarNet'}):
                        outputs = self.model(batch_x, batch_solar)
                    elif any(substr in self.args.model for substr in {'Linear', 'iTransformer', 'GRU', 'PatchTST', 'LSTM', 'SparseTSF', 'MPFGL', 'SegRNN', 'DUET', 'TimeBase'}):
                        outputs = self.model(batch_x)
                    else:
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]

                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                    # print(outputs.shape,batch_y.shape)
                    f_dim = -1 if self.args.features == 'MS' else 0
                    outputs = outputs[:, -self.args.pred_len:, f_dim:]
                    batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device, non_blocking=self.non_blocking)
                    loss = self._add_auxiliary_loss(criterion(outputs, batch_y), training=True)
                    train_loss.append(loss.item())

                if (i + 1) % 100 == 0:
                    print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()

                if self.args.use_amp:
                    scaler.scale(loss).backward()
                    scaler.step(model_optim)
                    scaler.update()
                else:
                    loss.backward()
                    model_optim.step()

                # current_memory = torch.cuda.max_memory_allocated() / 1024 ** 2
                # max_memory = max(max_memory, current_memory)

                if self.args.lradj == 'TST':
                    adjust_learning_rate(model_optim, scheduler, epoch + 1, self.args, printout=False)
                    scheduler.step()

            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            train_loss = np.average(train_loss)
            vali_loss = self.vali(vali_data, vali_loader, criterion)

            print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f}".format(
                epoch + 1, train_steps, train_loss, vali_loss))
            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break

            if self.args.lradj != 'TST':
                adjust_learning_rate(model_optim, scheduler, epoch + 1, self.args)
            else:
                print('Updating learning rate to {}'.format(scheduler.get_last_lr()[0]))

        best_model_path = path + '/' + 'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path, map_location=self.device))

        # print(f"Max Memory (MB): {max_memory}")

        return self.model



    def test(self, setting, test=0):
        test_data, test_loader = self._get_data(flag='test')

        if test:
            print('loading model')
            self.model.load_state_dict(torch.load(
                os.path.join(self.args.checkpoints, setting, 'checkpoint.pth'),
                map_location=self.device,
            ))

        result_dir = ensure_dir(os.path.join(self.args.results_dir, setting))
        plot_dir = ensure_dir(os.path.join(result_dir, 'plots'))
        normalized_metrics = StreamingMetricAccumulator()
        denormalized_metrics = StreamingMetricAccumulator()
        prediction_writer = None
        target_writer = None
        write_offset = 0
        if self.args.save_predictions:
            output_channels = 1 if self.args.features == 'MS' else self.args.enc_in
            output_shape = (len(test_data), self.args.pred_len, output_channels)
            prediction_writer = np.lib.format.open_memmap(
                os.path.join(result_dir, 'predictions.npy'), mode='w+', dtype=np.float32, shape=output_shape
            )
            target_writer = np.lib.format.open_memmap(
                os.path.join(result_dir, 'targets.npy'), mode='w+', dtype=np.float32, shape=output_shape
            )

        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark, batch_cycle, batch_solar) in enumerate(test_loader):
                batch_x = batch_x.float().to(self.device, non_blocking=self.non_blocking)
                batch_y = batch_y.float().to(self.device, non_blocking=self.non_blocking)

                batch_x_mark = batch_x_mark.float().to(self.device, non_blocking=self.non_blocking)
                batch_y_mark = batch_y_mark.float().to(self.device, non_blocking=self.non_blocking)
                batch_cycle = batch_cycle.int().to(self.device, non_blocking=self.non_blocking)
                batch_solar = batch_solar.float().to(self.device, non_blocking=self.non_blocking)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device, non_blocking=self.non_blocking)
                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if any(substr in self.args.model for substr in {'CycleNet', 'TQNet'}):
                            outputs = self.model(batch_x, batch_cycle)
                        elif any(substr in self.args.model for substr in {'SolarNet'}):
                            outputs = self.model(batch_x, batch_solar)
                        elif any(substr in self.args.model for substr in
                                 {'Linear', 'iTransformer', 'GRU', 'PatchTST', 'LSTM', 'SparseTSF', 'MPFGL', 'SegRNN', 'DUET', 'TimeBase'}):
                            outputs = self.model(batch_x)
                        else:
                            if self.args.output_attention:
                                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                            else:
                                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    if any(substr in self.args.model for substr in {'CycleNet', 'TQNet'}):
                        outputs = self.model(batch_x, batch_cycle)
                    elif any(substr in self.args.model for substr in {'SolarNet'}):
                        outputs = self.model(batch_x, batch_solar)
                    elif any(substr in self.args.model for substr in {'Linear', 'iTransformer', 'GRU', 'PatchTST', 'LSTM', 'SparseTSF', 'MPFGL', 'SegRNN', 'DUET', 'TimeBase'}):
                        outputs = self.model(batch_x)
                    else:
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]

                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                f_dim = -1 if self.args.features == 'MS' else 0
                # print(outputs.shape,batch_y.shape)
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device, non_blocking=self.non_blocking)
                outputs = outputs.detach().cpu().numpy()
                batch_y = batch_y.detach().cpu().numpy()

                pred = outputs
                true = batch_y
                normalized_metrics.update(pred, true)

                denorm_pred = _inverse_transform_batch(test_data, pred, self.args.features, self.args.enc_in)
                denorm_true = _inverse_transform_batch(test_data, true, self.args.features, self.args.enc_in)
                denormalized_metrics.update(denorm_pred, denorm_true)

                if prediction_writer is not None:
                    batch_end = write_offset + len(denorm_pred)
                    prediction_writer[write_offset:batch_end] = denorm_pred
                    target_writer[write_offset:batch_end] = denorm_true
                    write_offset = batch_end

                if i % 20 == 0:
                    input = batch_x.detach().cpu().numpy()

                    # gt = np.concatenate((input[0, :, -1], true[0, :, -1]), axis=0)
                    # pd = np.concatenate((input[0, :, -1], pred[0, :, -1]), axis=0)
                    
                    gt = np.concatenate((input[0, :, 0], true[0, :, 0]), axis=0)
                    pd = np.concatenate((input[0, :, 0], pred[0, :, 0]), axis=0)

                    visual(gt, pd, os.path.join(plot_dir, str(i) + '.pdf'))


        if self.args.test_flop:
            test_params_flop(self.model, (batch_x.shape[1], batch_x.shape[2]))
            exit()
        mae_norm, mse_norm, rmse_norm, mape_norm, mspe_norm, rse_norm = normalized_metrics.compute()
        mae_denorm, mse_denorm, rmse_denorm, mape_denorm, mspe_denorm, rse_denorm = denormalized_metrics.compute()
        mae_channel, rmse_channel = denormalized_metrics.channel_metrics()

        if prediction_writer is not None:
            prediction_writer.flush()
            target_writer.flush()
            del prediction_writer, target_writer

        # 2. 获取各通道Cap
        import re, pandas as pd
        data_path = getattr(test_data, 'data_path', None)
        root_path = getattr(test_data, 'root_path', None)
        if data_path is not None and root_path is not None:
            df_header = pd.read_csv(os.path.join(root_path, data_path), nrows=0)
            col_names = df_header.columns[1:]
            caps = []
            for name in col_names:
                m = re.search(r'_(\d+)MW', name)
                caps.append(float(m.group(1)) if m else 1.0)
            caps = np.array(caps)  # shape: [channel,]
        else:
            caps = np.ones(mae_channel.shape)  # fallback

        # 防止除零异常
        caps = np.where(caps == 0, 1.0, caps)
        acc_mae = 1.0 - mae_channel / caps
        acc_rmse = 1.0 - rmse_channel / caps
        # 平均准确率
        mean_acc_mae = np.mean(acc_mae)
        mean_acc_rmse = np.mean(acc_rmse)

        print('Normalized    mse:{}, mae:{}, rmse:{}'.format(mse_norm, mae_norm, rmse_norm))
        print('De-Normalized mse:{}, mae:{}, rmse:{}'.format(mse_denorm, mae_denorm, rmse_denorm))
        print(f'De-Norm Accuracy (1-MAE/Cap): {mean_acc_mae:.4f}')
        print(f'De-Norm Accuracy (1-RMSE/Cap): {mean_acc_rmse:.4f}')

        metrics_payload = {
            'normalized': {
                'mae': _to_scalar(mae_norm),
                'mse': _to_scalar(mse_norm),
                'rmse': _to_scalar(rmse_norm),
                'mape': _to_scalar(mape_norm),
                'mspe': _to_scalar(mspe_norm),
                'rse': _to_scalar(rse_norm),
            },
            'denormalized': {
                'mae': _to_scalar(mae_denorm),
                'mse': _to_scalar(mse_denorm),
                'rmse': _to_scalar(rmse_denorm),
                'mape': _to_scalar(mape_denorm),
                'mspe': _to_scalar(mspe_denorm),
                'rse': _to_scalar(rse_denorm),
            },
            'capacity_accuracy': {
                'mean_acc_mae': _to_scalar(mean_acc_mae),
                'mean_acc_rmse': _to_scalar(mean_acc_rmse),
            }
        }
        run_metadata = build_run_metadata(self.args, setting, metrics_payload)
        write_json(os.path.join(result_dir, 'metrics.json'), metrics_payload)
        write_json(os.path.join(result_dir, 'config.json'), vars(self.args))
        write_json(os.path.join(result_dir, 'run_summary.json'), run_metadata)
        append_results_row(self.args.results_csv, flatten_results_row(run_metadata))


        return

    def predict(self, setting, load=False):
        pred_data, pred_loader = self._get_data(flag='pred')

        if load:
            path = os.path.join(self.args.checkpoints, setting)
            best_model_path = path + '/' + 'checkpoint.pth'
            self.model.load_state_dict(torch.load(best_model_path, map_location=self.device))

        preds = []

        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark, batch_cycle, batch_solar) in enumerate(pred_loader):
                batch_x = batch_x.float().to(self.device, non_blocking=self.non_blocking)
                batch_y = batch_y.float()
                batch_x_mark = batch_x_mark.float().to(self.device, non_blocking=self.non_blocking)
                batch_y_mark = batch_y_mark.float().to(self.device, non_blocking=self.non_blocking)
                batch_cycle = batch_cycle.int().to(self.device, non_blocking=self.non_blocking)
                batch_solar = batch_solar.float().to(self.device, non_blocking=self.non_blocking)

                # decoder input
                dec_inp = torch.zeros([batch_y.shape[0], self.args.pred_len, batch_y.shape[2]]).float().to(
                    batch_y.device, non_blocking=self.non_blocking)
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device, non_blocking=self.non_blocking)
                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if any(substr in self.args.model for substr in {'CycleNet', 'TQNet'}):
                            outputs = self.model(batch_x, batch_cycle)
                        elif any(substr in self.args.model for substr in {'SolarNet'}):
                            outputs = self.model(batch_x, batch_solar)
                        elif any(substr in self.args.model for substr in
                                 {'Linear', 'iTransformer', 'GRU', 'PatchTST', 'LSTM', 'SparseTSF', 'MPFGL', 'SegRNN', 'DUET', 'TimeBase'}):
                            outputs = self.model(batch_x)
                        else:
                            if self.args.output_attention:
                                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                            else:
                                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    if any(substr in self.args.model for substr in {'CycleNet', 'TQNet'}):
                        outputs = self.model(batch_x, batch_cycle)
                    elif any(substr in self.args.model for substr in {'SolarNet'}):
                        outputs = self.model(batch_x, batch_solar)
                    elif any(substr in self.args.model for substr in {'Linear', 'iTransformer', 'GRU', 'PatchTST', 'LSTM', 'SparseTSF', 'MPFGL', 'SegRNN', 'DUET', 'TimeBase'}):
                        outputs = self.model(batch_x)
                    else:
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                pred = outputs.detach().cpu().numpy()  # .squeeze()
                preds.append(pred)

        preds = np.array(preds)
        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])

        # result save
        result_dir = ensure_dir(os.path.join(self.args.results_dir, setting))

        np.save(os.path.join(result_dir, 'real_prediction.npy'), preds)

        return
