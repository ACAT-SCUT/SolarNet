import argparse
import os
from models import SparseTSF
import torch
from exp.exp_main import Exp_Main
import random
import numpy as np
from utils.solar import resolve_site_metadata

parser = argparse.ArgumentParser(description='Model family for Time Series Forecasting')

# random seed
parser.add_argument('--random_seed', type=int, default=2024, help='random seed')

# basic config
parser.add_argument('--is_training', type=int, required=True, default=1, help='status')
parser.add_argument('--model_id', type=str, required=True, default='test', help='model id')
parser.add_argument('--model', type=str, required=True, default='SolarNet',
                    help='model name, options: [SolarNet, SegRNN, DUET, TimeBase, MPFGL, TQNet, Informer, Autoformer, ...]')

# data loader
parser.add_argument('--data', type=str, required=True, default='ETTh1', help='dataset type')
parser.add_argument('--root_path', type=str, default='./data/ETT/', help='root path of the data file')
parser.add_argument('--data_path', type=str, default='ETTh1.csv', help='data file')
parser.add_argument('--features', type=str, default='M',
                    help='forecasting task, options:[M, S, MS]; M:multivariate predict multivariate, S:univariate predict univariate, MS:multivariate predict univariate')
parser.add_argument('--target', type=str, default='OT', help='target feature in S or MS task')
parser.add_argument('--freq', type=str, default='h',
                    help='freq for time features encoding, options:[s:secondly, t:minutely, h:hourly, d:daily, b:business days, w:weekly, m:monthly], you can also use more detailed freq like 15min or 3h')
parser.add_argument('--checkpoints', type=str, default='./checkpoints/', help='location of model checkpoints')
parser.add_argument('--results_dir', type=str, default='./results/', help='directory for per-run artifacts')
parser.add_argument('--results_csv', type=str, default='./results.csv', help='CSV file for experiment summaries')
parser.add_argument('--save_predictions', type=int, default=0, help='save denormalized predictions and targets under results/<setting>/')

# forecasting task
parser.add_argument('--seq_len', type=int, default=96, help='input sequence length')
parser.add_argument('--label_len', type=int, default=48, help='start token length')  #fixed
parser.add_argument('--pred_len', type=int, default=96, help='prediction sequence length')
parser.add_argument('--train_ratio', type=float, default=0.6,
                    help='fraction of the original training split used for fitting; validation/test boundaries stay fixed')

# SolarNet
parser.add_argument('--latitude', type=float, default=None, help='latitude of the PV station, range [-90, 90]')
parser.add_argument('--longitude', type=float, default=None, help='longitude of the PV station, range [-180, 180]')
parser.add_argument('--solar_timezone', type=str, default=None, help='IANA timezone for solar-position calculation')
parser.add_argument('--solar_site', type=str, default=None, help='named site metadata for solar-position calculation')
parser.add_argument('--channel_mode', type=str, default='cd', help='channel model type, options: [cd, ci]')
parser.add_argument('--solar_cycle_mode', type=str, default='full',
                    help='SolarNet cycle ablation mode, options: [full, gain, bias, base, none]')
parser.add_argument('--solar_query_mode', choices=['residual', 'cycle', 'learned'], default='residual',
                    help='SolarNet attention Q: residual input, dynamic cycle, or independent TQNet-style learned queries')
parser.add_argument('--solar_input_attention', choices=['none', 'tqnet'], default='none',
                    help='SolarNet input attention before cycle removal; tqnet uses a separate learned periodic Q')
parser.add_argument('--solar_residual_attention', choices=['none', 'tqnet_gated'], default='none',
                    help='SolarNet parallel learned-Q attention on residuals, with a zero-initialized scalar gate')
parser.add_argument('--solar_gain_limit', type=float, default=0.25,
                    help='SolarNetV2/V3 maximum absolute multiplicative correction')
parser.add_argument('--solar_bias_limit', type=float, default=0.5,
                    help='SolarNetV2/V3 maximum absolute additive correction')


# TQNet & CycleNet
parser.add_argument('--cycle', type=int, default=24, help='cycle length')
parser.add_argument('--use_revin', type=int, default=0, help='1: use revin or 0: no revin')
parser.add_argument('--model_type', type=str, default='mlp', help='model type, options: [linear, mlp]')


# SparseTSF
parser.add_argument('--period_len', type=int, default=4, help='period length')

# SegRNN
parser.add_argument('--rnn_type', type=str, choices=['rnn', 'gru', 'lstm'], default='gru')
parser.add_argument('--dec_way', type=str, choices=['rmf', 'pmf'], default='pmf')
parser.add_argument('--seg_len', type=int, default=12, help='SegRNN segment length')
parser.add_argument('--channel_id', type=int, default=0, help='SegRNN channel embedding')
parser.add_argument('--revin', type=int, default=1, help='SegRNN RevIN switch')

# DUET
parser.add_argument('--CI', type=int, default=1, help='DUET channel-independent temporal clustering')
parser.add_argument('--hidden_size', type=int, default=256, help='DUET router hidden dimension')
parser.add_argument('--num_experts', type=int, default=4, help='DUET temporal expert count')
parser.add_argument('--k', type=int, default=2, help='DUET top-k experts')
parser.add_argument('--noisy_gating', type=int, default=1, help='DUET noisy top-k routing')

# TimeBase
parser.add_argument('--basis_num', type=int, default=6, help='TimeBase basis count')
parser.add_argument('--use_period_norm', type=int, choices=[0, 1], default=1,
                    help='TimeBase period normalization')
parser.add_argument('--use_orthogonal', type=int, choices=[0, 1], default=0,
                    help='TimeBase orthogonal regularization')
parser.add_argument('--orthogonal_weight', type=float, default=0.2,
                    help='TimeBase orthogonal regularization weight')


# PatchTST
parser.add_argument('--fc_dropout', type=float, default=0.05, help='fully connected dropout')
parser.add_argument('--head_dropout', type=float, default=0.0, help='head dropout')
parser.add_argument('--patch_len', type=int, default=16, help='patch length')
parser.add_argument('--stride', type=int, default=8, help='stride')
parser.add_argument('--padding_patch', default='end', help='None: None; end: padding on the end')
# parser.add_argument('--revin', type=int, default=0, help='RevIN; True 1 False 0')
parser.add_argument('--affine', type=int, default=0, help='RevIN-affine; True 1 False 0')
parser.add_argument('--subtract_last', type=int, default=0, help='0: subtract mean; 1: subtract last')
# parser.add_argument('--decomposition', type=int, default=0, help='decomposition; True 1 False 0')
# parser.add_argument('--kernel_size', type=int, default=25, help='decomposition-kernel')
parser.add_argument('--individual', type=int, default=0, help='individual head; True 1 False 0')

# Formers 
parser.add_argument('--embed_type', type=int, default=0, help='0: default 1: value embedding + temporal embedding + positional embedding 2: value embedding + temporal embedding 3: value embedding + positional embedding 4: value embedding')
parser.add_argument('--enc_in', type=int, default=7, help='encoder input size') # DLinear with --individual, use this hyperparameter as the number of channels
parser.add_argument('--dec_in', type=int, default=7, help='decoder input size')
parser.add_argument('--c_out', type=int, default=7, help='output size')
parser.add_argument('--d_model', type=int, default=512, help='dimension of model')
parser.add_argument('--d_core', type=int, default=128, help='SOFTS STAR core dimension')
parser.add_argument('--n_heads', type=int, default=8, help='num of heads')
parser.add_argument('--e_layers', type=int, default=2, help='num of encoder layers')
parser.add_argument('--d_layers', type=int, default=1, help='num of decoder layers')
parser.add_argument('--d_ff', type=int, default=512, help='dimension of fcn')
parser.add_argument('--moving_avg', type=int, default=25, help='window size of moving average')
parser.add_argument('--factor', type=int, default=1, help='attn factor')
parser.add_argument('--distil', action='store_false',
                    help='whether to use distilling in encoder, using this argument means not using distilling',
                    default=True)
parser.add_argument('--dropout', type=float, default=0, help='dropout')
parser.add_argument('--embed', type=str, default='timeF',
                    help='time features encoding, options:[timeF, fixed, learned]')
parser.add_argument('--activation', type=str, default='gelu', help='activation')
parser.add_argument('--output_attention', action='store_true', help='whether to output attention in ecoder')
parser.add_argument('--do_predict', action='store_true', help='whether to predict unseen future data')

# optimization
parser.add_argument('--num_workers', type=int, default=8, help='data loader num workers')
parser.add_argument('--pin_memory', type=int, default=1, help='pin host memory for faster GPU transfer')
parser.add_argument('--persistent_workers', type=int, default=1, help='keep dataloader workers alive across epochs')
parser.add_argument('--prefetch_factor', type=int, default=4, help='number of prefetched batches per worker')
parser.add_argument('--itr', type=int, default=1, help='experiments times')
parser.add_argument('--train_epochs', type=int, default=30, help='train epochs')
parser.add_argument('--batch_size', type=int, default=128, help='batch size of train input data')
parser.add_argument('--patience', type=int, default=5, help='early stopping patience')
parser.add_argument('--learning_rate', type=float, default=0.0001, help='optimizer learning rate')
parser.add_argument('--des', type=str, default='test', help='exp description')
parser.add_argument('--loss', type=str, default='mse', help='loss function')
parser.add_argument('--lradj', type=str, default='type3', help='adjust learning rate')
parser.add_argument('--pct_start', type=float, default=0.3, help='pct_start')
parser.add_argument('--use_amp', action='store_true', help='use automatic mixed precision training', default=False)

# GPU
parser.add_argument('--use_gpu', type=bool, default=True, help='use gpu')
parser.add_argument('--gpu', type=int, default=0, help='gpu')
parser.add_argument('--use_multi_gpu', action='store_true', help='use multiple gpus', default=False)
parser.add_argument('--devices', type=str, default='0,1', help='device ids of multile gpus')
parser.add_argument('--test_flop', action='store_true', default=False, help='See utils/tools for usage')

args = parser.parse_args()

if not 0 < args.train_ratio <= 0.6:
    parser.error('--train_ratio must be in the interval (0, 0.6]')

args.latitude, args.longitude, args.solar_timezone, args.site_name = resolve_site_metadata(
    args.latitude,
    args.longitude,
    args.solar_timezone,
    args.solar_site,
    args.data_path,
    args.model_id,
)

# random seed
fix_seed = args.random_seed
random.seed(fix_seed)
torch.manual_seed(fix_seed)
np.random.seed(fix_seed)





args.use_gpu = True if torch.cuda.is_available() and args.use_gpu else False

if hasattr(torch, 'set_float32_matmul_precision'):
    torch.set_float32_matmul_precision('high')

setting_channel_tag = args.channel_mode
if args.model in {'SolarNet', 'SolarNetV2', 'SolarNetV3', 'SolarNetV4', 'SolarNetV5', 'SolarNetV6', 'SolarNetLinear', 'SolarNetGRU', 'SolarNetDLinear', 'SolarNetPatchTST'}:
    setting_channel_tag = args.channel_mode + '_scm' + args.solar_cycle_mode
if args.model == 'SolarNetV5':
    setting_channel_tag += '_patchinput'
if args.model == 'SolarNetV6':
    setting_channel_tag += '_softs'
if args.model == 'SolarNet' and args.solar_query_mode != 'residual':
    setting_channel_tag += '_q' + args.solar_query_mode
if args.model == 'SolarNet' and args.solar_input_attention != 'none':
    setting_channel_tag += '_input' + args.solar_input_attention
if args.model == 'SolarNet' and args.solar_residual_attention != 'none':
    setting_channel_tag += '_resattn' + args.solar_residual_attention

train_ratio_tag = '' if abs(args.train_ratio - 0.6) < 1e-9 else '_train{:02d}'.format(round(args.train_ratio * 100))

if args.use_gpu and args.use_multi_gpu:
    args.devices = args.devices.replace(' ', '')
    device_ids = args.devices.split(',')
    args.device_ids = [int(id_) for id_ in device_ids]
    args.gpu = args.device_ids[0]

print('Args in experiment:')
print(args)

Exp = Exp_Main


if args.is_training:
    for ii in range(args.itr):

        # setting record of experiments
        setting = '{}_{}_{}_ft{}_sl{}_pl{}_{}{}_cycle{}_seed{}'.format(
            args.model_id,
            args.model,
            args.data,
            args.features,
            args.seq_len,
            args.pred_len,
            setting_channel_tag,
            train_ratio_tag,
            args.cycle,
            fix_seed)

        exp = Exp(args)  # set experiments
        print('>>>>>>>start training : {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(setting))
        exp.train(setting)

        print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
        exp.test(setting)

        if args.do_predict:
            print('>>>>>>>predicting : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
            exp.predict(setting, True)

        torch.cuda.empty_cache()
else:
    ii = 0
    setting = '{}_{}_{}_ft{}_sl{}_pl{}_{}{}_cycle{}_seed{}'.format(
        args.model_id,
        args.model,
        args.data,
        args.features,
        args.seq_len,
        args.pred_len,
        setting_channel_tag,
        train_ratio_tag,
        args.cycle,
        fix_seed)

    exp = Exp(args)  # set experiments
    print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
    exp.test(setting, test=1)
    torch.cuda.empty_cache()
