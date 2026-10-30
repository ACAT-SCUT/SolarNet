from data_provider.data_loader import Dataset_Custom, Dataset_Pred, Dataset_Solar, Dataset_PEMS
from torch.utils.data import DataLoader

data_dict = {
    'custom': Dataset_Custom,
    'Solar': Dataset_Solar,
    'PEMS': Dataset_PEMS
}


def data_provider(args, flag):
    Data = data_dict[args.data]
    timeenc = 0 if args.embed != 'timeF' else 1

    if flag == 'test':
        shuffle_flag = False
        drop_last = False
        batch_size = args.batch_size
        freq = args.freq
    elif flag == 'pred':
        shuffle_flag = False
        drop_last = False
        batch_size = 1
        freq = args.freq
        Data = Dataset_Pred
    else:
        shuffle_flag = True
        drop_last = True
        batch_size = args.batch_size
        freq = args.freq

    dataset_kwargs = dict(
        root_path=args.root_path,
        data_path=args.data_path,
        flag=flag,
        size=[args.seq_len, args.label_len, args.pred_len],
        features=args.features,
        target=args.target,
        timeenc=timeenc,
        freq=freq,
        cycle=args.cycle
    )
    if Data is Dataset_Custom:
        dataset_kwargs['train_ratio'] = getattr(args, 'train_ratio', 0.6)
        dataset_kwargs['lat'] = args.latitude
        dataset_kwargs['lon'] = args.longitude
        dataset_kwargs['timezone_name'] = getattr(args, 'solar_timezone', 'UTC')
        # All SolarNet variants, including residual hybrids, need the
        # cycle index and solar geometry in ``batch_solar``.  Omitting the
        # hybrid model names here silently replaces the physics input with a
        # zero-filled pack, so the backbone would run as a plain baseline.
        dataset_kwargs['use_solar_features'] = args.model in {
            'SolarNet', 'SolarNetV2', 'SolarNetV3', 'SolarNetV4', 'SolarNetV5',
            'SolarNetV6', 'SolarNetLinear', 'SolarNetGRU', 'SolarNetDLinear',
            'SolarNetPatchTST'
        }
        if args.model == 'SolarNet' and getattr(args, 'solar_cycle_mode', 'full') in {'base', 'none'}:
            dataset_kwargs['use_solar_features'] = False
        dataset_kwargs['solar_feature_version'] = 3 if args.model == 'SolarNetV3' else 1

    data_set = Data(**dataset_kwargs)
    print(flag, len(data_set))
    loader_kwargs = dict(
        batch_size=batch_size,
        shuffle=shuffle_flag,
        num_workers=args.num_workers,
        drop_last=drop_last,
        pin_memory=bool(getattr(args, 'pin_memory', 0) and args.use_gpu),
        persistent_workers=bool(getattr(args, 'persistent_workers', 0) and args.num_workers > 0),
    )
    if args.num_workers > 0:
        loader_kwargs['prefetch_factor'] = getattr(args, 'prefetch_factor', 2)

    data_loader = DataLoader(data_set, **loader_kwargs)
    return data_set, data_loader
