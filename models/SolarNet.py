import torch
import torch.nn as nn
import torch.nn.functional as F


class SolarCycle(torch.nn.Module):
    def __init__(self, cycle_len, channel_size, cycle_mode='full'):
        super(SolarCycle, self).__init__()
        self.cycle_len = cycle_len
        self.channel_size = channel_size
        self.cycle_mode = cycle_mode

        assert self.cycle_mode in ['full', 'gain', 'bias', 'base', 'none']

        self.base_profile = torch.nn.Parameter(torch.zeros(cycle_len, channel_size), requires_grad=True)

        # 2. 物理增益器 (Physics Scaler) - 学习根据太阳位置调整
        # 输入维度 3: [Elevation, Sin_Az, Cos_Az]
        self.solar_mlp = nn.Sequential(
            nn.Linear(3, 32),
            nn.ReLU(),
            nn.Linear(32, self.channel_size)
        )

        # 3. 物理偏移器 (Physics Bias) - 学习根据太阳位置的加法偏移
        self.solar_bias_mlp = nn.Sequential(
            nn.Linear(3, 32),
            nn.ReLU(),
            nn.Linear(32, self.channel_size)
        )

    def forward(self, solar_pack_slice):
            """
            solar_pack_slice: [Batch, Length, 4]
            """
            indices = solar_pack_slice[:, :, 0].long()
            base_cycle = self.base_profile[indices % self.cycle_len]
            if self.cycle_mode == 'none':
                return torch.zeros_like(base_cycle)

            if self.cycle_mode == 'base':
                return base_cycle

            # 物理特征必须是 Float 类型 (用于 MLP)
            solar_feats = solar_pack_slice[:, :, 1:].float()

            if self.cycle_mode == 'full':
                gain = self.solar_mlp(solar_feats)
                bias = self.solar_bias_mlp(solar_feats)
                dynamic_cycle = base_cycle * (1 + gain) + bias
            elif self.cycle_mode == 'gain':
                gain = self.solar_mlp(solar_feats)
                dynamic_cycle = base_cycle * (1 + gain)
            else:
                bias = self.solar_bias_mlp(solar_feats)
                dynamic_cycle = base_cycle + bias

            return dynamic_cycle

class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()

        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.enc_in = configs.enc_in
        self.cycle_len = configs.cycle
        self.channel_mode = configs.channel_mode
        self.d_model = configs.d_model
        self.solar_cycle_mode = configs.solar_cycle_mode
        self.solar_query_mode = getattr(configs, 'solar_query_mode', 'residual')
        self.solar_input_attention = getattr(configs, 'solar_input_attention', 'none')
        self.solar_residual_attention = getattr(configs, 'solar_residual_attention', 'none')

        assert self.channel_mode in ['cd', 'ci']
        if self.solar_query_mode not in {'residual', 'cycle', 'learned'}:
            raise ValueError('solar_query_mode must be residual, cycle, or learned')
        if self.solar_input_attention not in {'none', 'tqnet'}:
            raise ValueError('solar_input_attention must be none or tqnet')
        if self.solar_input_attention == 'tqnet' and self.channel_mode != 'cd':
            raise ValueError('TQNet input attention requires channel_mode=cd')
        if self.solar_residual_attention not in {'none', 'tqnet_gated'}:
            raise ValueError('solar_residual_attention must be none or tqnet_gated')
        if self.solar_residual_attention == 'tqnet_gated':
            if self.channel_mode != 'cd' or self.solar_cycle_mode == 'none':
                raise ValueError('Gated residual attention requires CD and an active solar cycle')
            if self.solar_query_mode != 'residual' or self.solar_input_attention != 'none':
                raise ValueError('Gated residual attention requires residual Q and no input attention')

        # 使用新的 SolarCycle 模块
        self.cycleQueue = SolarCycle(
            cycle_len=self.cycle_len,
            channel_size=self.enc_in,
            cycle_mode=self.solar_cycle_mode
        )

        if self.channel_mode == 'cd':
            self.channelAggregator = nn.MultiheadAttention(embed_dim=self.seq_len, num_heads=4, batch_first=True, dropout=0)
            if self.solar_query_mode == 'learned':
                # TQNet-style queries are independent of the physical cycle.
                self.temporalQuery = nn.Parameter(torch.zeros(self.cycle_len, self.enc_in))
            if self.solar_input_attention == 'tqnet':
                # A separate Q and attention layer enhance X before cycle removal.
                self.inputTemporalQuery = nn.Parameter(torch.zeros(self.cycle_len, self.enc_in))
                self.inputChannelAggregator = nn.MultiheadAttention(
                    embed_dim=self.seq_len, num_heads=4, batch_first=True, dropout=0.5
                )

        self.input_proj = nn.Linear(self.seq_len, self.d_model)

        self.model = nn.Sequential(
            nn.Linear(self.d_model, self.d_model),
            nn.GELU(),
            nn.Linear(self.d_model, self.d_model),
            nn.GELU()
        )
        self.output_proj = nn.Sequential(
            nn.Linear(self.d_model, self.pred_len),
        )

        # Construct the extra branch last to preserve the baseline initialization
        # under the same seed. At alpha=0 the initial forecast matches it exactly.
        if self.solar_residual_attention == 'tqnet_gated':
            self.residualTemporalQuery = nn.Parameter(torch.zeros(self.cycle_len, self.enc_in))
            self.residualChannelAggregator = nn.MultiheadAttention(
                embed_dim=self.seq_len, num_heads=4, batch_first=True, dropout=0.5
            )
            self.residual_attention_alpha = nn.Parameter(torch.zeros(()))
        
    def forward(self, x, solar_pack):
        """
        x: [Batch, Seq_Len, Channel]
        solar_pack: [Batch, Seq_Len + Pred_Len, 4] 
                    包含了 cycle_idx 和 solar info
        """


        # --- 1. 拆分 Cycle Pack ---
        # 前半部分对应输入序列 (History)
        solar_pack_x = solar_pack[:, :self.seq_len, :]
        # 后半部分对应预测序列 (Future)
        solar_pack_y = solar_pack[:, self.seq_len:, :]

        use_cycle = self.solar_cycle_mode != 'none'

        if self.solar_input_attention == 'tqnet':
            x_input = x.permute(0, 2, 1)
            indices = solar_pack_x[:, :, 0].long() % self.cycle_len
            query = self.inputTemporalQuery[indices].permute(0, 2, 1)
            channel_information = self.inputChannelAggregator(
                query=query, key=x_input, value=x_input, need_weights=False
            )[0]
            x = (x_input + channel_information).permute(0, 2, 1)

        # --- 2. Remove Cycle (Input) ---
        # 使用历史的太阳信息来去除周期
        if use_cycle:
            cycle_val_x = self.cycleQueue(solar_pack_x)
            x = x - cycle_val_x
        else:
            cycle_val_x = None

        # --- 3. Forecasting ---
        if self.channel_mode == 'cd' and use_cycle:
            residual = x.permute(0, 2, 1)
            if self.solar_query_mode == 'learned':
                indices = solar_pack_x[:, :, 0].long() % self.cycle_len
                query = self.temporalQuery[indices].permute(0, 2, 1)
            elif self.solar_query_mode == 'cycle':
                query = cycle_val_x.permute(0, 2, 1)
            else:
                query = residual
            channel_information = self.channelAggregator(query=query, key=residual, value=residual)[0]
            enhanced_residual = residual + channel_information
            if self.solar_residual_attention == 'tqnet_gated':
                indices = solar_pack_x[:, :, 0].long() % self.cycle_len
                learned_query = self.residualTemporalQuery[indices].permute(0, 2, 1)
                temporal_information = self.residualChannelAggregator(
                    query=learned_query, key=residual, value=residual, need_weights=False
                )[0]
                enhanced_residual = enhanced_residual + self.residual_attention_alpha * temporal_information
            input = self.input_proj(enhanced_residual)
        else:
            input = self.input_proj(x.permute(0, 2, 1))

        hidden = self.model(input)
        
        y = self.output_proj(hidden+input).permute(0, 2, 1)

        # --- 4. Add Back Cycle (Output) ---
        # 使用未来的太阳信息(已知)来加回周期
        if use_cycle:
            cycle_val_y = self.cycleQueue(solar_pack_y)
            y = y + cycle_val_y


        return y
