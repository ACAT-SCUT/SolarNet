import torch
import torch.nn as nn


class Model(nn.Module):

    def __init__(self, configs):
        super(Model, self).__init__()

        # get parameters
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.enc_in = configs.enc_in
        self.d_model = configs.d_model

        self.rnn = nn.GRU(input_size=self.enc_in, hidden_size=self.d_model, num_layers=1, bias=True,
                            batch_first=True, bidirectional=False)

        self.predict = nn.Sequential(
            nn.Linear(self.d_model, self.enc_in)
        )


    def forward(self, x):


        # encoding
        _, hn = self.rnn(x) # b,s,d  1,b,d

        # decoding
        y = []
        for i in range(self.pred_len):
            yy = self.predict(hn)    # 1,b,c
            yy = yy.permute(1,0,2) # b,1,c
            y.append(yy)
            _, hn = self.rnn(yy, hn)
        y = torch.stack(y, dim=1).squeeze(2) # bc,s,1

        return y