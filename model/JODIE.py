import torch
import torch.nn as nn
import math
from torch.nn import functional as F


class JODIE(nn.Module):
    def __init__(self, args):
        super(JODIE, self).__init__()
        self.user_num = args.dataset.user_num
        self.item_num = args.dataset.item_num        
        self.emb_dim = args.model.emb_dim
        self.static_user_emb_dim = args.dataset.user_num
        self.static_item_emb_dim = args.dataset.item_num
        
        self.initial_user_emb = nn.Parameter(torch.Tensor(self.emb_dim))
        self.initial_item_emb = nn.Parameter(torch.Tensor(self.emb_dim))

        # 원래 코드에서는 feature가 존재하지만, 이를 임베딩으로 바꾼다.
        rnn_input_size_user = 2 * self.emb_dim + 1
        rnn_input_size_item = 2 * self.emb_dim + 1  

        self.user_rnn = nn.RNNCell(rnn_input_size_user, self.emb_dim)
        self.item_rnn = nn.RNNCell(rnn_input_size_item, self.emb_dim)

        self.linear_layer1 = nn.Linear(self.emb_dim, 50)
        self.linear_layer2 = nn.Linear(50, 2)
        self.prediction_layer = nn.Linear(
            in_features=self.static_user_emb_dim + self.static_item_emb_dim + self.emb_dim * 2, 
            out_features=self.static_item_emb_dim + self.emb_dim
        )
        self.embedding_layer = nn.Linear(1, self.emb_dim)
        self._init_normal_linear(self.embedding_layer)

    def _init_normal_linear(self, layer):
        stdv = 1.0 / math.sqrt(layer.weight.size(1))
        nn.init.normal_(layer.weight, mean=0.0, std=stdv)
        if layer.bias is not None:
            nn.init.normal_(layer.bias, mean=0.0, std=stdv)

    def forward(self, user_emb, item_emb, time_diff=None, features=None, select=None):
        if select == 'item_update':
            input1 = torch.cat([user_emb, time_diff, features], dim=1)
            item_emb_out = self.item_rnn(input1, item_emb)
            return F.normalize(item_emb_out)

        elif select == 'user_update':
            input2 = torch.cat([item_emb, time_diff, features], dim=1)
            user_emb_out = self.user_rnn(input2, user_emb)
            return F.normalize(user_emb_out)

        elif select == 'project':
            user_proj_emb = self.context_convert(user_emb, time_diff, features)
            return user_proj_emb

    def context_convert(self, embeddings, time_diff, features):
        new_embeddings = embeddings * (1 + self.embedding_layer(time_diff))
        return new_embeddings

    def predict_label(self, user_emb):
        X_out = nn.ReLU()(self.linear_layer1(user_emb))
        X_out = self.linear_layer2(X_out)
        return X_out

    def predict_item_embedding(self, user_emb):
        X_out = self.prediction_layer(user_emb)
        return X_out

    def loss_fn(self):
        return