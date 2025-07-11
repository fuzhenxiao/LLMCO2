import math
import torch
from torch import nn
import torch.nn.functional as F
import torch_geometric.nn as tgnn
from torch_scatter import scatter


def init_tensor(tensor, init_type, nonlinearity):
    if tensor is None or init_type is None:
        return
    if init_type == "thomas":
        size = tensor.size(-1)
        stdv = 1.0 / math.sqrt(size)
        nn.init.uniform_(tensor, -stdv, stdv)
    elif init_type == "kaiming_normal_in":
        nn.init.kaiming_normal_(tensor, mode="fan_in", nonlinearity=nonlinearity)
    elif init_type == "kaiming_normal_out":
        nn.init.kaiming_normal_(tensor, mode="fan_out", nonlinearity=nonlinearity)
    elif init_type == "kaiming_uniform_in":
        nn.init.kaiming_uniform_(tensor, mode="fan_in", nonlinearity=nonlinearity)
    elif init_type == "kaiming_uniform_out":
        nn.init.kaiming_uniform_(tensor, mode="fan_out", nonlinearity=nonlinearity)
    else:
        raise ValueError(f"Unknown initialization type: {init_type}")


class GraphNet(torch.nn.Module):
    def __init__(
        self,
        num_node_features=44,
        gnn_layer="SAGEConv",
        num_other_features=33,
        gnn_hidden=512,
        fc_hidden=512,
        reduce_func="max",
    ):
        super(GraphNet, self).__init__()

        self.reduce_func = reduce_func
        self.num_node_features = num_node_features
        self.num_other_features = num_other_features
        self.gnn_layer_func = getattr(tgnn, gnn_layer)
        self.graph_conv_1 = self.gnn_layer_func(
            num_node_features, gnn_hidden, normalize=True
        )
        self.graph_conv_2 = self.gnn_layer_func(gnn_hidden, gnn_hidden, normalize=True)
        self.gnn_drop_1 = nn.Dropout(p=0.05)
        self.gnn_drop_2 = nn.Dropout(p=0.05)
        self.gnn_relu1 = nn.ReLU()
        self.gnn_relu2 = nn.ReLU()

        self.norm_sf_linear = nn.Linear(num_other_features, gnn_hidden)
        self.norm_sf_drop = nn.Dropout(p=0.05)
        self.norm_sf_relu = nn.ReLU()

        self.fc_1 = nn.Linear(gnn_hidden + gnn_hidden, fc_hidden)
        self.fc_2 = nn.Linear(fc_hidden, fc_hidden)
        self.fc_drop_1 = nn.Dropout(p=0.05)
        self.fc_drop_2 = nn.Dropout(p=0.05)
        self.fc_relu1 = nn.ReLU()
        self.fc_relu2 = nn.ReLU()
        self.predictor = nn.Linear(fc_hidden, 1)
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                init_tensor(m.weight, "thomas", "relu")
                init_tensor(m.bias, "thomas", "relu")
            elif isinstance(m, self.gnn_layer_func):
                pass

    def forward(self, graph, data):
        x, A = graph.x, graph.edge_index

        x = self.graph_conv_1(x, A)
        x = self.gnn_relu1(x)
        x = self.gnn_drop_1(x)
        x = self.graph_conv_2(x, A)
        x = self.gnn_relu2(x)
        x = self.gnn_drop_2(x)

        gnn_feat = scatter(x, graph.batch, dim=0, reduce=self.reduce_func)
        static_feature = self.norm_sf_linear(data)
        static_feature = self.norm_sf_drop(static_feature)
        static_feature = self.norm_sf_relu(static_feature)
        x = torch.cat([gnn_feat, static_feature], dim=1)

        x = self.fc_1(x)
        x = self.fc_relu1(x)
        x = self.fc_drop_1(x)
        x = self.fc_2(x)
        x = self.fc_relu2(x)
        feat = self.fc_drop_2(x)
        x = self.predictor(feat)

        return x


class DoubleGraphNet(torch.nn.Module):
    def __init__(
        self,
        num_node_features=44,
        gnn_layer="SAGEConv",
        num_other_features=33,
        gnn_hidden=512,
        fc_hidden=512,
        reduce_func="max",
    ):
        super(DoubleGraphNet, self).__init__()

        self.reduce_func = reduce_func
        self.num_node_features = num_node_features
        self.num_other_features = num_other_features
        self.gnn_layer_func = getattr(tgnn, gnn_layer)
        self.graph_conv_1 = self.gnn_layer_func(
            num_node_features, gnn_hidden, normalize=True
        )
        self.graph_conv_2 = self.gnn_layer_func(gnn_hidden, gnn_hidden, normalize=True)
        self.gnn_drop_1 = nn.Dropout(p=0.05)
        self.gnn_drop_2 = nn.Dropout(p=0.05)
        self.gnn_relu1 = nn.ReLU()
        self.gnn_relu2 = nn.ReLU()

        self.norm_sf_linear = nn.Linear(num_other_features, gnn_hidden)
        self.norm_sf_drop = nn.Dropout(p=0.05)
        self.norm_sf_relu = nn.ReLU()

        self.fc_1 = nn.Linear(gnn_hidden + gnn_hidden, fc_hidden)
        self.fc_2 = nn.Linear(fc_hidden, fc_hidden)
        self.fc_drop_1 = nn.Dropout(p=0.05)
        self.fc_drop_2 = nn.Dropout(p=0.05)
        self.fc_relu1 = nn.ReLU()
        self.fc_relu2 = nn.ReLU()
        self.predictor = nn.Linear(fc_hidden, 1)

        self.fc_12 = nn.Linear(gnn_hidden + gnn_hidden + 1, fc_hidden)
        self.fc_22 = nn.Linear(fc_hidden, fc_hidden)
        self.fc_drop_12 = nn.Dropout(p=0.05)
        self.fc_drop_22 = nn.Dropout(p=0.05)
        self.fc_relu12 = nn.ReLU()
        self.fc_relu22 = nn.ReLU()
        self.predictor2 = nn.Linear(fc_hidden, 1)

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                init_tensor(m.weight, "thomas", "relu")
                init_tensor(m.bias, "thomas", "relu")
            elif isinstance(m, self.gnn_layer_func):
                pass

    def forward(self, graph, data):
        x, A = graph.x, graph.edge_index

        x = self.graph_conv_1(x, A)
        x = self.gnn_relu1(x)
        x = self.gnn_drop_1(x)
        x = self.graph_conv_2(x, A)
        x = self.gnn_relu2(x)
        x = self.gnn_drop_2(x)

        gnn_feat = scatter(x, graph.batch, dim=0, reduce=self.reduce_func)
        static_feature = self.norm_sf_linear(data)
        static_feature = self.norm_sf_drop(static_feature)
        static_feature = self.norm_sf_relu(static_feature)
        x = torch.cat([gnn_feat, static_feature], dim=1)

        x = self.fc_1(x)
        x = self.fc_relu1(x)
        x = self.fc_drop_1(x)
        x = self.fc_2(x)
        x = self.fc_relu2(x)
        feat = self.fc_drop_2(x)
        prefill = self.predictor(feat)

        x = torch.cat([gnn_feat, static_feature, prefill], dim=1)
        x = self.fc_12(x)
        x = self.fc_relu12(x)
        x = self.fc_drop_12(x)
        x = self.fc_22(x)
        x = self.fc_relu22(x)
        feat = self.fc_drop_22(x)
        decode = self.predictor2(feat)

        return prefill + decode
