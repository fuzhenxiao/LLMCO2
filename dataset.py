import torch
from torch_geometric.data import Dataset, Data

import numpy as np


class GraphDataset(Dataset):
    def __init__(self, all_data, all_label):
        super(GraphDataset, self).__init__("./", None, None)
        self.graph_list = []
        self.length = len(all_data)

        for i in range(self.length):
            node = torch.from_numpy(all_data[i][0]).type(torch.float)
            edge = torch.from_numpy(all_data[i][1]).type(torch.long)
            sdata = torch.from_numpy(np.ravel(all_data[i][2])).type(torch.float)
            label = torch.tensor(all_label[i], dtype=torch.float)
            mdata = Data(x=node, edge_index=edge, y=label)
            self.graph_list.append([mdata, sdata])

    def get(self, idx):
        return self.graph_list[idx]

    def len(self):
        return self.length


class DoubleGraphDataset(Dataset):
    def __init__(self, all_data, prefill_label, gen_label):
        super(DoubleGraphDataset, self).__init__("./", None, None)
        self.graph_list = []
        self.length = len(all_data)
        
        for i in range(self.length):
            node = torch.from_numpy(all_data[i][0]).type(torch.float)
            edge = torch.from_numpy(all_data[i][1]).type(torch.long)
            sdata = torch.from_numpy(np.ravel(all_data[i][2])).type(torch.float)
            label_1 = torch.tensor(prefill_label[i], dtype=torch.float)
            label_2 = torch.tensor(gen_label[i], dtype=torch.float)
            mdata = Data(x=node, edge_index=edge, y=label_1)
            self.graph_list.append([mdata, label_2, sdata])

    def get(self, idx):
        return self.graph_list[idx]

    def len(self):
        return self.length
