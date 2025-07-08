import joblib
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR
from torch_geometric.loader import DataLoader
from tqdm.notebook import tqdm

from util import Metric
from dataset import DoubleGraphDataset
from gnn_model import GraphNet


if __name__ == "__main__":
    data_list = joblib.load("graph_all.joblib")

    X_train_output = data_list[0]
    X_val_output = data_list[1]
    X_test_output = data_list[2]
    y_train = data_list[3]
    y_val = data_list[4]
    y_test = data_list[5]

    train_dataset = DoubleGraphDataset(X_train_output, y_train[:, 0], y_train[:, 1])
    val_dataset = DoubleGraphDataset(X_val_output, y_val[:, 0], y_val[:, 1])
    test_dataset = DoubleGraphDataset(X_test_output, y_test[:, 0], y_test[:, 1])

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    EPOCHS = 50
    BATCH_SIZE = 512
    LEARNING_RATE = 0.01

    NUM_FEATURES = X_train_output[0][0].shape[1]
    NUM_STATIC_FEATURES = X_train_output[0][2].shape[0]
    GNN_DIM = 32
    FC_DIM = 32

    train_loader = DataLoader(
        dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True
    )
    val_loader = DataLoader(dataset=val_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=BATCH_SIZE)

    model = GraphNet(
        num_node_features=NUM_FEATURES,
        gnn_layer="SAGEConv",
        num_other_features=NUM_STATIC_FEATURES,
        gnn_hidden=GNN_DIM,
        fc_hidden=FC_DIM,
        reduce_func="max",
    )

    model2 = GraphNet(
        num_node_features=NUM_FEATURES,
        gnn_layer="SAGEConv",
        num_other_features=NUM_STATIC_FEATURES + 1,
        gnn_hidden=GNN_DIM,
        fc_hidden=FC_DIM,
        reduce_func="max",
    )

    model.to(device)
    model2.to(device)
    criterion = nn.MSELoss()
    criterion2 = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    optimizer2 = optim.Adam(model2.parameters(), lr=LEARNING_RATE)

    lambda2 = lambda epoch: 0.98**epoch
    scheduler = LambdaLR(optimizer, lr_lambda=[lambda2])
    scheduler2 = LambdaLR(optimizer2, lr_lambda=[lambda2])
    best_mape = 10000

    for e in tqdm(range(1, EPOCHS + 1)):
        metric = Metric()
        model.train()
        model2.train()
        num_iter = len(train_loader)
        iteration = 0

        for real_data_batch in train_loader:
            graph_train_batch = real_data_batch[0]
            label_train_batch = real_data_batch[1]
            data_train_batch = real_data_batch[2]
            graph_train_batch.y = graph_train_batch.y.view(-1, 1)
            label_train_batch = label_train_batch.view(-1, 1)
            graph_train_batch, data_train_batch, label_train_batch = (
                graph_train_batch.to(device),
                data_train_batch.to(device),
                label_train_batch.to(device),
            )
            optimizer.zero_grad()
            y_train_pred = model(graph_train_batch, data_train_batch)
            train_loss = criterion(y_train_pred, graph_train_batch.y)
            train_loss.backward()
            optimizer.step()

            data_train_batch2 = torch.cat((data_train_batch, y_train_pred), 1)
            optimizer2.zero_grad()
            y_train_pred2 = model2(graph_train_batch, data_train_batch2)
            train_loss2 = criterion2(y_train_pred2, label_train_batch)
            train_loss2.backward()
            optimizer2.step()

            ps = y_train_pred.data.cpu().numpy()[:, 0].tolist()
            ps2 = y_train_pred2.data.cpu().numpy()[:, 0].tolist()
            ps = ps + ps2
            gs = graph_train_batch.y.data.cpu().numpy()[:, 0].tolist()
            metric.update(ps, gs)
            acc, err, err1, err2, cnt = metric.get()
            if iteration % 10 == 0:
                print(
                    "Epoch[{}/{}]({}/{}) Loss:{:.5f} MAPE:{:.5f} "
                    "ErrBnd(0.3):{:.5f} ErrBnd(0.1):{:.5f} ErrBnd(0.05):{:.5f}".format(
                        e,
                        EPOCHS,
                        iteration + 1,
                        num_iter,
                        train_loss.item(),
                        acc,
                        err,
                        err1,
                        err2,
                    )
                )
                iteration += 1

        with torch.no_grad():
            model.eval()
            model2.eval()
            for real_data_batch in val_loader:
                graph_val_batch = real_data_batch[0]
                label_val_batch = real_data_batch[1]
                data_val_batch = real_data_batch[2]
                graph_val_batch.y = graph_val_batch.y.view(-1, 1)
                label_val_batch = label_val_batch.view(-1, 1)
                graph_val_batch, data_val_batch, label_val_batch = (
                    graph_val_batch.to(device),
                    data_val_batch.to(device),
                    label_val_batch.to(device),
                )
                y_val_pred = model(graph_val_batch, data_val_batch)
                val_loss = criterion(
                    y_val_pred / graph_val_batch.y,
                    graph_val_batch.y / graph_val_batch.y,
                )
                data_val_batch2 = torch.cat((data_val_batch, y_val_pred), 1)
                y_val_pred2 = model2(graph_val_batch, data_val_batch2)
                val_loss2 = criterion(
                    y_val_pred2 / label_val_batch,
                    label_val_batch / label_val_batch,
                )
                ps = y_val_pred.data.cpu().numpy()[:, 0].tolist()
                ps2 = y_val_pred2.data.cpu().numpy()[:, 0].tolist()
                ps = ps + ps2
                gs = graph_val_batch.y.data.cpu().numpy()[:, 0].tolist()
                metric.update(ps, gs)
            acc, err, err1, err2, cnt = metric.get()
            print(
                "Val - MAPE:{:.5f} ErrBnd(0.3):{:.5f} ErrBnd(0.1):{:.5f} ErrBnd(0.05):{:.5f}".format(
                    acc, err, err1, err2
                )
            )

            if acc < best_mape:
                best_mape = acc
                torch.save(model.state_dict(), "tgraph_model.pt")
                torch.save(model2.state_dict(), "tgraph_model2.pt")

        scheduler.step()

    model.load_state_dict(torch.load("tgraph_model.pt", weights_only=True))
    model2.load_state_dict(torch.load("tgraph_model2.pt", weights_only=True))

    metric4 = Metric()
    with torch.no_grad():
        model.eval()
        model2.eval()
        for real_data_batch in test_loader:
            graph_batch = real_data_batch[0]
            label_batch = real_data_batch[1]
            data_batch = real_data_batch[2]
            graph_batch.y = graph_batch.y.view(-1, 1)
            label_batch = label_batch.view(-1, 1)
            graph_batch, data_batch, label_batch = (
                graph_batch.to(device),
                data_batch.to(device),
                label_batch.to(device),
            )
            y_test_pred = model(graph_batch, data_batch)
            data_batch2 = torch.cat((data_batch, y_test_pred), 1)
            y_test_pred2 = model2(graph_batch, data_batch2)
            ps = y_test_pred.data.cpu().numpy()[:, 0].tolist()
            ps2 = y_test_pred2.data.cpu().numpy()[:, 0].tolist()
            ps = ps + ps2
            gs = graph_batch.y.data.cpu().numpy()[:, 0].tolist()
            metric4.update(ps, gs)
        acc, err, err1, err2, cnt = metric4.get()
        print(
            "GNN test - MAPE:{:.5f} ErrBnd(0.3):{:.5f} ErrBnd(0.1):{:.5f} ErrBnd(0.05):{:.5f}".format(
                acc, err, err1, err2
            )
        )
