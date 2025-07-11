import joblib
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR
from torch_geometric.loader import DataLoader
from tqdm.notebook import tqdm

from util import Metric
from dataset import GraphDataset
from gnn_model import GraphNet


def do_testing(model, X_test_output, y_test):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    BATCH_SIZE = 512

    y_test_output = y_test[:, 0] + y_test[:, 1]
    test_dataset = GraphDataset(X_test_output, y_test_output)
    test_loader = DataLoader(dataset=test_dataset, batch_size=BATCH_SIZE)

    metric4 = Metric()
    with torch.no_grad():
        model.eval()
        for real_data_batch in test_loader:
            graph_batch = real_data_batch[0]
            data_batch = real_data_batch[1]
            graph_batch.y = graph_batch.y.view(-1, 1)
            graph_batch, data_batch = graph_batch.to(device), data_batch.to(device)
            y_test_pred = model(graph_batch, data_batch)
            ps = y_test_pred.data.cpu().numpy()[:, 0].tolist()
            gs = graph_batch.y.data.cpu().numpy()[:, 0].tolist()
            metric4.update(ps, gs)

        acc, err, err1, err2, cnt = metric4.get()
        print(
            "GNN test - MAPE:{:.5f} ErrBnd(0.3):{:.5f} ErrBnd(0.1):{:.5f} ErrBnd(0.05):{:.5f}".format(
                acc, err, err1, err2
            )
        )


def do_training(X_train_output, X_val_output, y_train, y_val):

    y_train_output = y_train[:, 0] + y_train[:, 1]
    y_val_output = y_val[:, 0] + y_val[:, 1]

    train_dataset = GraphDataset(X_train_output, y_train_output)
    val_dataset = GraphDataset(X_val_output, y_val_output)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    EPOCHS = 50
    BATCH_SIZE = 512
    LEARNING_RATE = 0.01

    NUM_FEATURES = X_train_output[0][0].shape[1]
    NUM_STATIC_FEATURES = X_train_output[0][2].shape[0]

    train_loader = DataLoader(
        dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True
    )
    val_loader = DataLoader(dataset=val_dataset, batch_size=BATCH_SIZE, shuffle=True)

    model = GraphNet(
        num_node_features=NUM_FEATURES,
        gnn_layer="SAGEConv",
        num_other_features=NUM_STATIC_FEATURES,
        gnn_hidden=16,
        fc_hidden=16,
        reduce_func="sum",
    )

    model.to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    lambda2 = lambda epoch: 0.98**epoch
    scheduler = LambdaLR(optimizer, lr_lambda=[lambda2])

    best_mape = 10000

    for e in tqdm(range(1, EPOCHS + 1)):
        metric = Metric()
        model.train()
        num_iter = len(train_loader)
        iteration = 0

        for real_data_batch in train_loader:
            graph_train_batch = real_data_batch[0]
            data_train_batch = real_data_batch[1]
            graph_train_batch.y = graph_train_batch.y.view(-1, 1)
            graph_train_batch, data_train_batch = graph_train_batch.to(
                device
            ), data_train_batch.to(device)
            optimizer.zero_grad()
            y_train_pred = model(graph_train_batch, data_train_batch)
            train_loss = criterion(
                y_train_pred / graph_train_batch.y,
                graph_train_batch.y / graph_train_batch.y,
            )
            train_loss.backward()
            optimizer.step()
            ps = y_train_pred.data.cpu().numpy()[:, 0].tolist()
            gs = graph_train_batch.y.data.cpu().numpy()[:, 0].tolist()
            metric.update(ps, gs)
            acc, err, err1, err2, _ = metric.get()

        with torch.no_grad():
            model.eval()
            for real_data_batch in val_loader:
                graph_val_batch = real_data_batch[0]
                data_val_batch = real_data_batch[1]
                graph_val_batch.y = graph_val_batch.y.view(-1, 1)
                graph_val_batch, data_val_batch = graph_val_batch.to(
                    device
                ), data_val_batch.to(device)
                y_val_pred = model(graph_val_batch, data_val_batch)
                ps = y_val_pred.data.cpu().numpy()[:, 0].tolist()
                gs = graph_val_batch.y.data.cpu().numpy()[:, 0].tolist()
                metric.update(ps, gs)
            acc, err, err1, err2, _ = metric.get()
            print(
                "Val - MAPE:{:.5f} ErrBnd(0.3):{:.5f} ErrBnd(0.1):{:.5f} ErrBnd(0.05):{:.5f}".format(
                    acc, err, err1, err2
                )
            )

            if acc < best_mape:
                best_mape = acc
                torch.save(model.state_dict(), "graph_model.pt")

        scheduler.step()

    return model


if __name__ == "__main__":
    data_list = joblib.load("graph_all.joblib")

    X_train_output = data_list[0]
    X_val_output = data_list[1]
    y_train = data_list[2]
    y_val = data_list[3]

    model = do_training(X_train_output, X_val_output, y_train, y_val)

    model.load_state_dict(torch.load("graph_model.pt", weights_only=True))

    data_list = joblib.load("test.joblib")

    for i in range(3):
        X_test_output = data_list[0][i]
        y_test = data_list[1][i]
        do_testing(model, X_test_output, y_test)