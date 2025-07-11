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


def do_testing(prefill_model, decode_model, prefill_test_loader, decode_test_loader):
    metric = Metric()
    with torch.no_grad():
        prefill_model.eval()
        decode_model.eval()

        for prefill_data_batch, decode_data_batch in zip(
            prefill_test_loader, decode_test_loader
        ):
            prefill_graph_batch = prefill_data_batch[0]
            prefill_label_batch = prefill_data_batch[1]

            decode_graph_batch = decode_data_batch[0]
            decode_label_batch = decode_data_batch[1]

            prefill_graph_batch.y = prefill_graph_batch.y.view(-1, 1)
            prefill_graph_batch, prefill_label_batch = (
                prefill_graph_batch.to(device),
                prefill_label_batch.to(device),
            )
            prefill_y_test_pred = prefill_model(
                prefill_graph_batch, prefill_label_batch
            )

            decode_graph_batch.y = decode_graph_batch.y.view(-1, 1)
            decode_graph_batch, decode_label_batch = (
                decode_graph_batch.to(device),
                decode_label_batch.to(device),
            )
            decode_y_test_pred = decode_model(decode_graph_batch, decode_label_batch)

            prefill_ps = prefill_y_test_pred.data.cpu().numpy()[:, 0].tolist()
            decode_ps = decode_y_test_pred.data.cpu().numpy()[:, 0].tolist()

            prefill_gs = prefill_graph_batch.y.data.cpu().numpy()[:, 0].tolist()
            decode_gs = decode_graph_batch.y.data.cpu().numpy()[:, 0].tolist()

            ps = []
            gs = []
            for i in range(len(prefill_ps)):
                ps.append(prefill_ps[i] + decode_ps[i])
                gs.append(prefill_gs[i] + decode_gs[i])

            metric.update(ps, gs)
        acc, err, err1, err2, cnt = metric.get()
        print(
            "GNN test - MAPE:{:.5f} ErrBnd(0.3):{:.5f} ErrBnd(0.1):{:.5f} ErrBnd(0.05):{:.5f}".format(
                acc, err, err1, err2
            )
        )


def do_training(
    model,
    train_loader,
    val_loader,
    device,
    optimizer,
    model_name,
    scheduler,
    test_loader,
):
    EPOCHS = 50
    best_mape = 10000
    criterion = nn.MSELoss()

    for e in tqdm(range(1, EPOCHS + 1)):
        metric = Metric()
        model.train()

        for real_data_batch in train_loader:
            graph_train_batch = real_data_batch[0]
            data_train_batch = real_data_batch[1]
            graph_train_batch.y = graph_train_batch.y.view(-1, 1)
            graph_train_batch = graph_train_batch.to(device)
            data_train_batch = data_train_batch.to(device)
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
        print(
            "Train - MAPE:{:.5f} ErrBnd(0.3):{:.5f} ErrBnd(0.1):{:.5f} ErrBnd(0.05):{:.5f}".format(
                acc, err, err1, err2
            )
        )

        with torch.no_grad():
            metric2 = Metric()
            model.eval()
            for real_data_batch in val_loader:
                graph_val_batch = real_data_batch[0]
                data_val_batch = real_data_batch[1]
                graph_val_batch.y = graph_val_batch.y.view(-1, 1)
                graph_val_batch = graph_val_batch.to(device)
                data_val_batch = data_val_batch.to(device)
                y_val_pred = model(graph_val_batch, data_val_batch)
                ps = y_val_pred.data.cpu().numpy()[:, 0].tolist()
                gs = graph_val_batch.y.data.cpu().numpy()[:, 0].tolist()
                metric2.update(ps, gs)
            acc, err, err1, err2, _ = metric2.get()
            print(
                "Val - MAPE:{:.5f} ErrBnd(0.3):{:.5f} ErrBnd(0.1):{:.5f} ErrBnd(0.05):{:.5f}".format(
                    acc, err, err1, err2
                )
            )

            if acc < best_mape:
                best_mape = acc
                torch.save(model.state_dict(), model_name)

        scheduler.step()

    with torch.no_grad():
        model.eval()
        metric3 = Metric()
        for real_data_batch in test_loader:
            graph_batch = real_data_batch[0]
            data_batch = real_data_batch[1]
            graph_batch.y = graph_batch.y.view(-1, 1)
            graph_batch = graph_batch.to(device)
            data_batch = data_batch.to(device)
            y_test_pred = model(graph_batch, data_batch)
            ps = y_test_pred.data.cpu().numpy()[:, 0].tolist()
            gs = graph_batch.y.data.cpu().numpy()[:, 0].tolist()
            metric3.update(ps, gs)
        acc, err, err1, err2, _ = metric3.get()
        print(
            "Test - MAPE:{:.5f} ErrBnd(0.3):{:.5f} ErrBnd(0.1):{:.5f} ErrBnd(0.05):{:.5f}".format(
                acc, err, err1, err2
            )
        )


if __name__ == "__main__":
    data_list = joblib.load("graph_llm.joblib")

    X_train_output = data_list[0]
    X_val_output = data_list[1]
    y_train = data_list[2]
    y_val = data_list[3]

    prefill_train_dataset = GraphDataset(X_train_output, y_train[:, 0])
    prefill_val_dataset = GraphDataset(X_val_output, y_val[:, 0])
    decode_train_dataset = GraphDataset(X_train_output, y_train[:, 1])
    decode_val_dataset = GraphDataset(X_val_output, y_val[:, 1])

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    BATCH_SIZE = 512
    LEARNING_RATE = 0.01

    NUM_FEATURES = X_train_output[0][0].shape[1]
    NUM_STATIC_FEATURES = X_train_output[0][2].shape[0]
    GNN_DIM = 64
    FC_DIM = 64

    prefill_train_loader = DataLoader(
        dataset=prefill_train_dataset, batch_size=BATCH_SIZE, shuffle=True
    )
    prefill_val_loader = DataLoader(dataset=prefill_val_dataset, batch_size=BATCH_SIZE)
    decode_train_loader = DataLoader(
        dataset=decode_train_dataset, batch_size=BATCH_SIZE, shuffle=True
    )
    decode_val_loader = DataLoader(dataset=decode_val_dataset, batch_size=BATCH_SIZE)

    prefill_model = GraphNet(
        num_node_features=NUM_FEATURES,
        gnn_layer="SAGEConv",
        num_other_features=NUM_STATIC_FEATURES,
        gnn_hidden=GNN_DIM,
        fc_hidden=FC_DIM,
        reduce_func="max",
    )

    decode_model = GraphNet(
        num_node_features=NUM_FEATURES,
        gnn_layer="SAGEConv",
        num_other_features=NUM_STATIC_FEATURES,
        gnn_hidden=GNN_DIM,
        fc_hidden=FC_DIM,
        reduce_func="max",
    )

    prefill_model.to(device)
    decode_model.to(device)

    prefill_optimizer = optim.Adam(prefill_model.parameters(), lr=LEARNING_RATE)
    decode_optimizer = optim.Adam(decode_model.parameters(), lr=LEARNING_RATE)

    lambda2 = lambda epoch: 0.98**epoch
    prefill_scheduler = LambdaLR(prefill_optimizer, lr_lambda=[lambda2])
    decode_scheduler = LambdaLR(decode_optimizer, lr_lambda=[lambda2])

    # prefill_model.load_state_dict(torch.load("prefill_model.pt", weights_only=True))
    # decode_model.load_state_dict(torch.load("decode_model.pt", weights_only=True))

    data_list = joblib.load("test.joblib")

    X_test_output = data_list[0]
    y_test = data_list[1]

    prefill_test_dataset = GraphDataset(X_test_output[0], y_test[0][:, 0])
    decode_test_dataset = GraphDataset(X_test_output[0], y_test[0][:, 1])
    prefill_test_loader = DataLoader(
        dataset=prefill_test_dataset, batch_size=BATCH_SIZE
    )
    decode_test_loader = DataLoader(dataset=decode_test_dataset, batch_size=BATCH_SIZE)

    do_training(
        prefill_model,
        prefill_train_loader,
        prefill_val_loader,
        device,
        prefill_optimizer,
        "prefill_model.pt",
        prefill_scheduler,
        prefill_test_loader,
    )

    do_training(
        decode_model,
        decode_train_loader,
        decode_val_loader,
        device,
        decode_optimizer,
        "decode_model.pt",
        decode_scheduler,
        decode_test_loader,
    )

    prefill_model.load_state_dict(torch.load("prefill_model.pt", weights_only=True))
    decode_model.load_state_dict(torch.load("decode_model.pt", weights_only=True))

    for i in range(3):
        prefill_test_dataset = GraphDataset(X_test_output[i], y_test[i][:, 0])
        decode_test_dataset = GraphDataset(X_test_output[i], y_test[i][:, 1])
        prefill_test_loader = DataLoader(
            dataset=prefill_test_dataset, batch_size=BATCH_SIZE
        )
        decode_test_loader = DataLoader(
            dataset=decode_test_dataset, batch_size=BATCH_SIZE
        )
        do_testing(prefill_model, decode_model, prefill_test_loader, decode_test_loader)
