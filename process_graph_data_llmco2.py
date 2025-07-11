import joblib
import numpy as np
import random
from sklearn.preprocessing import StandardScaler
from make_graph import extract_graph_features


def single_task(m_data):
    llm = m_data[0]
    gpu = m_data[1]
    batch = m_data[2]
    tp = m_data[3]
    prompt_length = m_data[4]
    token_length = m_data[5]

    if gpu == 0.0:
        gpu_name = "nvidia_A100"
    else:
        gpu_name = "nvidia_H100"

    if llm == 0.0:
        llm_name = "meta-llama/Llama-2-70b-hf"
        act = "silu"
        hidden = 8192
        inter = 28672
        layer = 80
        head = 64
        vob = 32000
        width = "FP16"
    elif llm == 1.0:
        llm_name = "meta-llama/Llama-2-7b-hf"
        act = "silu"
        hidden = 4096
        inter = 11008
        layer = 32
        head = 32
        vob = 32000
        width = "FP16"
    else:
        llm_name = "internlm/internlm-20b"
        act = "silu"
        hidden = 5120
        inter = 13824
        layer = 60
        head = 40
        vob = 32000
        width = "FP16"

    if token_length > 1:
        stage_temp = "decode"
    else:
        stage_temp = "prefill"

    inference_config = {
        "stage": stage_temp,
        "batch_size": batch,
        "seq_length": prompt_length,
        "gen_length": token_length,
        "w_quant": width,
        "a_quant": width,
        "kv_quant": width,
        "use_flashattention": True,
        "activation": act,
        "hidden_size": hidden,
        "inter_size": inter,
        "layer_num": layer,
        "head_num": head,
        "vob_size": vob,
    }

    nodes, edges, global_f = extract_graph_features(
        llm_name, gpu_name, inference_config
    )

    global_data = np.array(
        [
            global_f[0],
            global_f[1],
            gpu,
            llm,
            hidden,
            inter,
            layer,
            head,
            vob,
            batch,
            prompt_length,
            token_length,
        ],
        dtype=np.float32,
    )

    global_data = np.ravel(global_data)
    edge_index = np.array(np.where(edges > 0), dtype=np.int32)
    node_features = np.array(nodes, dtype=np.float32)

    # print(node_features.shape)
    # print(edge_index.shape)
    # print(global_data.shape)

    return (node_features, edge_index, global_data)


def main():

    all_data = joblib.load("real_data_all.joblib")
    # [X_train, X_val, X_test, y_train, y_val, y_test]

    X_train = all_data[0]
    X_val = all_data[1]
    X_test = all_data[2]
    y_train = all_data[3]
    y_val = all_data[4]
    y_test = all_data[5]

    X_train_output = []
    X_val_output = []
    X_test_output = [[], [], []]

    node_list = []
    edge_list = []
    glob_list = []

    node_max = -1
    for i in range(X_train.shape[0]):
        node, edge, glob = single_task(X_train[i])
        current_max = np.max(node)
        if current_max > node_max:
            node_max = current_max

        node_list.append(node)
        edge_list.append(edge)
        glob_list.append(glob)

    glob_array = np.stack(glob_list, axis=0)
    print(glob_array.shape)
    glob_scale = StandardScaler()
    scaled_glob_array = glob_scale.fit_transform(glob_array)

    for i in range(X_train.shape[0]):
        node = node_list[i] / node_max
        edge = edge_list[i]
        glob = scaled_glob_array[i]
        X_train_output.append([node, edge, glob])

    node_list = []
    edge_list = []
    glob_list = []
    for i in range(X_val.shape[0]):
        node, edge, glob = single_task(X_val[i])
        node_list.append(node)
        edge_list.append(edge)
        glob_list.append(glob)

    glob_array = np.stack(glob_list, axis=0)
    scaled_glob_array = glob_scale.transform(glob_array)

    for i in range(X_val.shape[0]):
        node = node_list[i] / node_max
        edge = edge_list[i]
        glob = scaled_glob_array[i]
        X_val_output.append([node, edge, glob])

    node_list = [[], [], []]
    edge_list = [[], [], []]
    glob_list = [[], [], []]
    for i in range(len(X_test)):
        for j in range(X_test[i].shape[0]):
            node, edge, glob = single_task(X_test[i][j])
            node_list[i].append(node)
            edge_list[i].append(edge)
            glob_list[i].append(glob)

        glob_array = np.stack(glob_list[i], axis=0)
        scaled_glob_array = glob_scale.transform(glob_array)

        for j in range(X_test[i].shape[0]):
            node = node_list[i][j] / node_max
            edge = edge_list[i][j]
            glob = scaled_glob_array[j]
            X_test_output[i].append([node, edge, glob])

    print(len(X_train_output))
    print(len(y_train))
    print(len(X_val_output))
    print(len(y_val))

    y_train = y_train * 150
    y_val = y_val * 150

    for i in range(3):
        print(len(X_test_output[i]))
        print(y_test[i].shape)
        y_test[i] = y_test[i] * 150

    joblib.dump(
        [X_train_output, X_val_output, y_train, y_val],
        "graph_llm.joblib",
    )

    joblib.dump([X_test_output, y_test], "test.joblib")


if __name__ == "__main__":
    main()
