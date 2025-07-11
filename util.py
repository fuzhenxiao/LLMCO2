import torch
import numpy as np
import joblib


def make_dataset():
    real_data = joblib.load("real_data_all.joblib")
    fake_data = joblib.load("fake_data_all.joblib")
    # [X_train, X_val, X_test, y_train, y_val, y_test]
    joblib.dump(
        [fake_data[0], fake_data[1], fake_data[3], fake_data[4]], "uniform_train.joblib"
    )
    # joblib.dump([real_data[2], real_data[5]], "test_data.joblib")
    llm = [[], [], []]
    label= [[], [], []]

    print(real_data[2].shape)

    for i in range(real_data[2].shape[0]):
        index = int(real_data[2][i][0])
        llm[index].append(real_data[2][i])
        label[index].append(real_data[5][i])

    lama7b = np.stack(llm[0], axis=0)
    lama7b_la = np.stack(label[0], axis=0)

    print(lama7b.shape)

    intm = np.stack(llm[1], axis=0)
    intm_la = np.stack(label[1], axis=0)
    
    print(intm.shape)

    lama70b = np.stack(llm[2], axis=0)
    lama70b_la = np.stack(label[2], axis=0)
    
    print(lama70b.shape)

    joblib.dump([lama7b, lama7b_la], "llama7b_test.joblib")
    joblib.dump([intm, intm_la], "intllm_test.joblib")
    joblib.dump([lama70b, lama70b_la], "llama70b_test.joblib")

class Metric(object):
    def __init__(self):
        self.all = self.init_pack()

    def init_pack(self):
        return {
            "cnt": 0,
            "apes": [],  # absolute percentage error
            "errbnd_cnt": np.array([0.0, 0.0, 0.0]),  # error bound count
            "errbnd_val": np.array(
                [0.3, 0.1, 0.05]
            ),  # error bound value: 0.1, 0.05, 0.01
        }

    def update_pack(self, ps, gs, pack):
        for i in range(len(ps)):
            ape = np.abs(ps[i] - gs[i]) / (np.abs(gs[i]) + 0.000001)
            pack["errbnd_cnt"][ape <= pack["errbnd_val"]] += 1
            pack["apes"].append(ape)
        pack["cnt"] += len(ps)

    def measure_pack(self, pack):
        acc = np.mean(pack["apes"])
        err = (pack["errbnd_cnt"] / pack["cnt"])[0]
        err1 = (pack["errbnd_cnt"] / pack["cnt"])[1]
        err2 = (pack["errbnd_cnt"] / pack["cnt"])[2]
        return acc, err, err1, err2, pack["cnt"]

    def update(self, ps, gs):
        self.update_pack(ps, gs, self.all)

    def get(self):
        return self.measure_pack(self.all)


hardware_params = {
    "nvidia_V100": {
        "mem_bandwidth": 900e9,
        "FP32": 14e12,
        "FP16": 112e12,
        "INT8": 62e12,
        "onchip_buffer": 20480e3,
        "network": 300e9,
        "num": 4,
    },
    "nvidia_A100": {
        "mem_bandwidth": 1555e9,
        "FP16": 312e12,
        "INT8": 624e12,
        "onchip_buffer": 27648e3,
        "network": 600e9,
        "num": 4,
    },
    "nvidia_H100": {
        "mem_bandwidth": 3072e9,
        "FP16": 1979e12 / 2,
        "INT8": 3958e12 / 2,
        "onchip_buffer": 33792e3,
        "network": 900e9,
        "num": 4,
    },
}

GPU_type = {
    "a100": {"code": 1},
    "h100": {"code": 2},
}

avaliable_hardwares = [_ for _ in hardware_params.keys()]


def roofline_analyze(mem_bandwidth, max_OPS, OPs, memory_access):
    y_max = max_OPS
    memory_access_bytes = memory_access
    turning_point = y_max / mem_bandwidth
    arithmetic_intensity = OPs / memory_access_bytes
    if arithmetic_intensity < turning_point:
        bound = 1
        performance = arithmetic_intensity * mem_bandwidth
    else:
        bound = 0
        performance = y_max

    return arithmetic_intensity, performance, bound


kernel_type = {
    "input": {"code": 1},
    "attn_norm": {"code": 2},
    "q_proj": {"code": 3},
    "k_proj": {"code": 4},
    "v_proj": {"code": 5},
    "qk_matmul": {"code": 6},
    "softmax": {"code": 7},
    "sv_matmul": {"code": 8},
    "out_proj": {"code": 9},
    "attn_add": {"code": 10},
    "mlp_norm": {"code": 11},
    "gate_proj": {"code": 12},
    "up_proj": {"code": 13},
    "mlp_act": {"code": 14},
    "down_proj": {"code": 15},
    "mlp_add": {"code": 16},
    "output": {"code": 17},
    "fused_attention": {"code": 18},
}

llm_type = {
    "internlm": {"code": 1},
    "llama": {"code": 2},
}

act_type = {
    "silu": {"code": 1},
    "gelu": {"code": 2},
    "gelu_pytorch_tanh": {"code": 3},
}


avaliable_model_ids_sources = {
    "meta-llama/Llama-2-7b-hf": {"file": "Llama.py"},
    "meta-llama/Llama-2-70b-hf": {"file": "Llama.py"},
    "internlm/internlm-20b": {"file": "Inter.py"},
}
avaliable_model_ids = [_ for _ in avaliable_model_ids_sources.keys()]


if __name__ == "__main__":
    make_dataset()