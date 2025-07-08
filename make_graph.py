import numpy as np

from util import kernel_type
from extract_feature import ModelAnalyzer


class EmbedValue:
    @staticmethod
    def embed_int(x, center=0, scale=1):
        x = np.array([int(x)], dtype="float32")
        return (x - center) / np.abs(scale)

    @staticmethod
    def embed_float(x, center=0, scale=1):
        x = np.array([float(x)], dtype="float32")
        return (x - center) / np.abs(scale)

    @staticmethod
    def embed_bool(x, center=0, scale=1):
        x = np.array([int(bool(x))], dtype="float32")
        return (x - center) / np.abs(scale)

    @staticmethod
    def embed_tuple(x, length, center=0, scale=1):
        x = np.array(x, dtype="float32").reshape(-1)
        if x.size > length:
            x = x[:length]
        if x.size < length:
            x = np.concatenate([x, np.zeros(length - x.size, dtype="float32")])
        if not isinstance(center, list):
            center = [center] * x.size
        if not isinstance(scale, list):
            scale = [scale] * x.size
        center = np.array(center, dtype="float32")
        scale = np.array(scale, dtype="float32")
        return (x - center) / np.abs(scale)

    @staticmethod
    def embed_kernel(kernel):
        length = len(kernel_type)
        if kernel not in kernel_type:
            return np.zeros(length, dtype="float32")

        kernel_code = kernel_type[kernel]["code"] - 1
        if kernel_code >= length:
            raise Exception(
                "kernel code of {}: {} greater than one-hot length {}!".format(
                    kernel, kernel_code, length
                )
            )
        return np.eye(length, dtype="float32")[kernel_code]


config_cache = {}


def get_analyer(model_id, hardware) -> ModelAnalyzer:
    config = f"{model_id}_{hardware}"
    if config not in config_cache:
        config_cache[config] = ModelAnalyzer(model_id, hardware)
    return config_cache[config]


def get_quant_bit(dtype):
    if dtype == "FP32":
        return 32
    elif dtype == "FP16":
        return 16
    elif dtype == "INT8":
        return 8
    elif dtype == "INT4":
        return 4
    else:
        return 16


def get_model_graph(model_id, hardware, inference_config):

    w_bit = get_quant_bit(inference_config["w_quant"])
    a_bit = get_quant_bit(inference_config["a_quant"])
    kv_bit = get_quant_bit(inference_config["kv_quant"])
    seq_length = int(inference_config["seq_length"])
    batch_size = int(inference_config["batch_size"])
    use_flashattention = bool(inference_config["use_flashattention"])
    gen_length = int(inference_config["gen_length"])
    hidden_size = int(inference_config["hidden_size"])
    act = str(inference_config["activation"])
    inter_size = int(inference_config["inter_size"])
    layer_num = int(inference_config["layer_num"])
    head_num = int(inference_config["head_num"])
    vob_size = int(inference_config["vob_size"])

    analyzer = get_analyer(model_id, hardware)
    result = analyzer.analyze(
        seqlen=seq_length,
        batchsize=batch_size,
        w_bit=w_bit,
        a_bit=a_bit,
        kv_bit=kv_bit,
        use_flashattention=use_flashattention,
        gen_token_num=1,
        act=act,
        hidden_size=hidden_size,
        inter_size=inter_size,
        layer_num=layer_num,
        head_num=head_num,
        vob=vob_size,
    )
    nodes = []
    edges = []

    def write_to_node(name, OPs, memory_access, bound, input_names=[]):
        node = {"id": name, "OPs": OPs, "memory_access": memory_access, "bound": bound}
        nodes.append(node)
        for input_name in input_names:
            edge = {"source": input_name, "target": name}
            edges.append(edge)

    if use_flashattention:
        layer_graph = analyzer.config.flashattention_transformer_layer_graph
    else:
        layer_graph = analyzer.config.transformer_layer_graph

    stage = inference_config["stage"]
    total_results = result["total_results"][stage]
    result = result[stage]

    for name, input_names in layer_graph.items():
        if name in ["input", "output"]:
            OPs = 0
            memory_access = 0
            bound = 1
        else:
            OPs = result[name]["OPs"]
            memory_access = result[name]["memory_access"]
            bound = result[name]["bound"]
        write_to_node(name, OPs, memory_access, bound, input_names)

    if gen_length > 1:
        n_divide = min(10, gen_length)
        for lengthi in np.linspace(seq_length + 1, seq_length + gen_length, n_divide):
            gen_result = analyzer.analyze(
                seqlen=lengthi,
                batchsize=batch_size,
                w_bit=w_bit,
                a_bit=a_bit,
                kv_bit=kv_bit,
                use_flashattention=use_flashattention,
                gen_token_num=1,
                act=act,
                hidden_size=hidden_size,
                inter_size=inter_size,
                layer_num=layer_num,
                head_num=head_num,
                vob=vob_size,
            )

            for k, v in gen_result["total_results"]["decode"].items():
                total_results[k] += v * gen_length / n_divide

            for name, input_names in layer_graph.items():
                if name in gen_result["decode"]:
                    result[name]["OPs"] += (
                        gen_result["decode"][name]["OPs"] * gen_length / n_divide
                    )
                    result[name]["memory_access"] += (
                        gen_result["decode"][name]["memory_access"]
                        * gen_length
                        / n_divide
                    )
        for name, input_names in layer_graph.items():
            if name in ["input", "output"]:
                OPs = 0
                memory_access = 0
                bound = 1
            else:
                OPs = result[name]["OPs"]
                memory_access = result[name]["memory_access"]
                bound = result[name]["bound"]
            write_to_node(name, OPs, memory_access, bound, input_names)

    return nodes, edges, total_results


def extract_graph_features(model_id, hardware, inference_config):
    node_embeddings = {}
    nodes, edges, useful_info = get_model_graph(model_id, hardware, inference_config)

    for node in nodes:
        e_ops = node["OPs"]
        e_mem_acc = node["memory_access"]
        e_bound = node["bound"]

        node_embeddings[node["id"]] = [
            e_ops,
            e_mem_acc,
            e_bound,
        ]

    features = []
    name2id = {}
    id2name = {}
    index = 0
    for node in nodes:
        features.append(node_embeddings[node["id"]])
        name2id[node["id"]] = index
        id2name[index] = node["id"]
        index = index + 1

    node_num = len(nodes)
    adjacent = np.zeros((node_num, node_num), dtype="float32")

    for edge in edges:
        s_id = name2id[edge["source"]]
        t_id = name2id[edge["target"]]
        adjacent[s_id][t_id] = 1

    e_ops_u = EmbedValue.embed_float(useful_info["OPs"])
    e_ma_u = EmbedValue.embed_float(useful_info["memory_access"])
    global_feature = np.concatenate(
        [
            e_ops_u,
            e_ma_u,
        ]
    )

    return features, adjacent, global_feature
