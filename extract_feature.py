import os
import math
import importlib
from transformers import AutoConfig
from util import avaliable_model_ids_sources, hardware_params, roofline_analyze

ALL_DATA_NAMES = [
    "OPs",
    "memory_access",
]

access_token = ""


class ModelAnalyzer:
    def __init__(self, model_id, hardware):
        self.model_id = model_id
        self.hardware = hardware

        config_file = "llm_config/" + avaliable_model_ids_sources[model_id]["file"]

        assert (
            config_file is not None
        ), "config file is not found, please specify it manually."

        print(f"use config file {config_file} for {model_id}")

        self.model_params = AutoConfig.from_pretrained(
            model_id, token=access_token, trust_remote_code=True
        )

        self.config = importlib.import_module(
            config_file.replace("/", ".").replace(".py", "")
        )

        self.results = {}
        self.w_bit = 4
        self.a_bit = 4
        self.kv_bit = 4
        self.batchsize = 1
        self.seqlen = 1
        self.node_num = 0
        self.gen_token_num = 0
        self.use_flashattention = True
        self.activation = "silu"
        self.hidden_size = 0
        self.inter_size = 0
        self.layer_num = 0
        self.head_num = 0
        self.vob = 0

    def _analyze_to_results(
        self,
        stage,
        name,
        OPs=0,
        load_weight=0,
        load_act=0,
        store_act=0,
        load_kv_cache=0,
        store_kv_cache=0,
    ):
        mem_bandwidth, max_OPS, _ = self.get_hardware_info()

        memory_access = (
            load_weight + load_act + store_act + load_kv_cache + store_kv_cache
        )
        _, _, bound = roofline_analyze(mem_bandwidth, max_OPS, OPs, memory_access)
        self.results[stage][name] = {
            "OPs": OPs,
            "memory_access": memory_access,
            "bound": bound,
        }

    def analyze(
        self,
        seqlen,
        batchsize,
        w_bit=16,
        a_bit=16,
        kv_bit=16,
        use_flashattention=True,
        gen_token_num=1,
        act="silu",
        hidden_size=0,
        inter_size=0,
        layer_num=0,
        head_num=0,
        vob=0,
    ):
        assert seqlen > 0
        assert batchsize > 0
        self.results = {"decode": {}, "prefill": {}}
        if kv_bit is None:
            kv_bit = a_bit
        self.w_bit = w_bit
        self.a_bit = a_bit
        self.kv_bit = kv_bit
        self.batchsize = batchsize
        self.seqlen = seqlen

        self.gen_token_num = gen_token_num
        self.act = act
        self.hidden_size = hidden_size
        self.inter_size = inter_size
        self.layer_num = layer_num
        self.head_num = head_num
        self.vob = vob

        w_byte = self.w_bit / 8
        a_byte = self.a_bit / 8
        kv_byte = self.kv_bit / 8

        config = self.config
        model_params = self.model_params

        num_attention_heads = config.get_num_attention_heads(model_params)
        hidden_size = config.get_hidden_size(model_params)
        num_key_value_heads = config.get_num_key_value_heads(model_params)
        num_hidden_layers = config.get_num_hidden_layers(model_params)

        for name, (ic, oc) in config.get_linear_layers(model_params).items():
            is_kv_proj = name in ["k_proj", "v_proj"]
            is_normal_proj = not is_kv_proj
            self._analyze_to_results(
                "decode",
                name,
                OPs=ic * oc * batchsize * 2,
                load_weight=ic * oc * w_byte,
                load_act=ic * batchsize * a_byte,
                store_act=0 if is_kv_proj else oc * batchsize * a_byte,
                load_kv_cache=0,
                store_kv_cache=(0 if is_normal_proj else oc * batchsize * kv_byte),
            )
            # for prefill
            self._analyze_to_results(
                "prefill",
                name,
                OPs=ic * oc * batchsize * seqlen * 2,
                load_weight=ic * oc * w_byte,
                load_act=ic * batchsize * seqlen * a_byte,
                store_act=(0 if is_kv_proj else oc * batchsize * seqlen * a_byte),
                load_kv_cache=0,
                store_kv_cache=(
                    0 if is_normal_proj else oc * batchsize * seqlen * kv_byte
                ),
            )

        # for attention
        head_size = hidden_size // num_attention_heads
        # for decode
        qk_matmul_OPs = seqlen * head_size * num_attention_heads * batchsize * 2
        sv_matmul_OPs = 1 * head_size * seqlen * num_attention_heads * batchsize * 2
        # the softmax operation takes five steps:
        # max_x=max(x)
        # x=x-max_x
        # x_exp=exp(x)
        # sum_x_exp=sum(x_exp)
        # y=x_exp/sum(x_exp)
        softmax_OPs = batchsize * num_attention_heads * seqlen * 1 * 5
        if use_flashattention:
            name = f"fused_attention"
            mem_bandwidth, max_OPS, onchip_buffer = self.get_hardware_info()
            # flashattention-2 https://arxiv.org/pdf/2307.08691.pdf
            block_size_r = min(
                math.ceil(onchip_buffer / (kv_byte * head_size)), head_size
            )
            n_blocks_r = math.ceil(1 / block_size_r)
            q_numel = (1) * head_size * batchsize * num_attention_heads * a_byte
            o_numel = 1 * seqlen * batchsize * num_attention_heads * a_byte
            self._analyze_to_results(
                "decode",
                name,
                OPs=qk_matmul_OPs + sv_matmul_OPs + softmax_OPs,
                load_weight=0,
                load_act=q_numel,
                store_act=o_numel * 2,  # initialize O and save O
                load_kv_cache=n_blocks_r
                * (seqlen)
                * head_size
                * batchsize
                * num_key_value_heads
                * kv_byte
                * 2,
                store_kv_cache=0,
            )

        else:
            name = f"qk_matmul"
            self._analyze_to_results(
                "decode",
                name,
                OPs=qk_matmul_OPs,
                load_weight=0,
                load_act=(1) * head_size * batchsize * num_attention_heads * a_byte,
                store_act=1 * seqlen * batchsize * num_attention_heads * a_byte,
                load_kv_cache=(seqlen)
                * head_size
                * batchsize
                * num_key_value_heads
                * kv_byte,
                store_kv_cache=0,
            )
            name = f"sv_matmul"
            self._analyze_to_results(
                "decode",
                name,
                OPs=sv_matmul_OPs,
                load_weight=0,
                load_act=(1 * seqlen * batchsize * num_attention_heads) * a_byte,
                store_act=1 * head_size * batchsize * num_attention_heads * a_byte,
                load_kv_cache=(seqlen * head_size * batchsize * num_key_value_heads)
                * kv_byte,
                store_kv_cache=0,
            )

            name = f"softmax"
            # max sub exp sum div
            self._analyze_to_results(
                "decode",
                name,
                OPs=softmax_OPs,
                load_weight=0,
                load_act=batchsize * num_attention_heads * seqlen * 1 * a_byte,
                store_act=batchsize * num_attention_heads * seqlen * 1 * a_byte,
                load_kv_cache=0,
                store_kv_cache=0,
            )

        for name in config.get_norm_layers(model_params):
            # sum sub pow sum div mul add
            self._analyze_to_results(
                "decode",
                name,
                OPs=batchsize * hidden_size * 1 * 7,
                load_weight=0,
                load_act=batchsize * hidden_size * 1 * a_byte,
                store_act=batchsize * hidden_size * 1 * a_byte,
                load_kv_cache=0,
                store_kv_cache=0,
            )

        for name in ["attn_add", "mlp_add"]:
            self._analyze_to_results(
                "decode",
                name,
                OPs=batchsize * hidden_size * 1,
                load_weight=0,
                load_act=batchsize * hidden_size * 1 * a_byte,
                store_act=batchsize * hidden_size * 1 * a_byte,
                load_kv_cache=0,
                store_kv_cache=0,
            )
        for name in ["mlp_act"]:
            self._analyze_to_results(
                "decode",
                name,
                OPs=batchsize * hidden_size * 1 * 2,
                load_weight=0,
                load_act=batchsize * hidden_size * 1 * a_byte * 2,
                store_act=batchsize * hidden_size * 1 * a_byte,
                load_kv_cache=0,
                store_kv_cache=0,
            )

        # for prefill
        qk_matmul_OPs = (
            seqlen * seqlen * head_size * num_attention_heads * batchsize * 2
        )
        sv_matmul_OPs = (
            seqlen * head_size * seqlen * num_attention_heads * batchsize * 2
        )
        softmax_OPs = batchsize * num_attention_heads * seqlen * seqlen * 5
        if use_flashattention:
            name = f"fused_attention"
            mem_bandwidth, max_OPS, onchip_buffer = self.get_hardware_info()
            # flashattention-2 https://arxiv.org/pdf/2307.08691.pdf
            block_size_r = min(
                math.ceil(onchip_buffer / (kv_byte * head_size)), head_size
            )
            n_blocks_r = math.ceil(seqlen / block_size_r)
            q_numel = seqlen * head_size * batchsize * num_attention_heads * a_byte
            o_numel = seqlen * seqlen * batchsize * num_attention_heads * a_byte
            self._analyze_to_results(
                "prefill",
                name,
                OPs=qk_matmul_OPs + sv_matmul_OPs + softmax_OPs,
                load_weight=0,
                load_act=q_numel,
                store_act=o_numel * 2,  # initialize O and save O
                load_kv_cache=n_blocks_r
                * (seqlen)
                * head_size
                * batchsize
                * num_key_value_heads
                * kv_byte
                * 2,
                store_kv_cache=0,
            )
        else:
            name = f"qk_matmul"
            self._analyze_to_results(
                "prefill",
                name,
                OPs=qk_matmul_OPs,
                load_weight=0,
                load_act=seqlen * head_size * batchsize * num_key_value_heads * a_byte,
                store_act=seqlen * seqlen * batchsize * num_attention_heads * a_byte,
                load_kv_cache=seqlen
                * head_size
                * batchsize
                * num_key_value_heads
                * kv_byte,
                store_kv_cache=0,
            )
            name = f"sv_matmul"
            self._analyze_to_results(
                "prefill",
                name,
                OPs=sv_matmul_OPs,
                load_weight=0,
                load_act=seqlen * seqlen * batchsize * num_attention_heads * a_byte,
                store_act=seqlen * head_size * batchsize * num_attention_heads * a_byte,
                load_kv_cache=seqlen
                * head_size
                * batchsize
                * num_key_value_heads
                * kv_byte,
                store_kv_cache=0,
            )
            name = f"softmax"
            self._analyze_to_results(
                "prefill",
                name,
                OPs=softmax_OPs,
                load_weight=0,
                load_act=batchsize * num_attention_heads * seqlen * seqlen * a_byte,
                store_act=batchsize * num_attention_heads * seqlen * seqlen * a_byte,
                load_kv_cache=0,
                store_kv_cache=0,
            )
        for name in config.get_norm_layers(model_params):
            self._analyze_to_results(
                "prefill",
                name,
                OPs=batchsize * hidden_size * seqlen * 7,
                load_weight=0,
                load_act=batchsize * hidden_size * seqlen * a_byte,
                store_act=batchsize * hidden_size * seqlen * a_byte,
                load_kv_cache=0,
                store_kv_cache=0,
            )
        for name in ["attn_add", "mlp_add"]:
            self._analyze_to_results(
                "prefill",
                name,
                OPs=batchsize * hidden_size * seqlen * 1,
                load_weight=0,
                load_act=batchsize * hidden_size * seqlen * a_byte,
                store_act=batchsize * hidden_size * seqlen * a_byte,
                load_kv_cache=0,
                store_kv_cache=0,
            )
        for name in ["mlp_act"]:
            self._analyze_to_results(
                "prefill",
                name,
                OPs=batchsize * hidden_size * seqlen * 1 * 2,
                load_weight=0,
                load_act=batchsize * hidden_size * seqlen * a_byte * 2,
                store_act=batchsize * hidden_size * seqlen * a_byte,
                load_kv_cache=0,
                store_kv_cache=0,
            )

        # compute total
        total_results = {"decode": {}, "prefill": {}}
        for data_name in ALL_DATA_NAMES:
            total_results["decode"][data_name] = 0
            total_results["prefill"][data_name] = 0
        for stage in ["decode", "prefill"]:
            for layer_name, result in self.results[stage].items():
                for data_name in ALL_DATA_NAMES:
                    total_results[stage][data_name] += (
                        result[data_name] * num_hidden_layers
                    )

        # lm_head
        name = "lm_head"
        args = {"batchsize": batchsize, "a_byte": a_byte, "w_byte": w_byte}
        for layer_info in self.config.post_process(self.model_params, args):
            self._analyze_to_results(**layer_info)
            for data_name in ALL_DATA_NAMES:
                total_results[layer_info["stage"]][data_name] += self.results[
                    layer_info["stage"]
                ][layer_info["name"]][data_name]

        self.results["total_results"] = total_results
        return self.results

    def get_hardware_info(self):
        mem_bandwidth = hardware_params[self.hardware]["mem_bandwidth"]

        if self.w_bit <= 4 and self.a_bit <= 4 and self.kv_bit <= 4:
            max_OPS = hardware_params[self.hardware]["INT4"]
        elif self.w_bit <= 8 and self.a_bit <= 8 and self.kv_bit <= 8:
            max_OPS = hardware_params[self.hardware]["INT8"]
        elif self.w_bit <= 16 and self.a_bit <= 16 and self.kv_bit <= 16:
            max_OPS = hardware_params[self.hardware]["FP16"]
        else:
            max_OPS = hardware_params[self.hardware]["FP32"]
        onchip_buffer = hardware_params[self.hardware]["onchip_buffer"]
        return mem_bandwidth, max_OPS, onchip_buffer

    def get_model_info(self):
        if self.config.get_num_attention_heads(
            self.model_params
        ) != self.config.get_num_key_value_heads(self.model_params):
            GQA = True
        else:
            GQA = False
        info = {"GQA": GQA}  # group query attention
        return info
