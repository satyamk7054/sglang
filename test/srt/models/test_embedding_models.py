# Copyright 2023-2024 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

import multiprocessing as mp
import random
import unittest
from pathlib import Path

import torch
from transformers import AutoConfig, AutoTokenizer

from sglang.srt.layers.pooling_types import PoolingType
from sglang.test.runners import DEFAULT_PROMPTS, HFRunner, SRTRunner
from sglang.test.test_utils import (
    CustomTestCase,
    get_similarities,
    is_in_amd_ci,
    is_in_ci,
)

MODELS = [
    ("Alibaba-NLP/gte-Qwen2-1.5B-instruct", 1, 1e-5),
    ("intfloat/e5-mistral-7b-instruct", 1, 1e-5),
    ("marco/mcdse-2b-v1", 1, 1e-5),
    ("Qwen/Qwen3-Embedding-8B", 1, 1e-5),
    # Temporarily disable before this model is fixed
    # ("jason9693/Qwen2.5-1.5B-apeach", 1, 1e-5),
]
TORCH_DTYPES = [torch.float16]

LOCAL_MODEL_FOR_MEAN_POOLING_TEST = "/shared/public/elr-models/meta-llama/Llama-3.2-3B/5cc0ffe09ee49f7be6ca7c794ee6bd7245e84e60/"


class TestEmbeddingModels(CustomTestCase):

    @classmethod
    def setUpClass(cls):
        mp.set_start_method("spawn", force=True)

    def _truncate_prompts(self, prompts, model_path):
        config = AutoConfig.from_pretrained(model_path)
        max_length = getattr(config, "max_position_embeddings", 2048)

        tokenizer = AutoTokenizer.from_pretrained(model_path)

        truncated_prompts = []
        for prompt in prompts:
            tokens = tokenizer(prompt, return_tensors="pt", truncation=False)
            if len(tokens.input_ids[0]) > max_length:
                truncated_text = tokenizer.decode(
                    tokens.input_ids[0][: max_length - 1], skip_special_tokens=True
                )
                truncated_prompts.append(truncated_text)
            else:
                truncated_prompts.append(prompt)
        return truncated_prompts

    def assert_close_prefill_logits(
        self,
        prompts,
        model_path,
        tp_size,
        torch_dtype,
        prefill_tolerance,
    ) -> None:
        truncated_prompts = self._truncate_prompts(prompts, model_path)

        with HFRunner(
            model_path,
            torch_dtype=torch_dtype,
            model_type="embedding",
        ) as hf_runner:
            hf_outputs = hf_runner.forward(truncated_prompts)

        attention_backend = "triton" if is_in_amd_ci() else None
        with SRTRunner(
            model_path,
            tp_size=tp_size,
            torch_dtype=torch_dtype,
            model_type="embedding",
            attention_backend=attention_backend,
        ) as srt_runner:
            srt_outputs = srt_runner.forward(truncated_prompts)

        for i in range(len(prompts)):
            hf_logits = torch.Tensor(hf_outputs.embed_logits[i])
            srt_logits = torch.Tensor(srt_outputs.embed_logits[i])

            similarity = torch.tensor(get_similarities(hf_logits, srt_logits))
            print("similarity diff", abs(similarity - 1))

            if len(prompts[i]) <= 1000:
                assert torch.all(
                    abs(similarity - 1) < prefill_tolerance
                ), "embeddings are not all close"

    def test_prefill_logits(self):
        models_to_test = MODELS

        if is_in_ci():
            models_to_test = [random.choice(MODELS)]

        for model, tp_size, prefill_tolerance in models_to_test:
            for torch_dtype in TORCH_DTYPES:
                self.assert_close_prefill_logits(
                    DEFAULT_PROMPTS, model, tp_size, torch_dtype, prefill_tolerance
                )

    @unittest.skipIf(
        not Path(LOCAL_MODEL_FOR_MEAN_POOLING_TEST).exists(),
        f"Model path {LOCAL_MODEL_FOR_MEAN_POOLING_TEST} does not exist",
    )
    def test_mean_pooling(self):
        model_path = LOCAL_MODEL_FOR_MEAN_POOLING_TEST
        torch_dtype = torch.float16
        tp_size = 1
        prompts = DEFAULT_PROMPTS

        truncated_prompts = self._truncate_prompts(prompts, model_path)
        prefill_tolerance = 1e-5

        with HFRunner(
            model_path,
            torch_dtype=torch.float32,
            model_type="embedding",
            pooling_type="MEAN",
        ) as hf_runner:
            hf_outputs = hf_runner.forward(truncated_prompts)

        with SRTRunner(
            model_path,
            tp_size=tp_size,
            torch_dtype=torch_dtype,
            model_type="embedding",
            pooling_type=PoolingType.MEAN,
            chunked_prefill_size=-1,  # Disable chunked prefill for mean pooling
            disable_radix_cache=True,  # Disable radix cache
        ) as srt_runner:
            srt_outputs = srt_runner.forward(truncated_prompts)

        for i in range(len(prompts)):
            hf_logits = torch.Tensor(hf_outputs.embed_logits[i])
            srt_logits = torch.Tensor(srt_outputs.embed_logits[i])

            similarity = torch.tensor(get_similarities(hf_logits, srt_logits))
            print("similarity diff", abs(similarity - 1))

            if len(prompts[i]) <= 1000:
                assert torch.all(
                    abs(similarity - 1) < prefill_tolerance
                ), "embeddings are not all close"


if __name__ == "__main__":
    unittest.main()
