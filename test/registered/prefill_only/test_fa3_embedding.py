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
import unittest

import torch

from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.runners import DEFAULT_PROMPTS, HFRunner, SRTRunner
from sglang.test.test_utils import (
    CustomTestCase,
    get_similarities,
    is_in_amd_ci,
    is_in_ci,
)

# Split from test_embedding_models.py so it can take its own CI registration: this
# exercises the FA3 backend (SM 80-90, Ampere/Ada/Hopper), while the rest of the
# prefill_only embedding suite runs on SM120 where FA3 is absent.
register_cuda_ci(est_time=180, stage="base-b", runner_config="1-gpu-large")

_FA3_SM_MIN = (8, 0)
_FA3_SM_MAX = (9, 0)

# Non-MLA embedding model (the fa_skip_kv_cache path is restricted to non-MLA).
_MODEL_PATH = "Alibaba-NLP/gte-Qwen2-1.5B-instruct"
# Matches MODEL_TO_CONFIG[_MODEL_PATH] in test_embedding_models.py.
_PREFILL_TOLERANCE = 1e-5


def _fa3_supported() -> bool:
    if not torch.cuda.is_available() or is_in_amd_ci():
        return False
    return _FA3_SM_MIN <= torch.cuda.get_device_capability() <= _FA3_SM_MAX


def _fmt_sm(cap: tuple[int, int]) -> str:
    return f"{cap[0]}.{cap[1]}"


class TestFa3Embedding(CustomTestCase):
    """Validate fa3 + piecewise CUDA graph embeddings over both attention paths.

    Runs the production-shaped embedding config (fa3 + piecewise CUDA graph +
    chunked_prefill_size=-1 + disable_radix_cache) and asserts no NaN + SRT-vs-HF
    cosine for both prefill_only_disable_kv_cache=True (the opt-in raw-K/V
    fa_skip_kv_cache fast path) and =False (the default paged path that embedding
    deployments run unless they opt in).
    """

    @classmethod
    def setUpClass(cls):
        mp.set_start_method("spawn", force=True)

    def setUp(self):
        if _fa3_supported():
            return
        got = (
            _fmt_sm(torch.cuda.get_device_capability())
            if torch.cuda.is_available() and not is_in_amd_ci()
            else "none"
        )
        msg = (
            f"requires the FA3 backend (CUDA SM {_fmt_sm(_FA3_SM_MIN)}-"
            f"{_fmt_sm(_FA3_SM_MAX)}, non-AMD); got SM {got}"
        )
        # In CI this test is pinned to FA3-capable hardware, so a mismatch means it
        # was dispatched to the wrong SM capability -> fail loudly rather than pass
        # silently. Locally (other GPUs) just skip.
        if is_in_ci():
            self.fail(msg)
        self.skipTest(msg)

    def _srt_embeddings(self, prompts, prefill_only_disable_kv_cache):
        with SRTRunner(
            _MODEL_PATH,
            tp_size=1,
            torch_dtype=torch.float16,
            model_type="embedding",
            attention_backend="fa3",
            chunked_prefill_size=-1,
            disable_radix_cache=True,
            prefill_only_disable_kv_cache=prefill_only_disable_kv_cache,
            enforce_piecewise_cuda_graph=True,
        ) as srt_runner:
            return srt_runner.forward(prompts).embed_logits

    def test_piecewise_embeddings_match_hf_no_nan(self):
        with HFRunner(
            _MODEL_PATH, torch_dtype=torch.float16, model_type="embedding"
        ) as hf_runner:
            hf_logits = hf_runner.forward(DEFAULT_PROMPTS).embed_logits

        # Both prefill_only_disable_kv_cache=True (the opt-in raw-K/V fast path)
        # and =False (the default paged path) must stay NaN-free under piecewise
        # CUDA graph and match HF.
        for prefill_only_disable_kv_cache in (True, False):
            label = f"prefill_only_disable_kv_cache={prefill_only_disable_kv_cache}"
            with self.subTest(label):
                srt_logits = self._srt_embeddings(
                    DEFAULT_PROMPTS, prefill_only_disable_kv_cache
                )
                for i in range(len(DEFAULT_PROMPTS)):
                    hf_vec = torch.tensor(hf_logits[i], dtype=torch.float32)
                    srt_vec = torch.tensor(srt_logits[i], dtype=torch.float32)

                    self.assertFalse(
                        torch.isnan(srt_vec).any(),
                        f"{label}, prompt {i}: SRT embedding contains NaN",
                    )

                    similarity = float(get_similarities(hf_vec, srt_vec))
                    if len(DEFAULT_PROMPTS[i]) <= 1000:
                        self.assertLess(
                            abs(similarity - 1),
                            _PREFILL_TOLERANCE,
                            f"{label}, prompt {i}: embeddings are not all close",
                        )


if __name__ == "__main__":
    unittest.main()
