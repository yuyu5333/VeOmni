# Copyright 2025 Bytedance Ltd. and/or its affiliates
#
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

from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

from ...ops.attention import flash_attention_3_forward, flash_attention_forward


ALL_ATTENTION_FUNCTIONS.register("flash_attention_2", flash_attention_forward)
ALL_ATTENTION_FUNCTIONS.register("flash_attention_3", flash_attention_3_forward)

from . import deepseek_v3, flux, llama, qwen2, qwen2_vl, qwen3, qwen3_moe, wan, wan2_2


__all__ = ["qwen2_vl", "deepseek_v3", "qwen2", "llama", "qwen3", "qwen3_moe", "wan", "flux", "wan2_2"]
