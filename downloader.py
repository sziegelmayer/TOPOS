# Copyright 2025 The TOPOS Authors
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

from huggingface_hub import hf_hub_download
import os

def get_checkpoint_path():
    ckpt_dir = os.path.join(os.path.expanduser("~"), ".topos")
    os.makedirs(ckpt_dir, exist_ok=True)

    # Download files
    ckpt_path = hf_hub_download(
        repo_id="tristanlemke/TOPOS",
        filename="checkpoint_best.pth",
        cache_dir=ckpt_dir
    )

    plans_path = hf_hub_download(
        repo_id="tristanlemke/TOPOS",
        filename="plans.json",
        cache_dir=ckpt_dir
    )

    dataset_json_path = hf_hub_download(
        repo_id="tristanlemke/TOPOS",
        filename="dataset.json",
        cache_dir=ckpt_dir
    )

    return ckpt_path, plans_path, dataset_json_path

