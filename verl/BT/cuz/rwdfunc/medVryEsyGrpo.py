# Copyright 2024 Bytedance Ltd. and/or its affiliates
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

import re

ThinkL="<think>"
ThinkR="</think>"
rewardCorrect = 0.2
def format_reward(response):
    foundLs = response.split(ThinkR)
    foundLsLen = len(foundLs)
    if foundLsLen > 1:
        answ = foundLs[-1].strip()
        if answ and answ.count(ThinkL) == 0:
            return rewardCorrect
    return 0.0


def compute_score(data_sources, solution_strs, ground_truths, extra_infos, data, **reward_kwargs):
    # print("9999999999999999999999999999999"*10)
    assert 'bt_rm_scores' in data.batch.keys()
    # print(f"9999999999999999999999999999999>>{data.batch['bt_rm_scores'].shape}")
    rm_scores = data.batch['bt_rm_scores'].tolist()
    bsz = len(solution_strs)
    assert len(rm_scores) == bsz
    # print(f"9999999999999999999{str(rm_scores)}999999999999__{bsz}")
    rewards = []
    for i in range(bsz):
        # print(f"9999999999999999999{rm_scores[i]}")
        r = rm_scores[i] + format_reward(solution_strs[i])
        rewards.append(r)
    return rewards