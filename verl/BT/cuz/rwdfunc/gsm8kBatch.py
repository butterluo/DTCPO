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


def extract_solution(solution_str, method='strict'):
    assert method in ['strict', 'flexible']

    if method == 'strict':
        # this also tests the formatting of the model
        solution = re.search("#### (\\-?[0-9\\.\\,]+)", solution_str)
        if solution is None:
            final_answer = None
        else:
            final_answer = solution.group(0)
            final_answer = final_answer.split('#### ')[1].replace(',', '').replace('$', '')
    elif method == 'flexible':
        answer = re.findall("(\\-?[0-9\\.\\,]+)", solution_str)
        final_answer = None
        if len(answer) == 0:
            # no reward is there is no answer
            pass
        else:
            invalid_str = ['', '.']
            # find the last number that is not '.'
            for final_answer in reversed(answer):
                if final_answer not in invalid_str:
                    break
    return final_answer


def compute_score(data_sources, solution_strs, ground_truths, extra_infos, data, **reward_kwargs):
    """#@#这是为了测试 batch verifier功能，原来的代码请看gsm8k_origin.py
    """
    # print("9999999999999999999999999999999"*20)
    if 'rm_scores' in data.batch.keys():
        rm_scores = data.batch['rm_scores']
        assert rm_scores.size(0) == len(solution_strs)
    else:
        rm_scores = None
    # print(f"9999999999999999999{str(rm_scores)}999999999999__{len(solution_strs)}")
    rewards = []
    for solution_str, ground_truth in zip(solution_strs, ground_truths):
        answer = extract_solution(solution_str=solution_str, method='strict')
        if answer is None:
            rewards.append(0.0)
        else:
            if answer == ground_truth:
                rewards.append(1.0)
            else:
                rewards.append(1.0)
    return rewards