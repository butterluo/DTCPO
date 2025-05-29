import random

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


def compute_score(data_source=None,
                solution_str=None,
                ground_truth=None,
                extra_info=None,
                bt_rm_score=None,
                dorandom = False):
    if dorandom:
        rdn = random.uniform(-0.001, 0.002)
        if rdn >= 0.001:
            rdn = 0.0
    else:
        rdn = 0.0
    r = float(bt_rm_score + rdn) + format_reward(solution_str)
    return {
        "score": r,
    }