import re
import os
import datasets

import argparse


DATA_FLAG="vqVEsy1"
SYSTEM_PROMPT = f"You are an export in the field of medicine and biology. When you answer a question, you should first think about the reasoning process in your mind and then provide the answer. The reasoning process should be included in <think> </think>, and the answer should be located after the last </think>, i.e., <think> reasoning process here </think> answer here."


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', default='/home/azureuser/cloudfiles/code/SRC/O1/reasoningimprove/domain/Med/1D5B/Demo/data/tmpds/VerifiableQAllRnd1K')
    parser.add_argument('--local_dir', default='/home/azureuser/cloudfiles/code/SRC/O1/reasoningimprove/domain/Med/1D5B/Demo/data/tmpds/VerifiableQAllRnd1K_vrl')

    args = parser.parse_args()

    data_source = args.data_dir

    train_dataset = datasets.load_from_disk(data_source) #You are trying to load a dataset that was saved using `save_to_disk`. Please use `load_from_disk` instead.
    test_dataset = train_dataset.select(range(8)) #@#verl@bt250409的版本非得要一个dummy test dataset
    # add a row to each data item that represents a unique id
    def make_map_fn(split):

        def process_fn(example, idx):
            question = example.pop('question')
            solution = example.pop('gt_answ')
            data = {
                "data_source": DATA_FLAG,
                "prompt": [
                  {"role": "system", "content": SYSTEM_PROMPT},
                  {"role": "user", "content": question,}
                ],
                "ability": "med",
                "reward_model": {
                    "style": "rule",
                    "ground_truth": solution
                },
                "extra_info": {
                    'split': split,
                    'index': idx,
                }
            }
            return data

        return process_fn

    train_dataset = train_dataset.map(function=make_map_fn('train'), with_indices=True)
    test_dataset = test_dataset.map(function=make_map_fn('test'), with_indices=True)

    local_dir = args.local_dir
    train_dataset.to_parquet(os.path.join(local_dir, 'train.parquet'))
    test_dataset.to_parquet(os.path.join(local_dir, 'test.parquet'))
