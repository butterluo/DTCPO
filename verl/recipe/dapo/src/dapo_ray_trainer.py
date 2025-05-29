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
"""
FSDP PPO Trainer with Ray-based single controller.
This trainer supports model-agonistic model initialization with huggingface
"""

import uuid
import time
from pprint import pprint
from copy import deepcopy
from collections import defaultdict
from tqdm import tqdm
import numpy as np
import torch
from torch.utils.data.sampler import RandomSampler,SequentialSampler,Sampler
from torchdata.stateful_dataloader import StatefulDataLoader
import math
from typing import Iterable, Iterator, Dict, List, Union
from omegaconf import OmegaConf, open_dict

from verl import DataProto
from verl.utils.dataset.rl_dataset import RLHFDataset, collate_fn
from verl.trainer.ppo.ray_trainer import RayPPOTrainer, _timer, apply_kl_penalty, compute_advantage, AdvantageEstimator, Role, WorkerType, ResourcePoolManager, RayWorkerGroup
from verl.trainer.ppo.metric_utils import (compute_data_metrics, compute_throughout_metrics, compute_timing_metrics,
                                           reduce_metrics)

class BtBatchSampler(Sampler[List[int]]):
    def __init__(
        self,
        sampler: Union[Sampler[int], Iterable[int]],
        batch_size: int,
        drop_last: bool,
        _stdZroSmpDicLs,
        _config
    ) -> None:
        # Since collections.abc.Iterable does not check for `__getitem__`, which
        # is one way for an object to be an iterable, we don't do an `isinstance`
        # check here.
        if (
            not isinstance(batch_size, int)
            or isinstance(batch_size, bool)
            or batch_size <= 0
        ):
            raise ValueError(
                f"batch_size should be a positive integer value, but got batch_size={batch_size}"
            )
        if not isinstance(drop_last, bool):
            raise ValueError(
                f"drop_last should be a boolean value, but got drop_last={drop_last}"
            )
        self.sampler = sampler
        self.batch_size = batch_size
        self.drop_last = drop_last
        self._stdZroSmpDicLs = _stdZroSmpDicLs
        self._config = _config
        self._stdZroSmpThrshld = self._config.data.get('stdZroSmpThrshld', 3) # +data.stdZroSmpThrshld=3
        self._stdZroHrdSmpEsyThrshld = self._config.data.get('stdZroHrdSmpEsyThrshld', 0.95) # +data.stdZroHrdSmpEsyThrshld=0.2+0.94
        self._stdZroHrdSmpRetrainThrshld = self._config.data.get('stdZroHrdSmpRetrainThrshld', 3) # +data.stdZroHrdSmpRetrainThrshld=3
        self._stdZroHrdSmpRetrainThrshld4Esy = self._stdZroHrdSmpRetrainThrshld * 2 #对于简单样本，赞提的epoch数是难样本的2倍
        self._stdZroHrdSmpThrshld = self._config.data.get('stdZroHrdSmpThrshld', 0.3)


    def __iter__(self) -> Iterator[List[int]]:
        batch = [0] * self.batch_size
        idx_in_batch = 0
        for idx in self.sampler:
            if idx in self._stdZroSmpDicLs.keys():
                if len(self._stdZroSmpDicLs[idx]) == 1 and isinstance(self._stdZroSmpDicLs[idx][0], dict):
                    if 'cnt' in self._stdZroSmpDicLs[idx][0].keys():
                        if self._stdZroSmpDicLs[idx][0]['cnt'] >= self._stdZroHrdSmpRetrainThrshld:
                            self._stdZroSmpDicLs[idx] = []   #让该idx对应的样本重新投入训练
                            print(f"@#***stdZroSmpDicLs[idx] = []>>>{idx}")
                        else:
                            self._stdZroSmpDicLs[idx][0]['cnt'] = self._stdZroSmpDicLs[idx][0]['cnt'] + 1  #暂停若干epoch的处理,并对跳过的epoch进行计数
                            print(f"@#***stdZroSmpDicLs[idx][0]['cnt']+=1>>>{idx}")
                            continue
                    elif 'cntH' in self._stdZroSmpDicLs[idx][0].keys():
                        if self._stdZroSmpDicLs[idx][0]['cntH'] >= self._stdZroHrdSmpRetrainThrshld:
                            self._stdZroSmpDicLs[idx] = []   #让该idx对应的样本重新投入训练
                            print(f"@#***stdZroSmpDicLs[idx] = [] HHHHH>>>{idx}")
                        else:
                            self._stdZroSmpDicLs[idx][0]['cntH'] = self._stdZroSmpDicLs[idx][0]['cntH'] + 1  #暂停若干epoch的处理,并对跳过的epoch进行计数
                            print(f"@#***stdZroSmpDicLs[idx][0]['cntH']+=1 HHHHH>>>{idx}")
                            continue
                    else: # 'cntE' in self._stdZroSmpDicLs[idx][0].keys():
                        if self._stdZroSmpDicLs[idx][0]['cntE'] >= self._stdZroHrdSmpRetrainThrshld4Esy:
                            self._stdZroSmpDicLs[idx] = []   #让该idx对应的样本重新投入训练
                            print(f"@#***stdZroSmpDicLs[idx] = [] EEEEE>>>{idx}")
                        else:
                            self._stdZroSmpDicLs[idx][0]['cntE'] = self._stdZroSmpDicLs[idx][0]['cntE'] + 1  #暂停若干epoch的处理,并对跳过的epoch进行计数
                            print(f"@#***stdZroSmpDicLs[idx][0]['cntE']+=1 EEEEEE>>>{idx}")
                            continue
                else:
                    if len(self._stdZroSmpDicLs[idx]) >= self._stdZroSmpThrshld:
                        if self._stdZroSmpDicLs[idx][-1] > self._stdZroHrdSmpEsyThrshld:
                            print(f"***IGNORE EEEEEE>>>{idx} @ {self._stdZroSmpDicLs[idx][-1]}")
                            self._stdZroSmpDicLs[idx] = [{'cntE':0}] #暂停若干epoch的处理,并对跳过的epoch进行计数  易题
                            continue
                        elif self._stdZroSmpDicLs[idx][-1] < self._stdZroHrdSmpThrshld:
                            print(f"***IGNORE HHHHHH>>>{idx} @ {self._stdZroSmpDicLs[idx][-1]}")
                            self._stdZroSmpDicLs[idx] = [{'cntH':0}] #暂停若干epoch的处理,并对跳过的epoch进行计数  难题
                            continue
                        else:
                            print(f"***IGNORE HHHHHH>>>{idx} @ {self._stdZroSmpDicLs[idx][-1]}")
                            self._stdZroSmpDicLs[idx] = [{'cnt':0}] #暂停若干epoch的处理,并对跳过的epoch进行计数
                            continue
            batch[idx_in_batch] = idx
            idx_in_batch += 1
            if idx_in_batch == self.batch_size:
                yield batch
                idx_in_batch = 0
                batch = [0] * self.batch_size
        if (not self.drop_last) and idx_in_batch > 0:
            yield batch[:idx_in_batch]

    def __len__(self) -> int:
        if self.drop_last:
            return len(self.sampler) // self.batch_size  # type: ignore[arg-type]
        else:
            return (len(self.sampler) + self.batch_size - 1) // self.batch_size  # type: ignore[arg-type]

class RayDAPOTrainer(RayPPOTrainer):
    """
    Note that this trainer runs on the driver process on a single CPU/GPU node.
    """

    def __init__(self,
                 config,
                 tokenizer,
                 role_worker_mapping: dict[Role, WorkerType],
                 resource_pool_manager: ResourcePoolManager,
                 ray_worker_group_cls: RayWorkerGroup = RayWorkerGroup,
                 processor=None,
                 reward_fn=None,
                 val_reward_fn=None):
        # assert config.data.gen_batch_size == config.data.train_batch_size, "@#避免gen_batch_size太大造成浪费样本,同时与batch = batch[:traj_bsz]的修改相呼应"
        self.stdZroSmpDicLs = defaultdict(list) 
        super().__init__(config,
                 tokenizer,
                 role_worker_mapping,
                 resource_pool_manager,
                 ray_worker_group_cls,
                 processor,
                 reward_fn,
                 val_reward_fn)

    def _create_dataloader(self):#@#ADD
        # TODO: we have to make sure the batch size is divisible by the dp size
        self.train_dataset = RLHFDataset(parquet_files=self.config.data.train_files,
                                         tokenizer=self.tokenizer,
                                         processor=self.processor,
                                         prompt_key=self.config.data.prompt_key,
                                         image_key=self.config.data.get('image_key', 'images'),
                                         max_prompt_length=self.config.data.max_prompt_length,
                                         return_raw_chat=self.config.data.get('return_raw_chat', False),
                                         truncation=self.config.data.get('truncation', 'error'),
                                         filter_overlong_prompts=self.config.data.filter_overlong_prompts,
                                         num_workers=self.config.data.get('filter_overlong_prompts_workers', None))
        assert self.train_dataset.truncation == self.config.data.get(
            'truncation', 'error'
        ), f'dataset truncation {self.train_dataset.truncation} must be the same as config {self.config.data.get("truncation", "error")}'
        # use sampler for better ckpt resume
        if self.config.data.shuffle:
            train_dataloader_generator = torch.Generator()
            train_dataloader_generator.manual_seed(self.config.data.get('seed', 1))
            sampler = RandomSampler(data_source=self.train_dataset, generator=train_dataloader_generator)
        else:
            assert self.config.data.shuffle
            sampler = SequentialSampler(data_source=self.train_dataset)
        batch_size=self.config.data.get('gen_batch_size', self.config.data.train_batch_size)
        Bt_batch_sampler = BtBatchSampler(sampler, batch_size, drop_last=True, _stdZroSmpDicLs=self.stdZroSmpDicLs, _config=self.config)
        self.train_dataloader = StatefulDataLoader(dataset=self.train_dataset,
                                                   batch_sampler=Bt_batch_sampler, #@#训练循环
                                                   num_workers=8,
                                                   collate_fn=collate_fn,
                                                  )

        self.val_dataset = RLHFDataset(parquet_files=self.config.data.val_files,
                                       tokenizer=self.tokenizer,
                                       processor=self.processor,
                                       prompt_key=self.config.data.prompt_key,
                                       image_key=self.config.data.get('image_key', 'images'),
                                       max_prompt_length=self.config.data.max_prompt_length,
                                       return_raw_chat=self.config.data.get('return_raw_chat', False),
                                       truncation=self.config.data.get('truncation', 'error'),
                                       filter_overlong_prompts=self.config.data.filter_overlong_prompts,
                                       num_workers=self.config.data.get('filter_overlong_prompts_workers', None))
        assert self.val_dataset.truncation == self.config.data.get(
            'truncation', 'error'
        ), f'dataset truncation {self.val_dataset.truncation} must be the same as config {self.config.data.get("truncation", "error")}'
        self.val_dataloader = StatefulDataLoader(
            dataset=self.val_dataset,
            # Validation datasets are sent to inference engines as a whole batch,
            # which will schedule the memory themselves.
            batch_size=len(self.val_dataset),
            num_workers=8,
            shuffle=False,
            drop_last=False,
            collate_fn=collate_fn)

        assert len(self.train_dataloader) >= 1
        assert len(
            self.val_dataloader
        ) == 1, "Validation dataloader must have a single batch, which inference engines will schedule the memory themselves."

        print(f'Size of train dataloader: {len(self.train_dataloader)}')

        # inject total_training_steps to actor/critic optim_config. This is hacky.  #@#训练循环 可见训练总步数是由配置中的train_batch_size和要训多少total_epochs决定的
        total_training_steps = len(self.train_dataloader) * self.config.trainer.total_epochs

        if self.config.trainer.total_training_steps is not None:
            total_training_steps = self.config.trainer.total_training_steps

        self.total_training_steps = total_training_steps
        print(f'Total training steps: {self.total_training_steps}')

        OmegaConf.set_struct(self.config, True)
        with open_dict(self.config):
            self.config.actor_rollout_ref.actor.optim.total_training_steps = total_training_steps
            self.config.critic.optim.total_training_steps = total_training_steps


    def fit(self):
        """
        The training loop of PPO.
        The driver process only need to call the compute functions of the worker group through RPC to construct the PPO dataflow.
        The light-weight advantage computation is done on the driver process.
        """
        from verl.utils.tracking import Tracking
        from omegaconf import OmegaConf

        logger = Tracking(project_name=self.config.trainer.project_name,
                          experiment_name=self.config.trainer.experiment_name,
                          default_backend=self.config.trainer.logger,
                          config=OmegaConf.to_container(self.config, resolve=True))

        self.global_steps = 0

        # load checkpoint before doing anything @#这里会恢复 global_steps, total_training_steps 等状态
        self._load_checkpoint()

        # perform validation before training
        # currently, we only support validation using the reward_function.
        if self.val_reward_fn is not None and self.config.trainer.get('val_before_train', True):
            val_metrics = self._validate()
            pprint(f'Initial validation metrics: {val_metrics}')
            logger.log(data=val_metrics, step=self.global_steps)
            if self.config.trainer.get('val_only', False):
                return

        # add tqdm
        progress_bar = tqdm(total=self.total_training_steps, initial=self.global_steps, desc="Training Progress")

        # we start from step 1
        self.global_steps += 1
        last_val_metrics = None

        timing_raw = defaultdict(float)
        batch = None
        num_prompt_in_batch = 0
        num_gen_batches = 0
        assert self.config.algorithm.filter_groups.enable, "@# 必须 filter_groups.enable"
        idxset = set()#@#ADD START
        seedChangeEpoch = self.config.actor_rollout_ref.rollout.get('seedChangeEpoch', False) # +actor_rollout_ref.rollout.seedChangeEpoch=True
        seedChangeEpochStepTrshld = int(self.config.actor_rollout_ref.rollout.get('seedChangeEpochStepTrshldFactor', 0.8) * self.total_training_steps) # +actor_rollout_ref.rollout.seedChangeEpochStepTrshldFactor=0.1
        stdZroHrdSmpEsyThrshld = self.config.data.get('stdZroHrdSmpEsyThrshld', 0.95)
        stdZroSmpReszFactr = self.config.data.get('stdZroSmpReszFactr', 0.6) #+data.stdZroSmpReszFactr=0.6
        stdZroHrdSmpThrshld = self.config.data.get('stdZroHrdSmpThrshld', 0.6) #+data.stdZroHrdSmpThrshld=0.6
        stdZroSmpIncrTmperTrshld = self.config.actor_rollout_ref.rollout.get('stdZroSmpIncrTmperTrshld', -1.0) # 0.3
        stdZroSmpDecrTmperTrshld = self.config.actor_rollout_ref.rollout.get('stdZroSmpDecrTmperTrshld', 0.3) # 0.3
        stdZroSmpIncrTmper = self.config.actor_rollout_ref.rollout.get('stdZroSmpIncrTmper', 0.1) # 0.2
        stdZroSmpIncrTmperMx = self.config.actor_rollout_ref.rollout.get('stdZroSmpIncrTmperMx', 2.0) #
        lastEpochStep = self.total_training_steps - (self.total_training_steps // self.config.trainer.total_epochs) - 1
        datasetLen = len(self.train_dataloader.dataset)
        rolloutTmpr = -1 #self.config.actor_rollout_ref.rollout.get("temperature", 1.0)
        epochseed = -1
        smpParamKwargs = {}
        dorandom = False
        epoch = 0
        #@#ADD END
        # for epoch in range(self.config.trainer.total_epochs):  #@#MODIFIED 因为train_dataloader有时候因为数据std=0的缘故要取多次数据(跳过多个step)才凑够一个step，导致所有数据取完了也比原定的一个epoch包含的step数要少，所以按epoch循环的话可能无法跑完原定的total_training_steps数目
        while self.global_steps <= self.total_training_steps:
            if self.global_steps > lastEpochStep:   #@#ADD START
                print(f"@#** RESET ALL AT LAST**")
                rolloutTmpr = self.config.actor_rollout_ref.rollout.get("temperature", 1.0)
                dorandom = False
                for k in self.stdZroSmpDicLs.keys():
                    self.stdZroSmpDicLs[k] = []
            print(f"@#TMP ----> epoch:{epoch}, {lastEpochStep=}, {dorandom=}, {rolloutTmpr=}")
            if seedChangeEpoch and self.global_steps > seedChangeEpochStepTrshld:
                epochseed = int(time.time()) #@#TODO 没啥用
                print(f"@# epcohseed:{epochseed}")
                smpParamKwargs['seed']=epochseed
            if rolloutTmpr > 0:
                smpParamKwargs['temperature']=rolloutTmpr
            if len(smpParamKwargs.keys()) > 0:
                print(f"@#***setSampleParam: {smpParamKwargs} ")
                self.actor_rollout_wg.setSampleParam(**smpParamKwargs)
            #@#ADD END
            for batch_dict in self.train_dataloader:
                metrics = {}

                new_batch: DataProto = DataProto.from_single_dict(batch_dict)
                num_gen_batches += 1
                toprintidxls = []
                for xinf, idx in zip(new_batch.non_tensor_batch['extra_info'], new_batch.non_tensor_batch['index']):
                    assert xinf['index'] == idx
                    toprintidxls.append(idx)
                    if idx not in idxset:
                        idxset.add(idx)
                print(f"**IDX LS**{toprintidxls}")
                # pop those keys for generation
                if 'multi_modal_inputs' in new_batch.non_tensor_batch.keys():
                    gen_batch = new_batch.pop(
                        batch_keys=['input_ids', 'attention_mask', 'position_ids'],
                        non_tensor_batch_keys=['raw_prompt_ids', 'multi_modal_data', 'multi_modal_inputs'],
                    )
                else:
                    gen_batch = new_batch.pop(
                        batch_keys=['input_ids', 'attention_mask', 'position_ids'],
                        non_tensor_batch_keys=['raw_prompt_ids'],
                    )

                is_last_step = self.global_steps >= self.total_training_steps

                with _timer('step', timing_raw):
                    # generate a batch
                    with _timer('gen', timing_raw):
                        gen_batch_output = self.actor_rollout_wg.generate_sequences(gen_batch)

                    if self.config.algorithm.adv_estimator == AdvantageEstimator.REMAX:
                        with _timer('gen_max', timing_raw):
                            gen_baseline_batch = deepcopy(gen_batch)
                            gen_baseline_batch.meta_info['do_sample'] = False
                            gen_baseline_output = self.actor_rollout_wg.generate_sequences(gen_baseline_batch)

                            new_batch = new_batch.union(gen_baseline_output)
                            reward_baseline_tensor = self.reward_fn(new_batch)
                            reward_baseline_tensor = reward_baseline_tensor.sum(dim=-1)

                            new_batch.pop(batch_keys=list(gen_baseline_output.batch.keys()))

                            new_batch.batch['reward_baselines'] = reward_baseline_tensor

                            del gen_baseline_batch, gen_baseline_output

                    new_batch.non_tensor_batch['uid'] = np.array(
                        [str(uuid.uuid4()) for _ in range(len(new_batch.batch))], dtype=object)
                    # repeat to align with repeated responses in rollout
                    new_batch = new_batch.repeat(repeat_times=self.config.actor_rollout_ref.rollout.n, interleave=True)
                    new_batch = new_batch.union(gen_batch_output)

                    with _timer('reward', timing_raw):
                        # compute scores. Support both model and function-based.
                        # We first compute the scores using reward model. Then, we call reward_fn to combine
                        # the results from reward model and rule-based results.
                        if self.use_rm:
                            # we first compute reward model score
                            reward_tensor = self.rm_wg.compute_rm_score(new_batch)
                            new_batch = new_batch.union(reward_tensor)

                        # we combine with rule-based rm
                        reward_extra_infos_dict: dict[str, list]
                        try:
                            reward_result = self.reward_fn(new_batch, return_dict=True, dorandom=dorandom)
                            reward_tensor = reward_result['reward_tensor']
                            reward_extra_infos_dict = reward_result['reward_extra_info']
                        except Exception as e:
                            print(f'Error in reward_fn: {e}')
                            reward_tensor = self.reward_fn(new_batch)
                            reward_extra_infos_dict = {}

                        new_batch.batch['token_level_scores'] = reward_tensor

                        print(f'{list(reward_extra_infos_dict.keys())=}')
                        if reward_extra_infos_dict:
                            new_batch.non_tensor_batch.update({
                                k: np.array(v) for k, v in reward_extra_infos_dict.items()
                            })

                        # compute rewards. apply_kl_penalty if available
                        if self.config.algorithm.use_kl_in_reward:
                            new_batch, kl_metrics = apply_kl_penalty(new_batch,
                                                                     kl_ctrl=self.kl_ctrl_in_reward,
                                                                     kl_penalty=self.config.algorithm.kl_penalty)
                            metrics.update(
                                kl_metrics)  # TODO: This will be cleared if we use multiple genenration batches
                        else:
                            new_batch.batch['token_level_rewards'] = new_batch.batch['token_level_scores']

                    if not self.config.algorithm.filter_groups.enable:
                        batch = new_batch
                    else:  # NOTE: When prompts after filtering is less than train batch size, we skip to the next generation batch
                        metric_name = self.config.algorithm.filter_groups.metric
                        if metric_name == "seq_final_reward":
                            # Turn to numpy for easier filtering
                            new_batch.non_tensor_batch["seq_final_reward"] = new_batch.batch['token_level_rewards'].sum(
                                dim=-1).numpy() #@#MODIFIED change 'token_level_scores' to 'token_level_rewards' from main@250417
                        elif metric_name == "seq_reward":
                            new_batch.non_tensor_batch["seq_reward"] = new_batch.batch['token_level_scores'].sum(
                                dim=-1).numpy()

                        # Collect the sequence reward for each trajectory
                        prompt_uid2metric_vals = defaultdict(list)
                        for uid, xinf, metric_val in zip(new_batch.non_tensor_batch['uid'],
                                                   new_batch.non_tensor_batch['extra_info'],
                                                   new_batch.non_tensor_batch[metric_name]): #@#MODIFIED
                            idx = xinf['index']
                            prompt_uid2metric_vals[(uid, idx)].append(metric_val)

                        prompt_uid2metric_std = {}
                        for tpl, metric_vals in prompt_uid2metric_vals.items():#@#MODIFIED
                            prompt_uid2metric_std[tpl] = np.std(metric_vals)

                        kept_prompt_uids = [#@#MODIFIED
                            tpl[0] for tpl, std in prompt_uid2metric_std.items()
                            if std > 0 or len(prompt_uid2metric_vals[tpl]) == 1
                        ] #@#??? 这里一个原始prmp生成的多个rollout的rwd标准差是不是大于一个极小值就好了? '> 0'会不会导致几乎相似的rwd也纳入了计算?
                        num_prompt_in_batch += len(kept_prompt_uids)

                        kept_traj_idxs = []
                        for idx, traj_from_prompt_uid in enumerate(new_batch.non_tensor_batch['uid']):
                            if traj_from_prompt_uid in kept_prompt_uids:
                                kept_traj_idxs.append(idx)
                        #@#上面就是找到生成的rollout的rwd都相同的原始prmp,在下面把这些prmp过滤掉,若过滤后剩下的prmp不足train_batch_size则continue让train_dataloader再拿一批gen_batch_size的数据
                        new_batch = new_batch[kept_traj_idxs]
                        if batch is None:
                            batch = new_batch
                        else:
                            batch = DataProto.concat([batch, new_batch])

                        prompt_bsz = self.config.data.train_batch_size
                        if num_prompt_in_batch < prompt_bsz:
                            print(f'{num_prompt_in_batch=} < {prompt_bsz=}')
                            max_num_gen_batches = self.config.algorithm.filter_groups.max_num_gen_batches
                            toPrint=[]#@#ADD START
                            for tpl, metric_vals in prompt_uid2metric_vals.items():
                                prompt_uid = tpl[0]
                                idx = tpl[1]
                                if prompt_uid not in kept_prompt_uids:
                                    newval = metric_vals[0]
                                    toPrint.append({idx:metric_vals})
                                    if len(self.stdZroSmpDicLs[idx]) > 0:
                                        oldval = self.stdZroSmpDicLs[idx][-1]
                                        if math.isclose(oldval, newval, abs_tol=0.0001):
                                            self.stdZroSmpDicLs[idx].append(newval)
                                        else:
                                            self.stdZroSmpDicLs[idx] = []
                                    else:
                                        self.stdZroSmpDicLs[idx].append(newval)
                            print(f"@#***Std=0 metric>>{toPrint}")#@#ADD END
                            if max_num_gen_batches <= 0 or num_gen_batches < max_num_gen_batches:
                                print(f'{num_gen_batches=}. Keep generating...')
                                continue #@#BUG 如果data.gen_batch_size比data.train_batch_size大很多,会占用大量显存.如果再由于触发了这里的'Keep generating'对于小服务器则会爆显存
                            else:
                                raise ValueError(
                                    f'{num_gen_batches=} >= {max_num_gen_batches=}. Generated too many. Please check your data.'
                                )
                        else:
                            # Align the batch
                            traj_bsz = self.config.data.train_batch_size * self.config.actor_rollout_ref.rollout.n
                            # batch = batch[:traj_bsz] #@#BUG 这里会导致一些一些样本没纳入训练,比如第一次没找到足够合格样本训练,第二次又找到大量符合训练的样本,这样一截断,第二次找到的合格样本就会有一批在这次epoch中没被训练
                            # 正如上一行所描述，所以这里不再以traj_bsz做截断,但在训练循环以外要保证gen_batch_size=train_batch_size,以免为积累合格样本时超过traj_bsz太多,而由于后面都是基于batch_size_per_gpu做计算的,所以应该不会OOM

                    # balance the number of valid tokens on each dp rank.
                    # Note that this breaks the order of data inside the batch.
                    # Please take care when you implement group based adv computation such as GRPO and rloo
                    if self.config.trainer.balance_batch:
                        self._balance_batch(batch, metrics=metrics)

                    # compute global_valid tokens
                    batch.meta_info['global_token_num'] = torch.sum(batch.batch['attention_mask'], dim=-1).tolist()

                    # recompute old_log_probs
                    with _timer('old_log_prob', timing_raw):
                        old_log_prob = self.actor_rollout_wg.compute_log_prob(batch)
                        batch = batch.union(old_log_prob)

                    if self.use_reference_policy:
                        # compute reference log_prob
                        with _timer('ref', timing_raw):
                            ref_log_prob = self.ref_policy_wg.compute_ref_log_prob(batch)
                            batch = batch.union(ref_log_prob)

                    # compute values
                    if self.use_critic:
                        with _timer('values', timing_raw):
                            values = self.critic_wg.compute_values(batch)
                            batch = batch.union(values)

                    with _timer('adv', timing_raw):
                        # compute advantages, executed on the driver process
                        batch = compute_advantage(batch,
                                                  adv_estimator=self.config.algorithm.adv_estimator,
                                                  gamma=self.config.algorithm.gamma,
                                                  lam=self.config.algorithm.lam,
                                                  num_repeat=self.config.actor_rollout_ref.rollout.n)

                    # update critic
                    if self.use_critic:
                        with _timer('update_critic', timing_raw):
                            critic_output = self.critic_wg.update_critic(batch)
                        critic_output_metrics = reduce_metrics(critic_output.meta_info['metrics'])
                        metrics.update(critic_output_metrics)

                    # implement critic warmup
                    if self.config.trainer.critic_warmup <= self.global_steps:
                        # update actor
                        with _timer('update_actor', timing_raw):
                            actor_output = self.actor_rollout_wg.update_actor(batch)
                        actor_output_metrics = reduce_metrics(actor_output.meta_info['metrics'])
                        metrics.update(actor_output_metrics)

                    # validate
                    if self.val_reward_fn is not None and self.config.trainer.test_freq > 0 and \
                            (is_last_step or self.global_steps % self.config.trainer.test_freq == 0):
                        with _timer('testing', timing_raw):
                            val_metrics: dict = self._validate()
                            if is_last_step:
                                last_val_metrics = val_metrics
                        metrics.update(val_metrics)

                    if self.config.trainer.save_freq > 0 and (is_last_step or
                                                              self.global_steps % self.config.trainer.save_freq == 0):
                        with _timer('save_checkpoint', timing_raw):
                            self._save_checkpoint()

                # collect metrics
                metrics.update(compute_data_metrics(batch=batch, use_critic=self.use_critic))
                metrics.update(compute_timing_metrics(batch=batch, timing_raw=timing_raw))
                # TODO: implement actual tflpo and theoretical tflpo
                n_gpus = self.resource_pool_manager.get_n_gpus()
                metrics.update(compute_throughout_metrics(batch=batch, timing_raw=timing_raw, n_gpus=n_gpus))
                timing_raw = defaultdict(float)  # clear timing

                metrics["train/num_gen_batches"] = num_gen_batches
                batch = None
                num_prompt_in_batch = 0
                num_gen_batches = 0

                # TODO: make a canonical logger that supports various backend
                logger.log(data=metrics, step=self.global_steps)

                if is_last_step:
                    pprint(f'Final validation metrics: {last_val_metrics}')
                    progress_bar.close()
                    #@#ADD START #@#TODO 要在0123上测试
                    stdZroSmpLen = len(self.stdZroSmpDicLs.keys())
                    esyStdZroSmpLs, midStdZroSmpLs, hrdStdZroSmpLs = self.getEsyMidHrdLsFrmStdZroDicLs(stdZroHrdSmpEsyThrshld)
                    esyStdZroSmpLen = len(esyStdZroSmpLs)
                    midStdZroSmpLen = len(midStdZroSmpLs)
                    hrdStdZroSmpLen = len(hrdStdZroSmpLs)
                    print(f"@#**IDX SET LEN**{len(idxset)} --》stdZroSmpDicLs LEN:{stdZroSmpLen}, {esyStdZroSmpLen=}, {midStdZroSmpLen=} , {hrdStdZroSmpLen=}")
                    self._save_checkpoint() 
                    #@#ADD END
                    return

                progress_bar.update(1)
                self.global_steps += 1
                new_batch = None  #@#ADD
            #@#ADD  START
            if epoch >= self.config.trainer.get("saveEpoch", 10000):
                self._save_checkpoint()
            stdZroSmpLen = len(self.stdZroSmpDicLs.keys())
            esyStdZroSmpLs, midStdZroSmpLs, hrdStdZroSmpLs = self.getEsyMidHrdLsFrmStdZroDicLs(stdZroHrdSmpEsyThrshld)
            esyStdZroSmpLen = len(esyStdZroSmpLs)
            midStdZroSmpLen = len(midStdZroSmpLs)
            hrdStdZroSmpLen = len(hrdStdZroSmpLs)
            print(f"@#**IDX SET LEN**{len(idxset)} --》stdZroSmpDicLs LEN:{stdZroSmpLen}, {esyStdZroSmpLen=}, {midStdZroSmpLen=} , {hrdStdZroSmpLen=}")
            if len(idxset) == datasetLen:
                print(f"@#**RESET IDX SET**")
                idxset = set()
            if stdZroSmpIncrTmperTrshld > 0 :
                if (hrdStdZroSmpLen + midStdZroSmpLen) > (datasetLen * stdZroSmpIncrTmperTrshld):
                    if rolloutTmpr <= 0:
                        rolloutTmpr = self.config.actor_rollout_ref.rollout.get("temperature", 1.0)
                    rolloutTmpr = min(rolloutTmpr + stdZroSmpIncrTmper, stdZroSmpIncrTmperMx)
                    print(f"@#*** Increase Tmpr: {rolloutTmpr=}")
                elif esyStdZroSmpLen > (datasetLen * stdZroSmpDecrTmperTrshld): #@#TODO
                    rolloutTmpr = max(rolloutTmpr - 0.05, self.config.actor_rollout_ref.rollout.get("temperature", 1.0)) #@#TODO
                    print(f"@#*** DEcrease Tmpr: {rolloutTmpr=}")
                # else:
                #     rolloutTmpr = self.config.actor_rollout_ref.rollout.get("temperature", 1.0)
                #     print(f"@#*** Reset Tmpr: {rolloutTmpr=}")
            if stdZroSmpLen > (datasetLen * stdZroSmpReszFactr):
                print(f"@#***Befor ReSz stdZroSmpDicLs >> {self.stdZroSmpDicLs}")
                if stdZroSmpLen > (datasetLen * stdZroSmpReszFactr * 1.1): #@#TODO dorandom有风险
                    dorandom = True
                    for k in hrdStdZroSmpLs:
                        self.stdZroSmpDicLs.pop(k)
                for k in midStdZroSmpLs:
                    self.stdZroSmpDicLs.pop(k)
                print(f"@#***AFter ReSz stdZroSmpDicLs LEN:{len(self.stdZroSmpDicLs.keys())} >> {self.stdZroSmpDicLs}")
            else:
                dorandom = False
                print(f"@#***self.stdZroSmpDicLs** >> {self.stdZroSmpDicLs}")
            epoch = epoch + 1
            #@#ADD  END
    
    def getEsyMidHrdLsFrmStdZroDicLs(self, stdZroHrdSmpEsyThrshld):
        esyStdZroSmpLs = []
        for kk in self.stdZroSmpDicLs.keys(): 
            if len(self.stdZroSmpDicLs[kk]) == 1 and isinstance(self.stdZroSmpDicLs[kk][0], dict) and 'cntE' in self._stdZroSmpDicLs[kk][0].keys():
                esyStdZroSmpLs.append(kk)
            elif len(self.stdZroSmpDicLs[kk]) > 0 and not isinstance(self.stdZroSmpDicLs[kk][0], dict) and self.stdZroSmpDicLs[kk][-1] > stdZroHrdSmpEsyThrshld:
                esyStdZroSmpLs.append(kk)

        hrdStdZroSmpLs = []
        for k in self.stdZroSmpDicLs.keys(): 
            if len(self.stdZroSmpDicLs[k]) == 1 and isinstance(self.stdZroSmpDicLs[k][0], dict) and 'cntH' in self._stdZroSmpDicLs[k][0].keys():
                hrdStdZroSmpLs.append(k)
        
        midStdZroSmpLs = []
        for ky in self.stdZroSmpDicLs.keys(): 
            if ky not in esyStdZroSmpLs or ky not in hrdStdZroSmpLs:
                midStdZroSmpLs.append(ky)
        return esyStdZroSmpLs, midStdZroSmpLs, hrdStdZroSmpLs
            