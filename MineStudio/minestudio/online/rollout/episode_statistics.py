import minestudio.online.utils.train.wandb_logger as wandb_logger
import ray
import torchmetrics
from typing import Dict, Any, Optional, List
import numpy as np
from collections import deque
import logging

logger = logging.getLogger("ray")


@ray.remote
class EpisodeStatistics:
    def __init__(self, discount: float):
        self.discount = discount
        self.episode_info = {}
        # Maintain separate metrics for each task
        self.sum_rewards_metrics = {}  # torchmetrics.MeanMetric()
        self.discounted_rewards_metrics = {}  # torchmetrics.MeanMetric()
        self.episode_lengths_metrics = {}  # torchmetrics.MeanMetric()
        self.acc_episode_count = 0
        self.record_requests = deque()

    def update_training_session(self):
        wandb_logger.define_metric("episode_statistics/step")
        wandb_logger.define_metric("episode_statistics/*", step_metric="episode_statistics/step")

    def log_statistics(self, step: int, record_next_episode: bool):
        if self.acc_episode_count == 0:
            pass
        else:
            sum_train_reward = 0
            num_train_tasks = 0
            sum_test_reward = 0
            num_test_tasks = 0
            sum_discounted_reward = 0
            sum_episode_length = 0

            for task in self.sum_rewards_metrics.keys():
                mean_sum_reward = self.sum_rewards_metrics[task].compute()
                mean_discounted_reward = self.discounted_rewards_metrics[task].compute()
                mean_episode_length = self.episode_lengths_metrics[task].compute()

                self.sum_rewards_metrics[task].reset()
                self.discounted_rewards_metrics[task].reset()
                self.episode_lengths_metrics[task].reset()
                wandb_logger.log(
                    {
                        "episode_statistics/step": step,
                    }
                )

                if not np.isnan(mean_sum_reward) and "4train" in task:
                    sum_train_reward += mean_sum_reward
                    sum_discounted_reward += mean_discounted_reward
                    num_train_tasks += 1
                if not np.isnan(mean_sum_reward) and "4test" in task:
                    sum_test_reward += mean_sum_reward
                    num_test_tasks += 1
                sum_episode_length += mean_episode_length

            self.episode_info = {
                "steps": step,
                "episode_count": self.acc_episode_count,
                "mean_sum_reward": sum_train_reward / num_train_tasks if num_train_tasks > 0 else 0,
                "mean_discounted_reward": sum_discounted_reward / num_train_tasks if num_train_tasks > 0 else 0,
                "mean_episode_length": sum_episode_length / (num_train_tasks + num_test_tasks)
                if num_train_tasks + num_test_tasks > 0
                else 0,
            }
            wandb_logger.log(
                {
                    "episode_statistics/steps": step,
                    "episode_statistics/episode_count": self.acc_episode_count,
                    "episode_statistics/mean_sum_reward": sum_train_reward / num_train_tasks
                    if num_train_tasks > 0
                    else 0,
                    "episode_statistics/mean_test_sum_reward": sum_test_reward / num_test_tasks
                    if num_test_tasks > 0
                    else 0,
                    "episode_statistics/mean_discounted_reward": sum_discounted_reward / num_train_tasks
                    if num_train_tasks > 0
                    else 0,
                    "episode_statistics/mean_episode_length": sum_episode_length / (num_train_tasks + num_test_tasks)
                    if num_train_tasks + num_test_tasks > 0
                    else 0,
                }
            )

            self.acc_episode_count = 0

        print("received_episode_statistics:" + str(step) + str(record_next_episode))
        if record_next_episode:
            if len(self.record_requests) > 0:
                print("There are still unprocessed record requests.")
                logger.warning("There are still unprocessed record requests.")
            else:
                self.record_requests.append(step)
                print("append_record_requests:" + str(self.record_requests))

    def report_episode(
        self, rewards: np.ndarray, its_specfg: str = "", additional_des: str = "4train"
    ) -> Optional[int]:
        its_specfg = its_specfg + additional_des
        if its_specfg not in self.sum_rewards_metrics:
            self.sum_rewards_metrics[its_specfg] = torchmetrics.MeanMetric()
            self.discounted_rewards_metrics[its_specfg] = torchmetrics.MeanMetric()
            self.episode_lengths_metrics[its_specfg] = torchmetrics.MeanMetric()
        sum_reward = rewards.sum()

        discounted_reward = ((self.discount ** np.arange(len(rewards))) * rewards).sum()
        episode_length = len(rewards)
        self.sum_rewards_metrics[its_specfg].update(sum_reward)
        self.discounted_rewards_metrics[its_specfg].update(discounted_reward)
        self.episode_lengths_metrics[its_specfg].update(episode_length)
        self.acc_episode_count += 1
        if len(self.record_requests) > 0:
            print("episode, cord_requests>0:" + str(self.record_requests))
            step = self.record_requests.popleft()
            return step, self.episode_info
        else:
            return None, self.episode_info
