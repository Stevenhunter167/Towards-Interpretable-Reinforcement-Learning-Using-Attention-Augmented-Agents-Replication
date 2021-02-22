# tensorboard
from torch.utils.tensorboard import SummaryWriter

from datetime import datetime


class Tensorboardlog:

    now = str(datetime.now().strftime(r"%d%m%Y_%H_%M"))
    writer = SummaryWriter(f"runs/{now}")

    @classmethod
    def tensorboardlog(cls, episode_num, average_reward):
        """
        episode_num: current episode number
        average_reward: average reward until current episode
        """
        cls.writer.add_scalar("episode_reward_mean", average_reward, episode_num)