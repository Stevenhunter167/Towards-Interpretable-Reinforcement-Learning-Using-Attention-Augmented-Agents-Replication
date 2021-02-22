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


# video recorder helper class
from datetime import datetime
# pip install opencv-python
# pip install PIL
import numpy as np
from PIL import Image
import cv2


class VideoRecord:

    def __init__(self, path, name, fps=5):
        """
        param:
            path: video save path (str)
            name: video name (str)
            fps: frames per second (int) (default=5)
        example usage:
            rec = VideoRecord('path/to/', 'filename', 10)
        """
        self.path = path
        self.name = name
        self.fps = fps
        self.frames = []
    def record_frame(self, env_frame):
        """
            records video frame in this object
        param:
            env_frame: a frame from thor environment or any RGB np array
        example usage:
            env = gym.make("someenv")
            lastframe, _, _, _ = env.step()
            rec.record_frame(lastframe)
        """
        curr_image = Image.fromarray((env_frame))
        img = cv2.cvtColor(np.asarray(curr_image), cv2.COLOR_RGB2BGR)
        self.frames.append(img)
    def savemp4(self):
        """
            writes video to file at specified location, finalize video file
        example usage:
            rec.savemp4()
        """
        if len(self.frames) == 0:
            raise Exception("Can't write video file with no frames recorded")
        height, width, layers = self.frames[0].shape
        size = (width,height)
        out = cv2.VideoWriter(f"{self.path}{self.name}.mp4", 0x7634706d, self.fps, size)
        for i in range(len(self.frames)):
            out.write(self.frames[i])
        out.release()