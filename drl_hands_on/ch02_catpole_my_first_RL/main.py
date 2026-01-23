from dataclasses import dataclass
import torch
from torch import nn
from torch import optim
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import gymnasium as gym
from typing import Generator

@dataclass
class Hp:
    """
    超参数
    """
    # 神经网络隐藏层的神经元数量（维度）
    HIDDEN_SIZE = 128
    # 每次迭代训练时送入网络的样本数量
    BATCH_SIZE = 16
    # 用于精英样本筛选的百分位阈值（0-100）
    PERCENTILE = 70

class Net(nn.Module):
    """
    根据环境观测值采取相应动作的神经网络
    """
    def __init__(self, obs_size: int, hidden_size: int, n_actions: int):
        """
        :param obs_size: 环境观察值的个数
        :param hidden_size: 隐藏层的大小
        :param n_actions: 动作个数
        """
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, n_actions)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

@dataclass
class EpisodeStep:
    """
    episode中简单的一步
    """
    # 环境的观测值
    observation: np.ndarray
    # 这一步采取的动作
    action: int

@dataclass
class Episode:
    """
    一个episode
    """
    # 这个episode的总奖励值
    reward: float
    # steps: 这个episode的所有步骤
    steps: list[EpisodeStep]

def filter_batch(
        batch: list[Episode],
        percentile: float
) -> tuple[torch.Tensor, torch.Tensor, float, float]:
    """筛选精英样本并转换为张量。"""

    # 1. 统计计算
    rewards = [e.reward for e in batch]
    reward_bound = float(np.percentile(rewards, percentile))
    reward_mean = float(np.mean(rewards))

    # 2. 样本筛选 (利用列表推导式进一步精简)
    # 只保留奖励不低于阈值的精英剧集
    elite_episodes = [e for e in batch if e.reward >= reward_bound]

    # 提取观察值与动作 (Flatten 展平操作)
    train_obs = [s.observation for e in elite_episodes for s in e.steps]
    train_act = [s.action for e in elite_episodes for s in e.steps]

    # 3. 数据转换 (使用更现代的 as_tensor 提升性能)
    obs_v = torch.as_tensor(np.vstack(train_obs), dtype=torch.float32)
    act_v = torch.as_tensor(train_act, dtype=torch.long)

    return obs_v, act_v, reward_bound, reward_mean


class EpisodeCollector:
    """回合收集器，用于环境交互与数据封装"""

    def __init__(self, env: gym.Env, net: Net, batch_size: int):
        self.env = env
        self.net = net
        self.batch_size = batch_size

        self.obs, _ = env.reset()
        self.softmax = nn.Softmax(dim=1)

        # 内部缓存
        self._episode_reward = 0.0
        self._episode_steps: list[EpisodeStep] = []

    def get_batches(self) -> Generator[list[Episode], None, None]:
        batch: list[Episode] = []

        while True:
            action = self._get_action()
            next_obs, reward, is_done, is_trunc, _ = self.env.step(action)

            # 1. 累加当前回合数据
            self._episode_reward += float(reward)
            self._episode_steps.append(EpisodeStep(observation=self.obs, action=action))
            self.obs = next_obs

            # 2. 检查回合是否结束，如果结束则返回 Episode 对象
            episode = self._check_and_reset_episode(is_done, is_trunc)

            if episode is not None:
                batch.append(episode)
                if len(batch) == self.batch_size:
                    yield batch
                    batch = []

    def _get_action(self) -> int:
        """保持你的逻辑，增加了 no_grad 优化内存"""
        obs_v = torch.as_tensor(self.obs, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            act_probs_v = self.softmax(self.net(obs_v))

        act_probs = act_probs_v.cpu().numpy()[0]
        return int(np.random.choice(len(act_probs), p=act_probs))

    def _check_and_reset_episode(self, is_done: bool, is_trunc: bool) -> Episode | None:
        """
        检查回合状态。如果结束，重置环境并封装 Episode；否则返回 None。
        """
        if not (is_done or is_trunc):
            return None

        # 封装当前回合
        finished_episode = Episode(reward=self._episode_reward, steps=self._episode_steps)

        # 重置内部计数器
        self._episode_reward = 0.0
        self._episode_steps = []
        self.obs, _ = self.env.reset()

        return finished_episode

if __name__ == '__main__':
    env = gym.make("CartPole-v1", render_mode="rgb_array")
    env = gym.wrappers.RecordVideo(env, video_folder="video")

    assert env.observation_space.shape is not None
    obs_size = env.observation_space.shape[0]
    assert isinstance(env.action_space, gym.spaces.Discrete)
    n_actions = env.action_space.n

    net = Net(obs_size, Hp.HIDDEN_SIZE, n_actions)
    objective = nn.CrossEntropyLoss()
    optimizer = optim.Adam(params=net.parameters(), lr=0.01)

    episode_collector = EpisodeCollector(env, net, Hp.BATCH_SIZE)

    writer = SummaryWriter(comment="-cartpole")
    for iter_no, batch in enumerate(episode_collector.get_batches()):
        obs_v, acts_v, reward_b, reward_m = filter_batch(batch, Hp.PERCENTILE)
        optimizer.zero_grad()
        action_scores_v = net(obs_v)
        loss_v = objective(action_scores_v, acts_v)
        assert isinstance(loss_v, torch.Tensor)
        loss_v.backward()
        optimizer.step()
        print("%d: loss=%.3f, reward_mean=%.1f, rw_bound=%.1f" % (
            iter_no, loss_v.item(), reward_m, reward_b))
        writer.add_scalar("loss", loss_v.item(), iter_no)
        writer.add_scalar("reward_bound", reward_b, iter_no)
        writer.add_scalar("reward_mean", reward_m, iter_no)
        if reward_m >= 500:
             print("Solved!")
             # 1. 定义保存路径
             model_save_path = "solved_cartpole.pth"

             # 2. 保存模型的状态字典 (State Dict)
             # 这是工程界推荐的做法，只保存参数而不保存整个类结构
             torch.save(net.state_dict(), model_save_path)

             print(f"模型已成功保存至: {model_save_path}")
             break
    writer.close()

