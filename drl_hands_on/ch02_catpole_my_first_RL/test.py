import torch
import gymnasium as gym
import time
from main import Net

def test_infinite_balance(model_path: str = "solved_cartpole.pth"):
    """测试模型在无限制环境下的极限坚持时间。"""

    # 1. 创建环境并解包，彻底移除 500 步限制
    base_env = gym.make("CartPole-v1", render_mode="human")
    env = base_env.unwrapped  # 关键点：unwrapped 移除了 TimeLimit 限制

    # 2. 重新加载模型架构与权重
    # 注意：这里的参数需与训练时保持一致
    obs_size = env.observation_space.shape[0]
    n_actions = env.action_space.n

    # 假设你在训练后保存了模型，这里进行加载
    # 如果你当前内存中已经有 net 对象，可以直接使用
    net = Net(obs_size, 128, n_actions)
    try:
        net.load_state_dict(torch.load(model_path))
        print(f"成功加载模型: {model_path}")
    except FileNotFoundError:
        print("未找到保存的模型文件，将使用当前内存中的模型。")

    net.eval()  # 切换到评估模式（关闭 Dropout 等）
    softmax = torch.nn.Softmax(dim=1)

    # 3. 开始极限挑战
    obs, _ = env.reset()
    total_reward = 0
    steps = 0

    print("开始极限挑战... 按下 Ctrl+C 退出")

    try:
        while True:
            # 推理不计算梯度
            with torch.no_grad():
                obs_v = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)
                probs_v = softmax(net(obs_v))
                # 在测试时，通常选择概率最大的动作（确定性策略）
                # 或者依然采样（随机策略），取决于你想看稳定性还是极限性能
                action = torch.argmax(probs_v).item()

            obs, reward, terminated, truncated, _ = env.step(action)
            total_reward += reward
            steps += 1

            # 在 unwrapped 环境下，truncated 理论上永远为 False
            # 只有当杆子倒下（角度超限）或小车出界，terminated 才会为 True
            if terminated:
                print(f"挑战结束！倒立摆最终坚持了 {steps} 步。")
                break

            # 控制一下渲染速度，否则 CPU 会跑得飞快
            time.sleep(0.01)

    except KeyboardInterrupt:
        print("\n手动停止测试。")
    finally:
        env.close()


if __name__ == "__main__":
    # 在主训练循环结束后调用
    test_infinite_balance()