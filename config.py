import os

# 数据集路径
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
JOB_DATASET_PATH = os.path.join(BASE_DIR, "./dataset/DC_dataset/new_small_job_info_df.csv")
HOST_DATASET_PATH = os.path.join(BASE_DIR, "./dataset/DC_dataset/node_info_df.csv")


# 环境基本参数
NUM_DATACENTERS = 5          # 数据中心数量
NUM_HOST = 50                # 全局生成的主机 (Host) 总数量
NUM_JOBS = 1000              # 全局生成的任务 (Job) 总数量
LAMBDA_RATE = 0.5           # LAMBDA_RATE 越大，任务到达越密集（时间间隔越短）
CLOUD_LATENCY_RANGE = (5, 20) # 边缘节点到云数据中心的时延范围
EDGE_LATENCY_RANGE = (2,5)    # 边缘节点之间的时延范围
DROP_DEADLINE_RATE = 0.7
ENV_KEEP_PATH = os.path.join(BASE_DIR, "./environment/env_keep")

# 模型基本参数
ACTOR_HIDDEN_DIM = 256      # actor-net 隐藏层维度
Q_NET_HIDDEN_DIM = 256      # Q-net 隐藏层维度
ACTOR_GAIN = 0.01
Q_NET_GAIN = 0.01
GAMMA = 0.99
TUA = 0.005
ACTOR_LR = 1e-4
CRITIC_LR = 3e-4
ALPHA_LR = 1e-4
INITIAL_ALPHA = 0.1
TARGET_ENTROPY_RATIO = 0.2
MAX_GRAD_NORM = 5.0
Policy_Updata_Interval = 1
Target_Update_Interval = 1
DEVICE = "cuda:0"





# 训练基本参数
Episodes = 1000         # 训练轮次
ReplyBuffer_Capacity = 200000           # 经验池容量
Batch_Size = 256            # 采样批量
Seed = 42           # 宇宙的终极答案
Random_warmup_step = 5000           # 前期随机预热启动
Learning_Starts = 1000          # 至少积累多少个普通动作后再开始更新网络
Train_Every = 4                 # 收集多少经验训一次网络
Updates_Per_Train = 4            # 每条经验最多执行多少次网络更新
Log_interval = 1            # 每多少episode打印一次统计信息
Checkpoint_Interval = 100        # 每多少个episode保存一次带编号的checkpoint
Checkpoint_Dir = "model/MASAC/checkpoints"
Log_csv_Path = "result/MASAC/train_log.csv"
Old_Env_Path = None         #可选的旧环境文件路径,为 None 时，CloudEdgeEnv 会按自己的默认逻辑生成新环境
Resume_Checkpoint = None         #可选的断点模型路径,为 None 表示从头训练。
Vary_Episode_Seed: bool = True          # 是否在每个 episode 使用不同但可复现的 seed



# 奖励参数（先这样写着吧，目前没什么好办法
TASK_COMPLETION_REWARD = 2.5            # 一个任务真正完成获得
WAITING_TIME_COST_WEIGHT = 0.5          # 等待惩罚
EXECUTION_TIME_COST_WEIGHT = 1.0        # 任务长短的权重，越大越容易接受小任务
EDGE_FORWARD_BASE_PENALTY = 0.25        # 转发固定成本
EDGE_LATENCY_COST_WEIGHT = 1.0          # 边边时延惩罚
CLOUD_LATENCY_COST_WEIGHT = 1.0         # 云边时延惩罚
TIMEOUT_DROP_PENALTY = 2.5              # 超时惩罚
RESOURCE_DROP_PENALTY = 2.0             # 资源不足惩罚，其实已经不会触发了，因为后面搞了掩码


