import os

# 数据集路径
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
JOB_DATASET_PATH = os.path.join(BASE_DIR, "./dataset/DC_dataset/new_small_job_info_df.csv")
HOST_DATASET_PATH = os.path.join(BASE_DIR, "./dataset/DC_dataset/node_info_df.csv")


# 环境基本参数
NUM_DATACENTERS = 5          # 数据中心数量
NUM_HOST = 50                # 全局生成的主机 (Host) 总数量
NUM_JOBS = 1000              # 全局生成的任务 (Job) 总数量
LAMBDA_RATE = 0.25           # LAMBDA_RATE 越大，任务到达越密集（时间间隔越短）
CLOUD_LATENCY_RANGE = (5, 20) # 边缘节点到云数据中心的时延范围
EDGE_LATENCY_RANGE = (2,5)    # 边缘节点之间的时延范围
# DROP_DEADLINE_RATE = 2.0      # 丢弃任务超时倍数
SLA_DEADLINE_RATIO = 2.0
DROP_DEADLINE_RATIO = 2.5
QUEUE_LENGTH_SCALE = 10.0      # 队列长度归一化使用参数（根据训练后最大等待队列的一半）
ENV_KEEP_PATH = os.path.join(BASE_DIR, "./environment/env_keep")
SIMULATION_TIME_UNIT = "s"     # 系统时间基本单位
ENERGY_UNIT = "J"              # 系统能量基本单位
EDGE_CPU_IDLE_POWER_W = 110.0       # edge host CPU 空闲功率
EDGE_CPU_FULL_POWER_W = 170.0       # edge host CPU 满载功率
EDGE_GPU_IDLE_POWER_W = 25.0        # edge host GPU 空闲功率
EDGE_GPU_FULL_POWER_W = 250.0       # edge host CPU 满载功率
EDGE_EDGE_TRANSFER_ENERGY_RATIO = 0.04      # e2e传输能耗系数
EDGE_CLOUD_TRANSFER_ENERGY_RATIO = 0.10     # e2c传输能耗系数
ENERGY_NORMALIZATION_PERCENTILE = 95.0      # 能耗归一化参数

# 模型基本参数
ACTOR_HIDDEN_DIM = 256      # actor-net 隐藏层维度
Q_NET_HIDDEN_DIM = 256      # Q-net 隐藏层维度
ACTOR_GAIN = 0.01           # Actor 输出层初始化幅度
Q_NET_GAIN = 0.01           # Critic 输出层初始化幅度
GAMMA = 0.99                # 长期奖励折扣
TUA = 0.005                 # Target Critic 软更新速度
ACTOR_LR = 1e-4             # Actor 学习率
CRITIC_LR = 2e-4            # Critic 学习率
ALPHA_LR = 1e-4             # 温度参数 α 的学习率
INITIAL_ALPHA = 0.1         # 初始探索强度
TARGET_ENTROPY_RATIO = 0.2  # 随机性保留
MAX_GRAD_NORM = 3.0         # 梯度裁剪阈值
Policy_Updata_Interval = 2  # Actor 相对 Critic 的更新频率
Target_Update_Interval = 1  # Target Critic 更新频率
DEVICE = "cuda:0"





# 训练基本参数
Episodes = 1500         # 训练轮次
ReplyBuffer_Capacity = 200000           # 经验池容量
Batch_Size = 512            # 采样批量
Seed = 42           # 宇宙的终极答案
Random_warmup_step = 5000           # 前期随机预热启动
Learning_Starts = 5000          # 至少积累多少个普通动作后再开始更新网络
Train_Every = 4                 # 收集多少经验训一次网络
Updates_Per_Train = 2            # 每条经验最多执行多少次网络更新
Log_interval = 1            # 每多少episode打印一次统计信息
Checkpoint_Interval = 100        # 每多少个episode保存一次带编号的checkpoint
Checkpoint_Dir = "model/MASAC/checkpoints"
Log_csv_Path = "result/MASAC/train_log.csv"
Old_Env_Path = None         #可选的旧环境文件路径,为 None 时，CloudEdgeEnv 会按自己的默认逻辑生成新环境
Resume_Checkpoint = None         #可选的断点模型路径,为 None 表示从头训练。
Vary_Episode_Seed: bool = True          # 是否在每个 episode 使用不同但可复现的 seed



# 奖励参数（先这样写着吧，目前没什么好办法
TASK_COMPLETION_REWARD = 2.5            # 一个任务真正完成获得
# WAITING_TIME_COST_WEIGHT = 0.75          # 等待惩罚
# EXECUTION_TIME_COST_WEIGHT = 1.0        # 任务长短的权重，越大越容易接受小任务
COMPLETION_TIME_COST_WEIGHT = 1.0       # 实际任务完成时间成本权重
SLA_VIOLATION_COST_WEIGHT = 1.5         # SLA 违约严重程度权重
QUEUE_ADMISSION_COST_WEIGHT = 0.5       # 排队风险参数，帮助模型认识到进入等待队列和立即执行是不同的
EDGE_FORWARD_BASE_PENALTY = 0.05        # 转发固定成本
EDGE_LATENCY_COST_WEIGHT = 1.0          # 边边时延惩罚
CLOUD_LATENCY_COST_WEIGHT = 1.0         # 云边时延惩罚
SLA_RISK_COST_WEIGHT = 1.0              # 违约风险参数
EDGE_DEADLINE_RISK_COST_WEIGHT = 1.0    # Edge 转发行为对任务剩余 deadline 的侵蚀
TIMEOUT_DROP_PENALTY = 4.0              # 超时惩罚
RESOURCE_DROP_PENALTY = 2.0             # 资源不足惩罚，其实已经不会触发了，因为后面搞了掩码
COMPLETION_CREDIT_DECAY = 0.8           # 调度链奖励衰减参数
FAILURE_CREDIT_DECAY = 0.8              # 调度链惩罚衰减参数

# 能耗成本权重。
#
# 当前阶段必须保持为 0.0：
# 先完成 Edge / Cloud / Transmission 能耗模型和日志统计，
# 验证物理能耗计算正确以后，再逐步调高该权重。
#
# 这样当前这一步只建立能耗配置，不会改变现有 MASAC 的奖励函数和训练行为。
ENERGY_COST_WEIGHT = 0.0