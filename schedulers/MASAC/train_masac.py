from __future__ import annotations
import argparse
import csv
import json
import random
import sys

import time

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union
import numpy as np
import torch
from datetime import datetime
# 找根目录
PROJECT_ROOT = Path(__file__).resolve().parents[2]
# 找环境代码
ENVIRONMENT_DIR = PROJECT_ROOT / "environment"
from schedulers.MASAC.masac_agent import (DiscreteMASAC, MASACConfig)
from schedulers.MASAC.replay_buffer import ReplayBuffer
from schedulers.MASAC.transition_collector import (DecisionSnapshot, TransitionCollector)
from environment.cloud_edge_env import CloudEdgeEnv
import config as conf

TERMINAL_FAILURE_REASONS = frozenset({
    "waiting_timeout",
    "cloud_arrival_timeout",
    "cloud_resource_failure",
})

@dataclass(frozen=True)
class TrainConfig:
    num_episodes: int = conf.Episodes
    replay_capacity: int = conf.ReplyBuffer_Capacity
    batch_size: int = conf.Batch_Size
    random_warmup_steps: int = conf.Random_warmup_step
    learning_starts: int = conf.Learning_Starts
    # updates_per_step: int = conf.Updates_Per_Step
    # max_decisions_per_episode: int = 100_000
    train_every: int = conf.Train_Every
    updates_per_train: int = conf.Updates_Per_Train
    log_interval: int = conf.Log_interval
    checkpoint_interval: int = conf.Checkpoint_Interval
    seed: int = conf.Seed
    checkpoint_dir: str = conf.Checkpoint_Dir
    log_csv_path: str = conf.Log_csv_Path
    old_env_path: Optional[str] = conf.Old_Env_Path
    resume_checkpoint: Optional[str] = conf.Resume_Checkpoint
    vary_episode_seed: bool = conf.Vary_Episode_Seed
    completion_credit_decay: float = conf.COMPLETION_CREDIT_DECAY
    failure_credit_decay: float = (conf.FAILURE_CREDIT_DECAY)

# 统计一个 episode 运行期间的统计信息
@dataclass
class EpisodeStatistics:
    episode: int
    episode_seed: int
    per_agent_returns: Dict[str, float]
    episode_return: float = 0.0
    decision_count: int = 0
    normal_action_count: int = 0
    forced_action_count: int = 0
    random_action_count: int = 0
    policy_action_count: int = 0
    local_action_count: int = 0
    edge_action_count: int = 0
    cloud_action_count: int = 0
    drop_action_count: int = 0
    update_count: int = 0
    update_metric_sums: Dict[str, float] = field(default_factory=dict)
    update_metric_counts: Dict[str, int] = field(default_factory=dict)

    # 记录 Transition 奖励和动作来源
    def record_transition(
            self,
            agent_id: str,
            reward: float,
            action_type: str,
            action_source: str,
    ) -> None:

        reward = float(reward)
        self.episode_return += reward
        self.per_agent_returns[agent_id] = (self.per_agent_returns.get(agent_id, 0.0) + reward)
        self.decision_count += 1

        if action_source == "forced":
            self.forced_action_count += 1
        elif action_source == "random":
            self.random_action_count += 1
            self.normal_action_count += 1
        elif action_source == "policy":
            self.policy_action_count += 1
            self.normal_action_count += 1

        if action_type == "local_host":
            self.local_action_count += 1
        elif action_type == "edge_dc":
            self.edge_action_count += 1
        elif action_type == "cloud":
            self.cloud_action_count += 1
        elif action_type == "drop":
            self.drop_action_count += 1

    # 记录由于 Job 后续完成/超时产生的延迟 reward
    def record_reward_correction(self, agent_id: str, reward_delta: float,) -> None:
        agent_id = str(agent_id)
        reward_delta = float(reward_delta)
        self.episode_return += (reward_delta)
        self.per_agent_returns[agent_id] = (self.per_agent_returns.get(agent_id, 0.0,) + reward_delta)


    # 记录一次 MASAC.update() 返回的训练指标
    def record_update(self, update_info: Dict[str, float]) -> None:
        self.update_count += 1

        for metric_name, metric_value in update_info.items():
            metric_value = float(metric_value)
            if not np.isfinite(metric_value):
                continue
            self.update_metric_sums[metric_name] = (self.update_metric_sums.get(metric_name, 0.0)+ metric_value)
            self.update_metric_counts[metric_name] = (self.update_metric_counts.get(metric_name, 0)+ 1)

    # 返回某个训练指标在当前episode中的平均值
    def mean_metric(self, metric_name: str) -> float:
        count = self.update_metric_counts.get(metric_name, 0)
        if count == 0:
            return float("nan")
        return float(
            self.update_metric_sums[metric_name] / count
        )

UPDATE_TENSOR_METRIC_NAMES = (
        "critic_loss",
        "q1_loss",
        "q2_loss",
        "mean_q1",
        "mean_q2",
        "mean_target_q",
        "actor_loss",
        "alpha_loss",
        "alpha",
        "policy_entropy",
        "target_entropy",
    )

# 批量记录连续若干次 MASAC.update() 的训练指标，避免每次 update 内部执行大量 cuda_tensor.item()
def record_update_block(stats: EpisodeStatistics, update_infos: list[ Dict[str, Union[float, torch.Tensor]]],) -> None:
    if not update_infos:
        return

    update_rows = []

    for update_info in update_infos:
        metric_row = torch.stack([update_info[metric_name]
                for metric_name
                in UPDATE_TENSOR_METRIC_NAMES
            ],
            dim=0,
        )

        update_rows.append(
            metric_row
        )
    update_matrix = torch.stack(update_rows, dim=0,)
    update_matrix_cpu = (update_matrix.detach().cpu().numpy())

    for row in update_matrix_cpu:
        cpu_update_info = {
            metric_name: float(
                row[metric_index]
            )
            for metric_index, metric_name
            in enumerate(
                UPDATE_TENSOR_METRIC_NAMES
            )
         }
        stats.record_update(cpu_update_info)

def set_global_random_seeds(seed: int) -> None:
    """统一设置 Python、NumPy 和 PyTorch 随机种子。"""

    # 转换成标准 Python int。
    seed = int(seed)

    # 设置 Python random 随机种子。
    random.seed(seed)

    # 设置 NumPy 旧式全局随机接口的种子。
    # ReplayBuffer 和预热动作仍会使用各自独立的 default_rng。
    np.random.seed(seed)

    # 设置 CPU 上的 PyTorch 随机种子。
    torch.manual_seed(seed)

    # CUDA 可用时设置所有 GPU 的随机种子。
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def build_environment(seed: int, old_env_path: Optional[str] = None,) -> Any:
    if old_env_path is None:
        return CloudEdgeEnv(seed=int(seed))
    return CloudEdgeEnv(
        old_env_path=str(old_env_path),
        seed=int(seed),
    )

# 从 DecisionSnapshot 的 action_mask 中随机选择一个合法动作'
def choose_random_legal_action(decision: DecisionSnapshot, rng: np.random.Generator,) -> int:
    valid_actions = np.flatnonzero(np.asarray(decision.action_mask, dtype=np.int8))
    action = rng.choice(valid_actions)
    return int(action)

# 根据动作解码函数获取动作类型
def infer_action_type(env: Any, agent_id: str, action: int,) -> str:
    if int(action) == int(env.drop_action):
        return "drop"
    decode_action = getattr(env, "_decode_action", None)
    if callable(decode_action):
        decoded_action = decode_action(
            agent_id=str(agent_id),
            action=int(action),
        )
        return str(decoded_action.get("action_type", "unknown"))
    return "unknown"

# 遍历所有数据中心和 host，统计完成队列中的任务数量
def count_completed_jobs(env: Any) -> int:
    completed_count = 0
    for datacenter in getattr(env, "datacenters", []):
        # 遍历当前数据中心的全部 host。
        for host in getattr(datacenter, "host_list", []):
            # 读取 host 完成队列。
            completed_queue = getattr(host, "completed_queue", None)

            # 没有完成队列时跳过。
            if completed_queue is None:
                continue

            # 将该 host 的完成任务数量加入总数。
            completed_count += len(completed_queue)
    return int(completed_count)

# 统计当前 episode 的 SLA 与任务完成时间指标
def calculate_service_metrics(env: Any) -> Dict[str, float]:
    completed_jobs = []
    for job in getattr(env, "jobs", []):
        if getattr(job, "finish_time", None) is not None:
            completed_jobs.append(job)

    total_jobs = int(len(getattr(env, "jobs", [])))

    dropped_jobs = int(len(getattr(env, "dropped_jobs_info", [],)))

    total_completion_time = 0.0
    sla_satisfied_jobs = 0
    sla_violated_completed_jobs = 0
    total_violation_degree = 0.0

    sla_ratio = float(env.sla_deadline_ratio)
    drop_ratio = float(env.drop_deadline_ratio)

    for job in completed_jobs:
        turnaround_time = (job.get_turnaround_time())
        if turnaround_time is None:
            continue

        turnaround_time = float(turnaround_time)
        duration = max(float(job.duration),1e-8,)
        sla_limit = (sla_ratio * duration)
        drop_limit = (drop_ratio * duration)
        total_completion_time += (turnaround_time)

        if turnaround_time <= sla_limit:
            sla_satisfied_jobs += 1

        else:
            sla_violated_completed_jobs += 1
            violation_degree = (
                                       turnaround_time - sla_limit
                               ) / max(
                drop_limit - sla_limit,
                1e-8,
            )

            total_violation_degree += float(
                np.clip(
                    violation_degree,
                    0.0,
                    1.0,
                )
            )

    completed_count = len(
        completed_jobs
    )

    avg_completion_time = (
        total_completion_time
        / completed_count
        if completed_count > 0
        else 0.0
    )

    # Drop 直接视为最严重 SLA violation，degree = 1。
    total_violation_degree += float(
        dropped_jobs
    )

    sla_satisfaction_rate = (
        sla_satisfied_jobs / total_jobs
        if total_jobs > 0
        else 0.0
    )

    sla_violation_rate = (
        (
                sla_violated_completed_jobs
                + dropped_jobs
        )
        / total_jobs
        if total_jobs > 0
        else 0.0
    )

    mean_sla_violation_degree = (
        total_violation_degree
        / total_jobs
        if total_jobs > 0
        else 0.0
    )

    return {
        "sla_satisfied_jobs": int(
            sla_satisfied_jobs
        ),
        "sla_violated_completed_jobs": int(
            sla_violated_completed_jobs
        ),
        "sla_satisfaction_rate": float(
            sla_satisfaction_rate
        ),
        "sla_violation_rate": float(
            sla_violation_rate
        ),
        "mean_sla_violation_degree": float(
            mean_sla_violation_degree
        ),
        "total_completion_time": float(
            total_completion_time
        ),
        "avg_completion_time": float(
            avg_completion_time
        ),
    }

# 统计当前环境所有 Host 中剩余的 waiting job 数量
def count_waiting_jobs(env: Any) -> int:
    waiting_count = 0
    for datacenter in getattr(env, "datacenters", [],):
        for host in getattr(datacenter, "host_list", [],):
            waiting_queue = getattr(host, "waiting_queue", None,)
            if waiting_queue is None:
                continue
            waiting_count += len(waiting_queue)
    return int(waiting_count)

# 根据模型文件路径生成配套的训练器状态 JSON 路径
def checkpoint_state_path(model_path: Path) -> Path:
    return model_path.with_suffix(".trainer.json")

# 同时保存 MASAC 模型和训练主循环状态
def save_checkpoint(
    masac: DiscreteMASAC,
    model_path: Path,
    next_episode: int,
    global_decision_steps: int,
    global_normal_action_steps: int,
    best_episode_return: float,
) -> None:
    model_path.parent.mkdir(parents=True, exist_ok=True)
    masac.save(model_path)
    trainer_state = {
        "next_episode": int(next_episode),
        "global_decision_steps": int(global_decision_steps),
        "global_normal_action_steps": int(global_normal_action_steps),
        "best_episode_return": float(best_episode_return),
    }
    state_path = checkpoint_state_path(model_path)
    state_path.write_text(
        json.dumps(
            trainer_state,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

# 按需恢复模型和训练器状态
def load_checkpoint_if_needed(
    masac: DiscreteMASAC,
    resume_checkpoint: Optional[str],
) -> Tuple[int, int, int, float]:

    # 没有指定 checkpoint 时，从第 1 个 episode 开始
    if resume_checkpoint is None:
        return 1, 0, 0, float("-inf")

    model_path = Path(resume_checkpoint)
    masac.load(
        file_path=model_path,
        load_optimizers=True,
    )
    state_path = checkpoint_state_path(model_path)

    if not state_path.exists():
        print(
            "已恢复模型，但没有找到训练器状态文件："
            f"{state_path}。训练 episode 计数将从 1 开始。"
        )
        return 1, 0, 0, float("-inf")

    trainer_state = json.loads(state_path.read_text(encoding="utf-8"))

    start_episode = int(
        trainer_state.get("next_episode", 1)
    )
    global_decision_steps = int(
        trainer_state.get("global_decision_steps", 0)
    )
    global_normal_action_steps = int(
        trainer_state.get("global_normal_action_steps", 0)
    )
    best_episode_return = float(
        trainer_state.get("best_episode_return", float("-inf"))
    )

    # 返回恢复后的训练进度。
    return (
        start_episode,
        global_decision_steps,
        global_normal_action_steps,
        best_episode_return,
    )
# 写训练日志用的
def build_run_log_path(base_log_path: str) -> Path:
    base_path = Path(base_log_path)
    run_start_time = datetime.now().strftime(
        "%Y%m%d_%H%M%S"
    )
    log_file_name = (
        f"{base_path.stem}_"
        f"{run_start_time}"
        f"{base_path.suffix}"
    )
    return base_path.parent / log_file_name



# 把一个 episode 的统计信息追加到 CSV 文件
def append_csv_log(csv_path: Path, row: Dict[str, Any],) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    # 判断文件是否已经存在且不是空文件。
    write_header = (
            not csv_path.exists()
            or csv_path.stat().st_size == 0
    )

    # 以追加模式打开文件。
    with csv_path.open(
            mode="a",
            newline="",
            encoding="utf-8-sig",
    ) as csv_file:
        # 使用当前 row 的键作为固定列名。
        writer = csv.DictWriter(
            csv_file,
            fieldnames=list(row.keys()),
        )

        # 新文件先写入表头。
        if write_header:
            writer.writeheader()

        # 写入当前 episode 数据。
        writer.writerow(row)

# 把 episode 统计整理成固定结构的 CSV 行
def build_episode_log_row(
    stats: EpisodeStatistics,
    env: Any,
    replay_buffer: ReplayBuffer,
    masac: DiscreteMASAC,
    global_decision_steps: int,
    global_normal_action_steps: int,
    wall_time_seconds: float,
) -> Dict[str, Any]:
    # 当前 episode 总任务数。
    total_jobs = int(len(getattr(env, "jobs", [])))

    # 从 host 完成队列统计完成任务数。
    completed_jobs = count_completed_jobs(env)

    # 从环境丢弃记录统计丢弃任务数。
    dropped_jobs = int(
        len(getattr(env, "dropped_jobs_info", []))
    )

    # 没有进入完成或丢弃终态的任务数量。
    unresolved_jobs = int(total_jobs - completed_jobs - dropped_jobs)
    queued_jobs = int(getattr( env, "queued_jobs", 0,))
    started_from_waiting_jobs = int(getattr( env, "started_from_waiting_jobs", 0,))
    waiting_timeout_drops = int(getattr( env, "waiting_timeout_drops", 0,))
    max_waiting_queue_length = int(getattr( env, "max_waiting_queue_length", 0,))
    remaining_waiting_jobs = count_waiting_jobs(env)
    service_metrics = calculate_service_metrics(env)

    # 返回固定列顺序的字典。
    return {
        "episode": int(stats.episode),
        "episode_seed": int(stats.episode_seed),
        "episode_return": float(stats.episode_return),
        "decision_count": int(stats.decision_count),
        "normal_action_count": int(stats.normal_action_count),
        "forced_action_count": int(stats.forced_action_count),
        "random_action_count": int(stats.random_action_count),
        "policy_action_count": int(stats.policy_action_count),
        "local_action_count": int(stats.local_action_count),
        "edge_action_count": int(stats.edge_action_count),
        "cloud_action_count": int(stats.cloud_action_count),
        "drop_action_count": int(stats.drop_action_count),
        # "unknown_action_count": int(stats.unknown_action_count),
        "total_jobs": total_jobs,
        "completed_jobs": completed_jobs,
        "dropped_jobs": dropped_jobs,
        "sla_satisfied_jobs": int(service_metrics["sla_satisfied_jobs"]),
        "sla_violated_completed_jobs": int(service_metrics["sla_violated_completed_jobs"]),
        "sla_satisfaction_rate": float(service_metrics["sla_satisfaction_rate"]),
        "sla_violation_rate": float(service_metrics["sla_violation_rate"]),
        "mean_sla_violation_degree": float(service_metrics["mean_sla_violation_degree"]),
        "total_completion_time": float(service_metrics["total_completion_time"]),
        "avg_completion_time": float(service_metrics["avg_completion_time"]),
        "unresolved_jobs": unresolved_jobs,
        "queued_jobs": queued_jobs,
        "started_from_waiting_jobs": started_from_waiting_jobs,
        "waiting_timeout_drops": waiting_timeout_drops,
        "max_waiting_queue_length": max_waiting_queue_length,
        "remaining_waiting_jobs": remaining_waiting_jobs,
        "simulation_end_time": float(getattr(env, "current_time", 0.0)),
        "episode_updates": int(stats.update_count),
        "global_decision_steps": int(global_decision_steps),
        "global_normal_action_steps": int(global_normal_action_steps),
        "replay_size": int(len(replay_buffer)),
        "replay_trainable_size": int(replay_buffer.num_trainable_actions),
        "replay_forced_size": int(replay_buffer.num_forced_actions),
        "masac_update_step": int(masac.update_step),
        "critic_loss": stats.mean_metric("critic_loss"),
        "q1_loss": stats.mean_metric("q1_loss"),
        "q2_loss": stats.mean_metric("q2_loss"),
        "actor_loss": stats.mean_metric("actor_loss"),
        "alpha_loss": stats.mean_metric("alpha_loss"),
        "alpha": float(masac.alpha.detach().cpu().item()),
        "policy_entropy": stats.mean_metric("policy_entropy"),
        "target_entropy": stats.mean_metric("target_entropy"),
        "mean_q1": stats.mean_metric("mean_q1"),
        "mean_q2": stats.mean_metric("mean_q2"),
        "mean_target_q": stats.mean_metric("mean_target_q"),
        "wall_time_seconds": float(wall_time_seconds),
        "per_agent_returns": json.dumps(
            stats.per_agent_returns,
            ensure_ascii=False,
            sort_keys=True,
        ),
    }

# 打印episode摘要
def print_episode_summary(row: Dict[str, Any]) -> None:
    # 提取可能为 NaN 的 Critic 损失。
    critic_loss = float(row["critic_loss"])

    # 有限数值正常格式化，否则显示 nan。
    critic_loss_text = (
        f"{critic_loss:.6f}"
        if np.isfinite(critic_loss)
        else "nan"
    )

    # 打印主要训练指标。
    print(
        f"Episode {int(row['episode']):5d} | "
        f"return={float(row['episode_return']):9.4f} | "
        f"sla_sat={float(row['sla_satisfaction_rate']):6.2%} | "
        f"sla_vio={float(row['sla_violation_rate']):6.2%} | "
        f"avg_T={float(row['avg_completion_time']):8.2f} | "
        f"sum_T={float(row['total_completion_time']):10.2f} | "
        f"decisions={int(row['decision_count']):6d} | "
        f"completed={int(row['completed_jobs']):5d} | "
        f"dropped={int(row['dropped_jobs']):5d} | "
        f"queued={int(row['queued_jobs']):5d} | "
        f"wait_started={int(row['started_from_waiting_jobs']):5d} | "
        f"wait_timeout={int(row['waiting_timeout_drops']):5d} | "
        f"max_wait={int(row['max_waiting_queue_length']):4d} | "
        f"remain_wait={int(row['remaining_waiting_jobs']):4d} | "
        f"buffer={int(row['replay_trainable_size']):7d} | "
        f"updates={int(row['episode_updates']):5d} | "
        f"critic_loss={critic_loss_text} | "
        f"alpha={float(row['alpha']):.5f}"
    )



def train(
    train_config: TrainConfig,
    masac_config: Optional[MASACConfig] = None,
) -> DiscreteMASAC:

    set_global_random_seeds(train_config.seed)

    # 创建初始环境
    env = build_environment(
        seed=train_config.seed,
        old_env_path=train_config.old_env_path,
    )

    # 创建 Transition 采集器
    collector = TransitionCollector(env)

    # 创建经验回放池
    replay_buffer = ReplayBuffer(
        capacity=int(train_config.replay_capacity),
        local_obs_dim=int(env.local_obs_dim),
        global_state_dim=int(env.global_state_dim),
        action_dim=int(env.action_dim),
        seed=int(train_config.seed),
        forced_action_value=int(env.drop_action),
    )

    # 没有传入算法配置时，使用 MASACConfig 默认值，但让算法随机种子与训练配置保持一致
    if masac_config is None:
        masac_config = MASACConfig(
            seed=int(train_config.seed)
        )

    # 创建离散多智能体 SAC 算法
    masac = DiscreteMASAC(
        local_obs_dim=int(env.local_obs_dim),
        global_state_dim=int(env.global_state_dim),
        action_dim=int(env.action_dim),
        num_agents=int(len(env.possible_agents)),
        config=masac_config,
    )

    masac.train_mode()

    action_rng = np.random.default_rng(int(train_config.seed))

    # 按需恢复 checkpoint 和训练进度
    (
        start_episode,
        global_decision_steps,
        global_normal_action_steps,
        best_episode_return,
    ) = load_checkpoint_if_needed(
        masac=masac,
        resume_checkpoint=train_config.resume_checkpoint,
    )

    # 把保存路径转换成 Path
    checkpoint_dir = Path(train_config.checkpoint_dir)
    log_csv_path = build_run_log_path(train_config.log_csv_path)

    # 创建 checkpoint 目录
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    log_csv_path.parent.mkdir(parents=True, exist_ok=True,)

    try:
        # 从 start_episode 训练到 num_episodes，包含最后一个 episode。
        for episode in range(int(start_episode), int(train_config.num_episodes) + 1):

            ##################### 一轮 episode 开始前的准备 ####################
            # 根据配置决定当前 episode 使用哪个环境 seed
            if train_config.vary_episode_seed:
                episode_seed = int(train_config.seed + episode - 1)
            else:
                episode_seed = int(train_config.seed)

            # 重启环境
            env.reset(seed=episode_seed)
            replay_buffer.reset_episode_job_tracking()
            # 为每个智能体创建奖励累计字典
            per_agent_returns = {}
            for agent_id in env.possible_agents:
                per_agent_returns[str(agent_id)] = 0.0

            # 创建当前 episode 的统计对象
            stats = EpisodeStatistics(
                episode=int(episode),
                episode_seed=episode_seed,
                per_agent_returns=per_agent_returns,
            )

            # 记录episode开始的时间
            episode_wall_start = time.perf_counter()
            ###################################################################
            decision: Optional[DecisionSnapshot] = None

            # 只要 PettingZoo 的活跃智能体列表不为空，就继续循环
            while env.agents:

                # 如果智能体状态标记是死（应该不存在情况才对），把它从 AEC 环境中清理掉，然后跳过本轮训练逻辑
                if collector.drain_one_dead_agent():
                    decision = None
                    continue

                # 获取当前决策状态
                if decision is None:
                    decision = collector.capture_decision()



                # 处理超时任务
                if decision.forced_action is not None:
                    action = int(decision.forced_action)
                    action_source = "forced"

                # 预热阶段的普通动作
                elif (global_normal_action_steps < int(train_config.random_warmup_steps)):
                    # 随机选个动作
                    action = choose_random_legal_action(
                        decision=decision,
                        rng=action_rng,
                    )
                    action_source = "random"

                # 预热结束，actor根据概率分布采样动作
                else:
                    action = masac.select_action(
                        local_obs=decision.local_obs,
                        agent_index=decision.agent_index,
                        action_mask=decision.action_mask,
                        deterministic=False,
                    )
                    action_source = "policy"

                # 在环境执行前解码动作
                action_type = infer_action_type(env=env, agent_id=decision.agent_id, action=action)

                # 执行动作、推进事件队列并构造一条经验放入经验池
                transition, next_decision = collector.execute_and_collect(decision=decision, action=action,action_type=action_type,)
                replay_buffer.add(transition)
                decision = next_decision

                # 记录这条经验的奖励和动作类型
                stats.record_transition(
                    agent_id=transition.agent_id,
                    reward=transition.reward,
                    action_type=action_type,
                    action_source=action_source,
                )

                if (
                        action_source == "forced"
                        and action_type == "drop"
                        and float(transition.reward) < 0.0
                ):
                    replay_buffer.apply_discounted_terminal_reward(
                        job_id=str(
                            transition.job_id
                        ),
                        reward_delta=float(
                            transition.reward
                        ),
                        credit_decay=float(
                            train_config.failure_credit_decay
                        ),
                    )


                # 消费 env.step() 期间产生的延迟任务结果奖励
                reward_corrections = (
                    env.pop_reward_corrections()
                )

                for correction in reward_corrections:
                    correction_job_id = str(correction["job_id"])
                    reward_delta = float(correction["reward_delta"])

                    correction_reason = str(correction.get("reason", "",))

                    if correction_reason == "completed":
                        applied_credits = (
                            replay_buffer.apply_discounted_terminal_reward(
                                job_id=correction_job_id,
                                reward_delta=reward_delta,
                                credit_decay=float(
                                    train_config.completion_credit_decay
                                ),
                            )
                        )

                        if not applied_credits:
                            correction_agent_id = (
                                replay_buffer.apply_reward_correction(
                                    job_id=correction_job_id,
                                    reward_delta=reward_delta,
                                )
                            )
                            if correction_agent_id is not None:
                                stats.record_reward_correction(
                                    agent_id=correction_agent_id,
                                    reward_delta=reward_delta,
                                )
                            continue
                        for (correction_agent_id, terminal_credit,) in applied_credits:
                            stats.record_reward_correction(
                                agent_id=correction_agent_id,
                                reward_delta=terminal_credit,
                            )
                        continue
                    if correction_reason in TERMINAL_FAILURE_REASONS:
                        applied_penalties = (
                            replay_buffer.apply_discounted_terminal_reward(
                                job_id=correction_job_id,
                                reward_delta=reward_delta,
                                credit_decay=float(
                                    train_config.failure_credit_decay
                                ),
                            )
                        )
                        if not applied_penalties:
                            # 防御性 fallback。
                            correction_agent_id = (
                                replay_buffer.apply_reward_correction(
                                    job_id=correction_job_id,
                                    reward_delta=reward_delta,
                                )
                            )

                            if correction_agent_id is not None:
                                stats.record_reward_correction(
                                    agent_id=correction_agent_id,
                                    reward_delta=reward_delta,
                                )

                            continue
                        for (
                                correction_agent_id,
                                terminal_penalty,
                        ) in applied_penalties:
                            stats.record_reward_correction(
                                agent_id=correction_agent_id,
                                reward_delta=terminal_penalty,
                            )

                        continue


                    # 把奖励修正到这个 Job 最新一次调度经验。
                    correction_agent_id = (
                        replay_buffer.apply_reward_correction(
                            job_id=correction_job_id,
                            reward_delta=reward_delta,
                        )
                    )

                    # 如果对应 experience 仍然存在于 ReplayBuffer，
                    # 同时修正当前 episode 日志中的 return。
                    if correction_agent_id is not None:
                        stats.record_reward_correction(
                            agent_id=correction_agent_id,
                            reward_delta=reward_delta,
                        )

                # 增加计数
                global_decision_steps += 1
                if action_source != "forced":
                    global_normal_action_steps += 1
                    if (
                            global_normal_action_steps
                            == int(train_config.random_warmup_steps)
                    ):
                        print(
                            "\n"
                            "============================================================\n"
                            "✅ 随机动作预热结束\n"
                            f"普通动作步数: {global_normal_action_steps}\n"
                            f"当前 Episode: {episode}\n"
                            f"ReplayBuffer 大小: {len(replay_buffer)}\n"
                            f"MASAC 更新次数: {masac.update_step}\n"
                            "从下一条普通动作开始使用 Actor 策略进行动作采样。\n"
                            "============================================================\n"
                        )

                # 同时满足以下条件才允许更新网络：
                # 1. 普通动作总数已经达到 learning_starts；
                # 2. ReplayBuffer 中普通经验足够采样一个 batch。
                ready_to_update = (
                                    action_source != "forced" and
                                    global_normal_action_steps >= int(train_config.learning_starts) and
                                    global_normal_action_steps % int(train_config.train_every) == 0 and
                                    replay_buffer.can_sample(batch_size=int(train_config.batch_size),include_forced_actions=False))

                # 网络更新
                if ready_to_update:
                    update_info_block = []
                    for _ in range(int(train_config.updates_per_train)):
                        # 从 ReplayBuffer 采样并执行一次完整 SAC 更新
                        update_info = masac.update(replay_buffer=replay_buffer, batch_size=int(train_config.batch_size))
                        # stats.record_update(update_info)
                        update_info_block.append(update_info)
                    record_update_block(stats=stats,update_infos=update_info_block,)

            # 计算当前 episode 的真实运行秒数。
            wall_time_seconds = (time.perf_counter() - episode_wall_start)

            # 日志记录
            log_row = build_episode_log_row(
                stats=stats,
                env=env,
                replay_buffer=replay_buffer,
                masac=masac,
                global_decision_steps=global_decision_steps,
                global_normal_action_steps=global_normal_action_steps,
                wall_time_seconds=wall_time_seconds,
            )
            append_csv_log(csv_path=log_csv_path, row=log_row,)

            # 到达日志打印间隔时，在终端输出摘要。
            if episode % int(train_config.log_interval) == 0:
                print_episode_summary(log_row)

            # 当前 episode return 高于历史最佳值时保存 best checkpoint
            if stats.episode_return > best_episode_return:
                best_episode_return = float(
                    stats.episode_return
                )

                save_checkpoint(
                    masac=masac,
                    model_path=checkpoint_dir / "best.pt",
                    next_episode=episode + 1,
                    global_decision_steps=global_decision_steps,
                    global_normal_action_steps=global_normal_action_steps,
                    best_episode_return=best_episode_return,
                )

            # 断点恢复
            save_checkpoint(
                masac=masac,
                model_path=checkpoint_dir / "latest.pt",
                next_episode=episode + 1,
                global_decision_steps=global_decision_steps,
                global_normal_action_steps=global_normal_action_steps,
                best_episode_return=best_episode_return,
            )
            if (episode % int(train_config.checkpoint_interval) == 0):
                save_checkpoint(
                    masac=masac,
                    model_path=(checkpoint_dir / f"episode_{episode:06d}.pt"),
                    next_episode=episode + 1,
                    global_decision_steps=global_decision_steps,
                    global_normal_action_steps=global_normal_action_steps,
                    best_episode_return=best_episode_return,
                )

    finally:
        close_method = getattr(env, "close", None)
        if callable(close_method):
            close_method()

    # 全部 episode 完成后保存 final checkpoint
    save_checkpoint(
        masac=masac,
        model_path=checkpoint_dir / "final.pt",
        next_episode=int(train_config.num_episodes) + 1,
        global_decision_steps=global_decision_steps,
        global_normal_action_steps=global_normal_action_steps,
        best_episode_return=best_episode_return,
    )

    return masac

def main() -> None:
    train_config = TrainConfig(
        num_episodes=conf.Episodes,
        replay_capacity=conf.ReplyBuffer_Capacity,
        batch_size=conf.Batch_Size,
        random_warmup_steps=conf.Random_warmup_step,
        learning_starts=conf.Learning_Starts,
        # updates_per_step=conf.Updates_Per_Step,
        train_every=conf.Train_Every,
        updates_per_train=conf.Updates_Per_Train,
        log_interval=conf.Log_interval,
        checkpoint_interval=conf.Checkpoint_Interval,
        seed=conf.Seed,
        checkpoint_dir=conf.Checkpoint_Dir,
        log_csv_path=conf.Log_csv_Path,
        old_env_path=conf.Old_Env_Path,
        resume_checkpoint=conf.Resume_Checkpoint,
        vary_episode_seed=conf.Vary_Episode_Seed
    )

    masac_config = MASACConfig(
        gamma=conf.GAMMA,
        tau=conf.TUA,
        actor_lr=conf.ACTOR_LR,
        critic_lr=conf.CRITIC_LR,
        alpha_lr=conf.ALPHA_LR,
        actor_hidden_dim=conf.ACTOR_HIDDEN_DIM,
        critic_hidden_dim=conf.Q_NET_HIDDEN_DIM,
        initial_alpha=conf.INITIAL_ALPHA,
        target_entropy_ratio=conf.TARGET_ENTROPY_RATIO,
        max_grad_norm=conf.MAX_GRAD_NORM,
        policy_update_interval=conf.Policy_Updata_Interval,
        target_update_interval=conf.Target_Update_Interval,
        device=conf.DEVICE,
        seed=conf.Seed
    )

    train(
        train_config=train_config,
        masac_config=masac_config,
    )

if __name__ == "__main__":
    main()













