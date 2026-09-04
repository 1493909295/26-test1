from __future__ import annotations

import csv
import json
import random
import sys
import os
import time

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union
import numpy as np
import torch
from datetime import datetime
# 找根目录
PROJECT_ROOT = Path(__file__).resolve().parents[2]
H_MASAC_DIR = Path(__file__).resolve().parent
# 找环境代码
ENVIRONMENT_DIR = PROJECT_ROOT / "environment"
if str(H_MASAC_DIR) not in sys.path:
    sys.path.insert(0, str(H_MASAC_DIR))

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from h_masac_agent import (
    RoutingMASAC,
    RoutingMASACConfig,

    LocalHostSAC,
    HostSACConfig,
)
from routing_replay_buffer import (RoutingReplayBuffer,)
from host_replay_buffer import (HostReplayBuffer,)
from pending_job_trace import (PendingJobTraceStore, FinalizedJobTrace,)
from transition_collector import (DecisionSnapshot,TransitionCollector,)
from environment.cloud_edge_env import CloudEdgeEnv
import config as conf
from routing_observation import (RoutingObservationBuilder,)
from routing_centralized_state import (RoutingCentralizedStateBuilder,)
from host_observation import (HostObservationBuilder,)

CHECKPOINT_SCHEMA_VERSION = 2

CHECKPOINT_ARCHITECTURE = (
    "h_masac_two_layer_routing_host_v1"
)

TERMINAL_FAILURE_REASONS = frozenset({
    "waiting_timeout",
    "cloud_arrival_timeout",
    "cloud_resource_failure",
    "local_host_arrival_timeout",
    "local_host_resource_failure",
})

UPDATE_TENSOR_METRIC_NAMES = ("critic_loss",
        "q1_loss",
        "q2_loss",
        "mean_q1",
        "mean_q2",
        "mean_target_q",
        "actor_loss",
        "alpha_loss",
        "alpha",
        "policy_entropy",
        "target_entropy",)

@dataclass(frozen=True)
class TrainConfig:
    num_episodes: int = (
        conf.Episodes
    )

    # ==========================================================
    # Replay
    # ==========================================================

    routing_replay_capacity: int = (
        conf.ROUTING_REPLAY_CAPACITY
    )

    host_replay_capacity: int = (
        conf.HOST_REPLAY_CAPACITY
    )

    # ==========================================================
    # Routing MASAC training schedule
    # ==========================================================

    routing_batch_size: int = (
        conf.ROUTING_BATCH_SIZE
    )

    routing_random_warmup_steps: int = (
        conf.ROUTING_RANDOM_WARMUP_STEPS
    )

    routing_learning_starts: int = (
        conf.ROUTING_LEARNING_STARTS
    )

    routing_train_every: int = (
        conf.ROUTING_TRAIN_EVERY
    )

    routing_updates_per_train: int = (
        conf.ROUTING_UPDATES_PER_TRAIN
    )

    # ==========================================================
    # Local Host SAC training schedule
    # ==========================================================

    host_batch_size: int = (
        conf.HOST_BATCH_SIZE
    )

    host_random_warmup_steps: int = (
        conf.HOST_RANDOM_WARMUP_STEPS
    )

    host_learning_starts: int = (
        conf.HOST_LEARNING_STARTS
    )

    host_train_every: int = (
        conf.HOST_TRAIN_EVERY
    )

    host_updates_per_train: int = (
        conf.HOST_UPDATES_PER_TRAIN
    )

    # ==========================================================
    # Three-stage training
    # ==========================================================

    host_pretrain_episodes: int = (
        conf.HOST_PRETRAIN_EPISODES
    )

    routing_train_episodes: int = (
        conf.ROUTING_TRAIN_EPISODES
    )

    joint_finetune_episodes: int = (
        conf.JOINT_FINETUNE_EPISODES
    )

    log_interval: int = (
        conf.Log_interval
    )

    checkpoint_interval: int = (
        conf.Checkpoint_Interval
    )

    seed: int = conf.Seed

    checkpoint_dir: str = (
        conf.Checkpoint_Dir
    )

    log_csv_path: str = (
        conf.Log_csv_Path
    )

    old_env_path: Optional[str] = (
        conf.Old_Env_Path
    )

    resume_checkpoint: Optional[str] = (
        conf.Resume_Checkpoint
    )

    vary_episode_seed: bool = (
        conf.Vary_Episode_Seed
    )

    use_neighbor_historical_feedback: bool = (
        conf.USE_NEIGHBOR_HISTORICAL_FEEDBACK
    )

class TrainingStage( str,Enum,):
    """
    Two-Level Scheduler 三阶段训练状态。
    """

    HOST_PRETRAIN = (
        "host_pretrain"
    )

    ROUTING_TRAIN = (
        "routing_train"
    )

    JOINT_FINETUNE = (
        "joint_finetune"
    )

# 统计一个 episode 运行期间的统计信息
@dataclass
class EpisodeStatistics:
    episode: int
    episode_seed: int
    training_stage: str
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
    # Routing MASAC 当前 Episode 的梯度更新次数。
    routing_update_count: int = 0

    # 所有 Local Host SAC 在当前 Episode
    # 合计完成的梯度更新次数。
    host_update_count: int = 0
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

        if action_type == "self":
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
    def record_routing_update(
            self,
            update_info: Dict[
                str,
                float,
            ],
    ) -> None:

        self.routing_update_count += 1

        for metric_name, metric_value in (
                update_info.items()
        ):

            metric_value = float(
                metric_value
            )

            if not np.isfinite(
                    metric_value
            ):
                continue

            self.update_metric_sums[
                metric_name
            ] = (
                    self.update_metric_sums.get(
                        metric_name,
                        0.0,
                    )
                    + metric_value
            )

            self.update_metric_counts[
                metric_name
            ] = (
                    self.update_metric_counts.get(
                        metric_name,
                        0,
                    )
                    + 1
            )

    # 返回某个训练指标在当前episode中的平均值
    def mean_metric(self, metric_name: str) -> float:
        count = self.update_metric_counts.get(metric_name, 0)
        if count == 0:
            return float("nan")
        return float(
            self.update_metric_sums[metric_name] / count
        )

def validate_training_stage_config(
        train_config: TrainConfig,
) -> None:

    host_pretrain = int(
        train_config
        .host_pretrain_episodes
    )

    routing_train = int(
        train_config
        .routing_train_episodes
    )

    joint_finetune = int(
        train_config
        .joint_finetune_episodes
    )

    if min(
        host_pretrain,
        routing_train,
        joint_finetune,
    ) < 0:
        raise ValueError(
            "三阶段 Episode 数不能为负数。"
        )

    configured_total = (
        host_pretrain
        + routing_train
        + joint_finetune
    )

    if (
        configured_total
        != int(
            train_config.num_episodes
        )
    ):
        raise ValueError(
            "三阶段 Episode 总数与 "
            "num_episodes 不一致："
            f"stages={configured_total}, "
            f"num_episodes="
            f"{train_config.num_episodes}"
        )

def resolve_training_stage(
        episode: int,
        train_config: TrainConfig,
) -> TrainingStage:

    episode = int(
        episode
    )

    host_end = int(
        train_config
        .host_pretrain_episodes
    )

    routing_end = (
        host_end
        + int(
            train_config
            .routing_train_episodes
        )
    )

    if episode <= host_end:
        return (
            TrainingStage
            .HOST_PRETRAIN
        )

    if episode <= routing_end:
        return (
            TrainingStage
            .ROUTING_TRAIN
        )

    return (
        TrainingStage
        .JOINT_FINETUNE
    )

def training_stage_start_episode(
        stage: TrainingStage,
        train_config: TrainConfig,
) -> int:
    """
    返回某个 Training Stage 的第一个 Episode。

    Episode 使用 1-based 编号：
        Stage 1: 1
        Stage 2: HOST_PRETRAIN_EPISODES + 1
        Stage 3: HOST_PRETRAIN_EPISODES
                 + ROUTING_TRAIN_EPISODES + 1
    """

    host_end = int(
        train_config.host_pretrain_episodes
    )

    routing_end = (
        host_end
        + int(
            train_config.routing_train_episodes
        )
    )

    if stage == TrainingStage.HOST_PRETRAIN:
        return 1

    if stage == TrainingStage.ROUTING_TRAIN:
        return host_end + 1

    return routing_end + 1


def training_stage_end_episode(
        stage: TrainingStage,
        train_config: TrainConfig,
) -> int:
    """
    返回某个 Training Stage 的最后一个 Episode。
    """

    host_end = int(
        train_config.host_pretrain_episodes
    )

    routing_end = (
        host_end
        + int(
            train_config.routing_train_episodes
        )
    )

    if stage == TrainingStage.HOST_PRETRAIN:
        return host_end

    if stage == TrainingStage.ROUTING_TRAIN:
        return routing_end

    return int(
        train_config.num_episodes
    )


def training_stage_boundary_checkpoint_name(
        stage: TrainingStage,
) -> Optional[str]:
    """
    为前两个训练阶段生成固定的阶段结束 checkpoint。

    Stage 3 已经由训练器最终的 final.pt 保存，
    因此这里不重复保存。
    """

    if stage == TrainingStage.HOST_PRETRAIN:
        return "host_pretrain_final.pt"

    if stage == TrainingStage.ROUTING_TRAIN:
        return "routing_train_final.pt"

    return None

def stage_trains_routing(
        stage: TrainingStage,
) -> bool:

    return stage in {
        TrainingStage.ROUTING_TRAIN,
        TrainingStage.JOINT_FINETUNE,
    }


def stage_trains_host(
        stage: TrainingStage,
) -> bool:

    return stage in {
        TrainingStage.HOST_PRETRAIN,
        TrainingStage.JOINT_FINETUNE,
    }

def apply_training_stage_modes(
        stage: TrainingStage,
        routing_masac: RoutingMASAC,
        host_sac_agents:
        Dict[str, LocalHostSAC],
) -> None:
    """
    根据 Training Stage 明确设置两层网络模式。

    真正是否更新参数仍由 Orchestrator
    是否调用 update() 决定。
    """

    if (
        stage
        == TrainingStage.HOST_PRETRAIN
    ):

        routing_masac.eval_mode()

        for host_agent in (
            host_sac_agents.values()
        ):
            host_agent.train_mode()

        return

    if (
        stage
        == TrainingStage.ROUTING_TRAIN
    ):

        routing_masac.train_mode()

        for host_agent in (
            host_sac_agents.values()
        ):
            host_agent.eval_mode()

        return

    # Joint fine-tune
    routing_masac.train_mode()

    for host_agent in (
        host_sac_agents.values()
    ):
        host_agent.train_mode()

def get_self_routing_action(
        env: CloudEdgeEnv,
        agent_id: str,
) -> int:
    """
    Stage 1 Host Pretrain 时，
    Orchestrator 直接让当前 Job 进入当前 DC Host 层。

    不是 action mask，
    也不是 Environment forced action。
    """

    agent_id = str(
        agent_id
    )

    action = (
        env.routing_dc_id_to_action
        .get(
            agent_id
        )
    )

    if action is None:
        raise RuntimeError(
            "找不到当前 DC 对应的 Self "
            "Routing action："
            f"dc={agent_id}"
        )

    decoded = env._decode_action(
        agent_id=agent_id,
        action=int(action),
    )

    if (
        str(
            decoded["action_type"]
        )
        != "self"
    ):
        raise RuntimeError(
            "Stage 1 Self bypass "
            "动作编码错误："
            f"dc={agent_id}, "
            f"action={action}"
        )

    return int(
        action
    )

def choose_random_host_action(
        action_dim: int,
        rng: np.random.Generator,
) -> int:

    action_dim = int(
        action_dim
    )

    if action_dim <= 0:
        raise ValueError(
            "Host action_dim 必须 > 0"
        )

    return int(
        rng.integers(
            low=0,
            high=action_dim,
        )
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
        stats.record_routing_update(
            cpu_update_info
        )

def flush_finalized_trace_to_replay(
        finalized_trace: FinalizedJobTrace,
        routing_replay_buffer:
        RoutingReplayBuffer,
        host_replay_buffers:
        Dict[str, HostReplayBuffer],
) -> None:
    """
    把一个已经完整 Finalize 的 Job
    一次性写入两个正式 Replay System。

    顺序：

        FinalizedJobTrace
            ↓
        RoutingTransitions
            ↓
        RoutingReplayBuffer

    如果该 Job 最终 Routing=Self：

        HostTransition
            ↓
        对应 DC 的 HostReplayBuffer

    Cloud / Drop：
        不产生 HostTransition。
    """

    # ==========================================================
    # 先检查 Host Buffer 是否存在。
    #
    # 在真正写 Routing Replay 前先检查，
    # 避免 Host buffer 配置错误造成半写入状态。
    # ==========================================================

    host_transition = (
        finalized_trace
        .host_transition
    )

    host_replay_buffer = None

    if host_transition is not None:

        host_dc_id = str(
            host_transition.dc_id
        )

        host_replay_buffer = (
            host_replay_buffers.get(
                host_dc_id
            )
        )

        if host_replay_buffer is None:
            raise RuntimeError(
                "找不到 HostTransition 对应的 "
                "HostReplayBuffer："
                f"job={finalized_trace.job_id}, "
                f"dc={host_dc_id}"
            )

    # ==========================================================
    # Routing Replay
    # ==========================================================

    for routing_transition in (
        finalized_trace
        .routing_transitions
    ):

        routing_replay_buffer.add(
            routing_transition
        )

    # ==========================================================
    # Host Replay
    #
    # 一个 Self Job 最多写一条。
    # ==========================================================

    if (
        host_transition is not None
        and host_replay_buffer is not None
    ):

        host_replay_buffer.add(
            host_transition
        )

def consume_environment_reward_corrections(
        env: CloudEdgeEnv,
        pending_trace_store:
        PendingJobTraceStore,
        routing_replay_buffer:
        RoutingReplayBuffer,
        host_replay_buffers:
        Dict[str, HostReplayBuffer],
        stats: EpisodeStatistics,
) -> None:
    """
    第十九步以后：

        Environment delayed outcome
                ↓
        Pending Job Causal Trace
                ↓
        Job terminal
                ↓
        FinalizedJobTrace
                ↓
        ┌──────────────────────┐
        │                      │
        ▼                      ▼
    RoutingReplay        HostReplay

    Job 没有 terminal 前：
        不允许修改正式 ReplayBuffer。
    """

    reward_corrections = (
        env.pop_reward_corrections()
    )

    if not reward_corrections:
        return

    for correction in reward_corrections:

        job_id = str(
            correction["job_id"]
        )

        reward_delta = float(
            correction["reward_delta"]
        )

        reason = str(
            correction.get(
                "reason",
                "",
            )
        )

        env_time = float(
            correction.get(
                "env_time",
                env.current_time,
            )
        )

        is_terminal = bool(
            reason == "completed"
            or reason
            in TERMINAL_FAILURE_REASONS
        )

        # ======================================================
        # 1. Reward Event 永远先进入因果链。
        # ======================================================

        pending_trace_store.record_reward_event(
            job_id=job_id,
            env_time=env_time,
            reward_delta=reward_delta,
            reason=reason,
            terminal=is_terminal,
        )

        # ======================================================
        # 2. Episode statistics：
        #
        # 这里只做统计归因。
        # 不再修改任何 ReplayBuffer。
        # ======================================================

        if is_terminal:

            finalized_trace = (
                pending_trace_store
                .finalize_terminal_trace(
                    job_id=job_id
                )
            )

            correction_agent_id = None

            if (
                finalized_trace
                .routing_transitions
            ):
                correction_agent_id = str(
                    finalized_trace
                    .routing_transitions[-1]
                    .agent_id
                )

            # ==================================================
            # 3. Job terminal 后，
            #    一次性写入两个正式经验池。
            # ==================================================

            flush_finalized_trace_to_replay(
                finalized_trace=(
                    finalized_trace
                ),

                routing_replay_buffer=(
                    routing_replay_buffer
                ),

                host_replay_buffers=(
                    host_replay_buffers
                ),
            )

            # ==================================================
            # 4. Replay 写入成功以后，
            #    才允许从 Finalized Trace Store 移除。
            # ==================================================

            pending_trace_store.pop_finalized_trace(
                job_id
            )

            if correction_agent_id is not None:

                stats.record_reward_correction(
                    agent_id=(
                        correction_agent_id
                    ),

                    reward_delta=(
                        reward_delta
                    ),
                )

            continue

        # ======================================================
        # Non-terminal delayed reward：
        #
        # 只记录在 Causal Trace。
        # 不修改 Replay。
        #
        # Episode statistics 仍然正常记录。
        # ======================================================

        pending_trace = (
            pending_trace_store
            .get_trace(
                job_id
            )
        )

        correction_agent_id = None

        for routing_step in reversed(
            pending_trace.routing_steps
        ):

            if (
                routing_step.action_source
                != "forced"
            ):
                correction_agent_id = str(
                    routing_step.agent_id
                )
                break

        if correction_agent_id is not None:

            stats.record_reward_correction(
                agent_id=(
                    correction_agent_id
                ),

                reward_delta=(
                    reward_delta
                ),
            )

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
def choose_random_routing_action(
    action_dim: int,
    rng: np.random.Generator,
) -> int:

    action_dim = int(
        action_dim
    )

    if action_dim <= 0:
        raise ValueError(
            "Routing action_dim 必须大于 0。"
        )

    return int(
        rng.integers(
            low=0,
            high=action_dim,
        )
    )



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

# 功能函数计算安全比例，避免分母为0日志出问题
def safe_ratio(numerator: float, denominator: float,) -> float:
    denominator = float(denominator)
    if denominator <= 0.0:
        return 0.0
    return float(float(numerator) / denominator)

# 日志用，计算资源占用情况
def calculate_resource_timeline_metrics(jobs: list, episode_duration_s: float,) -> Dict[str, float]:
    episode_duration_s = max(float(episode_duration_s), 0.0,)
    cpu_resource_seconds = 0.0
    gpu_resource_seconds = 0.0
    running_job_seconds = 0.0
    events: Dict[float, list] = {}
    valid_running_jobs = 0

    for job in jobs:
        start_time = getattr(job, "start_time", None,)
        finish_time = getattr(job, "finish_time", None,)

        if (start_time is None or finish_time is None):
            continue

        start_time = float(start_time)
        finish_time = float(finish_time)
        run_time_s = max(finish_time - start_time, 0.0,)
        cpu_request = float(getattr(job, "cpu_request", 0.0,))
        gpu_request = float(getattr( job, "gpu_request", 0.0,))

        cpu_resource_seconds += (cpu_request * run_time_s)
        gpu_resource_seconds += (gpu_request * run_time_s)
        running_job_seconds += (run_time_s)
        valid_running_jobs += 1

        if start_time not in events:
            events[start_time] = [0.0, 0.0, 0,]
        if finish_time not in events:
            events[finish_time] = [0.0, 0.0, 0,]

        # 任务开始：CPU/GPU/Running Count 增加。
        events[start_time][0] += (cpu_request)
        events[start_time][1] += (gpu_request)
        events[start_time][2] += 1

        # 任务结束：CPU/GPU/Running Count 释放。
        events[finish_time][0] -= (cpu_request)
        events[finish_time][1] -= (gpu_request)
        events[finish_time][2] -= 1

    current_used_cpu = 0.0
    current_used_gpu = 0.0
    current_running_jobs = 0

    peak_used_cpu = 0.0
    peak_used_gpu = 0.0
    peak_running_jobs = 0

    busy_time_s = 0.0
    previous_event_time = 0.0

    for event_time in sorted(events.keys()):
        event_time = float(event_time)
        interval_s = max(event_time - previous_event_time, 0.0,)

        # 在前一个事件到当前事件之间，如果至少有一个 Job 在运行，则 Host/System 是 busy 状态。
        if current_running_jobs > 0:
            busy_time_s += (interval_s)

        cpu_delta, gpu_delta, running_delta = (events[event_time])

        current_used_cpu += float(cpu_delta)
        current_used_gpu += float(gpu_delta)
        current_running_jobs += int(running_delta)
        # 防止浮点残差产生 -1e-15 一类值。
        current_used_cpu = max(current_used_cpu, 0.0,)
        current_used_gpu = max(current_used_gpu, 0.0,)
        current_running_jobs = max(current_running_jobs, 0,)

        peak_used_cpu = max(peak_used_cpu, current_used_cpu,)
        peak_used_gpu = max(peak_used_gpu, current_used_gpu,)
        peak_running_jobs = max(peak_running_jobs, current_running_jobs,)
        previous_event_time = (event_time)

    return {
        "valid_running_jobs":
            int(valid_running_jobs),
        "cpu_resource_seconds":
            float(cpu_resource_seconds),
        "gpu_resource_seconds":
            float(gpu_resource_seconds),
        "avg_used_cpu":
            safe_ratio(
                cpu_resource_seconds,
                episode_duration_s,
            ),
        "avg_used_gpu":
            safe_ratio(
                gpu_resource_seconds,
                episode_duration_s,
            ),
        "peak_used_cpu":
            float(peak_used_cpu),
        "peak_used_gpu":
            float(peak_used_gpu),
        "avg_running_jobs":
            safe_ratio(
                running_job_seconds,
                episode_duration_s,
            ),
        "peak_running_jobs":
            int(peak_running_jobs),
        "busy_time_s":
            float(busy_time_s),
        "busy_ratio":
            safe_ratio(
                busy_time_s,
                episode_duration_s,
            ),
    }

# Episode 级 Workload / Edge / Cloud Load 统计
def calculate_episode_load_metrics(env: Any,) -> Dict[str, Any]:

    episode_duration_s = max(float(getattr(env, "current_time", 0.0,)), 0.0,)
    all_jobs = list(getattr(env, "jobs", [],))
    cloud_id = str(getattr(env, "cloud_id", "cloud", ))

    cpu_requests = np.asarray([float(getattr(job, "cpu_request", 0.0,)) for job in all_jobs],dtype=np.float64,)
    gpu_requests = np.asarray([float(getattr(job, "gpu_request", 0.0,)) for job in all_jobs],dtype=np.float64,)
    durations = np.asarray([float( getattr( job, "duration", 0.0,)) for job in all_jobs], dtype=np.float64,)
    arrival_times = np.asarray([float(getattr(job, "arrive_time", 0.0,)) for job in all_jobs],dtype=np.float64,)

    edge_jobs = []
    cloud_jobs = []
    edge_total_cpu_capacity = 0.0
    edge_total_gpu_capacity = 0.0
    edge_host_cpu_avg_loads = []
    edge_host_gpu_avg_loads = []
    edge_host_busy_ratios = []
    edge_host_details = {}
    dc_load_details = {}

    for dc in getattr(env, "datacenters", [],):
        dc_id = str(dc.dc_id)
        is_cloud = (dc_id == cloud_id)
        dc_jobs = []
        dc_cpu_capacity = 0.0
        dc_gpu_capacity = 0.0

        for host_idx, host in enumerate(getattr(dc, "host_list", [],)):
            completed_queue = getattr(host, "completed_queue", None,)
            host_jobs = list(getattr(completed_queue, "_queue", [],))
            dc_jobs.extend(host_jobs)
            host_timeline = (
                calculate_resource_timeline_metrics(
                    jobs=host_jobs,
                    episode_duration_s=episode_duration_s,
                )
            )

            cpu_capacity = float(getattr(host, "cpu_num", 0.0,))
            gpu_capacity = float(getattr(host, "gpu_capacity_num", 0.0,))

            if not is_cloud:
                dc_cpu_capacity += (cpu_capacity)
                dc_gpu_capacity += (gpu_capacity)
                edge_total_cpu_capacity += (cpu_capacity)
                edge_total_gpu_capacity += (gpu_capacity)
                avg_cpu_load = safe_ratio(host_timeline["avg_used_cpu"], cpu_capacity,)
                peak_cpu_load = safe_ratio(host_timeline["peak_used_cpu"], cpu_capacity,)
                avg_gpu_load = safe_ratio(host_timeline["avg_used_gpu"], gpu_capacity,)
                peak_gpu_load = safe_ratio(host_timeline["peak_used_gpu"], gpu_capacity,)
                edge_host_cpu_avg_loads.append(avg_cpu_load)

                if gpu_capacity > 0.0:
                    edge_host_gpu_avg_loads.append(avg_gpu_load)

                edge_host_busy_ratios.append(float(host_timeline["busy_ratio"]))
                edge_host_details[
                    f"{dc_id}/{host.host_id}"
                ] = {
                    "cpu_capacity":cpu_capacity,
                    "gpu_capacity":gpu_capacity,
                    "completed_jobs":len(host_jobs),
                    "avg_cpu_load":avg_cpu_load,
                    "peak_cpu_load":peak_cpu_load,
                    "avg_gpu_load":avg_gpu_load,
                    "peak_gpu_load":peak_gpu_load,
                    "avg_running_jobs":host_timeline["avg_running_jobs"],
                    "peak_running_jobs":host_timeline["peak_running_jobs"],
                    "busy_ratio":host_timeline["busy_ratio"],
                }

        dc_timeline = (calculate_resource_timeline_metrics(jobs=dc_jobs, episode_duration_s=episode_duration_s,))

        if is_cloud:
            cloud_jobs.extend(dc_jobs)
            dc_load_details[dc_id] = {
                "completed_jobs":len(dc_jobs),
                "avg_used_cpu":dc_timeline["avg_used_cpu"],
                "peak_used_cpu":dc_timeline["peak_used_cpu"],
                "avg_used_gpu":dc_timeline["avg_used_gpu"],
                "peak_used_gpu":dc_timeline[ "peak_used_gpu"],
                "avg_running_jobs":dc_timeline["avg_running_jobs"],
                "peak_running_jobs":dc_timeline["peak_running_jobs"],
                "busy_ratio":dc_timeline[ "busy_ratio"],
            }

        else:
            edge_jobs.extend(dc_jobs)
            dc_load_details[dc_id] = {
                "cpu_capacity":dc_cpu_capacity,
                "gpu_capacity":dc_gpu_capacity,
                "completed_jobs":len(dc_jobs),
                "avg_cpu_load":safe_ratio(dc_timeline["avg_used_cpu"],dc_cpu_capacity,),
                "peak_cpu_load":safe_ratio(dc_timeline["peak_used_cpu"],dc_cpu_capacity,),
                "avg_gpu_load":safe_ratio(dc_timeline["avg_used_gpu"], dc_gpu_capacity,),
                "peak_gpu_load":safe_ratio(dc_timeline["peak_used_gpu"],dc_gpu_capacity,),
                "avg_running_jobs":dc_timeline["avg_running_jobs"],
                "peak_running_jobs": dc_timeline["peak_running_jobs"],
                "busy_ratio":dc_timeline["busy_ratio"],
            }

    edge_timeline = (calculate_resource_timeline_metrics(jobs=edge_jobs, episode_duration_s=episode_duration_s,))
    cloud_timeline = (calculate_resource_timeline_metrics(jobs=cloud_jobs,episode_duration_s=episode_duration_s,))
    edge_cpu_host_mean = (float(np.mean(edge_host_cpu_avg_loads))
        if edge_host_cpu_avg_loads
        else 0.0
    )
    edge_cpu_host_std = (float(np.std(edge_host_cpu_avg_loads))
        if edge_host_cpu_avg_loads
        else 0.0
    )
    edge_cpu_host_p95 = (float(np.percentile(edge_host_cpu_avg_loads,95,))
        if edge_host_cpu_avg_loads
        else 0.0
    )
    edge_gpu_host_mean = (float(np.mean(edge_host_gpu_avg_loads))
        if edge_host_gpu_avg_loads
        else 0.0
    )
    edge_gpu_host_std = (float(np.std(edge_host_gpu_avg_loads))
        if edge_host_gpu_avg_loads
        else 0.0
    )

    if len(arrival_times) >= 2:
        arrival_span_s = float(np.max(arrival_times) - np.min(arrival_times))
        observed_arrival_rate = safe_ratio(len(arrival_times) - 1, arrival_span_s,)

    else:
        arrival_span_s = 0.0
        observed_arrival_rate = 0.0

    return {

        "workload_total_cpu_request":
            float(np.sum(cpu_requests))
            if cpu_requests.size
            else 0.0,

        "workload_mean_cpu_request":
            float(
                np.mean(cpu_requests)
            )
            if cpu_requests.size
            else 0.0,

        "workload_max_cpu_request":
            float(
                np.max(cpu_requests)
            )
            if cpu_requests.size
            else 0.0,

        "workload_total_gpu_request":
            float(
                np.sum(gpu_requests)
            )
            if gpu_requests.size
            else 0.0,

        "workload_mean_gpu_request":
            float(
                np.mean(gpu_requests)
            )
            if gpu_requests.size
            else 0.0,

        "workload_max_gpu_request":
            float(
                np.max(gpu_requests)
            )
            if gpu_requests.size
            else 0.0,

        "workload_gpu_job_ratio":
            float(
                np.mean(
                    gpu_requests > 0.0
                )
            )
            if gpu_requests.size
            else 0.0,

        "workload_total_duration_s":
            float(
                np.sum(durations)
            )
            if durations.size
            else 0.0,

        "workload_mean_duration_s":
            float(
                np.mean(durations)
            )
            if durations.size
            else 0.0,

        "workload_p95_duration_s":
            float(
                np.percentile(
                    durations,
                    95,
                )
            )
            if durations.size
            else 0.0,

        "workload_max_duration_s":
            float(
                np.max(durations)
            )
            if durations.size
            else 0.0,

        "workload_arrival_span_s":
            arrival_span_s,

        "workload_observed_arrival_rate":
            observed_arrival_rate,

        "edge_completed_jobs":
            int(
                len(edge_jobs)
            ),

        "edge_total_cpu_capacity":
            float(
                edge_total_cpu_capacity
            ),

        "edge_total_gpu_capacity":
            float(
                edge_total_gpu_capacity
            ),

        "edge_cpu_resource_seconds":
            float(
                edge_timeline[
                    "cpu_resource_seconds"
                ]
            ),

        "edge_gpu_resource_seconds":
            float(
                edge_timeline[
                    "gpu_resource_seconds"
                ]
            ),

        # 整个 Edge 系统容量加权、时间加权平均负载。
        "edge_avg_cpu_load":
            safe_ratio(
                edge_timeline[
                    "avg_used_cpu"
                ],
                edge_total_cpu_capacity,
            ),

        "edge_peak_cpu_load":
            safe_ratio(
                edge_timeline[
                    "peak_used_cpu"
                ],
                edge_total_cpu_capacity,
            ),

        "edge_avg_gpu_load":
            safe_ratio(
                edge_timeline[
                    "avg_used_gpu"
                ],
                edge_total_gpu_capacity,
            ),

        "edge_peak_gpu_load":
            safe_ratio(
                edge_timeline[
                    "peak_used_gpu"
                ],
                edge_total_gpu_capacity,
            ),

        "edge_avg_running_jobs":
            float(
                edge_timeline[
                    "avg_running_jobs"
                ]
            ),

        "edge_peak_running_jobs":
            int(
                edge_timeline[
                    "peak_running_jobs"
                ]
            ),

        # 每台 Edge Host 平均 CPU Load 的均值、标准差、P95。
        #
        # std 越大说明长期负载越不均衡。
        "edge_host_avg_cpu_load_mean":
            edge_cpu_host_mean,

        "edge_host_avg_cpu_load_std":
            edge_cpu_host_std,

        "edge_host_avg_cpu_load_p95":
            edge_cpu_host_p95,

        "edge_host_avg_gpu_load_mean":
            edge_gpu_host_mean,

        "edge_host_avg_gpu_load_std":
            edge_gpu_host_std,

        "edge_host_busy_ratio_mean":
            float(
                np.mean(
                    edge_host_busy_ratios
                )
            )
            if edge_host_busy_ratios
            else 0.0,

        "edge_host_busy_ratio_max":
            float(
                np.max(
                    edge_host_busy_ratios
                )
            )
            if edge_host_busy_ratios
            else 0.0,

        "cloud_completed_jobs":
            int(
                len(cloud_jobs)
            ),

        "cloud_cpu_resource_seconds":
            float(
                cloud_timeline[
                    "cpu_resource_seconds"
                ]
            ),

        "cloud_gpu_resource_seconds":
            float(
                cloud_timeline[
                    "gpu_resource_seconds"
                ]
            ),

        "cloud_avg_used_cpu":
            float(
                cloud_timeline[
                    "avg_used_cpu"
                ]
            ),

        "cloud_peak_used_cpu":
            float(
                cloud_timeline[
                    "peak_used_cpu"
                ]
            ),

        "cloud_avg_used_gpu":
            float(
                cloud_timeline[
                    "avg_used_gpu"
                ]
            ),

        "cloud_peak_used_gpu":
            float(
                cloud_timeline[
                    "peak_used_gpu"
                ]
            ),

        "cloud_avg_running_jobs":
            float(
                cloud_timeline[
                    "avg_running_jobs"
                ]
            ),

        "cloud_peak_running_jobs":
            int(
                cloud_timeline[
                    "peak_running_jobs"
                ]
            ),

        "cloud_busy_ratio":
            float(
                cloud_timeline[
                    "busy_ratio"
                ]
            ),

        "dc_load_details":
            json.dumps(
                dc_load_details,
                ensure_ascii=False,
                sort_keys=True,
            ),

        "edge_host_load_details":
            json.dumps(
                edge_host_details,
                ensure_ascii=False,
                sort_keys=True,
            ),
    }

# Episode Energy / Power 统计
def calculate_episode_energy_metrics(env: Any,) -> Dict[str, float]:

    simulation_end_time_s = float(getattr(env,"current_time", 0.0,))

    energy_end_time_s = float(
        getattr(
            env,
            "last_energy_update_time",
            simulation_end_time_s,
        )
    )

    energy_time_s = max(
        energy_end_time_s,
        0.0,
    )


    edge_idle_energy_j = float(
        getattr(
            env,
            "edge_idle_energy_j",
            0.0,
        )
    )

    edge_cpu_dynamic_energy_j = float(
        getattr(
            env,
            "edge_cpu_dynamic_energy_j",
            0.0,
        )
    )

    edge_gpu_dynamic_energy_j = float(
        getattr(
            env,
            "edge_gpu_dynamic_energy_j",
            0.0,
        )
    )

    cloud_compute_energy_j = float(
        getattr(
            env,
            "cloud_compute_energy_j",
            0.0,
        )
    )

    transfer_energy_j = float(
        getattr(
            env,
            "transfer_energy_j",
            0.0,
        )
    )

    edge_edge_transfer_energy_j = float(
        getattr(
            env,
            "edge_edge_transfer_energy_j",
            0.0,
        )
    )

    edge_cloud_transfer_energy_j = float(
        getattr(
            env,
            "edge_cloud_transfer_energy_j",
            0.0,
        )
    )


    edge_total_energy_j = (
        edge_idle_energy_j
        + edge_cpu_dynamic_energy_j
        + edge_gpu_dynamic_energy_j
    )

    system_compute_energy_j = (
        edge_total_energy_j
        + cloud_compute_energy_j
    )

    system_dynamic_compute_energy_j = (
        edge_cpu_dynamic_energy_j
        + edge_gpu_dynamic_energy_j
        + cloud_compute_energy_j
    )

    total_system_energy_j = (
        system_compute_energy_j
        + transfer_energy_j
    )


    all_jobs = list(
        getattr(
            env,
            "jobs",
            [],
        )
    )

    task_compute_energy_j = sum(
        float(
            getattr(
                job,
                "compute_energy_j",
                0.0,
            )
        )
        for job in all_jobs
    )

    task_transfer_energy_j = sum(
        float(
            getattr(
                job,
                "transfer_energy_j",
                0.0,
            )
        )
        for job in all_jobs
    )

    task_edge_edge_transfer_energy_j = sum(
        float(
            getattr(
                job,
                "edge_edge_transfer_energy_j",
                0.0,
            )
        )
        for job in all_jobs
    )

    task_edge_cloud_transfer_energy_j = sum(
        float(
            getattr(
                job,
                "edge_cloud_transfer_energy_j",
                0.0,
            )
        )
        for job in all_jobs
    )

    task_attributable_energy_j = (
        task_compute_energy_j
        + task_transfer_energy_j
    )


    total_jobs = len(
        all_jobs
    )

    completed_jobs = count_completed_jobs(
        env
    )



    # 系统 Transmission 总量应该与 EE + EC 严格一致。
    transfer_split_gap_j = (
        transfer_energy_j
        - edge_edge_transfer_energy_j
        - edge_cloud_transfer_energy_j
    )

    # 系统账本与全部 Job 的 Transmission attribution
    # 正常情况下也应该一致。
    transfer_job_accounting_gap_j = (
        transfer_energy_j
        - task_transfer_energy_j
    )

    # Energy clock 与 Simulation clock 应在 Episode 结束时一致。
    energy_time_gap_s = (
        simulation_end_time_s
        - energy_end_time_s
    )


    return {

        "episode_energy_time_s":
            energy_time_s,

        "energy_time_gap_s":
            float(
                energy_time_gap_s
            ),

        "edge_idle_energy_j":
            edge_idle_energy_j,

        "edge_cpu_dynamic_energy_j":
            edge_cpu_dynamic_energy_j,

        "edge_gpu_dynamic_energy_j":
            edge_gpu_dynamic_energy_j,

        "edge_total_energy_j":
            edge_total_energy_j,

        "cloud_compute_energy_j":
            cloud_compute_energy_j,

        "system_compute_energy_j":
            system_compute_energy_j,

        "system_dynamic_compute_energy_j":
            system_dynamic_compute_energy_j,

        "transfer_energy_j":
            transfer_energy_j,

        "edge_edge_transfer_energy_j":
            edge_edge_transfer_energy_j,

        "edge_cloud_transfer_energy_j":
            edge_cloud_transfer_energy_j,

        "total_system_energy_j":
            total_system_energy_j,

        "total_system_energy_kwh":
            total_system_energy_j
            / 3_600_000.0,


        "edge_idle_avg_power_w":
            safe_ratio(
                edge_idle_energy_j,
                energy_time_s,
            ),

        "edge_cpu_dynamic_avg_power_w":
            safe_ratio(
                edge_cpu_dynamic_energy_j,
                energy_time_s,
            ),

        "edge_gpu_dynamic_avg_power_w":
            safe_ratio(
                edge_gpu_dynamic_energy_j,
                energy_time_s,
            ),

        "edge_total_avg_power_w":
            safe_ratio(
                edge_total_energy_j,
                energy_time_s,
            ),

        "cloud_compute_avg_power_w":
            safe_ratio(
                cloud_compute_energy_j,
                energy_time_s,
            ),

        "system_compute_avg_power_w":
            safe_ratio(
                system_compute_energy_j,
                energy_time_s,
            ),

        # Transmission 模型是离散一次性 Energy，
        # 这里写的是 Episode horizon 上的等效平均能量率，
        # 不是单条链路的瞬时物理功率。
        "transfer_equivalent_avg_power_w":
            safe_ratio(
                transfer_energy_j,
                energy_time_s,
            ),

        "system_total_equivalent_avg_power_w":
            safe_ratio(
                total_system_energy_j,
                energy_time_s,
            ),


        "edge_idle_energy_share":
            safe_ratio(
                edge_idle_energy_j,
                total_system_energy_j,
            ),

        "edge_cpu_dynamic_energy_share":
            safe_ratio(
                edge_cpu_dynamic_energy_j,
                total_system_energy_j,
            ),

        "edge_gpu_dynamic_energy_share":
            safe_ratio(
                edge_gpu_dynamic_energy_j,
                total_system_energy_j,
            ),

        "cloud_compute_energy_share":
            safe_ratio(
                cloud_compute_energy_j,
                total_system_energy_j,
            ),

        "transfer_energy_share":
            safe_ratio(
                transfer_energy_j,
                total_system_energy_j,
            ),


        "system_energy_per_completed_job_j":
            safe_ratio(
                total_system_energy_j,
                completed_jobs,
            ),

        "system_energy_per_total_job_j":
            safe_ratio(
                total_system_energy_j,
                total_jobs,
            ),

        "task_compute_energy_j":
            float(
                task_compute_energy_j
            ),

        "task_transfer_energy_j":
            float(
                task_transfer_energy_j
            ),

        "task_attributable_energy_j":
            float(
                task_attributable_energy_j
            ),

        "task_compute_energy_per_completed_job_j":
            safe_ratio(
                task_compute_energy_j,
                completed_jobs,
            ),

        "task_attributable_energy_per_total_job_j":
            safe_ratio(
                task_attributable_energy_j,
                total_jobs,
            ),


        "transfer_split_gap_j":
            float(
                transfer_split_gap_j
            ),

        "transfer_job_accounting_gap_j":
            float(
                transfer_job_accounting_gap_j
            ),

        "task_edge_edge_transfer_energy_j":
            float(
                task_edge_edge_transfer_energy_j
            ),

        "task_edge_cloud_transfer_energy_j":
            float(
                task_edge_cloud_transfer_energy_j
            ),
    }

def build_checkpoint_structure_metadata(
        env: CloudEdgeEnv,
        routing_masac: RoutingMASAC,
        host_sac_agents: Dict[
            str,
            LocalHostSAC,
        ],
) -> Dict[str, Any]:
    """
    构造 H-MASAC checkpoint 的结构身份信息。

    这些信息不是训练指标，而是判断：
        “当前运行环境是否仍然与该 checkpoint 兼容”

    的硬结构约束。

    特别需要保护：
        1. Cloud ON/OFF；
        2. Edge DC 数量及顺序；
        3. Routing action 语义及维度；
        4. Routing Observation / Global State 维度；
        5. 每个 DC 的 Host 数量及顺序；
        6. 每个 Host SAC 的 observation/action dimension。
    """

    edge_dc_ids = [
        str(dc_id)
        for dc_id
        in env.edge_dc_ids
    ]

    base_dc_map = {
        str(dc.dc_id): dc
        for dc
        in env.base_datacenters
    }

    host_ids_per_dc: Dict[
        str,
        list,
    ] = {}

    host_count_per_dc: Dict[
        str,
        int,
    ] = {}

    for dc_id in edge_dc_ids:
        dc = base_dc_map.get(
            dc_id
        )

        if dc is None:
            raise RuntimeError(
                "构造 checkpoint metadata 时 "
                "找不到 Edge DC："
                f"{dc_id}"
            )

        host_ids = [
            str(host.host_id)
            for host
            in dc.host_list
        ]

        host_ids_per_dc[
            dc_id
        ] = host_ids

        host_count_per_dc[
            dc_id
        ] = len(
            host_ids
        )

    host_model_metadata = {
        str(dc_id): {
            "obs_dim": int(
                host_agent.obs_dim
            ),

            "action_dim": int(
                host_agent.action_dim
            ),
        }
        for dc_id, host_agent
        in host_sac_agents.items()
    }

    return {
        "cloud_enabled": bool(
            env.enable_cloud_action
        ),

        # 顺序必须保存。
        # Routing Actor 中 agent one-hot 和 action index
        # 都依赖这些顺序。
        "edge_dc_ids": edge_dc_ids,

        "routing_action_target_dc_ids": [
            str(dc_id)
            for dc_id
            in env.routing_action_target_dc_ids
        ],

        # Host action index 同样依赖 host_list 顺序，
        # 不能只比较 Host 数量。
        "host_count_per_dc":
            host_count_per_dc,

        "host_ids_per_dc":
            host_ids_per_dc,

        "routing": {
            "local_obs_dim": int(
                routing_masac.local_obs_dim
            ),

            "global_state_dim": int(
                routing_masac.global_state_dim
            ),

            "action_dim": int(
                routing_masac.action_dim
            ),

            "num_agents": int(
                routing_masac.num_agents
            ),
        },

        "hosts":
            host_model_metadata,
    }

def validate_checkpoint_structure_metadata(
        checkpoint_metadata:
        Dict[str, Any],

        env: CloudEdgeEnv,

        routing_masac: RoutingMASAC,

        host_sac_agents:
        Dict[str, LocalHostSAC],
) -> None:
    """
    在真正加载任何网络参数以前，
    检查 checkpoint 与当前双层调度结构是否兼容。

    结构不一致时立即 fail-fast，
    禁止把语义不同的权重强行加载进当前模型。
    """

    schema_version = int(
        checkpoint_metadata.get(
            "schema_version",
            -1,
        )
    )

    if (
            schema_version
            != CHECKPOINT_SCHEMA_VERSION
    ):
        raise RuntimeError(
            "Checkpoint schema version 不兼容："
            f"saved={schema_version}, "
            f"current="
            f"{CHECKPOINT_SCHEMA_VERSION}"
        )

    architecture = str(
        checkpoint_metadata.get(
            "architecture",
            "",
        )
    )

    if (
            architecture
            != CHECKPOINT_ARCHITECTURE
    ):
        raise RuntimeError(
            "Checkpoint architecture 不兼容："
            f"saved={architecture!r}, "
            f"current="
            f"{CHECKPOINT_ARCHITECTURE!r}"
        )

    saved_structure = (
        checkpoint_metadata.get(
            "structure",
            {}
        )
    )

    current_structure = (
        build_checkpoint_structure_metadata(
            env=env,
            routing_masac=(
                routing_masac
            ),
            host_sac_agents=(
                host_sac_agents
            ),
        )
    )

    # ==========================================================
    # Cloud Action
    #
    # Cloud ON/OFF 会直接改变 Routing action space，
    # 因此不允许跨配置恢复。
    # ==========================================================

    saved_cloud_enabled = bool(
        saved_structure.get(
            "cloud_enabled",
            False,
        )
    )

    current_cloud_enabled = bool(
        current_structure[
            "cloud_enabled"
        ]
    )

    if (
            saved_cloud_enabled
            != current_cloud_enabled
    ):
        raise RuntimeError(
            "Checkpoint Cloud 配置不兼容："
            f"saved={saved_cloud_enabled}, "
            f"current={current_cloud_enabled}"
        )

    # ==========================================================
    # Edge DC identity / order
    # ==========================================================

    saved_edge_dc_ids = list(
        saved_structure.get(
            "edge_dc_ids",
            [],
        )
    )

    current_edge_dc_ids = list(
        current_structure[
            "edge_dc_ids"
        ]
    )

    if (
            saved_edge_dc_ids
            != current_edge_dc_ids
    ):
        raise RuntimeError(
            "Checkpoint Edge DC 列表或顺序不兼容："
            f"saved={saved_edge_dc_ids}, "
            f"current={current_edge_dc_ids}"
        )

    # ==========================================================
    # Routing action semantic mapping
    # ==========================================================

    saved_targets = list(
        saved_structure.get(
            "routing_action_target_dc_ids",
            [],
        )
    )

    current_targets = list(
        current_structure[
            "routing_action_target_dc_ids"
        ]
    )

    if (
            saved_targets
            != current_targets
    ):
        raise RuntimeError(
            "Checkpoint Routing action mapping 不兼容："
            f"saved={saved_targets}, "
            f"current={current_targets}"
        )

    # ==========================================================
    # Routing dimensions
    # ==========================================================

    saved_routing = (
        saved_structure.get(
            "routing",
            {}
        )
    )

    current_routing = (
        current_structure[
            "routing"
        ]
    )

    for field_name in (
        "local_obs_dim",
        "global_state_dim",
        "action_dim",
        "num_agents",
    ):
        if (
                int(
                    saved_routing.get(
                        field_name,
                        -1,
                    )
                )
                != int(
                    current_routing[
                        field_name
                    ]
                )
        ):
            raise RuntimeError(
                "Checkpoint Routing 结构不兼容："
                f"field={field_name}, "
                f"saved="
                f"{saved_routing.get(field_name)}, "
                f"current="
                f"{current_routing[field_name]}"
            )

    # ==========================================================
    # Host physical mapping
    # ==========================================================

    saved_host_ids = (
        saved_structure.get(
            "host_ids_per_dc",
            {}
        )
    )

    current_host_ids = (
        current_structure[
            "host_ids_per_dc"
        ]
    )

    if (
            saved_host_ids
            != current_host_ids
    ):
        raise RuntimeError(
            "Checkpoint Host ID / action mapping "
            "与当前环境不兼容。"
        )

    # ==========================================================
    # Host SAC dimensions
    # ==========================================================

    saved_hosts = (
        saved_structure.get(
            "hosts",
            {}
        )
    )

    current_hosts = (
        current_structure[
            "hosts"
        ]
    )

    if (
            set(saved_hosts.keys())
            != set(current_hosts.keys())
    ):
        raise RuntimeError(
            "Checkpoint Host SAC DC 集合不兼容："
            f"saved={sorted(saved_hosts.keys())}, "
            f"current={sorted(current_hosts.keys())}"
        )

    for dc_id in current_hosts.keys():
        for field_name in (
            "obs_dim",
            "action_dim",
        ):
            if (
                    int(
                        saved_hosts[
                            dc_id
                        ].get(
                            field_name,
                            -1,
                        )
                    )
                    != int(
                        current_hosts[
                            dc_id
                        ][
                            field_name
                        ]
                    )
            ):
                raise RuntimeError(
                    "Checkpoint Local Host SAC "
                    "结构不兼容："
                    f"dc={dc_id}, "
                    f"field={field_name}, "
                    f"saved="
                    f"{saved_hosts[dc_id].get(field_name)}, "
                    f"current="
                    f"{current_hosts[dc_id][field_name]}"
                )

# 根据模型文件路径生成配套的训练器状态 JSON 路径
def checkpoint_state_path(model_path: Path) -> Path:
    return model_path.with_suffix(".trainer.json")

def host_checkpoint_dir(
        routing_model_path: Path,
) -> Path:

    return routing_model_path.with_name(
        f"{routing_model_path.stem}"
        "_hosts"
    )

# 同时保存 MASAC 模型和训练主循环状态
def save_two_layer_checkpoint(
        env: CloudEdgeEnv,

        routing_masac: RoutingMASAC,

        host_sac_agents:
        Dict[str, LocalHostSAC],

        model_path: Path,

        training_stage: TrainingStage,

        train_config: TrainConfig,

        next_episode: int,

        global_decision_steps: int,

        routing_normal_action_steps: int,

        host_training_action_steps:
        Dict[str, int],

        best_episode_return: float,
) -> None:


    model_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    routing_masac.save(
        model_path
    )

    host_dir = host_checkpoint_dir(model_path)

    host_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    for dc_id, host_agent in host_sac_agents.items():
        host_agent.save(
            host_dir / f"{dc_id}.pt"
        )

    checkpoint_metadata = {
        "schema_version": int(
            CHECKPOINT_SCHEMA_VERSION
        ),

        "architecture":
            CHECKPOINT_ARCHITECTURE,

        "saved_training_stage":
            str(
                training_stage.value
            ),

        "structure":
            build_checkpoint_structure_metadata(
                env=env,

                routing_masac=(
                    routing_masac
                ),

                host_sac_agents=(
                    host_sac_agents
                ),
            ),

        # 训练阶段长度保存下来主要用于实验追溯。
        # 不把它作为神经网络结构兼容性的硬约束，
        # 因为后续可能人为延长 Joint Fine-tune。
        "training_schedule": {
            "num_episodes": int(
                train_config.num_episodes
            ),

            "host_pretrain_episodes": int(
                train_config
                    .host_pretrain_episodes
            ),

            "routing_train_episodes": int(
                train_config
                    .routing_train_episodes
            ),

            "joint_finetune_episodes": int(
                train_config
                    .joint_finetune_episodes
            ),
        },

        "trainer_state": {
            "next_episode": int(
                next_episode
            ),

            "global_decision_steps": int(
                global_decision_steps
            ),

            "routing_normal_action_steps":
                int(
                    routing_normal_action_steps
                ),

            # 作为全局诊断量保存。
            "routing_global_steps": int(
                routing_normal_action_steps
            ),

            "host_training_action_steps": {
                str(dc_id): int(
                    step_count
                )
                for dc_id, step_count
                in host_training_action_steps.items()
            },

            "host_global_steps": int(
                sum(
                    int(step_count)
                    for step_count
                    in host_training_action_steps.values()
                )
            ),

            "best_episode_return": float(
                best_episode_return
            ),
        },
    }

    state_path = (
        checkpoint_state_path(
            model_path
        )
    )

    state_path.write_text(
        json.dumps(
            checkpoint_metadata,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

def load_two_layer_checkpoint_if_needed(
        env: CloudEdgeEnv,

        train_config: TrainConfig,

        routing_masac: RoutingMASAC,

        host_sac_agents: Dict[
            str,
            LocalHostSAC,
        ],

        resume_checkpoint: Optional[str],
) -> Tuple[
        int,
        int,
        int,
        Dict[str, int],
        float,
]:
    """
    恢复完整 Two-Level Scheduler checkpoint。

    新的 Resume 顺序严格为：

        1. 检查 Routing checkpoint 是否存在；
        2. 检查 Host checkpoint 目录是否存在；
        3. 检查每个 DC 的 Host checkpoint 是否完整；
        4. 检查 trainer metadata 是否存在；
        5. 读取 checkpoint metadata；
        6. 检查 checkpoint schema / architecture；
        7. 检查当前 Environment 与 checkpoint 的结构兼容性；
        8. 检查 Trainer State；
        9. 所有检查通过后，才真正加载 Routing MASAC；
       10. 加载每个 DC 的 Local Host SAC；
       11. 恢复训练计数。

    这样可以避免：

        - Cloud ON/OFF 不一致；
        - Edge DC 数量或顺序变化；
        - Routing action mapping 变化；
        - Observation / Global State 维度变化；
        - Host 数量或顺序变化；
        - Host SAC action_dim 变化；
        - Trainer metadata 丢失；

    时仍然静默恢复旧 checkpoint。

    返回：
        start_episode
        global_decision_steps
        routing_normal_action_steps
        host_training_action_steps
        best_episode_return
    """

    # ==========================================================
    # 默认 Host Training Step
    #
    # 只有在“完全没有指定 Resume checkpoint”的情况下，
    # 才允许使用这些默认值开始全新训练。
    #
    # 一旦用户明确指定 checkpoint，
    # 就不允许因为 metadata 缺失而偷偷回到 0。
    # ==========================================================

    default_host_steps = {
        str(dc_id): 0
        for dc_id
        in host_sac_agents.keys()
    }

    # ==========================================================
    # 0. 没有指定 Resume checkpoint
    #
    # 这是唯一允许从 Episode 1 / Step 0 开始的情况。
    # ==========================================================

    if resume_checkpoint is None:
        return (
            1,                  # start_episode
            0,                  # global_decision_steps
            0,                  # routing_normal_action_steps
            default_host_steps,
            float("-inf"),      # best_episode_return
        )

    # ==========================================================
    # 1. Routing checkpoint 路径
    # ==========================================================

    model_path = Path(
        resume_checkpoint
    )

    if not model_path.exists():
        raise FileNotFoundError(
            "找不到 Routing MASAC checkpoint："
            f"{model_path}"
        )

    if not model_path.is_file():
        raise FileNotFoundError(
            "Routing MASAC checkpoint 不是有效文件："
            f"{model_path}"
        )

    # ==========================================================
    # 2. Host checkpoint 目录
    #
    # 例如：
    #
    #   latest.pt
    #
    # 对应：
    #
    #   latest_hosts/
    #       DC1.pt
    #       DC2.pt
    #       ...
    # ==========================================================

    host_dir = host_checkpoint_dir(
        model_path
    )

    if not host_dir.exists():
        raise FileNotFoundError(
            "恢复 Two-Level checkpoint 时找不到 "
            "Local Host SAC checkpoint 目录："
            f"{host_dir}"
        )

    if not host_dir.is_dir():
        raise FileNotFoundError(
            "Local Host SAC checkpoint 路径不是目录："
            f"{host_dir}"
        )

    # ==========================================================
    # 3. 在加载任何模型参数以前，
    #    先检查所有 DC 的 Host checkpoint 是否完整。
    #
    # 不能出现：
    #
    #   Routing 已经 load
    #       ↓
    #   DC3 Host checkpoint 不存在
    #       ↓
    #   当前进程模型进入“半恢复”状态
    #
    # 因此这里首先只检查文件，不修改模型。
    # ==========================================================

    expected_host_paths: Dict[
        str,
        Path,
    ] = {}

    for dc_id in host_sac_agents.keys():

        dc_id = str(
            dc_id
        )

        host_path = (
            host_dir
            / f"{dc_id}.pt"
        )

        expected_host_paths[
            dc_id
        ] = host_path

        if not host_path.exists():
            raise FileNotFoundError(
                "缺少 Local Host SAC checkpoint："
                f"dc={dc_id}, "
                f"path={host_path}"
            )

        if not host_path.is_file():
            raise FileNotFoundError(
                "Local Host SAC checkpoint "
                "不是有效文件："
                f"dc={dc_id}, "
                f"path={host_path}"
            )

    # ==========================================================
    # 4. Trainer Metadata
    #
    # 新 checkpoint 中：
    #
    #   *.trainer.json
    #
    # 不再是“可有可无”的辅助文件，
    # 而是整个 Two-Level checkpoint 的结构声明与
    # 完整写入标志。
    #
    # 因此缺少它时必须 fail-fast。
    # ==========================================================

    state_path = (
        checkpoint_state_path(
            model_path
        )
    )

    if not state_path.exists():
        raise FileNotFoundError(
            "Two-Level checkpoint 缺少必要的 "
            "trainer metadata："
            f"{state_path}"
        )

    if not state_path.is_file():
        raise FileNotFoundError(
            "Trainer metadata 不是有效文件："
            f"{state_path}"
        )

    # ==========================================================
    # 5. 读取完整 checkpoint metadata
    # ==========================================================

    try:
        checkpoint_metadata = json.loads(
            state_path.read_text(
                encoding="utf-8"
            )
        )
    except (
            json.JSONDecodeError,
            OSError,
    ) as exc:
        raise RuntimeError(
            "无法读取 Two-Level checkpoint "
            "trainer metadata："
            f"{state_path}"
        ) from exc

    if not isinstance(
            checkpoint_metadata,
            dict,
    ):
        raise RuntimeError(
            "Two-Level checkpoint metadata "
            "根节点必须是 dict："
            f"{state_path}"
        )

    # ==========================================================
    # 6. 验证 checkpoint 的结构身份
    #
    # validate_checkpoint_structure_metadata() 应检查：
    #
    #   schema_version
    #   architecture
    #   cloud_enabled
    #   edge_dc_ids
    #   routing_action_target_dc_ids
    #   Routing observation/state/action dimensions
    #   num_agents
    #   host_ids_per_dc
    #   每个 Host SAC obs_dim/action_dim
    #
    # 注意：
    # 这里仍然没有真正 load_state_dict()。
    # ==========================================================

    validate_checkpoint_structure_metadata(
        checkpoint_metadata=(
            checkpoint_metadata
        ),

        env=env,

        routing_masac=(
            routing_masac
        ),

        host_sac_agents=(
            host_sac_agents
        ),
    )

    # ==========================================================
    # 7. 检查 saved_training_stage
    #
    # 它主要用于：
    #   - checkpoint provenance；
    #   - Resume 日志；
    #   - 判断 checkpoint 是在哪个训练阶段产生的。
    #
    # Stage 本身不在这里强制要求与当前 TrainConfig
    # 完全相同，因为后续可能人为延长 Joint Fine-tune。
    # ==========================================================

    saved_training_stage_raw = (
        checkpoint_metadata.get(
            "saved_training_stage"
        )
    )

    if saved_training_stage_raw is None:
        raise RuntimeError(
            "Checkpoint metadata 缺少 "
            "saved_training_stage。"
        )

    try:
        saved_training_stage = (
            TrainingStage(
                str(
                    saved_training_stage_raw
                )
            )
        )
    except ValueError as exc:
        raise RuntimeError(
            "Checkpoint 中存在未知 Training Stage："
            f"{saved_training_stage_raw!r}"
        ) from exc

    # ==========================================================
    # 8. 检查 Training Schedule metadata
    #
    # Schedule 主要用于实验追溯。
    #
    # 不作为网络结构的硬兼容条件：
    # 例如可以在已有 checkpoint 基础上延长
    # JOINT_FINETUNE_EPISODES。
    #
    # 如果与当前配置不同，只给出明确警告。
    # ==========================================================

    saved_training_schedule = (
        checkpoint_metadata.get(
            "training_schedule"
        )
    )

    if not isinstance(
            saved_training_schedule,
            dict,
    ):
        raise RuntimeError(
            "Checkpoint metadata 缺少有效的 "
            "training_schedule。"
        )

    current_training_schedule = {
        "num_episodes": int(
            train_config.num_episodes
        ),

        "host_pretrain_episodes": int(
            train_config
                .host_pretrain_episodes
        ),

        "routing_train_episodes": int(
            train_config
                .routing_train_episodes
        ),

        "joint_finetune_episodes": int(
            train_config
                .joint_finetune_episodes
        ),
    }

    saved_schedule_normalized = {
        "num_episodes": int(
            saved_training_schedule.get(
                "num_episodes",
                -1,
            )
        ),

        "host_pretrain_episodes": int(
            saved_training_schedule.get(
                "host_pretrain_episodes",
                -1,
            )
        ),

        "routing_train_episodes": int(
            saved_training_schedule.get(
                "routing_train_episodes",
                -1,
            )
        ),

        "joint_finetune_episodes": int(
            saved_training_schedule.get(
                "joint_finetune_episodes",
                -1,
            )
        ),
    }

    if (
            saved_schedule_normalized
            != current_training_schedule
    ):
        print(
            "\n"
            "============================================================\n"
            "⚠️ Checkpoint Training Schedule 与当前配置不同\n"
            f"Saved   : {saved_schedule_normalized}\n"
            f"Current : {current_training_schedule}\n"
            "\n"
            "模型结构兼容，因此允许继续恢复；\n"
            "但请确认这是有意修改三阶段 Episode 配置。\n"
            "============================================================\n",
            flush=True,
        )

    # ==========================================================
    # 9. Trainer State 必须完整存在
    # ==========================================================

    trainer_state = (
        checkpoint_metadata.get(
            "trainer_state"
        )
    )

    if not isinstance(
            trainer_state,
            dict,
    ):
        raise RuntimeError(
            "Checkpoint metadata 缺少有效的 "
            "trainer_state。"
        )

    required_trainer_fields = {
        "next_episode",
        "global_decision_steps",
        "routing_normal_action_steps",
        "host_training_action_steps",
        "best_episode_return",
    }

    missing_trainer_fields = (
        required_trainer_fields
        - set(
            trainer_state.keys()
        )
    )

    if missing_trainer_fields:
        raise RuntimeError(
            "Checkpoint trainer_state 缺少必要字段："
            f"{sorted(missing_trainer_fields)}"
        )

    # ==========================================================
    # 10. 先解析 Trainer counters。
    #
    # 仍然没有加载模型。
    #
    # 目的是保证 metadata 有问题时，
    # 当前 Routing / Host 网络保持原始初始化状态。
    # ==========================================================

    start_episode = int(
        trainer_state[
            "next_episode"
        ]
    )

    global_decision_steps = int(
        trainer_state[
            "global_decision_steps"
        ]
    )

    routing_normal_action_steps = int(
        trainer_state[
            "routing_normal_action_steps"
        ]
    )

    best_episode_return = float(
        trainer_state[
            "best_episode_return"
        ]
    )

    # ==========================================================
    # next_episode 合法性检查
    #
    # num_episodes + 1 是允许的：
    #
    #   例如加载已经完成 Episode 1000 的 final checkpoint，
    #   next_episode 可以为 1001。
    # ==========================================================

    if start_episode < 1:
        raise RuntimeError(
            "Checkpoint next_episode 非法："
            f"{start_episode}"
        )

    if (
            start_episode
            > int(
                train_config.num_episodes
            ) + 1
    ):
        raise RuntimeError(
            "Checkpoint next_episode 超出当前训练范围："
            f"next_episode={start_episode}, "
            f"num_episodes="
            f"{train_config.num_episodes}"
        )

    if global_decision_steps < 0:
        raise RuntimeError(
            "Checkpoint global_decision_steps "
            "不能为负数："
            f"{global_decision_steps}"
        )

    if routing_normal_action_steps < 0:
        raise RuntimeError(
            "Checkpoint routing_normal_action_steps "
            "不能为负数："
            f"{routing_normal_action_steps}"
        )

    # ==========================================================
    # 11. Host Training Steps
    #
    # 每个 DC 的 counter 都必须存在。
    #
    # 不能像旧实现一样：
    #
    #   missing -> 0
    #
    # 因为这样 Resume 后某个 Host SAC 会错误地
    # 重新进入 random warmup。
    # ==========================================================

    saved_host_steps = (
        trainer_state[
            "host_training_action_steps"
        ]
    )

    if not isinstance(
            saved_host_steps,
            dict,
    ):
        raise RuntimeError(
            "Checkpoint "
            "host_training_action_steps "
            "必须是 dict。"
        )

    expected_host_dc_ids = {
        str(dc_id)
        for dc_id
        in host_sac_agents.keys()
    }

    saved_host_dc_ids = {
        str(dc_id)
        for dc_id
        in saved_host_steps.keys()
    }

    if (
            saved_host_dc_ids
            != expected_host_dc_ids
    ):
        raise RuntimeError(
            "Checkpoint Host training counter "
            "的 DC 集合与当前环境不一致："
            f"saved="
            f"{sorted(saved_host_dc_ids)}, "
            f"current="
            f"{sorted(expected_host_dc_ids)}"
        )

    host_training_action_steps: Dict[
        str,
        int,
    ] = {}

    for dc_id in sorted(
            expected_host_dc_ids
    ):

        step_count = int(
            saved_host_steps[
                dc_id
            ]
        )

        if step_count < 0:
            raise RuntimeError(
                "Checkpoint Host training step "
                "不能为负数："
                f"dc={dc_id}, "
                f"steps={step_count}"
            )

        host_training_action_steps[
            dc_id
        ] = step_count

    # ==========================================================
    # 到这里为止：
    #
    #   文件完整性
    #   Metadata
    #   Schema
    #   Environment structure
    #   Routing structure
    #   Host structure
    #   Training stage
    #   Trainer counters
    #
    # 已经全部验证成功。
    #
    # 从下面开始才允许真正修改当前模型参数。
    # ==========================================================

    # ==========================================================
    # 12. 加载 Routing MASAC
    # ==========================================================

    routing_masac.load(
        file_path=(
            model_path
        ),

        load_optimizers=True,
    )

    # ==========================================================
    # 13. 加载全部 Local Host SAC
    # ==========================================================

    for dc_id, host_agent in (
        host_sac_agents.items()
    ):

        dc_id = str(
            dc_id
        )

        host_path = (
            expected_host_paths[
                dc_id
            ]
        )

        host_agent.load(
            host_path,

            load_optimizers=True,
        )

    # ==========================================================
    # 14. Resume 成功信息
    # ==========================================================

    print(
        "\n"
        "============================================================\n"
        "✅ Two-Level H-MASAC checkpoint 恢复成功\n"
        f"Routing checkpoint : {model_path}\n"
        f"Host directory     : {host_dir}\n"
        f"Trainer metadata   : {state_path}\n"
        f"Saved stage        : {saved_training_stage.value}\n"
        f"Next episode       : {start_episode}\n"
        f"Global decisions   : {global_decision_steps}\n"
        f"Routing steps      : {routing_normal_action_steps}\n"
        f"Host steps         : {host_training_action_steps}\n"
        f"Best return        : {best_episode_return}\n"
        "============================================================\n",
        flush=True,
    )

    # ==========================================================
    # 15. 返回 Trainer Resume State
    # ==========================================================

    return (
        start_episode,
        global_decision_steps,
        routing_normal_action_steps,
        host_training_action_steps,
        best_episode_return,
    )

# 写训练日志用的
def build_run_log_path(base_log_path: str) -> Path:
    base_path = Path(base_log_path)
    if not base_path.is_absolute():
        base_path = (PROJECT_ROOT/base_path)
    base_path = base_path.resolve()
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
        csv_file.flush()
        os.fsync(csv_file.fileno())

# 把 episode 统计整理成固定结构的 CSV 行
def build_episode_log_row(
    stats: EpisodeStatistics,
    env: Any,

    routing_replay_buffer:
    RoutingReplayBuffer,

    host_replay_buffers:
    Dict[str, HostReplayBuffer],

    routing_masac: RoutingMASAC,

    global_decision_steps: int,
    routing_normal_action_steps: int,
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
    energy_metrics = (calculate_episode_energy_metrics(env))
    load_metrics = (calculate_episode_load_metrics(env))

    # 返回固定列顺序的字典。
    return {
        "episode": int(stats.episode),
        "episode_seed": int(stats.episode_seed),
        "training_stage": str(
            stats.training_stage
        ),
        "episode_return": float(stats.episode_return),
        "energy_normalization_j": float(conf.ENERGY_NORMALIZATION_J),
        "energy_cost_weight": float(conf.ENERGY_COST_WEIGHT),
        "energy_optimization_enabled": bool(float(conf.ENERGY_COST_WEIGHT) > 0.0),
        "decision_count": int(stats.decision_count),
        "normal_action_count": int(stats.normal_action_count),
        "forced_action_count": int(stats.forced_action_count),
        "random_action_count": int(stats.random_action_count),
        "policy_action_count": int(stats.policy_action_count),
        "local_action_count": int(stats.local_action_count),
        "edge_action_count": int(stats.edge_action_count),
        "cloud_action_count": int(stats.cloud_action_count),
        "drop_action_count": int(stats.drop_action_count),
        "local_action_rate": safe_ratio(stats.local_action_count, stats.decision_count,),
        "edge_action_rate": safe_ratio(stats.edge_action_count,stats.decision_count,),
        "cloud_action_rate": safe_ratio(stats.cloud_action_count,stats.decision_count,),
        "drop_action_rate": safe_ratio(stats.drop_action_count, stats.decision_count,),
        # "unknown_action_count": int(stats.unknown_action_count),
        "total_jobs": total_jobs,
        "completed_jobs": completed_jobs,
        "dropped_jobs": dropped_jobs,
        "completion_rate": safe_ratio(completed_jobs, total_jobs,),
        "drop_rate": safe_ratio(dropped_jobs, total_jobs,),
        "queue_admission_rate": safe_ratio( queued_jobs,total_jobs,),
        "waiting_timeout_drop_rate": safe_ratio(waiting_timeout_drops, total_jobs,),
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
        "routing_episode_updates": int(
    stats.routing_update_count
),

"host_episode_updates": int(
    stats.host_update_count
),
        "global_decision_steps": int(global_decision_steps),
        "routing_normal_action_steps": int(
            routing_normal_action_steps
        ),
        "routing_replay_size": int(len(routing_replay_buffer)),
        "routing_replay_trainable_size": int(routing_replay_buffer.num_trainable_actions),
        "host_replay_size_total": int(sum(len(buffer) for buffer in host_replay_buffers.values())),
        "host_replay_size_by_dc": json.dumps({dc_id: int(len(buffer)) for dc_id, buffer in host_replay_buffers.items()},ensure_ascii=False,sort_keys=True,),
        "routing_masac_update_step": int(
    routing_masac.update_step
),
        "critic_loss": stats.mean_metric("critic_loss"),
        "q1_loss": stats.mean_metric("q1_loss"),
        "q2_loss": stats.mean_metric("q2_loss"),
        "actor_loss": stats.mean_metric("actor_loss"),
        "alpha_loss": stats.mean_metric("alpha_loss"),
        "alpha": float( routing_masac.alpha.detach().cpu().item()),
        "policy_entropy": stats.mean_metric("policy_entropy"),
        "target_entropy": stats.mean_metric("target_entropy"),
        "mean_q1": stats.mean_metric("mean_q1"),
        "mean_q2": stats.mean_metric("mean_q2"),
        "mean_target_q": stats.mean_metric("mean_target_q"),
        "wall_time_seconds": float(wall_time_seconds),
        "workload_total_cpu_request":load_metrics["workload_total_cpu_request"],
        "workload_mean_cpu_request":load_metrics["workload_mean_cpu_request"],
        "workload_max_cpu_request":load_metrics["workload_max_cpu_request"],
        "workload_total_gpu_request":load_metrics["workload_total_gpu_request"],
        "workload_mean_gpu_request":load_metrics["workload_mean_gpu_request"],
        "workload_max_gpu_request":load_metrics["workload_max_gpu_request"],
        "workload_gpu_job_ratio":load_metrics["workload_gpu_job_ratio"],
        "workload_total_duration_s":load_metrics["workload_total_duration_s"],
        "workload_mean_duration_s": load_metrics["workload_mean_duration_s"],
        "workload_p95_duration_s":load_metrics["workload_p95_duration_s"],
        "workload_max_duration_s":load_metrics["workload_max_duration_s"],
        "workload_arrival_span_s":load_metrics["workload_arrival_span_s"],
        "workload_observed_arrival_rate":load_metrics["workload_observed_arrival_rate"],
        "edge_completed_jobs":load_metrics["edge_completed_jobs"],
        "edge_total_cpu_capacity":load_metrics["edge_total_cpu_capacity"],
        "edge_total_gpu_capacity":load_metrics["edge_total_gpu_capacity"],
        "edge_cpu_resource_seconds":load_metrics["edge_cpu_resource_seconds"],
        "edge_gpu_resource_seconds":load_metrics["edge_gpu_resource_seconds"],
        "edge_avg_cpu_load":load_metrics["edge_avg_cpu_load"],
        "edge_peak_cpu_load":load_metrics[ "edge_peak_cpu_load"],
        "edge_avg_gpu_load":load_metrics["edge_avg_gpu_load"],
        "edge_peak_gpu_load":load_metrics["edge_peak_gpu_load"],
        "edge_avg_running_jobs":load_metrics["edge_avg_running_jobs"],
        "edge_peak_running_jobs":load_metrics["edge_peak_running_jobs"],
        "edge_host_avg_cpu_load_mean":load_metrics["edge_host_avg_cpu_load_mean"],
        "edge_host_avg_cpu_load_std":load_metrics["edge_host_avg_cpu_load_std"],
        "edge_host_avg_cpu_load_p95":load_metrics["edge_host_avg_cpu_load_p95"],
        "edge_host_avg_gpu_load_mean": load_metrics[ "edge_host_avg_gpu_load_mean"],
        "edge_host_avg_gpu_load_std":load_metrics["edge_host_avg_gpu_load_std"],
        "edge_host_busy_ratio_mean":load_metrics["edge_host_busy_ratio_mean"],
        "edge_host_busy_ratio_max":load_metrics["edge_host_busy_ratio_max"],
        "cloud_completed_jobs":load_metrics["cloud_completed_jobs"],
        "cloud_cpu_resource_seconds":load_metrics[ "cloud_cpu_resource_seconds"],
        "cloud_gpu_resource_seconds":load_metrics[ "cloud_gpu_resource_seconds"],
        "cloud_avg_used_cpu":load_metrics["cloud_avg_used_cpu"],
        "cloud_peak_used_cpu":load_metrics[ "cloud_peak_used_cpu"],
        "cloud_avg_used_gpu": load_metrics["cloud_avg_used_gpu"],
        "cloud_peak_used_gpu":load_metrics["cloud_peak_used_gpu"],
        "cloud_avg_running_jobs":load_metrics["cloud_avg_running_jobs"],
        "cloud_peak_running_jobs":load_metrics["cloud_peak_running_jobs"],
        "cloud_busy_ratio": load_metrics["cloud_busy_ratio" ],
        "episode_energy_time_s":energy_metrics["episode_energy_time_s"],
        "energy_time_gap_s":energy_metrics["energy_time_gap_s"],
        "edge_idle_energy_j":energy_metrics["edge_idle_energy_j"],
        "edge_cpu_dynamic_energy_j":energy_metrics["edge_cpu_dynamic_energy_j"],
        "edge_gpu_dynamic_energy_j":energy_metrics["edge_gpu_dynamic_energy_j"],
        "edge_total_energy_j":energy_metrics["edge_total_energy_j"],
        "cloud_compute_energy_j":energy_metrics["cloud_compute_energy_j"],
        "system_compute_energy_j":energy_metrics["system_compute_energy_j"],
        "system_dynamic_compute_energy_j":energy_metrics["system_dynamic_compute_energy_j"],
        "transfer_energy_j":energy_metrics["transfer_energy_j"],
        "edge_edge_transfer_energy_j":energy_metrics["edge_edge_transfer_energy_j"],
        "edge_cloud_transfer_energy_j":energy_metrics["edge_cloud_transfer_energy_j"],
        "total_system_energy_j":energy_metrics["total_system_energy_j"],
        "total_system_energy_kwh":energy_metrics["total_system_energy_kwh"],
        "edge_idle_avg_power_w":
            energy_metrics[
                "edge_idle_avg_power_w"
            ],

        "edge_cpu_dynamic_avg_power_w":
            energy_metrics[
                "edge_cpu_dynamic_avg_power_w"
            ],

        "edge_gpu_dynamic_avg_power_w":
            energy_metrics[
                "edge_gpu_dynamic_avg_power_w"
            ],

        "edge_total_avg_power_w":
            energy_metrics[
                "edge_total_avg_power_w"
            ],

        "cloud_compute_avg_power_w":
            energy_metrics[
                "cloud_compute_avg_power_w"
            ],

        "system_compute_avg_power_w":
            energy_metrics[
                "system_compute_avg_power_w"
            ],

        "transfer_equivalent_avg_power_w":
            energy_metrics[
                "transfer_equivalent_avg_power_w"
            ],

        "system_total_equivalent_avg_power_w":
            energy_metrics[
                "system_total_equivalent_avg_power_w"
            ],
        "edge_idle_energy_share":
            energy_metrics[
                "edge_idle_energy_share"
            ],

        "edge_cpu_dynamic_energy_share":
            energy_metrics[
                "edge_cpu_dynamic_energy_share"
            ],

        "edge_gpu_dynamic_energy_share":
            energy_metrics[
                "edge_gpu_dynamic_energy_share"
            ],

        "cloud_compute_energy_share":
            energy_metrics[
                "cloud_compute_energy_share"
            ],

        "transfer_energy_share":
            energy_metrics[
                "transfer_energy_share"
            ],
        "task_compute_energy_j":
            energy_metrics[
                "task_compute_energy_j"
            ],

        "task_transfer_energy_j":
            energy_metrics[
                "task_transfer_energy_j"
            ],

        "task_attributable_energy_j":
            energy_metrics[
                "task_attributable_energy_j"
            ],

        "task_compute_energy_per_completed_job_j":
            energy_metrics[
                "task_compute_energy_per_completed_job_j"
            ],

        "task_attributable_energy_per_total_job_j":
            energy_metrics[
                "task_attributable_energy_per_total_job_j"
            ],

        "system_energy_per_completed_job_j":
            energy_metrics[
                "system_energy_per_completed_job_j"
            ],

        "system_energy_per_total_job_j":
            energy_metrics[
                "system_energy_per_total_job_j"
            ],
        "transfer_split_gap_j":
            energy_metrics[
                "transfer_split_gap_j"
            ],

        "transfer_job_accounting_gap_j":
            energy_metrics[
                "transfer_job_accounting_gap_j"
            ],
        "per_agent_returns": json.dumps(
            stats.per_agent_returns,
            ensure_ascii=False,
            sort_keys=True,
        ),
        # 每个 DC 的 Episode 负载详细信息。
        "dc_load_details":
            load_metrics[
                "dc_load_details"
            ],

        # 每一台 Edge Host 的 Episode 负载详细信息。
        "edge_host_load_details":
            load_metrics[
                "edge_host_load_details"
            ],

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
    # 打印 Episode 核心训练指标。
    print(
        f"Episode {int(row['episode']):5d} | "
        f"return={float(row['episode_return']):9.4f} | "
        f"sla_vio={float(row['sla_violation_rate']):6.2%} | "
        f"avg_T={float(row['avg_completion_time']):8.2f}s | "
        f"completed={int(row['completed_jobs']):5d} | "
        f"dropped={int(row['dropped_jobs']):5d} | "
        f"local={int(row['local_action_count']):5d} | "
        f"edge={int(row['edge_action_count']):5d} | "
        f"cloud={int(row['cloud_action_count']):5d}"
    )

    # 打印本轮负载情况。
    print(
        f"  Load   | "
        f"EdgeCPU(avg/peak)="
        f"{float(row['edge_avg_cpu_load']):6.2%}/"
        f"{float(row['edge_peak_cpu_load']):6.2%} | "
        f"EdgeGPU(avg/peak)="
        f"{float(row['edge_avg_gpu_load']):6.2%}/"
        f"{float(row['edge_peak_gpu_load']):6.2%} | "
        f"HostCPUStd="
        f"{float(row['edge_host_avg_cpu_load_std']):6.2%} | "
        f"CloudCPU(avg/peak)="
        f"{float(row['cloud_avg_used_cpu']):8.2f}/"
        f"{float(row['cloud_peak_used_cpu']):8.2f} | "
        f"CloudGPU(avg/peak)="
        f"{float(row['cloud_avg_used_gpu']):8.2f}/"
        f"{float(row['cloud_peak_used_gpu']):8.2f}"
    )

    # 打印本轮 Energy / Power。
    print(
        f"  Energy | "
        f"total="
        f"{float(row['total_system_energy_kwh']):10.6f} kWh | "
        f"per_job="
        f"{float(row['system_energy_per_completed_job_j']):10.2f} J | "
        f"avgP="
        f"{float(row['system_total_equivalent_avg_power_w']):10.2f} W | "
        f"EdgeIdle="
        f"{float(row['edge_idle_energy_share']):6.2%} | "
        f"EdgeCPU="
        f"{float(row['edge_cpu_dynamic_energy_share']):6.2%} | "
        f"EdgeGPU="
        f"{float(row['edge_gpu_dynamic_energy_share']):6.2%} | "
        f"Cloud="
        f"{float(row['cloud_compute_energy_share']):6.2%} | "
        f"Transfer="
        f"{float(row['transfer_energy_share']):6.2%}"
    )

    # MASAC 本身训练状态。
    print(
        f"  Routing MASAC | "
        f"routing_buffer="
        f"{int(row['routing_replay_trainable_size']):7d} | "
        f"host_buffer="
        f"{int(row['host_replay_size_total']):7d} | "
        f"routing_updates="
f"{int(row['routing_episode_updates']):5d} | "
f"host_updates="
f"{int(row['host_episode_updates']):5d} | "
        f"critic_loss={critic_loss_text} | "
        f"alpha={float(row['alpha']):.5f}"
    )



def train(
    train_config: TrainConfig,

    routing_masac_config:
    Optional[
        RoutingMASACConfig
    ] = None,

    host_sac_config:
    Optional[
        HostSACConfig
    ] = None,

) -> Tuple[
    RoutingMASAC,
    Dict[str, LocalHostSAC],
]:
    validate_training_stage_config(
        train_config
    )

    set_global_random_seeds(train_config.seed)

    # 创建初始环境
    env = build_environment(
        seed=train_config.seed,
        old_env_path=train_config.old_env_path,
    )
    host_observation_builder = (
        HostObservationBuilder(
            env=env,
        )
    )

    # ==========================================================
    # 每个 Edge DC 建立独立 Local Host SAC。
    #
    # Host 层：
    #   - 不使用 PettingZoo
    #   - 不共享 Actor
    #   - 不共享 Critic
    #   - 不使用 Mask
    # ==========================================================

    host_sac_agents: Dict[str, LocalHostSAC] = {}

    host_replay_buffers: Dict[
        str,
        HostReplayBuffer,
    ] = {}

    for host_dc_index, dc_id in enumerate(
            env.edge_dc_ids
    ):
        dc_id = str(dc_id)

        if host_sac_config is None:
            # ==========================================================
            # Local Host SAC fallback configuration
            #
            # 即使 train() 被其他入口直接调用，
            # 没有显式传入 HostSACConfig，
            # 也必须使用 HOST_* 专属参数。
            #
            # 禁止退回 Flat-MASAC / Routing 公共超参数。
            # ==========================================================

            host_sac_config = (
                HostSACConfig(
                    gamma=(
                        conf.HOST_GAMMA
                    ),

                    tau=(
                        conf.HOST_TAU
                    ),

                    actor_lr=(
                        conf.HOST_ACTOR_LR
                    ),

                    critic_lr=(
                        conf.HOST_CRITIC_LR
                    ),

                    alpha_lr=(
                        conf.HOST_ALPHA_LR
                    ),

                    actor_hidden_dim=(
                        conf.HOST_ACTOR_HIDDEN_DIM
                    ),

                    critic_hidden_dim=(
                        conf.HOST_CRITIC_HIDDEN_DIM
                    ),

                    initial_alpha=(
                        conf.HOST_INITIAL_ALPHA
                    ),

                    target_entropy_ratio=(
                        conf.HOST_TARGET_ENTROPY_RATIO
                    ),

                    max_grad_norm=(
                        conf.HOST_MAX_GRAD_NORM
                    ),

                    policy_update_interval=(
                        conf.HOST_POLICY_UPDATE_INTERVAL
                    ),

                    target_update_interval=(
                        conf.HOST_TARGET_UPDATE_INTERVAL
                    ),

                    device=(
                        conf.DEVICE
                    ),

                    seed=int(
                        train_config.seed
                    ),
                )
            )

        host_obs_dim = int(
            host_observation_builder
                .get_obs_dim(
                dc_id
            )
        )

        host_action_dim = int(
            host_observation_builder
                .get_action_dim(
                dc_id
            )
        )

        # ==========================================================
        # 每个 DC 独立 Local Host SAC。
        # ==========================================================

        host_sac_agents[
            dc_id
        ] = LocalHostSAC(
            obs_dim=(
                host_obs_dim
            ),

            action_dim=(
                host_action_dim
            ),

            config=(
                host_sac_config
            ),
        )

        # ==========================================================
        # 第十九步：
        # 每个 Local Host SAC 对应自己的 Host ReplayBuffer。
        #
        # 不共享：
        #   Actor
        #   Critic
        #   ReplayBuffer
        # ==========================================================

        host_replay_buffers[
            dc_id
        ] = HostReplayBuffer(
            dc_id=dc_id,

            capacity=int(
                train_config
                    .host_replay_capacity
            ),

            obs_dim=(
                host_obs_dim
            ),

            action_dim=(
                host_action_dim
            ),

            seed=(
                    int(train_config.seed)
                    + 20_000
                    + host_dc_index
            ),
        )
    host_training_action_steps: Dict[
        str,
        int,
    ] = {
        str(dc_id): 0
        for dc_id
        in env.edge_dc_ids
    }

    # ==============================================================
    # Local Host SAC Independent Action RNG
    #
    # 每个 DC 使用自己独立的 NumPy RNG，
    # 用于 Host random warmup 阶段随机选择 Host。
    #
    # 这样不同 DC 的随机 Host action stream 不会共用
    # 同一个随机生成器。
    # ==============================================================

    host_action_rngs: Dict[
        str,
        np.random.Generator,
    ] = {
        str(dc_id):
            np.random.default_rng(
                int(
                    train_config.seed
                )
                + 30_000
                + dc_index
            )

        for dc_index, dc_id
        in enumerate(
            env.edge_dc_ids
        )
    }


    routing_observation_builder = (
        RoutingObservationBuilder(
            env=env,

            # 当前默认 False：
            # 只保留 Feedback Observation 接口，
            # 不让历史结果参与 Routing 决策。
            use_neighbor_historical_feedback=(
                train_config
                    .use_neighbor_historical_feedback
            ),

            # 当前阶段尚未建立 Historical Feedback Store。
            neighbor_feedback_provider=None,
        )
    )
    routing_obs_dim = int(routing_observation_builder.obs_dim)

    routing_state_builder = (
        RoutingCentralizedStateBuilder(
            env=env,
            routing_observation_builder=(
                routing_observation_builder
            ),
        )
    )

    routing_global_state_dim = int(
        routing_state_builder.state_dim
    )

    pending_trace_store = PendingJobTraceStore()

    # 创建 Transition 采集器
    collector = TransitionCollector(
        env=env,

        routing_observation_builder=(
            routing_observation_builder
        ),

        routing_state_builder=(
            routing_state_builder
        ),

        pending_trace_store=(
            pending_trace_store
        ),
    )

    # ==============================================================
    # Routing MASAC 专用 ReplayBuffer
    #
    # 与 Host Replay 完全独立。
    #
    # 这里只接收 Job terminal 后生成的
    # Finalized RoutingTransition。
    # ==============================================================

    routing_replay_buffer = RoutingReplayBuffer(
        capacity=int(
            train_config
                .routing_replay_capacity
        ),

        local_obs_dim=(
            routing_obs_dim
        ),

        global_state_dim=(
            routing_global_state_dim
        ),

        seed=int(
            train_config.seed
        ),
    )


    # 没有传入算法配置时，使用 MASACConfig 默认值，但让算法随机种子与训练配置保持一致
    # ==============================================================
    # Routing MASAC + CTDE
    #
    # 这里创建的是整个系统唯一的一套 Routing MASAC。
    #
    # Host SAC 不使用本对象。
    # 每个 Edge DC 的 LocalHostSAC 已在前面独立创建。
    # ==============================================================

    if routing_masac_config is None:
        # ==========================================================
        # Routing MASAC fallback configuration
        #
        # config.py 是 H-MASAC 实验的统一参数入口。
        # 即使 train() 被直接调用，
        # Routing 也必须继续使用 ROUTING_* 配置。
        # ==========================================================

        routing_masac_config = (
            RoutingMASACConfig(
                gamma=(
                    conf.ROUTING_GAMMA
                ),

                tau=(
                    conf.ROUTING_TAU
                ),

                actor_lr=(
                    conf.ROUTING_ACTOR_LR
                ),

                critic_lr=(
                    conf.ROUTING_CRITIC_LR
                ),

                alpha_lr=(
                    conf.ROUTING_ALPHA_LR
                ),

                actor_hidden_dim=(
                    conf.ROUTING_ACTOR_HIDDEN_DIM
                ),

                critic_hidden_dim=(
                    conf.ROUTING_CRITIC_HIDDEN_DIM
                ),

                initial_alpha=(
                    conf.ROUTING_INITIAL_ALPHA
                ),

                target_entropy_ratio=(
                    conf.ROUTING_TARGET_ENTROPY_RATIO
                ),

                max_grad_norm=(
                    conf.ROUTING_MAX_GRAD_NORM
                ),

                policy_update_interval=(
                    conf.ROUTING_POLICY_UPDATE_INTERVAL
                ),

                target_update_interval=(
                    conf.ROUTING_TARGET_UPDATE_INTERVAL
                ),

                device=(
                    conf.DEVICE
                ),

                seed=int(
                    train_config.seed
                ),
            )
        )


    routing_masac = RoutingMASAC(
        # Routing Actor local input
        local_obs_dim=(
            routing_obs_dim
        ),

        # Routing CTDE centralized Critic input
        global_state_dim=(
            routing_global_state_dim
        ),

        action_dim=int(
            env.action_dim
        ),

        num_agents=int(
            len(
                env.possible_agents
            )
        ),

        config=(
            routing_masac_config
        ),
    )

    routing_masac.train_mode()

    action_rng = np.random.default_rng(int(train_config.seed))

    (
        start_episode,
        global_decision_steps,
        routing_normal_action_steps,
        host_training_action_steps,
        best_episode_return,
    ) = load_two_layer_checkpoint_if_needed(
        # ==========================================================
        # Resume 前需要当前 Environment / TrainConfig，
        # 用于验证 Cloud、DC、Host、Observation、
        # Action 以及三阶段训练 metadata。
        # ==========================================================

        env=env,

        train_config=(
            train_config
        ),

        routing_masac=(
            routing_masac
        ),

        host_sac_agents=(
            host_sac_agents
        ),

        resume_checkpoint=(
            train_config.resume_checkpoint
        ),
    )

    # 把保存路径转换成 Path
    checkpoint_dir = Path(train_config.checkpoint_dir)
    log_csv_path = build_run_log_path(train_config.log_csv_path)

    # 创建 checkpoint 目录
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    log_csv_path.parent.mkdir(parents=True, exist_ok=True,)

    current_log_pointer_path = (log_csv_path.parent / "current_train_log.txt")
    current_log_pointer_path.write_text(str(log_csv_path.resolve()), encoding="utf-8",)

    print(
        "\n"
        "============================================================\n"
        "📊 本次训练日志\n"
        f"CSV Path : {log_csv_path.resolve()}\n"
        f"Pointer  : {current_log_pointer_path.resolve()}\n"
        "\n"
        "每个 Episode 完成后会立即追加一行并写入磁盘。\n"
        "训练过程中可以直接读取该 CSV 查看已完成 Episode。\n"
        "============================================================\n",
        flush=True,
    )

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
            pending_trace_store.reset_episode()
            env.reset(seed=episode_seed)

            collector.reset_episode()

            training_stage = (
                resolve_training_stage(
                    episode=episode,
                    train_config=(
                        train_config
                    ),
                )
            )

            if (
                    episode
                    == training_stage_start_episode(
                stage=training_stage,
                train_config=train_config,
            )
            ):
                best_episode_return = float("-inf")

                print(
                    "\n"
                    "============================================================\n"
                    f"🚦 Training Stage Start: {training_stage.value}\n"
                    f"Episode Range: "
                    f"{training_stage_start_episode(training_stage, train_config)}"
                    f" -> "
                    f"{training_stage_end_episode(training_stage, train_config)}\n"
                    "============================================================\n",
                    flush=True,
                )

            apply_training_stage_modes(
                stage=(
                    training_stage
                ),

                routing_masac=(
                    routing_masac
                ),

                host_sac_agents=(
                    host_sac_agents
                ),
            )

            # 为每个智能体创建奖励累计字典
            per_agent_returns = {}
            for agent_id in env.possible_agents:
                per_agent_returns[str(agent_id)] = 0.0

            # 创建当前 episode 的统计对象
            stats = EpisodeStatistics(
                episode=int(
                    episode
                ),

                episode_seed=(
                    episode_seed
                ),

                per_agent_returns=(
                    per_agent_returns
                ),

                training_stage=(
                    training_stage.value
                ),
            )

            # 记录episode开始的时间
            episode_wall_start = time.perf_counter()
            ###################################################################
            decision: Optional[DecisionSnapshot] = None

            # 只要 PettingZoo 的活跃智能体列表不为空，就继续循环
            while env.agents:

                # ==========================================================
                # Phase 1：Host decision
                #
                # Host 必须优先于任何 PettingZoo 操作处理。
                # Self Routing 后：
                #
                #   agent_selection == None
                #   pending_host_job_id != None
                #
                # 因此此处绝对不能调用：
                #   collector.capture_decision()
                #   env.step()
                # ==========================================================
                if env.has_pending_host_decision():
                    # ==========================================================
                    # Host Phase
                    #
                    # Host 层完全脱离 PettingZoo。
                    # 这里使用与 Routing Collector 相同的
                    # pending_trace_store。
                    # ==========================================================

                    host_context = (
                        env.get_pending_host_decision()
                    )

                    host_job_id = str(
                        host_context["job_id"]
                    )

                    host_dc_id = str(
                        host_context["dc_id"]
                    )

                    # ==========================================================
                    # 保存 Host SAC 真正做决策的时间。
                    #
                    # execute_pending_host_action() 后环境可能已经向前推进，
                    # 因此不能在执行之后再读取 decision time。
                    # ==========================================================

                    host_decision_time = float(
                        env.current_time
                    )

                    # Host Observation 完全独立于 PettingZoo。
                    host_obs = (
                        host_observation_builder.build(
                            dc_id=host_dc_id,
                            job_id=host_job_id,
                        )
                    )

                    # 当前 DC 自己的 Local Host SAC 决策。
                    host_agent = (
                        host_sac_agents[
                            host_dc_id
                        ]
                    )

                    host_replay = (
                        host_replay_buffers[
                            host_dc_id
                        ]
                    )

                    # ==========================================================
                    # Training Stage 决定 Host action 行为。
                    # ==========================================================

                    if (
                            training_stage
                            == TrainingStage.ROUTING_TRAIN
                    ):
                        # ======================================================
                        # Stage 2：
                        #
                        # Host 网络完全冻结。
                        #
                        # 使用 deterministic policy，
                        # 降低 Routing MASAC 所面对环境的非平稳性。
                        # ======================================================

                        host_action = (
                            host_agent.select_action(
                                host_obs=host_obs,
                                deterministic=True,
                            )
                        )

                        host_action_source = (
                            "policy"
                        )

                    else:

                        # ======================================================
                        # Stage 1 / Stage 3：
                        #
                        # Host SAC 参与训练。
                        # 每个 DC 独立进行 random warmup。
                        # ======================================================

                        host_training_steps = int(
                            host_training_action_steps[
                                host_dc_id
                            ]
                        )

                        if (
                                host_training_steps
                                < int(
                            train_config
                                    .host_random_warmup_steps
                        )
                        ):

                            host_action = (
                                choose_random_host_action(
                                    action_dim=(
                                        host_agent.action_dim
                                    ),

                                    rng=(
                                        host_action_rngs[
                                            host_dc_id
                                        ]
                                    ),
                                )
                            )

                            host_action_source = (
                                "random"
                            )

                        else:

                            host_action = (
                                host_agent.select_action(
                                    host_obs=host_obs,
                                    deterministic=False,
                                )
                            )

                            host_action_source = (
                                "policy"
                            )

                        host_training_action_steps[
                            host_dc_id
                        ] += 1

                    # ==========================================================
                    # 非 PettingZoo Host execution。
                    #
                    # Environment 会返回：
                    #
                    #   job_id
                    #   dc_id
                    #   host_action
                    #   host_id
                    #   execution_result
                    #   env_time
                    # ==========================================================

                    host_result = (
                        env.execute_pending_host_action(
                            host_action=host_action,
                        )
                    )

                    # ==========================================================
                    # 防御性检查：
                    # Trace 中记录的 Job/DC/Action 必须与 Environment
                    # 真正执行的对象完全一致。
                    # ==========================================================

                    result_job_id = str(
                        host_result["job_id"]
                    )

                    result_dc_id = str(
                        host_result["dc_id"]
                    )

                    result_host_action = int(
                        host_result["host_action"]
                    )

                    if result_job_id != host_job_id:
                        raise RuntimeError(
                            "Host execution 返回 Job 不一致："
                            f"expected={host_job_id}, "
                            f"actual={result_job_id}"
                        )

                    if result_dc_id != host_dc_id:
                        raise RuntimeError(
                            "Host execution 返回 DC 不一致："
                            f"job={host_job_id}, "
                            f"expected={host_dc_id}, "
                            f"actual={result_dc_id}"
                        )

                    if (
                            result_host_action
                            != int(host_action)
                    ):
                        raise RuntimeError(
                            "Host execution 返回 action 不一致："
                            f"job={host_job_id}, "
                            f"expected={host_action}, "
                            f"actual={result_host_action}"
                        )

                    actual_host_id = str(
                        host_result["host_id"]
                    )

                    # ==========================================================
                    # 将 Host Decision 正式加入 Job Causal Trace。
                    #
                    # 此时仅记录事实。
                    # 仍然不写 Host ReplayBuffer。
                    # ==========================================================

                    pending_trace_store.record_host_step(
                        job_id=host_job_id,
                        dc_id=host_dc_id,
                        env_time=host_decision_time,
                        host_obs=host_obs,
                        action=int(
                            host_action
                        ),
                        host_id=actual_host_id,

                        # Stage 1 warmup 可以是 random；
                        # Stage 1/3 后期以及 Stage 2 为 policy。
                        action_source=(
                            host_action_source
                        ),
                    )

                    # ==========================================================
                    # 回填 Host placement 的即时执行结果：
                    #
                    #   started
                    #   queued
                    #   dropped
                    #
                    # 它不是最终 SLA/completion reward。
                    # ==========================================================

                    pending_trace_store.record_host_result(
                        job_id=host_job_id,

                        result=str(
                            host_result[
                                "execution_result"
                            ]
                        ),

                        env_time=float(
                            host_result[
                                "env_time"
                            ]
                        ),
                    )

                    # ==========================================================
                    # execute_pending_host_action() 内部会继续推进事件，
                    # 因而期间可能已经发生：
                    #
                    #   completed
                    #   waiting_timeout
                    #   resource failure
                    #
                    # 必须在 Host Step 已经写入以后，
                    # 再消费这些 delayed outcome。
                    # ==========================================================

                    consume_environment_reward_corrections(
                        env=env,

                        pending_trace_store=(
                            pending_trace_store
                        ),

                        routing_replay_buffer=(
                            routing_replay_buffer
                        ),

                        host_replay_buffers=(
                            host_replay_buffers
                        ),

                        stats=stats,


                    )
                    # ==========================================================
                    # Local Host SAC Update
                    #
                    # 只在：
                    #
                    #   Stage 1 Host Pretrain
                    #   Stage 3 Joint Fine-tune
                    #
                    # 更新。
                    #
                    # Stage 2 Host 完全冻结。
                    # ==========================================================

                    if stage_trains_host(
                            training_stage
                    ):

                        host_steps = int(
                            host_training_action_steps[
                                host_dc_id
                            ]
                        )

                        ready_to_update_host = (
                                host_steps
                                >= int(
                            train_config
                                .host_learning_starts
                        )

                                and host_steps
                                % int(
                            train_config
                                .host_train_every
                        )
                                == 0

                                and host_replay.can_sample(
                            batch_size=int(
                                train_config
                                    .host_batch_size
                            )
                        )
                        )

                        if ready_to_update_host:

                            host_update_infos = []

                            for _ in range(
                                    int(
                                        train_config
                                                .host_updates_per_train
                                    )
                            ):
                                host_update_infos.append(
                                    host_agent.update(
                                        replay_buffer=(
                                            host_replay
                                        ),

                                        batch_size=int(
                                            train_config
                                                .host_batch_size
                                        ),
                                    )
                                )

                            stats.host_update_count += int(
                                len(
                                    host_update_infos
                                )
                            )

                    decision = None


                    continue

                # ==========================================================
                # Phase 2：PettingZoo Routing decision
                # ==========================================================

                if collector.drain_one_dead_agent():
                    decision = None
                    continue

                if decision is None:
                    decision = collector.capture_decision()

                # ----------------------------------------------------------
                # 以下继续保留现有 Routing：
                #
                # forced
                # random warmup
                # Routing MASAC policy
                # ----------------------------------------------------------

                if decision.forced_action is not None:

                    # Environment lifecycle forced drop
                    action = int(
                        decision.forced_action
                    )

                    action_source = (
                        "forced"
                    )


                elif (
                        training_stage
                        == TrainingStage.HOST_PRETRAIN
                ):

                    # ======================================================
                    # Stage 1:
                    #
                    # Routing 不参与学习。
                    # 所有正常 Job 都进入当前 DC 的 Host 层。
                    #
                    # 该 Self action：
                    #   - 写入 Causal Trace；
                    #   - 不进入 Routing Replay；
                    #   - 不增加 Routing warmup step。
                    # ======================================================

                    action = (
                        get_self_routing_action(
                            env=env,
                            agent_id=(
                                decision.agent_id
                            ),
                        )
                    )

                    action_source = (
                        "orchestrator"
                    )


                elif (
                        routing_normal_action_steps
                        < int(
                    train_config
                            .routing_random_warmup_steps
                )
                ):

                    action = (
                        choose_random_routing_action(
                            action_dim=(
                                env.action_dim
                            ),

                            rng=(
                                action_rng
                            ),
                        )
                    )

                    action_source = (
                        "random"
                    )


                else:

                    action = (
                        routing_masac
                            .select_action(
                            local_obs=(
                                decision.local_obs
                            ),

                            agent_index=(
                                decision.agent_index
                            ),

                            deterministic=False,
                        )
                    )

                    action_source = (
                        "policy"
                    )

                # ==========================================================
                # Routing action 的真实语义统一由 Collector
                # 调用 Environment._decode_action() 决定。
                #
                # Trainer 不再复制 action_type 推断逻辑。
                # ==========================================================

                routing_result, next_decision = (
                    collector.execute_and_record(
                        decision=decision,

                        action=action,

                        action_source=(
                            action_source
                        ),
                    )
                )

                decision = next_decision

                # 记录这条经验的奖励和动作类型
                stats.record_transition(
                    agent_id=(
                        routing_result.agent_id
                    ),

                    reward=(
                        routing_result
                            .immediate_reward
                    ),

                    action_type=(
                        routing_result
                            .action_type
                    ),

                    action_source=(
                        routing_result
                            .action_source
                    ),
                )

                # ==========================================================
                # Forced Drop / Actor Drop 会在 execute_and_collect()
                # 内部直接完成 Job Finalize。
                #
                # 它不会产生后续 Environment reward correction，
                # 所以这里必须检查并立即 Flush。
                # ==========================================================

                # ==========================================================
                # 当前 action 如果让 Job 在 Collector 中直接 Finalize，
                # 典型情况就是 forced drop。
                #
                # 此类 Job 不会再产生 Environment terminal correction，
                # 因此这里立即 Flush。
                # ==========================================================

                if routing_result.job_finalized:
                    finalized_trace = (
                        pending_trace_store
                            .get_finalized_trace(
                            routing_result.job_id
                        )
                    )

                    flush_finalized_trace_to_replay(
                        finalized_trace=(
                            finalized_trace
                        ),

                        routing_replay_buffer=(
                            routing_replay_buffer
                        ),

                        host_replay_buffers=(
                            host_replay_buffers
                        ),
                    )

                    pending_trace_store.pop_finalized_trace(
                        routing_result.job_id
                    )



                consume_environment_reward_corrections(
                    env=env,

                    pending_trace_store=(
                        pending_trace_store
                    ),

                    routing_replay_buffer=(
                        routing_replay_buffer
                    ),

                    host_replay_buffers=(
                        host_replay_buffers
                    ),

                    stats=stats,


                )

                # 增加计数
                global_decision_steps += 1
                if action_source in {
                    "random",
                    "policy",
                }:

                    routing_normal_action_steps += 1
                    if (
                            routing_normal_action_steps
                            == int(
                        train_config
                                .routing_random_warmup_steps
                    )
                    ):
                        print(
                            "\n"
                            "============================================================\n"
                            "✅ Routing 随机动作预热结束\n"
                            "============================================================\n"
                        )

                # 同时满足以下条件才允许更新网络：
                # 1. 普通动作总数已经达到 learning_starts；
                # 2. ReplayBuffer 中普通经验足够采样一个 batch。
                ready_to_update_routing = (
                        stage_trains_routing(
                            training_stage
                        )

                        and action_source
                        in {
                            "random",
                            "policy",
                        }

                        and routing_normal_action_steps
                        >= int(
                    train_config
                        .routing_learning_starts
                )

                        and routing_normal_action_steps
                        % int(
                    train_config
                        .routing_train_every
                )
                        == 0

                        and routing_replay_buffer
                        .can_sample(
                    batch_size=int(
                        train_config
                            .routing_batch_size
                    ),

                    include_forced_actions=False,
                )
                )

                # 网络更新
                if ready_to_update_routing:

                    update_info_block = []

                    for _ in range(
                            int(
                                train_config
                                        .routing_updates_per_train
                            )
                    ):
                        update_info = (
                            routing_masac.update(
                                replay_buffer=(
                                    routing_replay_buffer
                                ),

                                batch_size=int(
                                    train_config
                                        .routing_batch_size
                                ),
                            )
                        )

                        update_info_block.append(
                            update_info
                        )

                    record_update_block(
                        stats=stats,
                        update_infos=(
                            update_info_block
                        ),
                    )

            # ==============================================================
            # Episode 结束后的最后一次 delayed outcome flush。
            #
            # 正常情况下 Routing/Host Branch 已经实时消费；
            # 这里作为 Episode tail 的防御性收尾，
            # 防止最后一批 terminal correction 留在 Environment 中。
            # ==============================================================

            consume_environment_reward_corrections(
                env=env,
                pending_trace_store=(
                    pending_trace_store
                ),
                routing_replay_buffer=(
                    routing_replay_buffer
                ),
                host_replay_buffers=(
                    host_replay_buffers
                ),
                stats=stats,
            )
            pending_trace_store.assert_no_open_trace()
            pending_trace_store.assert_no_unflushed_finalized_trace()

            # 计算当前 episode 的真实运行秒数。
            wall_time_seconds = (time.perf_counter() - episode_wall_start)

            # 日志记录
            log_row = build_episode_log_row(
                stats=stats,

                env=env,

                routing_replay_buffer=(
                    routing_replay_buffer
                ),

                host_replay_buffers=(
                    host_replay_buffers
                ),

                routing_masac=(
                    routing_masac
                ),

                global_decision_steps=(
                    global_decision_steps
                ),

                routing_normal_action_steps=(
                    routing_normal_action_steps
                ),

                wall_time_seconds=(
                    wall_time_seconds
                ),
            )
            append_csv_log(csv_path=log_csv_path, row=log_row,)

            # 到达日志打印间隔时，在终端输出摘要。
            if episode % int(train_config.log_interval) == 0:
                print_episode_summary(log_row)

            stage_boundary_checkpoint_name = (
                training_stage_boundary_checkpoint_name(
                    training_stage
                )
            )

            if (
                    stage_boundary_checkpoint_name is not None
                    and episode
                    == training_stage_end_episode(
                stage=training_stage,
                train_config=train_config,
            )
            ):
                save_two_layer_checkpoint(
                    env=env,

                    training_stage=(
                        training_stage
                    ),

                    train_config=(
                        train_config
                    ),
                    routing_masac=routing_masac,
                    host_sac_agents=host_sac_agents,

                    model_path=(
                            checkpoint_dir
                            / stage_boundary_checkpoint_name
                    ),

                    next_episode=(
                            episode + 1
                    ),

                    global_decision_steps=(
                        global_decision_steps
                    ),

                    routing_normal_action_steps=(
                        routing_normal_action_steps
                    ),

                    host_training_action_steps=(
                        host_training_action_steps
                    ),

                    best_episode_return=(
                        best_episode_return
                    ),
                )

                print(
                    "\n"
                    "============================================================\n"
                    f"✅ Training Stage Finished: {training_stage.value}\n"
                    f"Checkpoint: "
                    f"{checkpoint_dir / stage_boundary_checkpoint_name}\n"
                    "============================================================\n",
                    flush=True,
                )

            # 当前 episode return 高于历史最佳值时保存 best checkpoint
            if stats.episode_return > best_episode_return:
                best_episode_return = float(
                    stats.episode_return
                )

                save_two_layer_checkpoint(
                    env=env,

                    training_stage=(
                        training_stage
                    ),

                    train_config=(
                        train_config
                    ),
                    routing_masac=(
                        routing_masac
                    ),

                    host_sac_agents=(
                        host_sac_agents
                    ),

                    model_path=(
                            checkpoint_dir
                            / "best.pt"
                    ),

                    next_episode=(
                            episode + 1
                    ),

                    global_decision_steps=(
                        global_decision_steps
                    ),

                    routing_normal_action_steps=(
                        routing_normal_action_steps
                    ),

                    host_training_action_steps=(
                        host_training_action_steps
                    ),

                    best_episode_return=(
                        best_episode_return
                    ),
                )
            # 断点恢复
            save_two_layer_checkpoint(
                env=env,

                training_stage=(
                    training_stage
                ),

                train_config=(
                    train_config
                ),
                routing_masac=(
                    routing_masac
                ),

                host_sac_agents=(
                    host_sac_agents
                ),

                model_path=(
                        checkpoint_dir
                        / "latest.pt"
                ),

                next_episode=(
                        episode + 1
                ),

                global_decision_steps=(
                    global_decision_steps
                ),

                routing_normal_action_steps=(
                    routing_normal_action_steps
                ),

                host_training_action_steps=(
                    host_training_action_steps
                ),

                best_episode_return=(
                    best_episode_return
                ),
            )
            if (
                    episode
                    % int(
                train_config.checkpoint_interval
            )
                    == 0
            ):
                save_two_layer_checkpoint(
                    env=env,

                    training_stage=(
                        training_stage
                    ),

                    train_config=(
                        train_config
                    ),
                    routing_masac=(
                        routing_masac
                    ),

                    host_sac_agents=(
                        host_sac_agents
                    ),

                    model_path=(
                            checkpoint_dir
                            / f"episode_{episode:06d}.pt"
                    ),

                    next_episode=(
                            episode + 1
                    ),

                    global_decision_steps=(
                        global_decision_steps
                    ),

                    routing_normal_action_steps=(
                        routing_normal_action_steps
                    ),

                    host_training_action_steps=(
                        host_training_action_steps
                    ),

                    best_episode_return=(
                        best_episode_return
                    ),
                )

    finally:
        close_method = getattr(env, "close", None)
        if callable(close_method):
            close_method()

    # 全部 episode 完成后保存 final checkpoint
    save_two_layer_checkpoint(
        env=env,

        training_stage=(
            resolve_training_stage(
                episode=int(
                    train_config.num_episodes
                ),

                train_config=(
                    train_config
                ),
            )
        ),

        train_config=(
            train_config
        ),

        routing_masac=(
            routing_masac
        ),

        host_sac_agents=(
            host_sac_agents
        ),

        model_path=(
                checkpoint_dir
                / "final.pt"
        ),

        next_episode=(
                int(
                    train_config.num_episodes
                )
                + 1
        ),

        global_decision_steps=(
            global_decision_steps
        ),

        routing_normal_action_steps=(
            routing_normal_action_steps
        ),

        host_training_action_steps=(
            host_training_action_steps
        ),

        best_episode_return=(
            best_episode_return
        ),
    )

    return (
        routing_masac,
        host_sac_agents,
    )

def main() -> None:
    train_config = TrainConfig(
        num_episodes=conf.Episodes,
        routing_replay_capacity=(conf.ROUTING_REPLAY_CAPACITY),
        host_replay_capacity=(conf.HOST_REPLAY_CAPACITY),
        # ==========================================================
        # Routing MASAC Training Schedule
        # ==========================================================

        routing_batch_size=(
            conf.ROUTING_BATCH_SIZE
        ),

        routing_random_warmup_steps=(
            conf.ROUTING_RANDOM_WARMUP_STEPS
        ),

        routing_learning_starts=(
            conf.ROUTING_LEARNING_STARTS
        ),

        routing_train_every=(
            conf.ROUTING_TRAIN_EVERY
        ),

        routing_updates_per_train=(
            conf.ROUTING_UPDATES_PER_TRAIN
        ),

        # ==========================================================
        # Local Host SAC Training Schedule
        # ==========================================================

        host_batch_size=(
            conf.HOST_BATCH_SIZE
        ),

        host_random_warmup_steps=(
            conf.HOST_RANDOM_WARMUP_STEPS
        ),

        host_learning_starts=(
            conf.HOST_LEARNING_STARTS
        ),

        host_train_every=(
            conf.HOST_TRAIN_EVERY
        ),

        host_updates_per_train=(
            conf.HOST_UPDATES_PER_TRAIN
        ),

        # ==========================================================
        # Three-Stage Training
        # ==========================================================

        host_pretrain_episodes=(
            conf.HOST_PRETRAIN_EPISODES
        ),

        routing_train_episodes=(
            conf.ROUTING_TRAIN_EPISODES
        ),

        joint_finetune_episodes=(
            conf.JOINT_FINETUNE_EPISODES
        ),
        log_interval=conf.Log_interval,
        checkpoint_interval=conf.Checkpoint_Interval,
        seed=conf.Seed,
        checkpoint_dir=conf.Checkpoint_Dir,
        log_csv_path=conf.Log_csv_Path,
        old_env_path=conf.Old_Env_Path,
        resume_checkpoint=conf.Resume_Checkpoint,
        vary_episode_seed=conf.Vary_Episode_Seed
    )

    host_sac_config = (
        HostSACConfig(
            # ======================================================
            # Host SAC 使用完全独立的算法超参数。
            #
            # 正常 main() 入口与 train() fallback 必须保持一致，
            # 防止 Host 又退回旧 Flat-MASAC 公共参数。
            # ======================================================

            gamma=(
                conf.HOST_GAMMA
            ),

            tau=(
                conf.HOST_TAU
            ),

            actor_lr=(
                conf.HOST_ACTOR_LR
            ),

            critic_lr=(
                conf.HOST_CRITIC_LR
            ),

            alpha_lr=(
                conf.HOST_ALPHA_LR
            ),

            actor_hidden_dim=(
                conf.HOST_ACTOR_HIDDEN_DIM
            ),

            critic_hidden_dim=(
                conf.HOST_CRITIC_HIDDEN_DIM
            ),

            initial_alpha=(
                conf.HOST_INITIAL_ALPHA
            ),

            target_entropy_ratio=(
                conf.HOST_TARGET_ENTROPY_RATIO
            ),

            max_grad_norm=(
                conf.HOST_MAX_GRAD_NORM
            ),

            policy_update_interval=(
                conf.HOST_POLICY_UPDATE_INTERVAL
            ),

            target_update_interval=(
                conf.HOST_TARGET_UPDATE_INTERVAL
            ),

            device=(
                conf.DEVICE
            ),

            seed=(
                conf.Seed
            ),
        )
    )

    routing_masac_config = (
        RoutingMASACConfig(
            gamma=(
                conf.ROUTING_GAMMA
            ),

            tau=(
                conf.ROUTING_TAU
            ),

            actor_lr=(
                conf.ROUTING_ACTOR_LR
            ),

            critic_lr=(
                conf.ROUTING_CRITIC_LR
            ),

            alpha_lr=(
                conf.ROUTING_ALPHA_LR
            ),

            actor_hidden_dim=(
                conf.ROUTING_ACTOR_HIDDEN_DIM
            ),

            critic_hidden_dim=(
                conf.ROUTING_CRITIC_HIDDEN_DIM
            ),

            initial_alpha=(
                conf.ROUTING_INITIAL_ALPHA
            ),

            target_entropy_ratio=(
                conf.ROUTING_TARGET_ENTROPY_RATIO
            ),

            max_grad_norm=(
                conf.ROUTING_MAX_GRAD_NORM
            ),

            policy_update_interval=(
                conf.ROUTING_POLICY_UPDATE_INTERVAL
            ),

            target_update_interval=(
                conf.ROUTING_TARGET_UPDATE_INTERVAL
            ),

            device=(
                conf.DEVICE
            ),

            seed=(
                conf.Seed
            ),
        )
    )

    train(
        train_config=(
            train_config
        ),

        routing_masac_config=(
            routing_masac_config
        ),

        host_sac_config=(
            host_sac_config
        ),
    )

if __name__ == "__main__":
    main()













