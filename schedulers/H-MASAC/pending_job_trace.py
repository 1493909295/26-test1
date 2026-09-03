from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float32]

@dataclass
class PendingRoutingStep:
    job_id: str
    sequence_index: int

    agent_id: str
    agent_index: int

    env_time: float

    local_obs: FloatArray
    global_state: FloatArray

    action: int
    action_type: str
    action_source: str

    immediate_reward: float

    source_dc_id: str
    target_dc_id: Optional[str]

    # Edge -> Edge 的真实 same-job successor
    # 在任务真正到达下一 DC 时再填写。
    next_agent_id: Optional[str] = None
    next_agent_index: int = -1
    next_env_time: Optional[float] = None

    next_local_obs: Optional[FloatArray] = None
    next_global_state: Optional[FloatArray] = None

    successor_resolved: bool = False


@dataclass
class PendingHostStep:
    job_id: str
    dc_id: str

    env_time: float

    host_obs: FloatArray

    action: int
    host_id: str

    action_source: str

    execution_result: Optional[str] = None
    result_time: Optional[float] = None


@dataclass
class PendingRewardEvent:
    job_id: str

    env_time: float

    reward_delta: float
    reason: str

    terminal: bool


@dataclass
class PendingJobTrace:
    job_id: str

    routing_steps: list[
        PendingRoutingStep
    ] = field(default_factory=list)

    host_step: Optional[
        PendingHostStep
    ] = None

    reward_events: list[
        PendingRewardEvent
    ] = field(default_factory=list)

    terminal: bool = False
    terminal_reason: Optional[str] = None
    terminal_time: Optional[float] = None


class PendingJobTraceStore:
    """
    保存当前 Episode 内尚未完成生命周期闭环的 Job 因果链。

    注意：
        1. 本类不是 ReplayBuffer；
        2. 不提供 sample()；
        3. 不直接参与 SAC update；
        4. Job terminal 前，所有 Routing / Host 决策事实只记录在这里；
        5. 后续步骤负责将完整 Trace 转换成正式 Replay Transition。
    """

    def __init__(self) -> None:
        self._traces: Dict[
            str,
            PendingJobTrace,
        ] = {}

    def _get_or_create(
            self,
            job_id: str,
    ) -> PendingJobTrace:
        job_id = str(job_id)

        trace = self._traces.get(
            job_id
        )

        if trace is None:
            trace = PendingJobTrace(
                job_id=job_id
            )

            self._traces[job_id] = trace

        return trace

    def has_trace(
            self,
            job_id: str,
    ) -> bool:
        return str(job_id) in self._traces

    def get_trace(
            self,
            job_id: str,
    ) -> PendingJobTrace:

        job_id = str(job_id)

        trace = self._traces.get(
            job_id
        )

        if trace is None:
            raise KeyError(
                "找不到 Pending Job Trace："
                f"job={job_id}"
            )

        return trace

    def record_routing_step(
            self,
            *,
            job_id: str,
            agent_id: str,
            agent_index: int,
            env_time: float,
            local_obs: FloatArray,
            global_state: FloatArray,
            action: int,
            action_type: str,
            action_source: str,
            immediate_reward: float,
            target_dc_id: Optional[str],
    ) -> None:

        trace = self._get_or_create(
            job_id
        )

        if trace.terminal:
            raise RuntimeError(
                "不能向已经 terminal 的 Job Trace "
                "继续添加 Routing Step："
                f"job={job_id}"
            )

        sequence_index = len(
            trace.routing_steps
        )

        trace.routing_steps.append(
            PendingRoutingStep(
                job_id=str(job_id),
                sequence_index=(
                    sequence_index
                ),

                agent_id=str(agent_id),
                agent_index=int(agent_index),

                env_time=float(env_time),

                local_obs=np.asarray(
                    local_obs,
                    dtype=np.float32,
                ).copy(),

                global_state=np.asarray(
                    global_state,
                    dtype=np.float32,
                ).copy(),

                action=int(action),
                action_type=str(
                    action_type
                ),

                action_source=str(
                    action_source
                ),

                immediate_reward=float(
                    immediate_reward
                ),

                source_dc_id=str(
                    agent_id
                ),

                target_dc_id=(
                    None
                    if target_dc_id is None
                    else str(target_dc_id)
                ),
            )
        )

    def resolve_routing_successor(
            self,
            *,
            job_id: str,
            next_agent_id: str,
            next_agent_index: int,
            next_env_time: float,
            next_local_obs: FloatArray,
            next_global_state: FloatArray,
    ) -> bool:

        job_id = str(job_id)

        trace = self._traces.get(
            job_id
        )

        if trace is None:
            return False

        unresolved_steps = [
            step
            for step
            in trace.routing_steps
            if (
                    step.action_type == "edge_dc"
                    and not step.successor_resolved
            )
        ]

        if not unresolved_steps:
            return False

        if len(unresolved_steps) > 1:
            raise RuntimeError(
                "同一个 Job 同时出现多个未解析 "
                "Edge Routing predecessor："
                f"job={job_id}, "
                f"count={len(unresolved_steps)}"
            )

        previous_step = (
            unresolved_steps[-1]
        )

        previous_step.next_agent_id = str(
            next_agent_id
        )

        previous_step.next_agent_index = int(
            next_agent_index
        )

        previous_step.next_env_time = float(
            next_env_time
        )

        previous_step.next_local_obs = np.asarray(
            next_local_obs,
            dtype=np.float32,
        ).copy()

        previous_step.next_global_state = (
            np.asarray(
                next_global_state,
                dtype=np.float32,
            ).copy()
        )

        previous_step.successor_resolved = True

        return True

    def record_host_step(
            self,
            *,
            job_id: str,
            dc_id: str,
            env_time: float,
            host_obs: FloatArray,
            action: int,
            host_id: str,
            action_source: str,
    ) -> None:

        trace = self._get_or_create(
            job_id
        )

        if trace.terminal:
            raise RuntimeError(
                "不能为 terminal Job 添加 Host Step："
                f"job={job_id}"
            )

        if trace.host_step is not None:
            raise RuntimeError(
                "同一个 Job 出现了重复 Host Decision："
                f"job={job_id}"
            )

        trace.host_step = PendingHostStep(
            job_id=str(job_id),
            dc_id=str(dc_id),
            env_time=float(env_time),

            host_obs=np.asarray(
                host_obs,
                dtype=np.float32,
            ).copy(),

            action=int(action),
            host_id=str(host_id),

            action_source=str(
                action_source
            ),
        )

    def record_host_result(
            self,
            *,
            job_id: str,
            result: str,
            env_time: float,
    ) -> None:

        trace = self._traces.get(
            str(job_id)
        )

        if trace is None:
            raise RuntimeError(
                "找不到 Host Result 对应的 "
                f"Job Trace：{job_id}"
            )

        if trace.host_step is None:
            raise RuntimeError(
                "Job 尚无 Host Step，"
                "却收到 Host execution result："
                f"job={job_id}"
            )

        trace.host_step.execution_result = str(
            result
        )

        trace.host_step.result_time = float(
            env_time
        )

    def record_reward_event(
            self,
            *,
            job_id: str,
            env_time: float,
            reward_delta: float,
            reason: str,
            terminal: bool,
    ) -> None:

        trace = self._get_or_create(
            job_id
        )

        trace.reward_events.append(
            PendingRewardEvent(
                job_id=str(job_id),
                env_time=float(env_time),
                reward_delta=float(
                    reward_delta
                ),
                reason=str(reason),
                terminal=bool(terminal),
            )
        )

        if terminal:

            if trace.terminal:
                raise RuntimeError(
                    "同一个 Job 收到了重复 Terminal Event："
                    f"job={job_id}, "
                    f"old={trace.terminal_reason}, "
                    f"new={reason}"
                )

            trace.terminal = True
            trace.terminal_reason = str(
                reason
            )
            trace.terminal_time = float(
                env_time
            )

    def mark_terminal(
            self,
            *,
            job_id: str,
            env_time: float,
            reason: str,
    ) -> None:

        trace = self._get_or_create(
            job_id
        )

        if trace.terminal:
            raise RuntimeError(
                "Job Trace 重复 terminal："
                f"job={job_id}, "
                f"old={trace.terminal_reason}, "
                f"new={reason}"
            )

        trace.terminal = True
        trace.terminal_reason = str(
            reason
        )
        trace.terminal_time = float(
            env_time
        )

    def is_terminal(
            self,
            job_id: str,
    ) -> bool:

        trace = self._traces.get(
            str(job_id)
        )

        if trace is None:
            return False

        return bool(
            trace.terminal
        )

    def pop_terminal_trace(
            self,
            job_id: str,
    ) -> PendingJobTrace:

        job_id = str(job_id)

        trace = self.get_trace(
            job_id
        )

        if not trace.terminal:
            raise RuntimeError(
                "不能 pop 尚未 terminal 的 Job Trace："
                f"job={job_id}"
            )

        return self._traces.pop(
            job_id
        )

    def remove_trace(
            self,
            job_id: str,
    ) -> Optional[PendingJobTrace]:

        return self._traces.pop(
            str(job_id),
            None,
        )

    @property
    def open_trace_count(self) -> int:
        return sum(
            1
            for trace
            in self._traces.values()
            if not trace.terminal
        )

    @property
    def terminal_trace_count(self) -> int:
        return sum(
            1
            for trace
            in self._traces.values()
            if trace.terminal
        )

    @property
    def total_trace_count(self) -> int:
        return len(
            self._traces
        )

    def assert_no_open_trace(self) -> None:

        unfinished = [
            job_id
            for job_id, trace
            in self._traces.items()
            if not trace.terminal
        ]

        if unfinished:
            raise RuntimeError(
                "Episode 已结束，但仍存在未闭合 "
                "Pending Job Causal Trace："
                f"{unfinished[:20]}"
            )

    def reset_episode(self) -> None:

        if self._traces:
            self.assert_no_open_trace()

        self._traces.clear()




