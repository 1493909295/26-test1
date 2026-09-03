from __future__ import annotations
import copy
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

@dataclass(frozen=True)
class FinalizedJobTrace:
    """
    一个 Job 已经完成生命周期闭环以后生成的不可继续扩展因果链。

    本对象保存的是已经确认完成的事实：

        Routing Step 0
            ↓
        Routing Step 1
            ↓
        ...
            ↓
        Self / Cloud / Drop
            ↓
        Host（仅 Self）
            ↓
        Terminal Outcome

    注意：
        1. FinalizedJobTrace 不是 Replay Transition；
        2. 不参与 sample()；
        3. 不直接参与 SAC update；
        4. Reward attribution 将在后续步骤单独完成。
    """

    job_id: str

    routing_steps: tuple[
        PendingRoutingStep,
        ...
    ]

    host_step: Optional[
        PendingHostStep
    ]

    reward_events: tuple[
        PendingRewardEvent,
        ...
    ]

    terminal_reason: str
    terminal_time: float

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
        # ==========================================================
        # 尚未 Finalize 的 Job。
        #
        # Job 从第一次 Routing Decision 开始进入这里。
        # ==========================================================

        self._traces: Dict[
            str,
            PendingJobTrace,
        ] = {}

        # ==========================================================
        # 已完成 terminal + structural validation 的 Job。
        #
        # 注意：
        # 这里仍然不是 ReplayBuffer。
        #
        # 后续步骤会把 FinalizedJobTrace 转换为：
        #
        #   Routing Replay Transition
        #   Host Replay Transition
        # ==========================================================

        self._finalized_traces: Dict[
            str,
            FinalizedJobTrace,
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
        """
        向指定 Job 的 Pending Causal Trace
        追加一个真实发生的 Routing Decision。

        严格因果约束：

            1. 已 terminal Job 不能继续 Routing；
            2. 上一个 Edge->Edge predecessor 必须先解析完成，
               才允许追加新的 Routing Step；
            3. action_type 与 target_dc_id 必须语义一致；
            4. 不做 visited-DC / cycle 限制。

        因此：
            DC1 -> DC3 -> DC1

        仍然完全允许。

        本函数检查的是“因果正确性”，
        不是限制 Routing 策略。
        """

        job_id = str(
            job_id
        )

        agent_id = str(
            agent_id
        )

        action_type = str(
            action_type
        )

        action_source = str(
            action_source
        )

        target_dc_id = (
            None
            if target_dc_id is None
            else str(target_dc_id)
        )

        trace = self._get_or_create(
            job_id
        )

        # ==========================================================
        # 已 terminal Job 不允许继续产生 Routing decision。
        # ==========================================================

        if trace.terminal:
            raise RuntimeError(
                "不能向已经 terminal 的 Job Trace "
                "继续添加 Routing Step："
                f"job={job_id}"
            )

        # ==========================================================
        # Routing Action Type 合法性检查。
        # ==========================================================

        allowed_action_types = {
            "edge_dc",
            "self",
            "cloud",
            "drop",
        }

        if action_type not in allowed_action_types:
            raise ValueError(
                "PendingJobTrace 收到未知 Routing action_type："
                f"job={job_id}, "
                f"action_type={action_type}"
            )

        # ==========================================================
        # 严格因果链：
        #
        # 在添加新的 Routing Step 之前，
        # 不允许该 Job 还有一个尚未解析的 Edge predecessor。
        #
        # 正常顺序必须是：
        #
        #   record R0: DC1 -> DC3
        #       ↓
        #   Job 到 DC3
        #       ↓
        #   resolve R0.next_state
        #       ↓
        #   才允许 record R1
        # ==========================================================

        unresolved_edge_steps = [
            step
            for step
            in trace.routing_steps
            if (
                    step.action_type == "edge_dc"
                    and not step.successor_resolved
            )
        ]

        if unresolved_edge_steps:
            raise RuntimeError(
                "上一条 Edge Routing predecessor 尚未解析，"
                "不能继续追加新的 Routing Step："
                f"job={job_id}, "
                f"current_agent={agent_id}, "
                f"unresolved_count={len(unresolved_edge_steps)}"
            )

        # ==========================================================
        # Action Semantic Validation
        # ==========================================================

        if action_type == "edge_dc":

            if target_dc_id is None:
                raise RuntimeError(
                    "Edge Routing 缺少 target_dc_id："
                    f"job={job_id}, "
                    f"source={agent_id}"
                )

            # target == source 应该被编码为 Self，
            # 不能被错误标记为 edge_dc。
            if target_dc_id == agent_id:
                raise RuntimeError(
                    "edge_dc 的 target_dc_id "
                    "不能等于当前 source DC："
                    f"job={job_id}, "
                    f"dc={agent_id}"
                )

        elif action_type == "self":

            if target_dc_id != agent_id:
                raise RuntimeError(
                    "Self Routing 的 target_dc_id "
                    "必须等于当前 DC："
                    f"job={job_id}, "
                    f"source={agent_id}, "
                    f"target={target_dc_id}"
                )

        elif action_type == "cloud":

            if target_dc_id is None:
                raise RuntimeError(
                    "Cloud Routing 缺少 target_dc_id："
                    f"job={job_id}"
                )

        elif action_type == "drop":

            if target_dc_id is not None:
                raise RuntimeError(
                    "Drop action 不应该存在 target_dc_id："
                    f"job={job_id}, "
                    f"target={target_dc_id}"
                )

        # ==========================================================
        # Routing Step Sequence
        # ==========================================================

        sequence_index = len(
            trace.routing_steps
        )

        trace.routing_steps.append(
            PendingRoutingStep(
                job_id=job_id,

                sequence_index=(
                    sequence_index
                ),

                agent_id=agent_id,

                agent_index=int(
                    agent_index
                ),

                env_time=float(
                    env_time
                ),

                local_obs=np.asarray(
                    local_obs,
                    dtype=np.float32,
                ).copy(),

                global_state=np.asarray(
                    global_state,
                    dtype=np.float32,
                ).copy(),

                action=int(
                    action
                ),

                action_type=(
                    action_type
                ),

                action_source=(
                    action_source
                ),

                immediate_reward=float(
                    immediate_reward
                ),

                source_dc_id=(
                    agent_id
                ),

                target_dc_id=(
                    target_dc_id
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

        next_agent_id = str(
            next_agent_id
        )

        next_env_time = float(
            next_env_time
        )

        expected_target_dc_id = (
            previous_step.target_dc_id
        )

        if expected_target_dc_id is None:
            raise RuntimeError(
                "未解析 Edge predecessor 缺少 target_dc_id："
                f"job={job_id}, "
                f"sequence={previous_step.sequence_index}"
            )

        if next_agent_id != str(
                expected_target_dc_id
        ):
            raise RuntimeError(
                "Routing Causal Chain target 不一致："
                f"job={job_id}, "
                f"sequence={previous_step.sequence_index}, "
                f"source={previous_step.source_dc_id}, "
                f"expected_target={expected_target_dc_id}, "
                f"actual_next_agent={next_agent_id}"
            )

        # ==========================================================
        # 时间因果关系必须单调。
        #
        # successor 决策时间不能早于 predecessor 决策时间。
        # ==========================================================

        if (
                next_env_time
                < float(
            previous_step.env_time
        )
        ):
            raise RuntimeError(
                "Routing Causal Chain 时间倒退："
                f"job={job_id}, "
                f"previous_time={previous_step.env_time}, "
                f"next_time={next_env_time}"
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

        job_id = str(
            job_id
        )

        dc_id = str(
            dc_id
        )

        # ==========================================================
        # Host Step 绝不能凭空创建 Job Trace。
        #
        # 正确因果链必须已经存在：
        #
        #   Routing ... -> Self
        #
        # Host 才能发生。
        # ==========================================================

        trace = self.get_trace(
            job_id
        )

        if not trace.routing_steps:
            raise RuntimeError(
                "Host Decision 出现时 Job 没有任何 Routing Step："
                f"job={job_id}, "
                f"dc={dc_id}"
            )

        last_routing_step = (
            trace.routing_steps[-1]
        )

        if (
                last_routing_step.action_type
                != "self"
        ):
            raise RuntimeError(
                "Host Decision 的上一因果节点不是 Self Routing："
                f"job={job_id}, "
                f"last_routing_action="
                f"{last_routing_step.action_type}"
            )

        if (
                last_routing_step.source_dc_id
                != dc_id
        ):
            raise RuntimeError(
                "Host Decision DC 与 Self Routing DC 不一致："
                f"job={job_id}, "
                f"self_dc={last_routing_step.source_dc_id}, "
                f"host_dc={dc_id}"
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

        job_id = str(
            job_id
        )

        # ==========================================================
        # Environment terminal/delayed outcome 必须属于
        # 一个已经存在的 Job Causal Trace。
        #
        # 不允许 terminal event 凭空创建 Trace，
        # 否则会掩盖 Routing Decision 丢失问题。
        # ==========================================================

        trace = self.get_trace(
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

        job_id = str(
            job_id
        )

        trace = self.get_trace(
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

    def _validate_trace_for_finalize(self, trace: PendingJobTrace,) -> None:
        """
        Job Terminal 后，在真正 Finalize 之前，
        对整条 Job Causal Trace 做一次最终结构验证。

        本函数只验证“因果事实是否完整”，
        不进行 Reward Credit Assignment。
        """

        job_id = str(
            trace.job_id
        )

        # ==========================================================
        # 1. Job 必须已经 terminal
        # ==========================================================

        if not trace.terminal:
            raise RuntimeError(
                "不能 Finalize 尚未 terminal 的 Job："
                f"job={job_id}"
            )

        if trace.terminal_reason is None:
            raise RuntimeError(
                "Terminal Job 缺少 terminal_reason："
                f"job={job_id}"
            )

        if trace.terminal_time is None:
            raise RuntimeError(
                "Terminal Job 缺少 terminal_time："
                f"job={job_id}"
            )

        terminal_time = float(
            trace.terminal_time
        )

        # ==========================================================
        # 2. 必须至少发生过一次 Routing Decision
        # ==========================================================

        if not trace.routing_steps:
            raise RuntimeError(
                "Terminal Job 没有任何 Routing Step："
                f"job={job_id}"
            )

        routing_steps = (
            trace.routing_steps
        )

        # ==========================================================
        # 3. 逐条验证 Routing Chain
        # ==========================================================

        for index, step in enumerate(
                routing_steps
        ):
            if (
                    str(step.job_id)
                    != job_id
            ):
                raise RuntimeError(
                    "Routing Step job_id 与 Trace 不一致："
                    f"trace_job={job_id}, "
                    f"step_job={step.job_id}, "
                    f"sequence={index}"
                )

            if (
                    int(step.sequence_index)
                    != index
            ):
                raise RuntimeError(
                    "Routing sequence_index 不连续："
                    f"job={job_id}, "
                    f"expected={index}, "
                    f"actual={step.sequence_index}"
                )

            # ======================================================
            # Edge -> Edge 必须已经拥有真实 same-job successor
            # ======================================================

            if step.action_type == "edge_dc":

                if not step.successor_resolved:
                    raise RuntimeError(
                        "Finalize 时仍存在未解析的 "
                        "Edge Routing predecessor："
                        f"job={job_id}, "
                        f"sequence={index}, "
                        f"source={step.source_dc_id}, "
                        f"target={step.target_dc_id}"
                    )

                if step.next_agent_id is None:
                    raise RuntimeError(
                        "Edge Routing 已标记 resolved，"
                        "但 next_agent_id 为 None："
                        f"job={job_id}, "
                        f"sequence={index}"
                    )

                if step.next_env_time is None:
                    raise RuntimeError(
                        "Edge Routing 缺少 next_env_time："
                        f"job={job_id}, "
                        f"sequence={index}"
                    )

                if step.next_local_obs is None:
                    raise RuntimeError(
                        "Edge Routing 缺少 next_local_obs："
                        f"job={job_id}, "
                        f"sequence={index}"
                    )

                if step.next_global_state is None:
                    raise RuntimeError(
                        "Edge Routing 缺少 next_global_state："
                        f"job={job_id}, "
                        f"sequence={index}"
                    )

                if (
                        str(step.next_agent_id)
                        != str(step.target_dc_id)
                ):
                    raise RuntimeError(
                        "Edge Routing target 与 successor 不一致："
                        f"job={job_id}, "
                        f"sequence={index}, "
                        f"target={step.target_dc_id}, "
                        f"next_agent={step.next_agent_id}"
                    )

                # Terminal Job 不允许最后一条还是 Edge forwarding。
                if index + 1 >= len(
                        routing_steps
                ):
                    raise RuntimeError(
                        "Terminal Job 的 Routing Chain "
                        "不能以 edge_dc 结束："
                        f"job={job_id}, "
                        f"sequence={index}"
                    )

                next_step = (
                    routing_steps[
                        index + 1
                        ]
                )

                if (
                        str(next_step.agent_id)
                        != str(step.next_agent_id)
                ):
                    raise RuntimeError(
                        "相邻 Routing Step 因果断裂："
                        f"job={job_id}, "
                        f"sequence={index}, "
                        f"resolved_next={step.next_agent_id}, "
                        f"next_step_agent={next_step.agent_id}"
                    )

            else:
                # ==================================================
                # Self / Cloud / Drop 都是 Routing terminal action。
                #
                # 后面不能再有同 Job Routing Step。
                # ==================================================

                if (
                        index
                        != len(routing_steps) - 1
                ):
                    raise RuntimeError(
                        "Routing terminal action 后 "
                        "仍然存在新的 Routing Step："
                        f"job={job_id}, "
                        f"sequence={index}, "
                        f"action_type={step.action_type}"
                    )

        # ==========================================================
        # 4. 最后一条 Routing 必须明确终止 Routing
        # ==========================================================

        final_routing_step = (
            routing_steps[-1]
        )

        final_action_type = str(
            final_routing_step.action_type
        )

        if final_action_type not in {
            "self",
            "cloud",
            "drop",
        }:
            raise RuntimeError(
                "Terminal Job 的最后一条 Routing action "
                "不是 Routing terminal action："
                f"job={job_id}, "
                f"action_type={final_action_type}"
            )

        # ==========================================================
        # 5. Self -> Host 因果关系
        # ==========================================================

        if final_action_type == "self":

            if trace.host_step is None:
                raise RuntimeError(
                    "Self Routing 后 Job 已 terminal，"
                    "但缺少 Host Step："
                    f"job={job_id}"
                )

            host_step = (
                trace.host_step
            )

            if (
                    str(host_step.job_id)
                    != job_id
            ):
                raise RuntimeError(
                    "Host Step job_id 不一致："
                    f"trace_job={job_id}, "
                    f"host_job={host_step.job_id}"
                )

            if (
                    str(host_step.dc_id)
                    != str(
                final_routing_step
                        .source_dc_id
            )
            ):
                raise RuntimeError(
                    "Host Step DC 与最终 Self DC 不一致："
                    f"job={job_id}, "
                    f"self_dc="
                    f"{final_routing_step.source_dc_id}, "
                    f"host_dc={host_step.dc_id}"
                )

            if (
                    host_step.execution_result
                    is None
            ):
                raise RuntimeError(
                    "Host Step 尚未记录 execution_result："
                    f"job={job_id}"
                )

            if (
                    host_step.result_time
                    is None
            ):
                raise RuntimeError(
                    "Host Step 尚未记录 result_time："
                    f"job={job_id}"
                )

        # ==========================================================
        # 6. Cloud / Drop 不允许存在 Host Step
        # ==========================================================

        elif final_action_type in {
            "cloud",
            "drop",
        }:

            if trace.host_step is not None:
                raise RuntimeError(
                    "Cloud/Drop Routing 后不应该出现 Host Step："
                    f"job={job_id}, "
                    f"action_type={final_action_type}"
                )

        # ==========================================================
        # 7. Terminal Event 一致性
        #
        # Forced Drop 特殊：
        #   penalty 已经保存在 Routing immediate_reward 中，
        #   所以没有 terminal RewardEvent。
        #
        # 其他完成/失败：
        #   必须恰好有一个 terminal RewardEvent。
        # ==========================================================

        terminal_events = [
            event
            for event
            in trace.reward_events
            if event.terminal
        ]

        if final_action_type == "drop":

            if (
                    str(trace.terminal_reason)
                    != "forced_drop"
            ):
                raise RuntimeError(
                    "Drop Routing 的 terminal_reason "
                    "不是 forced_drop："
                    f"job={job_id}, "
                    f"reason={trace.terminal_reason}"
                )

            if terminal_events:
                raise RuntimeError(
                    "Forced Drop 不应该重复存在 "
                    "terminal RewardEvent："
                    f"job={job_id}, "
                    f"count={len(terminal_events)}"
                )

        else:

            if len(terminal_events) != 1:
                raise RuntimeError(
                    "非 Forced-Drop Terminal Job "
                    "必须恰好有一个 terminal RewardEvent："
                    f"job={job_id}, "
                    f"count={len(terminal_events)}"
                )

            terminal_event = (
                terminal_events[0]
            )

            if (
                    str(terminal_event.reason)
                    != str(trace.terminal_reason)
            ):
                raise RuntimeError(
                    "Terminal Event reason "
                    "与 Trace terminal_reason 不一致："
                    f"job={job_id}, "
                    f"event_reason={terminal_event.reason}, "
                    f"trace_reason={trace.terminal_reason}"
                )

        # ==========================================================
        # 8. 所有 Reward Event 都不能发生在 terminal 以后
        # ==========================================================

        for event in trace.reward_events:

            if (
                    float(event.env_time)
                    > terminal_time + 1e-9
            ):
                raise RuntimeError(
                    "Reward Event 时间晚于 Job terminal："
                    f"job={job_id}, "
                    f"event_time={event.env_time}, "
                    f"terminal_time={terminal_time}"
                )

    def _build_finalized_trace(self,trace: PendingJobTrace, ) -> FinalizedJobTrace:
        """
        从已经通过结构验证的 PendingJobTrace
        生成完全独立的 FinalizedJobTrace 副本。

        使用 deepcopy 的原因：
            Finalized Trace 不应该继续引用 Pending 对象中的
            numpy array / Step 实例。
        """

        if trace.terminal_reason is None:
            raise RuntimeError(
                "Finalized Trace 缺少 terminal_reason："
                f"job={trace.job_id}"
            )

        if trace.terminal_time is None:
            raise RuntimeError(
                "Finalized Trace 缺少 terminal_time："
                f"job={trace.job_id}"
            )

        finalized_routing_steps = tuple(
            copy.deepcopy(step)
            for step
            in trace.routing_steps
        )

        finalized_host_step = (
            None
            if trace.host_step is None
            else copy.deepcopy(
                trace.host_step
            )
        )

        finalized_reward_events = tuple(
            copy.deepcopy(event)
            for event
            in trace.reward_events
        )

        return FinalizedJobTrace(
            job_id=str(
                trace.job_id
            ),

            routing_steps=(
                finalized_routing_steps
            ),

            host_step=(
                finalized_host_step
            ),

            reward_events=(
                finalized_reward_events
            ),

            terminal_reason=str(
                trace.terminal_reason
            ),

            terminal_time=float(
                trace.terminal_time
            ),
        )

    def finalize_terminal_trace(self,job_id: str,) -> FinalizedJobTrace:
        """
        对一个已经 terminal 的 Job 一次性完成 Finalize。

        原子流程：

            Pending Trace
                ↓
            Structural Validation
                ↓
            Deep Copy
                ↓
            FinalizedJobTrace
                ↓
            从 Pending 区删除
                ↓
            写入 Finalized 区

        本函数不写 ReplayBuffer。
        """

        job_id = str(
            job_id
        )

        if (
                job_id
                in self._finalized_traces
        ):
            raise RuntimeError(
                "同一个 Job 被重复 Finalize："
                f"job={job_id}"
            )

        trace = self.get_trace(
            job_id
        )

        # ==========================================================
        # 第一阶段：
        # 必须先保证完整链结构正确。
        # ==========================================================

        self._validate_trace_for_finalize(
            trace
        )

        # ==========================================================
        # 第二阶段：
        # 创建 Finalized 副本。
        # ==========================================================

        finalized_trace = (
            self._build_finalized_trace(
                trace
            )
        )

        # ==========================================================
        # 第三阶段：
        # 只有前面全部成功以后才移动。
        #
        # 这样如果 validation / deepcopy 出错，
        # 原 Pending Trace 仍完整保留，便于调试。
        # ==========================================================

        self._finalized_traces[
            job_id
        ] = finalized_trace

        del self._traces[
            job_id
        ]

        return finalized_trace

    def has_finalized_trace(self,job_id: str,) -> bool:

        return (
                str(job_id)
                in self._finalized_traces
        )

    def get_finalized_trace(self,job_id: str,) -> FinalizedJobTrace:

        job_id = str(
            job_id
        )

        trace = self._finalized_traces.get(
            job_id
        )

        if trace is None:
            raise KeyError(
                "找不到 Finalized Job Trace："
                f"job={job_id}"
            )

        return trace

    def is_terminal(
            self,
            job_id: str,
    ) -> bool:

        job_id = str(
            job_id
        )

        # 已 Finalize 的 Job 必然已经 terminal。
        if (
                job_id
                in self._finalized_traces
        ):
            return True

        trace = self._traces.get(
            job_id
        )

        if trace is None:
            return False

        return bool(
            trace.terminal
        )

    def remove_trace(self,job_id: str,) -> Optional[PendingJobTrace]:

        return self._traces.pop(
            str(job_id),
            None,
        )

    @property
    def open_trace_count(self,) -> int:
        """
        尚未 terminal 的 Pending Trace 数。
        """

        return sum(
            1
            for trace
            in self._traces.values()
            if not trace.terminal
        )

    @property
    def pending_trace_count(self,) -> int:
        """
        尚未完成 Finalize 的 Trace 总数。

        包括：
            open
            terminal-but-not-finalized
        """

        return len(
            self._traces
        )

    @property
    def terminal_trace_count(self,) -> int:
        """
        已经 terminal 的 Job 总数。

        正常情况下 terminal 后会立即 Finalize，
        因而主要来自 _finalized_traces。
        """

        pending_terminal_count = sum(
            1
            for trace
            in self._traces.values()
            if trace.terminal
        )

        return (
                pending_terminal_count
                + len(
            self._finalized_traces
        )
        )

    @property
    def finalized_trace_count(self,) -> int:

        return len(
            self._finalized_traces
        )

    @property
    def total_trace_count(self,) -> int:

        return (
                len(self._traces)
                + len(
            self._finalized_traces
        )
        )

    def assert_no_open_trace(self,) -> None:
        """
        Episode 结束以后，Pending 区必须完全为空。

        因为：

            open Job
                → 错误

            terminal 但未 Finalize
                → 同样错误
        """

        if not self._traces:
            return

        remaining_details = [
            {
                "job_id": job_id,
                "terminal": bool(
                    trace.terminal
                ),
                "terminal_reason": (
                    trace.terminal_reason
                ),
                "routing_steps": len(
                    trace.routing_steps
                ),
                "has_host_step": (
                        trace.host_step
                        is not None
                ),
            }
            for job_id, trace
            in self._traces.items()
        ]

        raise RuntimeError(
            "Episode 已结束，但仍存在未 Finalize 的 "
            "Pending Job Causal Trace："
            f"{remaining_details[:20]}"
        )

    def reset_episode(self,) -> None:
        """
        开始新 Episode 前清理上一 Episode Trace。

        当前第十六步 Finalized Trace 只保留到 Episode 边界。

        后续双 ReplayBuffer 建成以后，
        Finalized Trace 会在这里之前被 pop 并写入经验池。
        """

        # Pending 区只要还有任何内容，都说明上一轮
        # 有 Job 没有完成 Finalize。
        if self._traces:
            self.assert_no_open_trace()

        self._traces.clear()

        # 当前 Finalized Trace 暂时作为本 Episode 的完整结果归档。
        # 下一 Episode 开始时清除。
        self._finalized_traces.clear()




