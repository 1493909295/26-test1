from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, Mapping, Optional, Protocol, Sequence, Tuple

import numpy as np
from numpy.typing import NDArray
from routing_observation import (RoutingObservationBuilder,)
from routing_centralized_state import (RoutingCentralizedStateBuilder,)
from pending_job_trace import (PendingJobTraceStore,)

FloatArray = NDArray[np.float32]


# 接口清单类 TransitionCollector 的对象，都应该至少具备它描述的这些接口
# CloudEdgeEnvLike 类不负责实现逻辑
class CloudEdgeEnvLike(Protocol):

    # pettingzoo 需要区分全部可能智能体与现在活着的智能体，在我的环境中这俩个没区别
    possible_agents: Sequence[str]
    agents: Sequence[str]

    # 当前执行step的智能体与当前做决策的智能体，在我环境中这两个也相同
    agent_selection: Optional[str]
    current_agent_id: Optional[str]

    current_job_id: Optional[str]
    pending_host_job_id: Optional[str]
    pending_host_dc_id: Optional[str]
    current_time: float
    # local_obs_dim: int
    # global_state_dim: int
    action_dim: int
    drop_action: int
    has_reset: bool

    # 把str类型的agent id映射成数字
    agent_name_mapping: Mapping[str, int]
    rewards: Mapping[str, float]

    # 正常停止与异常停止标记，但我其实没实现异常停止 ·_·
    terminations: Mapping[str, bool]
    truncations: Mapping[str, bool]

    # def observe(self, agent: str) -> Dict[str, np.ndarray]:
    #     ...
    # def state(self) -> np.ndarray:
    #     ...
    def step(self, action: Optional[int]) -> None:
        ...

# 动作执行前快照捕捉，True代表不可重新赋值
@dataclass(frozen=True)
class DecisionSnapshot:
    agent_id: str
    agent_index: int
    job_id: str
    env_time: float
    local_obs: FloatArray

    global_state: FloatArray

    # 等于None表示由actor给工作，等于-1表示动作是丢弃
    forced_action: Optional[int] = None

    # 转化成字典方便打印和传递
    def as_dict(self) -> Dict[str, Any]:
        return {
            "agent_id": self.agent_id,
            "agent_index": self.agent_index,
            "job_id": self.job_id,
            "env_time": self.env_time,
            "local_obs": self.local_obs.copy(),

            "global_state": self.global_state.copy(),
            "forced_action": self.forced_action,
        }

# 表示 replay buffer 中的一条训练经验
@dataclass(frozen=True)
class Transition:
    agent_id: str
    agent_index: int
    job_id: str
    env_time: float
    local_obs: FloatArray
    global_state: FloatArray

    action: int
    action_type: str
    reward: float
    next_agent_id: Optional[str]

    # next_agent_index 在终止状态下使用 -1 表示“不存在下一智能体”
    next_agent_index: int
    next_job_id: Optional[str]
    next_env_time: float
    next_local_obs: FloatArray
    next_global_state: FloatArray

    terminated: bool
    truncated: bool
    done: bool

    # 将 Transition 转成字典，ReplayBuffer 可以按键读取各字段
    def as_dict(self) -> Dict[str, Any]:
        return {
            "agent_id": self.agent_id,
            "agent_index": self.agent_index,
            "job_id": self.job_id,
            "env_time": self.env_time,
            "local_obs": self.local_obs.copy(),
            "global_state": self.global_state.copy(),

            "action": self.action,
            "action_type": self.action_type,
            "reward": self.reward,
            "next_agent_id": self.next_agent_id,
            "next_agent_index": self.next_agent_index,
            "next_job_id": self.next_job_id,
            "next_env_time": self.next_env_time,
            "next_local_obs": self.next_local_obs.copy(),
            "next_global_state": self.next_global_state.copy(),

            "terminated": self.terminated,
            "truncated": self.truncated,
            "done": self.done,
        }

# 采集器，从环境中抽取经验
class TransitionCollector:
    def __init__(
            self,
            env: CloudEdgeEnvLike,
            routing_observation_builder:
            RoutingObservationBuilder,
            routing_state_builder:
            RoutingCentralizedStateBuilder,
            pending_trace_store:
            PendingJobTraceStore,
            validate_actions: bool = True,
    ) -> None:
        self.env = env
        self.routing_observation_builder = (routing_observation_builder)
        self.validate_actions = bool(validate_actions)
        self.routing_state_builder = (routing_state_builder)
        self.pending_trace_store = (pending_trace_store)
        self.validate_actions = bool(validate_actions)
        # 全局计数，不随episode清空
        self.total_transition_count = 0
        # 当前episode生成多少条经验
        self.episode_transition_count = 0
        # 在采集器实例化时就用环境接口检查
        self._validate_environment_interface()

    # 重置计数器
    def reset_episode(self) -> None:
        self.episode_transition_count = 0
        self.routing_observation_builder.reset_episode()

    # 获取决策状态快照
    # def capture_decision(self) -> DecisionSnapshot:
    #     self._require_reset()
    #     agent_id = self._get_live_selected_agent()
    #     job_id = str(self.env.current_job_id)
    #     observation = self.env.observe(agent_id)
    #
    #     # 把局部观测转成独立 float32 数组，避免对同一块内存的引用
    #     local_obs = np.asarray(observation["observation"], dtype=np.float32).copy()
    #
    #     # 把动作掩码转成独立 int8 数组
    #     action_mask = np.asarray(observation["action_mask"],dtype=np.int8).copy()
    #
    #     # 获取同一决策时刻的集中式全局状态
    #     global_state = np.asarray(self.env.state(), dtype=np.float32).copy()
    #
    #     return DecisionSnapshot(
    #         agent_id=agent_id,
    #         agent_index=int(self.env.agent_name_mapping[agent_id]),
    #         job_id=job_id,
    #         env_time=float(self.env.current_time),
    #         local_obs=local_obs,
    #         action_mask=action_mask,
    #         global_state=global_state,
    #         forced_action=self._get_forced_action(job_id),
    #     )
    # 执行一次动作并生成完整的经验

    # 获取当前环境所对应的决策状态快照
    def _build_current_decision_snapshot(self,) -> DecisionSnapshot:
        """
        根据环境当前时刻构造 Routing DecisionSnapshot。

        本函数只服务于 Routing 层 PettingZoo Agent。

        主要职责：
            1. 获取当前真正需要做 Routing 决策的 Edge DC；
            2. 构造 Routing Actor 的 local observation；
            3. 构造 Routing centralized critic state；
            4. 构造当前 DecisionSnapshot；
            5. 如果当前 Job 是之前某个 Edge->Edge Routing action
               到达下一 Edge 后产生的 same-job successor，
               则在 PendingJobTraceStore 中完成上一跳 next_state 回填。

        注意：
            Host 层不会调用本函数。
        """

        # ==========================================================
        # 1. 基础环境状态检查
        # ==========================================================

        self._require_reset()

        # 当前必须存在真正活跃的 PettingZoo Routing Agent。
        # Host phase 下 agent_selection=None，因此不能进入这里。
        agent_id = (
            self._get_live_selected_agent()
        )

        if self.env.current_job_id is None:
            raise RuntimeError(
                "构造 Routing DecisionSnapshot 时 "
                "current_job_id 为 None："
                f"agent={agent_id}"
            )

        job_id = str(
            self.env.current_job_id
        )

        # ==========================================================
        # 2. Routing Actor Local Observation
        #
        # 这里只构造 Routing 层 Observation。
        # Host Observation 由 HostObservationBuilder 单独负责。
        # ==========================================================

        local_obs = np.asarray(
            self.routing_observation_builder.build(
                agent_id
            ),
            dtype=np.float32,
        ).copy()

        # ==========================================================
        # 3. Routing Centralized Critic State
        #
        # 该 State 只用于 CTDE 训练阶段的 Routing Critic。
        # Routing Actor 执行阶段仍然只使用 local_obs。
        # ==========================================================

        global_state = np.asarray(
            self.routing_state_builder.build(),
            dtype=np.float32,
        ).copy()

        # ==========================================================
        # 4. 构造当前 Routing DecisionSnapshot
        # ==========================================================

        snapshot = DecisionSnapshot(
            agent_id=agent_id,

            agent_index=int(
                self.env.agent_name_mapping[
                    agent_id
                ]
            ),

            job_id=job_id,

            env_time=float(
                self.env.current_time
            ),

            local_obs=local_obs,

            global_state=global_state,

            forced_action=(
                self._get_forced_action(
                    job_id
                )
            ),
        )

        # ==========================================================
        # 5. Pending Job Causal Trace：
        #    回填上一跳 Edge -> Edge Routing 的真实 successor
        #
        # 例如：
        #
        #   Job42 @ DC1
        #       ↓ action = DC3
        #
        # 前一跳只记录：
        #
        #   state      = obs(Job42 @ DC1)
        #   action     = DC3
        #   next_state = UNKNOWN
        #
        # 当 Job42 真正到达 DC3，并进入本函数时，
        # 当前 snapshot 就是上一跳真正的 same-job successor：
        #
        #   next_state = obs(Job42 @ DC3)
        #
        # 如果当前 Job 没有未解析的 Edge predecessor，
        # resolve_routing_successor() 返回 False，
        # 不执行任何修改。
        # ==========================================================

        self.pending_trace_store.resolve_routing_successor(
            job_id=snapshot.job_id,

            next_agent_id=(
                snapshot.agent_id
            ),

            next_agent_index=(
                snapshot.agent_index
            ),

            next_env_time=(
                snapshot.env_time
            ),

            next_local_obs=(
                snapshot.local_obs
            ),

            next_global_state=(
                snapshot.global_state
            ),
        )

        # ==========================================================
        # 6. 返回当前 Routing Decision
        # ==========================================================

        return snapshot

    def capture_decision(self) -> DecisionSnapshot:
        self._require_reset()
        return self._build_current_decision_snapshot()
    # 这里相当于把原来的 capture_decision() 拆成 _build_current_decision_snapshot()->capture_decision(),这样后面 execute_and_collect() 也可以调用同一个函数

    def execute_and_collect(
            self,
            decision: DecisionSnapshot,
            action: int,
            action_type: str,
            action_source: str,
    ) -> Tuple[
        Transition,
        Optional[DecisionSnapshot],
    ]:
        """
        执行一次 PettingZoo Routing action，
        同时完成两件事情：

            1. 继续构造当前 Legacy Routing Transition，
               供现阶段 ReplayBuffer 兼容使用；

            2. 将当前 Routing decision 记录到
               PendingJobTraceStore，
               为后续“Job terminal 后完整链一次性回填 Replay”
               做准备。

        这里必须严格区分：

            environment_done:
                整个环境 Episode 是否结束。

            routing_done:
                当前 Job 的 Routing trajectory 是否结束。

        Routing 动作语义：

            edge_dc:
                当前 Job 的 Routing 尚未结束。
                它真正的 next_state 必须等同一个 Job
                到达下一 Edge DC 后才能确定。

            self:
                当前 Job 的 Routing trajectory 结束，
                但 Job 生命周期没有结束。
                后续进入非 PettingZoo Local Host SAC。

            cloud:
                当前 Job 的 Edge Routing trajectory 结束，
                但 Job 生命周期没有结束。
                后续由 Cloud execution path 继续。

            drop:
                当前 Routing trajectory 与 Job 生命周期同时结束。

        注意：

            loop_next_decision 只是 Trainer 在环境时间线上
            接下来可以继续处理的 Routing DecisionSnapshot。

            它不一定属于当前 Job，
            因此绝不能无条件当作当前 Job 的因果 next_state。
        """

        # ==========================================================
        # 1. 基础参数标准化与检查
        # ==========================================================

        self._require_reset()

        action = int(
            action
        )

        action_type = str(
            action_type
        )

        action_source = str(
            action_source
        )

        if action_type not in {
            "self",
            "edge_dc",
            "cloud",
            "drop",
        }:
            raise ValueError(
                "TransitionCollector 收到未知 Routing action_type："
                f"{action_type}"
            )

        if action_source not in {
            "forced",
            "random",
            "policy",
        }:
            raise ValueError(
                "TransitionCollector 收到未知 action_source："
                f"{action_source}"
            )

        # ==========================================================
        # 2. 在执行动作前先解码 Routing action
        #
        # 这里主要获取：
        #     source_dc_id
        #     target_dc_id
        #
        # target_dc_id 后面会写入 Pending Routing Step。
        #
        # 使用 Environment 已有的 _decode_action()，
        # 不在 Collector 中重新复制一套动作映射逻辑。
        # ==========================================================

        decode_action = getattr(
            self.env,
            "_decode_action",
            None,
        )

        if not callable(
                decode_action
        ):
            raise AttributeError(
                "TransitionCollector 需要环境提供 "
                "_decode_action() 来解析 Routing action。"
            )

        decoded_action = decode_action(
            agent_id=(
                decision.agent_id
            ),
            action=action,
        )

        decoded_action_type = str(
            decoded_action.get(
                "action_type",
                "",
            )
        )

        if (
                decoded_action_type
                != action_type
        ):
            raise RuntimeError(
                "Trainer 给出的 action_type "
                "与 Environment 解码结果不一致："
                f"job={decision.job_id}, "
                f"agent={decision.agent_id}, "
                f"action={action}, "
                f"trainer_type={action_type}, "
                f"decoded_type={decoded_action_type}"
            )

        decoded_target_dc_id = (
            decoded_action.get(
                "target_dc_id",
                None,
            )
        )

        target_dc_id: Optional[str]

        if (
                decoded_target_dc_id
                is None
        ):
            target_dc_id = None
        else:
            target_dc_id = str(
                decoded_target_dc_id
            )

        # ==========================================================
        # 3. 执行 PettingZoo Routing action
        #
        # Host action 绝对不会经过本函数。
        #
        # Self:
        #     env.step() 只产生 pending Host decision。
        #
        # Edge:
        #     env.step() 创建传输并继续推进环境。
        #
        # Cloud:
        #     env.step() 进入 Cloud path。
        #
        # Drop:
        #     env.step() 直接终止当前 Job。
        # ==========================================================

        self.env.step(
            action
        )

        # ==========================================================
        # 4. 更新 Routing Observation History
        #
        # 当前动作已经实际发生，
        # 因此现在才可以写 Routing history。
        # ==========================================================

        self.routing_observation_builder.record_routing_action(
            job_id=(
                decision.job_id
            ),

            action_type=(
                action_type
            ),

            source_dc_id=(
                decision.agent_id
            ),

            action=action,
        )

        # ==========================================================
        # 5. 保存当前 Routing action 的即时 reward
        #
        # 必须使用动作执行前 decision.agent_id。
        #
        # 因为 env.step() 后：
        #
        #     agent_selection
        #     current_agent_id
        #     current_job_id
        #
        # 都可能已经发生变化。
        # ==========================================================

        reward = float(
            self.env.rewards[
                decision.agent_id
            ]
        )

        # ==========================================================
        # 6. 将当前 Routing decision 写入 Pending Job Causal Trace
        #
        # 重要：
        #
        # 这里只记录已经发生的“因果事实”：
        #
        #     state
        #     action
        #     action_type
        #     action_source
        #     immediate_reward
        #     source / target DC
        #
        # 如果当前是 Edge -> Edge：
        #
        #     next_state 现在仍然未知。
        #
        # 必须等同一个 Job 真正到达目标 Edge，
        # 再由 _build_current_decision_snapshot()
        # 调用 resolve_routing_successor() 回填。
        # ==========================================================

        self.pending_trace_store.record_routing_step(
            job_id=(
                decision.job_id
            ),

            agent_id=(
                decision.agent_id
            ),

            agent_index=(
                decision.agent_index
            ),

            env_time=(
                decision.env_time
            ),

            local_obs=(
                decision.local_obs
            ),

            global_state=(
                decision.global_state
            ),

            action=action,

            action_type=(
                action_type
            ),

            action_source=(
                action_source
            ),

            immediate_reward=(
                reward
            ),

            target_dc_id=(
                target_dc_id
            ),
        )

        # ==========================================================
        # 7. Forced Drop 是 Job 本身立即 terminal
        #
        # Drop reward 已经作为 immediate_reward
        # 写入 PendingRoutingStep。
        #
        # 因此这里不能再：
        #
        #     record_reward_event(
        #         reward_delta=reward
        #     )
        #
        # 否则后续 Finalize 时会把 Drop penalty 计算两遍。
        #
        # 这里只标记 Job terminal。
        # ==========================================================

        if action_type == "drop":
            self.pending_trace_store.record_reward_event(
                job_id=decision.job_id,

                env_time=float(
                    self.env.current_time
                ),

                reward_delta=reward,

                reason="forced_drop",

                terminal=True,
            )

        # ==========================================================
        # 8. PettingZoo Agent termination / truncation
        #
        # 这两个字段继续保留当前 Legacy Transition 的原语义。
        # ==========================================================

        terminated = bool(
            self.env.terminations.get(
                decision.agent_id,
                False,
            )
        )

        truncated = bool(
            self.env.truncations.get(
                decision.agent_id,
                False,
            )
        )

        # ==========================================================
        # 9. 判断整个 Environment Episode 是否结束
        # ==========================================================

        environment_done = (
            self._is_episode_done()
        )

        # ==========================================================
        # 10. 判断当前 Job 的 Routing trajectory 是否结束
        #
        # Self:
        #     后面转 Host SAC，所以 Routing terminal。
        #
        # Cloud:
        #     后面不会再产生 Edge Routing decision，
        #     所以 Routing terminal。
        #
        # Drop:
        #     Routing 和 Job 都 terminal。
        #
        # Edge -> Edge:
        #     当前 Job 后续仍会再次产生 Routing decision。
        # ==========================================================

        routing_terminal_action = (
                action_type
                in {
                    "self",
                    "cloud",
                    "drop",
                }
        )

        routing_done = bool(
            environment_done
            or routing_terminal_action
        )

        # ==========================================================
        # 11. 构造 Trainer 环境时间线上的下一 Routing Decision
        #
        # 注意：
        #
        # loop_next_decision
        #
        # 仅用于 Trainer 继续运行。
        #
        # 它可能属于完全不同的 Job。
        #
        # _build_current_decision_snapshot() 内部现在还会调用：
        #
        #     pending_trace_store.resolve_routing_successor(...)
        #
        # 因此：
        #
        # 如果这个 Decision 恰好是同一个 Job
        # Edge -> Edge 后真正到达目标 DC 的状态，
        # 上一 Routing Step 会在这里被正确闭合。
        #
        # 如果是其他 Job，则不会错误连接。
        # ==========================================================

        loop_next_decision: Optional[
            DecisionSnapshot
        ] = None

        if (
                not environment_done
                and self.env.agent_selection
                is not None
        ):
            loop_next_decision = (
                self._build_current_decision_snapshot()
            )

        # ==========================================================
        # 12. 构造当前 Legacy Routing Transition 的 next_state
        #
        # 这里暂时保留旧 ReplayBuffer 训练兼容逻辑。
        #
        # 第十四步真正可信的 Job 因果关系，
        # 已经由 PendingJobTraceStore 单独维护。
        #
        # 后续完成：
        #
        #     Terminal
        #       ↓
        #     Finalize Trace
        #       ↓
        #     RoutingReplayBuffer
        #
        # 后，这一整段 Legacy next_state 逻辑会被删除。
        # ==========================================================

        if routing_done:

            # ------------------------------------------------------
            # Self / Cloud / Drop
            #
            # 对 Routing SAC 来说已经是 terminal，
            # 因此不允许 bootstrap 到其他 Job。
            # ------------------------------------------------------

            next_agent_id = None

            next_agent_index = -1

            next_job_id = None

            next_local_obs = np.zeros(
                int(
                    self.routing_observation_builder
                        .obs_dim
                ),
                dtype=np.float32,
            )

            next_global_state = np.zeros(
                int(
                    self.routing_state_builder
                        .state_dim
                ),
                dtype=np.float32,
            )

        else:

            # ------------------------------------------------------
            # 正常情况下只有 Edge -> Edge 会进入这里。
            #
            # 当前仍然处于 Legacy ReplayBuffer 兼容阶段，
            # 所以先沿用旧的 loop_next_decision。
            #
            # 注意：
            # 这个 next state 不一定属于当前 Job。
            #
            # 当前 Legacy ReplayBuffer 后面仍会通过
            # same-job successor 机制进行修正。
            #
            # 真正新的因果链则已经由：
            #
            #     PendingJobTraceStore
            #
            # 单独维护。
            # ------------------------------------------------------

            if (
                    loop_next_decision
                    is None
            ):
                raise RuntimeError(
                    "非 terminal Routing action 后"
                    "没有得到下一 Routing decision："
                    f"job={decision.job_id}, "
                    f"action_type={action_type}"
                )

            next_agent_id = (
                loop_next_decision.agent_id
            )

            next_agent_index = (
                loop_next_decision.agent_index
            )

            next_job_id = (
                loop_next_decision.job_id
            )

            next_local_obs = (
                loop_next_decision
                    .local_obs
                    .copy()
            )

            next_global_state = (
                loop_next_decision
                    .global_state
                    .copy()
            )

        # ==========================================================
        # 13. 构造 Legacy Routing Transition
        #
        # 注意：
        #
        # 当前第十四步仍然返回 Transition，
        # 因为 train_h_masac.py 目前还需要：
        #
        #     replay_buffer.add(transition)
        #
        # 来维持现阶段 Routing MASAC 可训练。
        #
        # 后续“双 Replay + Terminal Finalize”完成后，
        # execute_and_collect() 将不再负责直接生成
        # 最终训练 Transition。
        # ==========================================================

        transition = Transition(

            agent_id=(
                decision.agent_id
            ),

            agent_index=(
                decision.agent_index
            ),

            job_id=(
                decision.job_id
            ),

            env_time=(
                decision.env_time
            ),

            local_obs=(
                decision.local_obs
                    .copy()
            ),

            global_state=(
                decision.global_state
                    .copy()
            ),

            action=action,

            action_type=(
                action_type
            ),

            reward=(
                reward
            ),

            next_agent_id=(
                next_agent_id
            ),

            next_agent_index=(
                next_agent_index
            ),

            next_job_id=(
                next_job_id
            ),

            next_env_time=float(
                self.env.current_time
            ),

            next_local_obs=(
                next_local_obs
            ),

            next_global_state=(
                next_global_state
            ),

            terminated=(
                terminated
            ),

            truncated=(
                truncated
            ),

            # ======================================================
            # done 表示 Routing-layer terminal，
            # 而不是只表示 Environment Episode terminal。
            # ======================================================
            done=(
                routing_done
            ),
        )

        # ==========================================================
        # 14. Collector 计数
        # ==========================================================

        self.total_transition_count += 1

        self.episode_transition_count += 1

        # ==========================================================
        # 15. 返回
        #
        # transition:
        #     当前阶段继续交给 Legacy Routing ReplayBuffer。
        #
        # loop_next_decision:
        #     Trainer 时间线上的下一 Routing decision。
        #
        # 对 Self：
        #
        #     env.agent_selection == None
        #
        # 因此：
        #
        #     loop_next_decision == None
        #
        # Trainer 下一轮应该转入 Host branch。
        # ==========================================================

        return (
            transition,
            loop_next_decision,
        )

    # 按 PettingZoo AEC 约定清理一个已经终止的智能体
    def drain_one_dead_agent(self) -> bool:
        if not self.env.agents:
            return False
        agent_id = str(self.env.agent_selection)
        is_dead = bool(
            self.env.terminations.get(agent_id, False)
            or self.env.truncations.get(agent_id, False)
        )
        if not is_dead:
            return False

        # 按 PettingZoo AEC 约定，对已经结束的智能体执行一次 None 动作。
        self.env.step(None)
        return True

    # 清理 episode 结束后的全部 dead agent，返回清理次数
    def drain_all_dead_agents(self) -> int:
        count = 0
        while self.env.agents:
            if not self.drain_one_dead_agent():
                break
            count += 1
        return count

    ############################## 辅助函数 ############################################

    # 构造阶段检查环境是否提供采集器需要的接口
    def _validate_environment_interface(self) -> None:
        required_attributes = (
            "possible_agents",
            "agents",
            "agent_selection",
            "current_agent_id",
            "current_job_id",
            "current_time",
            # "local_obs_dim",
            # "global_state_dim",
            "action_dim",
            "drop_action",
            "has_reset",
            "agent_name_mapping",
            "rewards",
            "terminations",
            "truncations",
        )

        for name in required_attributes:
            if not hasattr(self.env, name):
                raise AttributeError(
                    f"环境缺少 TransitionCollector 所需属性：{name}。"
                )

        # observe、state 和 step 不仅要存在，还必须可以调用。
        for method_name in (
                # "observe",
                "step",
        ):

            method = getattr(
                self.env,
                method_name,
                None,
            )

            if not callable(method):
                raise AttributeError(
                    "环境缺少可调用方法："
                    f"{method_name}()。"
                )


        if int(self.env.action_dim) <= 0:
            raise ValueError("action_dim 必须大于 0。")
        if (
                int(
                    self.routing_observation_builder
                            .obs_dim
                )
                <= 0
        ):
            raise ValueError(
                "routing_obs_dim 必须大于 0。"
            )

        if (
                int(
                    self.routing_state_builder
                            .state_dim
                )
                <= 0
        ):
            raise ValueError(
                "routing_global_state_dim "
                "必须大于 0。"
            )
    # 所有采集动作都要求环境先完成reset
    def _require_reset(self) -> None:
        if not bool(self.env.has_reset):
            raise RuntimeError(
                "环境尚未 reset，不能采集 Transition。"
            )

    # 返回活着的、当前决策的agent
    def _get_live_selected_agent(self) -> str:
        agent_id = str(self.env.agent_selection)
        return agent_id

    # 判断整个 episode 结束
    def _is_episode_done(self) -> bool:
        if len(self.env.possible_agents) == 0:
            return True
        return all(
            bool(
                self.env.terminations.get(agent_id, False)
                or self.env.truncations.get(agent_id, False)
            )
            for agent_id in self.env.possible_agents
        )

    # 返回当前决策动作
    def _get_forced_action(self, job_id: str) -> Optional[int]:
        should_drop_fn = getattr(
            self.env, "_should_drop_arrival_job", None
        )
        if bool(should_drop_fn(job_id)):
            return int(self.env.drop_action)
        return None






