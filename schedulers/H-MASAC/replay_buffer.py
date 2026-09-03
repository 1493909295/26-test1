from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, Optional, Protocol

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float32]
IntArray = NDArray[np.int64]

BoolArray = NDArray[np.bool_]

# ReplayBuffer.add() 所要求的最小 Transition 接口
class TransitionLike(Protocol):
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
    next_agent_index: int
    next_job_id: Optional[str]
    next_env_time: float
    next_local_obs: FloatArray
    next_global_state: FloatArray

    terminated: bool
    truncated: bool
    done: bool

@dataclass(frozen=True)
class ReplayBatch:
    agent_indices: IntArray
    local_obs: FloatArray
    global_states: FloatArray

    actions: IntArray
    rewards: FloatArray
    next_agent_indices: IntArray
    next_local_obs: FloatArray
    next_global_states: FloatArray

    terminated: BoolArray
    truncated: BoolArray
    done: BoolArray
    is_forced_action: BoolArray
    buffer_indices: IntArray
    agent_ids: NDArray[np.object_]
    job_ids: NDArray[np.object_]
    next_agent_ids: NDArray[np.object_]
    next_job_ids: NDArray[np.object_]
    env_times: NDArray[np.float64]
    next_env_times: NDArray[np.float64]

    # 把 ReplayBatch 转成普通字典
    def as_dict(self) -> Dict[str, np.ndarray]:
        return {
            "agent_indices": self.agent_indices.copy(),
            "local_obs": self.local_obs.copy(),
            "global_states": self.global_states.copy(),
            # "action_masks": self.action_masks.copy(),
            "actions": self.actions.copy(),
            "rewards": self.rewards.copy(),
            "next_agent_indices": self.next_agent_indices.copy(),
            "next_local_obs": self.next_local_obs.copy(),
            "next_global_states": self.next_global_states.copy(),
            # "next_action_masks": self.next_action_masks.copy(),
            "terminated": self.terminated.copy(),
            "truncated": self.truncated.copy(),
            "done": self.done.copy(),
            "is_forced_action": self.is_forced_action.copy(),
            "buffer_indices": self.buffer_indices.copy(),
            "agent_ids": self.agent_ids.copy(),
            "job_ids": self.job_ids.copy(),
            "next_agent_ids": self.next_agent_ids.copy(),
            "next_job_ids": self.next_job_ids.copy(),
            "env_times": self.env_times.copy(),
            "next_env_times": self.next_env_times.copy(),
        }

class ReplayBuffer:
    def __init__(
            self,
            capacity: int,
            local_obs_dim: int,
            global_state_dim: int,
            seed: Optional[int] = None,
            forced_action_value: int = -1,
    ) -> None:

        # 将输入统一转换成int
        capacity = int(capacity)
        local_obs_dim = int(local_obs_dim)
        global_state_dim = int(global_state_dim)

        forced_action_value = int(forced_action_value)

        self.capacity = capacity
        self.local_obs_dim = local_obs_dim
        self.global_state_dim = global_state_dim

        self.forced_action_value = forced_action_value

        self.rng = np.random.default_rng(seed)
        self.seed = seed
        # 下一条经验写入哪个槽位
        self._position = 0
        # 有覆盖经验数，判断当前能否采样
        self._size = 0
        # 无覆盖，没啥用其实，可以统计训练进度
        self.total_added = 0

        self.agent_indices = np.full(
            (capacity,),
            -1,
            dtype=np.int64,
        )

        self.local_obs = np.zeros(
            (capacity, local_obs_dim),
            dtype=np.float32,
        )

        self.global_states = np.zeros(
            (capacity, global_state_dim),
            dtype=np.float32,
        )


        self.actions = np.full(
            (capacity,),
            forced_action_value,
            dtype=np.int64,
        )

        self.actions = np.full(
            (capacity,),
            forced_action_value,
            dtype=np.int64,
        )

        self.rewards = np.zeros(
            (capacity,),
            dtype=np.float32,
        )

        self.next_agent_indices = np.full(
            (capacity,),
            -1,
            dtype=np.int64,
        )

        self.next_local_obs = np.zeros(
            (capacity, local_obs_dim),
            dtype=np.float32,
        )

        self.next_global_states = np.zeros(
            (capacity, global_state_dim),
            dtype=np.float32,
        )


        self.terminated = np.zeros(
            (capacity,),
            dtype=np.bool_,
        )

        self.truncated = np.zeros(
            (capacity,),
            dtype=np.bool_,
        )

        self.done = np.zeros(
            (capacity,),
            dtype=np.bool_,
        )

        self.is_forced_action = np.zeros(
            (capacity,),
            dtype=np.bool_,
        )
        # 这三条用于性能优化
        self._trainable_indices = np.full(
            (capacity,),
            -1,
            dtype=np.int64,
        )
        self._trainable_positions = np.full(
            (capacity,),
            -1,
            dtype=np.int64,
        )
        self._num_trainable = 0
        self._num_forced_actions = 0

        self.env_times = np.zeros(
            (capacity,),
            dtype=np.float64,
        )

        self.next_env_times = np.zeros(
            (capacity,),
            dtype=np.float64,
        )

        # 四个固定长度的列表，用来保存每条经验对应的字符串 ID 信息
        self.agent_ids: list[Optional[str]] = [None] * capacity
        self.job_ids: list[Optional[str]] = [None] * capacity
        self.next_agent_ids: list[Optional[str]] = [None] * capacity
        self.next_job_ids: list[Optional[str]] = [None] * capacity

        self.latest_actor_transition_index: Dict[
            str,
            int,
        ] = {}

        self.pending_edge_transition_index: Dict[
            str,
            int,
        ] = {}

        # 保存未获得后续决策状态的边边调度转移

        self.action_types: list[Optional[str]] = [None] * capacity

    def __len__(self) -> int:
        return self._size

    # 重置当前 episode 的任务级信用分配索引
    def reset_episode_job_tracking(self) -> None:

        self.latest_actor_transition_index.clear()
        self.pending_edge_transition_index.clear()



    # 返回下一条经验将写入的槽位
    @property
    def position(self) -> int:
        return self._position

    # 维护索引关系
    def _add_trainable_index(self, buffer_index: int,) -> None:
        buffer_index = int(buffer_index)
        if int(self._trainable_positions[buffer_index]) != -1:
            return
        trainable_position = self._num_trainable
        self._trainable_indices[trainable_position] = buffer_index
        self._trainable_positions[buffer_index] = trainable_position
        self._num_trainable += 1

    def _remove_trainable_index(self,buffer_index: int,) -> None:
        buffer_index = int(buffer_index)
        remove_position = int(self._trainable_positions[buffer_index])
        if remove_position == -1:
            return
        last_position = self._num_trainable - 1
        last_buffer_index = int(self._trainable_indices[last_position])
        self._trainable_indices[remove_position] = last_buffer_index
        self._trainable_positions[last_buffer_index] = remove_position
        self._trainable_indices[last_position] = -1
        self._trainable_positions[buffer_index] = -1
        self._num_trainable -= 1

    # 回填 Edge 后继状态
    def _resolve_pending_edge_with_same_job_successor(
            self,
            job_id: str,
            successor: TransitionLike,
    ) -> bool:

        job_id = str(job_id)

        pending_index = self.pending_edge_transition_index.get(
            job_id
        )

        # 当前 Job 没有等待回填的 Edge transition。
        if pending_index is None:
            return False

        pending_index = int(pending_index)

        # ============================================================
        # 防御检查 1：
        # ReplayBuffer 是环形结构。
        # 如果 pending_index 对应的槽位已经被其他 Job 覆盖，
        # 就必须删除失效映射，绝不能修改别的 Job 的经验。
        # ============================================================
        if self.job_ids[pending_index] != job_id:
            self.pending_edge_transition_index.pop(
                job_id,
                None,
            )
            return False

        # ============================================================
        # 防御检查 2：
        # Edge transition 的 next_state 必须来自“同一个 Job”
        # 真正到达目标 Edge 后形成的下一次决策状态。
        #
        # 不允许把环境时间线上另一个 Job 的状态作为 next_state。
        # ============================================================
        if str(successor.job_id) != job_id:
            return False

        # ============================================================
        # 核心修改：
        # 使用同一 Job 在目标 Edge 上的决策状态，
        # 回填上一条 Edge transition 的真实 next_state。
        # ============================================================

        self.next_agent_indices[pending_index] = int(
            successor.agent_index
        )

        self.next_local_obs[pending_index] = np.asarray(
            successor.local_obs,
            dtype=np.float32,
        )

        self.next_global_states[pending_index] = np.asarray(
            successor.global_state,
            dtype=np.float32,
        )


        self.next_agent_ids[pending_index] = str(
            successor.agent_id
        )

        self.next_job_ids[pending_index] = job_id

        self.next_env_times[pending_index] = float(
            successor.env_time
        )

        # 当前 Edge transition 的后继仍然是正常决策状态，
        # 所以不能被视为 terminal。
        self.terminated[pending_index] = False
        self.truncated[pending_index] = False
        self.done[pending_index] = False

        # Edge transition 现在已经拥有真实后继状态，
        # 可以正式加入 SAC 可训练经验集合。
        if (
                not bool(self.is_forced_action[pending_index])
                and int(self._trainable_positions[pending_index]) == -1
        ):
            self._add_trainable_index(
                pending_index
            )

        # 当前 pending Edge 已经成功解析。
        self.pending_edge_transition_index.pop(
            job_id,
            None,
        )

        return True

    # 当 Edge 转发后的 Job 在产生下一次正常 Actor 决策前已经直接失败时，把 pending Edge Transition 标记成终态
    def _finalize_pending_edge_as_terminal(self,job_id: str,) -> None:
        job_id = str(job_id)

        pending_index = self.pending_edge_transition_index.pop(
            job_id,
            None,
        )

        if pending_index is None:
            return

        pending_index = int(pending_index)

        if self.job_ids[pending_index] != job_id:
            return

        # ======================== 新增 ========================
        # 该 Edge 行为直接走向任务终止，不允许继续 bootstrap。
        self.next_agent_indices[pending_index] = -1
        self.next_local_obs[pending_index].fill(0.0)
        self.next_global_states[pending_index].fill(0.0)


        self.next_agent_ids[pending_index] = None
        self.next_job_ids[pending_index] = None

        self.next_env_times[pending_index] = self.env_times[
            pending_index
        ]

        # MASAC 当前使用 done 控制是否 bootstrap。
        self.done[pending_index] = True
        # ====================================================

        # 此时 terminal failure reward 会直接写入该 Job 的因果链，
        # 该 Edge Transition 已经完整，可以参与训练。
        if (
                not bool(self.is_forced_action[pending_index])
                and int(self._trainable_positions[pending_index]) == -1
        ):
            self._add_trainable_index(
                pending_index
            )

    # 返回当前保存的环境强制动作经验数量
    @property
    def num_forced_actions(self) -> int:
        return int(
            self._num_forced_actions
        )

    # 返回当前保存的普通 Actor 动作经验数量
    @property
    def num_trainable_actions(self) -> int:
        return int(
            self._num_trainable
        )

    # 当前所有尚未进入 trainable 集合的 transition 数量
    @property
    def num_untrainable_actions(self) -> int:
        return int(
            self._size
            - self._num_trainable
        )

    # 非 forced、但仍不能训练的 transition 数量
    @property
    def num_nonforced_untrainable_actions(self,) -> int:
        return max(
            int(
                self.num_untrainable_actions
                - self.num_forced_actions
            ),
            0,
        )

    # 当前仍等待同 Job 后继状态回填的 Edge transition 数量
    @property
    def num_pending_edge_actions(self,) -> int:
        return int(
            len(
                self.pending_edge_transition_index
            )
        )

    # 保存一条 Transition
    # 当回放池已满时，本次写入会覆盖最旧经验
    def add(self, transition: TransitionLike) -> None:
        agent_index = int(transition.agent_index)
        action = int(transition.action)
        is_forced_action = (action == self.forced_action_value)
        current_job_id = str(transition.job_id)
        current_action_type = str(transition.action_type)
        local_obs = np.asarray(transition.local_obs, dtype=np.float32).copy()
        global_state = np.asarray(transition.global_state, dtype=np.float32).copy()

        next_local_obs = np.asarray(transition.next_local_obs, dtype=np.float32).copy()
        next_global_state = np.asarray(transition.next_global_state, dtype=np.float32).copy()


        reward = float(transition.reward)
        env_time = float(transition.env_time)
        next_env_time = float(transition.next_env_time)
        terminated = bool(transition.terminated)
        truncated = bool(transition.truncated)
        done = bool(transition.done)
        next_agent_index = int(transition.next_agent_index)
        index = self._position

        # 如果当前槽位即将覆盖旧经验，
        # 先清理旧 Job 对该槽位的映射。
        if self._size == self.capacity:
            self._remove_trainable_index(index)

            if bool(
                    self.is_forced_action[
                        index
                    ]
            ):
                self._num_forced_actions -= 1

            old_job_id = self.job_ids[
                index
            ]

            if old_job_id is not None:
                old_job_id = str(old_job_id)

                if (
                        self.latest_actor_transition_index
                                .get(old_job_id)
                        == index
                ):
                    self.latest_actor_transition_index.pop(
                        old_job_id,
                        None,
                    )
                if (self.pending_edge_transition_index.get(old_job_id) == index):
                    self.pending_edge_transition_index.pop(
                        old_job_id,
                        None,
                    )

        if not is_forced_action:
            self._resolve_pending_edge_with_same_job_successor(
                job_id=current_job_id,
                successor=transition,
            )




        # 将经过检查的数据写入对应数组
        self.agent_indices[index] = agent_index
        self.local_obs[index] = local_obs
        self.global_states[index] = global_state

        self.actions[index] = action
        self.rewards[index] = reward
        self.next_agent_indices[index] = next_agent_index
        self.next_local_obs[index] = next_local_obs
        self.next_global_states[index] = next_global_state

        self.terminated[index] = terminated
        self.truncated[index] = truncated
        self.done[index] = done
        self.is_forced_action[index] = is_forced_action
        self.env_times[index] = env_time
        self.next_env_times[index] = next_env_time

        self.agent_ids[index] = str(transition.agent_id)

        # current_job_id = str(transition.job_id)
        self.job_ids[index] = (
            current_job_id
        )

        # ==========================================================
        # 只跟踪 Actor 真正选择的 Transition。
        #
        # forced action：
        #   - 可以保存到 ReplayBuffer 作为环境记录；
        #   - 但不应该成为 terminal reward 的训练归因目标。
        # ==========================================================

        if not is_forced_action:
            self.latest_actor_transition_index[
                current_job_id
            ] = index

        self.action_types[index] = (
            current_action_type
        )

        self.next_agent_ids[index] = (
            None
            if transition.next_agent_id is None
            else str(transition.next_agent_id)
        )
        self.next_job_ids[index] = (
            None
            if transition.next_job_id is None
            else str(transition.next_job_id)
        )

        if not is_forced_action:
            # e2e经验暂时不入池
            if current_action_type == "edge_dc":
                self.pending_edge_transition_index[current_job_id] = index
            else:
                self._add_trainable_index(index)

        # 写入后向后移一位
        self._position = (
            self._position + 1
        ) % self.capacity

        # 经验数量上限
        self._size = min(
            self._size + 1,
            self.capacity,
        )

        self.total_added += 1

    # ==============================================================
    # Delayed Non-Terminal Reward Correction
    # ==============================================================
    def apply_reward_correction(
            self,
            job_id: str,
            reward_delta: float,
    ) -> Optional[str]:
        """
        把一个非 terminal 的 delayed reward correction
        修正到该 Job 最近一次真正由 Actor 选择的
        Routing Transition 上。

        第十七步以后，本函数不再承担 terminal credit assignment。

        也就是说：

            普通 delayed correction
                -> apply_reward_correction()

            Job terminal reward
                -> apply_terminal_reward_to_last_actor_transition()

        本函数不会：
            1. 沿 Routing Chain 分配奖励；
            2. 修改 done；
            3. 修改 next_state；
            4. 修改 transition trainable 状态；
            5. 处理 forced action。

        它只修改最近一个 Actor-controlled Transition 的 reward。
        """

        job_id = str(
            job_id
        )

        reward_delta = float(
            reward_delta
        )

        # ==========================================================
        # 1. 找到该 Job 最近一次真正由 Actor 选择的 Transition。
        #
        # 这里不能再使用：
        #
        #     latest_job_transition_index
        #
        # 因为它可能指向 forced drop。
        #
        # 第十七步以后必须使用：
        #
        #     latest_actor_transition_index
        # ==========================================================

        buffer_index = (
            self.latest_actor_transition_index.get(
                job_id
            )
        )

        # ==========================================================
        # 2. 当前 Job 没有可归因的 Actor Transition。
        #
        # 可能原因：
        #
        #   - 这个 Job 没有产生 Actor-controlled action；
        #   - 对应 Replay 槽位已经被环形 Buffer 覆盖；
        #   - Episode tracking 已经清理。
        #
        # 这里不创建假 Transition，也不抛异常。
        # ==========================================================

        if buffer_index is None:
            return None

        buffer_index = int(
            buffer_index
        )

        # ==========================================================
        # 3. 环形 ReplayBuffer 防御性验证。
        #
        # ReplayBuffer 是固定容量循环覆盖结构。
        #
        # 即使字典里还留着：
        #
        #     Job42 -> index 100
        #
        # index 100 也有可能已经被 Job99 覆盖。
        #
        # 所以在修改 reward 前必须再次验证：
        #
        #     self.job_ids[index] == job_id
        # ==========================================================

        if (
                self.job_ids[
                    buffer_index
                ]
                != job_id
        ):
            # 发现 stale index 后立即删除，
            # 防止以后再次误修改其他 Job 的 reward。
            self.latest_actor_transition_index.pop(
                job_id,
                None,
            )

            return None

        # ==========================================================
        # 4. 防御性检查：
        #
        # latest_actor_transition_index 理论上绝不能指向
        # forced transition。
        #
        # 如果发生，说明 add() 中 Actor/Forced tracking 有 Bug。
        # ==========================================================

        if bool(
                self.is_forced_action[
                    buffer_index
                ]
        ):
            raise RuntimeError(
                "latest_actor_transition_index "
                "错误指向 forced transition："
                f"job={job_id}, "
                f"buffer_index={buffer_index}"
            )

        # ==========================================================
        # 5. 真正进行普通 delayed reward correction。
        #
        # 注意：
        # 这里只修改 reward。
        #
        # 不修改：
        #
        #     done
        #     terminated
        #     truncated
        #     next_state
        #     trainable index
        #
        # 因为本函数处理的是普通 non-terminal correction。
        # ==========================================================

        self.rewards[
            buffer_index
        ] = np.float32(
            float(
                self.rewards[
                    buffer_index
                ]
            )
            + reward_delta
        )

        # ==========================================================
        # 6. 返回发生 reward correction 的 Routing Agent。
        #
        # train_h_masac.py 使用这个返回值进行 Episode
        # reward statistics。
        # ==========================================================

        agent_id = (
            self.agent_ids[
                buffer_index
            ]
        )

        if agent_id is None:
            return None

        return str(
            agent_id
        )

    # ==============================================================
    # Job Terminal Reward Assignment
    # ==============================================================
    def apply_terminal_reward_to_last_actor_transition(
            self,
            job_id: str,
            reward_delta: float,
    ) -> Optional[str]:
        """
        Job terminal 后，把最终 outcome 只归因到该 Job
        最后一次真正由 Actor 选择的 Routing Transition。

        不再：

            1. 根据 Routing chain length 计算权重；
            2. 使用 credit_decay；
            3. 把 terminal reward 人工切分给所有历史动作。

        前序 Routing Step 的长期 credit 通过：

            r_t + gamma * V(s_{t+1})

        自然向前传播。

        特殊情况：
            如果 Job 在 Edge forwarding 后直接 terminal，
            或下一次决策只是 forced drop，
            则最后一个 Actor-controlled Edge Transition
            被直接收口为 terminal transition。
        """

        job_id = str(
            job_id
        )

        reward_delta = float(
            reward_delta
        )

        # ==========================================================
        # 如果最后一个 Edge action 尚未得到 same-job successor，
        # 说明 Job 在下一次正常 Routing Decision 之前已经终止。
        #
        # 将它收口成 terminal transition。
        # ==========================================================

        self._finalize_pending_edge_as_terminal(
            job_id=job_id
        )

        buffer_index = (
            self.latest_actor_transition_index
                .get(job_id)
        )

        if buffer_index is None:
            # 当前 Job 没有任何 Actor-controlled transition。
            # 例如首次状态就被环境强制 Drop。
            self.pending_edge_transition_index.pop(
                job_id,
                None,
            )

            return None

        buffer_index = int(
            buffer_index
        )

        # ==========================================================
        # 环形 Replay 防御。
        # ==========================================================

        if (
                self.job_ids[
                    buffer_index
                ]
                != job_id
        ):
            self.latest_actor_transition_index.pop(
                job_id,
                None,
            )

            self.pending_edge_transition_index.pop(
                job_id,
                None,
            )

            return None

        if bool(
                self.is_forced_action[
                    buffer_index
                ]
        ):
            raise RuntimeError(
                "latest_actor_transition_index "
                "错误指向 forced transition："
                f"job={job_id}, "
                f"buffer_index={buffer_index}"
            )

        # ==========================================================
        # Job 已经 terminal：
        #
        # 最后一个 Actor-controlled transition
        # 不允许再 bootstrap。
        #
        # 对 Self / Cloud：
        #     本来就应该是 done=True。
        #
        # 对 Edge -> forced-drop：
        #     这里把先前已 resolve 的 Edge successor
        #     收口成真正 terminal。
        # ==========================================================

        self.next_agent_indices[
            buffer_index
        ] = -1

        self.next_local_obs[
            buffer_index
        ].fill(
            0.0
        )

        self.next_global_states[
            buffer_index
        ].fill(
            0.0
        )

        self.next_agent_ids[
            buffer_index
        ] = None

        self.next_job_ids[
            buffer_index
        ] = None

        self.done[
            buffer_index
        ] = True

        # ==========================================================
        # Terminal reward 只加到最后一个 Actor transition。
        # ==========================================================

        self.rewards[
            buffer_index
        ] = np.float32(
            float(
                self.rewards[
                    buffer_index
                ]
            )
            + reward_delta
        )

        # 确保它可以被 SAC 采样。
        if (
                int(
                    self._trainable_positions[
                        buffer_index
                    ]
                )
                == -1
        ):
            self._add_trainable_index(
                buffer_index
            )

        agent_id = (
            self.agent_ids[
                buffer_index
            ]
        )

        # Job terminal 后，不再需要这些临时索引。
        self.latest_actor_transition_index.pop(
            job_id,
            None,
        )

        self.pending_edge_transition_index.pop(
            job_id,
            None,
        )

        if agent_id is None:
            return None

        return str(
            agent_id
        )

    # 判断当前是否有足够经验采样一个 batch
    def can_sample(self, batch_size: int, include_forced_actions: bool = False,) -> bool:
        batch_size = int(batch_size)
        if batch_size <= 0:
            return False
        if include_forced_actions:
            available_count = self._size
        else:
            available_count = self._num_trainable

        return int(available_count) >= batch_size

    # 随机采样一个ReplayBatch
    def sample(self,batch_size: int,include_forced_actions: bool = False,replace: bool = False,) -> ReplayBatch:

        # # 生成全部有效槽位下标
        # candidate_indices = np.arange(self._size,dtype=np.int64,)
        #
        # # 默认排除强制动作经验
        # if not include_forced_actions:
        #     forced_flags = self.is_forced_action[
        #         candidate_indices
        #     ]
        #     candidate_indices = candidate_indices[
        #         ~forced_flags
        #     ]
        #
        # # 当前满足条件的经验数量
        # available_count = int(candidate_indices.shape[0])
        #
        # # 随机选择经验槽位
        # sampled_indices = self.rng.choice(
        #     candidate_indices,
        #     size=batch_size,
        #     replace=bool(replace),
        # )

        batch_size = int(batch_size)
        replace = bool(replace)
        if include_forced_actions:
            available_count = self._size
            sampled_indices = self.rng.choice(
                available_count,
                size=batch_size,
                replace=replace,
            )
        else:
            available_count = (self._num_trainable)
            sampled_positions = self.rng.choice(
                available_count,
                size=batch_size,
                replace=replace,
            )
            sampled_indices = (
                self._trainable_indices[
                    sampled_positions
                ]
            )
        sampled_indices = np.asarray(
            sampled_indices,
            dtype=np.int64,
        )
        # 统一转换成 int64 数组
        sampled_indices = np.asarray(
            sampled_indices,
            dtype=np.int64,
        )

        # 根据槽位下标复制出一个独立 batch
        return ReplayBatch(
            agent_indices=self.agent_indices[sampled_indices].copy(),
            local_obs=self.local_obs[sampled_indices].copy(),
            global_states=self.global_states[sampled_indices].copy(),

            actions=self.actions[sampled_indices].copy(),
            rewards=self.rewards[sampled_indices].copy(),
            next_agent_indices=self.next_agent_indices[sampled_indices].copy(),
            next_local_obs=self.next_local_obs[sampled_indices].copy(),
            next_global_states=self.next_global_states[sampled_indices].copy(),

            terminated=self.terminated[sampled_indices].copy(),
            truncated=self.truncated[sampled_indices].copy(),
            done=self.done[sampled_indices].copy(),
            is_forced_action=self.is_forced_action[sampled_indices].copy(),
            buffer_indices=sampled_indices.copy(),
            agent_ids=np.asarray(
                [self.agent_ids[index] for index in sampled_indices],
                dtype=object,
            ),
            job_ids=np.asarray(
                [self.job_ids[index] for index in sampled_indices],
                dtype=object,
            ),
            next_agent_ids=np.asarray(
                [self.next_agent_ids[index] for index in sampled_indices],
                dtype=object,
            ),
            next_job_ids=np.asarray(
                [self.next_job_ids[index] for index in sampled_indices],
                dtype=object,
            ),
            env_times=self.env_times[sampled_indices].copy(),
            next_env_times=self.next_env_times[sampled_indices].copy(),
        )


    # 清空全部经验
    def clear(self) -> None:
        """
        完全清空 ReplayBuffer。

        清理内容包括：

            1. ReplayBuffer 基本位置与数量统计；
            2. 所有数值 Transition 数组；
            3. 所有字符串 ID 元数据；
            4. trainable / forced action 索引；
            5. Job -> 最近 Actor Transition 临时索引；
            6. Edge -> same-job successor 临时索引。

        第十七步以后不再维护：

            latest_job_transition_index
            job_transition_indices

        因为 terminal reward 不再按照整条 Routing Chain
        进行人工 credit allocation。
        """

        # ==========================================================
        # 1. ReplayBuffer 基本状态
        # ==========================================================

        self._position = 0
        self._size = 0

        self.total_added = 0

        # ==========================================================
        # 2. Trainable / Forced Action 统计
        # ==========================================================

        self._num_trainable = 0
        self._num_forced_actions = 0

        self._trainable_indices.fill(
            -1
        )

        self._trainable_positions.fill(
            -1
        )

        # ==========================================================
        # 3. 当前状态
        # ==========================================================

        self.agent_indices.fill(
            -1
        )

        self.local_obs.fill(
            0.0
        )

        self.global_states.fill(
            0.0
        )

        # ==========================================================
        # 4. Action / Reward
        # ==========================================================

        self.actions.fill(
            self.forced_action_value
        )

        self.rewards.fill(
            0.0
        )

        # ==========================================================
        # 5. Next State
        # ==========================================================

        self.next_agent_indices.fill(
            -1
        )

        self.next_local_obs.fill(
            0.0
        )

        self.next_global_states.fill(
            0.0
        )

        # ==========================================================
        # 6. Terminal Flags
        # ==========================================================

        self.terminated.fill(
            False
        )

        self.truncated.fill(
            False
        )

        self.done.fill(
            False
        )

        # ==========================================================
        # 7. Forced Action Flags
        # ==========================================================

        self.is_forced_action.fill(
            False
        )

        # ==========================================================
        # 8. Environment Time
        # ==========================================================

        self.env_times.fill(
            0.0
        )

        self.next_env_times.fill(
            0.0
        )

        # ==========================================================
        # 9. String Metadata
        #
        # 这些是 Python list，
        # 不能使用 numpy.fill()。
        # ==========================================================

        self.agent_ids = (
                [None] * self.capacity
        )

        self.job_ids = (
                [None] * self.capacity
        )

        self.next_agent_ids = (
                [None] * self.capacity
        )

        self.next_job_ids = (
                [None] * self.capacity
        )

        self.action_types = (
                [None] * self.capacity
        )

        # ==========================================================
        # 10. Job-level Temporary Tracking
        #
        # 第十七步以后只保留两类临时索引：
        #
        # latest_actor_transition_index
        #     -> 最近一次真正由 Actor 选择的 Routing Transition
        #
        # pending_edge_transition_index
        #     -> Edge action 等待同 Job successor 回填
        #
        # 不再存在：
        #
        # latest_job_transition_index
        # job_transition_indices
        # ==========================================================

        self.latest_actor_transition_index.clear()

        self.pending_edge_transition_index.clear()




