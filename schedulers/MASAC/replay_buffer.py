from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, Optional, Protocol

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float32]
IntArray = NDArray[np.int64]
MaskArray = NDArray[np.int8]
BoolArray = NDArray[np.bool_]

# ReplayBuffer.add() 所要求的最小 Transition 接口
class TransitionLike(Protocol):
    agent_id: str
    agent_index: int
    job_id: str
    env_time: float
    local_obs: FloatArray
    global_state: FloatArray
    action_mask: MaskArray
    action: int
    reward: float
    next_agent_id: Optional[str]
    next_agent_index: int
    next_job_id: Optional[str]
    next_env_time: float
    next_local_obs: FloatArray
    next_global_state: FloatArray
    next_action_mask: MaskArray
    terminated: bool
    truncated: bool
    done: bool

@dataclass(frozen=True)
class ReplayBatch:
    agent_indices: IntArray
    local_obs: FloatArray
    global_states: FloatArray
    action_masks: MaskArray
    actions: IntArray
    rewards: FloatArray
    next_agent_indices: IntArray
    next_local_obs: FloatArray
    next_global_states: FloatArray
    next_action_masks: MaskArray
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
            "action_masks": self.action_masks.copy(),
            "actions": self.actions.copy(),
            "rewards": self.rewards.copy(),
            "next_agent_indices": self.next_agent_indices.copy(),
            "next_local_obs": self.next_local_obs.copy(),
            "next_global_states": self.next_global_states.copy(),
            "next_action_masks": self.next_action_masks.copy(),
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
            action_dim: int,
            seed: Optional[int] = None,
            forced_action_value: int = -1,
    ) -> None:

        # 将输入统一转换成int
        capacity = int(capacity)
        local_obs_dim = int(local_obs_dim)
        global_state_dim = int(global_state_dim)
        action_dim = int(action_dim)
        forced_action_value = int(forced_action_value)

        self.capacity = capacity
        self.local_obs_dim = local_obs_dim
        self.global_state_dim = global_state_dim
        self.action_dim = action_dim
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

        self.action_masks = np.zeros(
            (capacity, action_dim),
            dtype=np.int8,
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

        self.next_action_masks = np.zeros(
            (capacity, action_dim),
            dtype=np.int8,
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
        self.latest_job_transition_index: Dict[str, int,] = {}

    def __len__(self) -> int:
        return self._size

    # 返回下一条经验将写入的槽位
    @property
    def position(self) -> int:
        return self._position

    # 维护索引关系
    def _add_trainable_index(self, buffer_index: int,) -> None:
        buffer_index = int(buffer_index)
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


    # 返回当前保存的环境强制动作经验数量
    @property
    def num_forced_actions(self) -> int:
        return int(
            self._size - self._num_trainable
        )

    # 返回当前保存的普通 Actor 动作经验数量
    @property
    def num_trainable_actions(self) -> int:
        return int(
            self._num_trainable
        )

    # 保存一条 Transition
    # 当回放池已满时，本次写入会覆盖最旧经验
    def add(self, transition: TransitionLike) -> None:
        agent_index = int(transition.agent_index)
        action = int(transition.action)
        is_forced_action = (action == self.forced_action_value)

        local_obs = np.asarray(transition.local_obs, dtype=np.float32).copy()
        global_state = np.asarray(transition.global_state, dtype=np.float32).copy()
        action_mask = np.asarray(transition.action_mask, dtype=np.int8).copy()
        next_local_obs = np.asarray(transition.next_local_obs, dtype=np.float32).copy()
        next_global_state = np.asarray(transition.next_global_state, dtype=np.float32).copy()
        next_action_mask = np.asarray(transition.next_action_mask,dtype=np.int8).copy()

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
            self._remove_trainable_index(
                index
            )

            old_job_id = self.job_ids[index]

            if old_job_id is not None:
                old_job_id = str(
                    old_job_id
                )
            if (
                    self.latest_job_transition_index.get(
                        old_job_id
                    )
                    == index
            ):
                self.latest_job_transition_index.pop(
                    old_job_id,
                    None,
                )

        # 将经过检查的数据写入对应数组
        self.agent_indices[index] = agent_index
        self.local_obs[index] = local_obs
        self.global_states[index] = global_state
        self.action_masks[index] = action_mask
        self.actions[index] = action
        self.rewards[index] = reward
        self.next_agent_indices[index] = next_agent_index
        self.next_local_obs[index] = next_local_obs
        self.next_global_states[index] = next_global_state
        self.next_action_masks[index] = next_action_mask
        self.terminated[index] = terminated
        self.truncated[index] = truncated
        self.done[index] = done
        self.is_forced_action[index] = is_forced_action
        self.env_times[index] = env_time
        self.next_env_times[index] = next_env_time
        self.agent_ids[index] = str(transition.agent_id)
        self.job_ids[index] = str(transition.job_id)
        self.latest_job_transition_index[str(transition.job_id)] = index
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

    # 把一个延迟出现的任务结果奖励，修正到该 Job 最后一次调度 Transition 上。
    def apply_reward_correction(self, job_id: str, reward_delta: float,) -> Optional[str]:
        job_id = str(job_id)
        reward_delta = float(reward_delta)

        # O(1) 找到这个 Job 最新经验的位置。
        buffer_index = (self.latest_job_transition_index.get(job_id))

        # 经验可能已经被 ReplayBuffer 覆盖。
        if buffer_index is None:
            return None

        buffer_index = int(buffer_index)

        # 真正修正 SAC 将来采样到的 reward
        self.rewards[buffer_index] = np.float32(float(self.rewards[buffer_index]) + reward_delta)

        agent_id = self.agent_ids[buffer_index]

        if agent_id is None:
            return None

        return str(agent_id)

    # 判断当前是否有足够经验采样一个 batch
    def can_sample(self, batch_size: int, include_forced_actions: bool = False,) -> bool:
        batch_size = int(batch_size)
        if batch_size <= 0:
            return False
        if include_forced_actions:
            available_count = self._size
        else:
            available_count = (
                self._num_trainable
            )
        return self.num_trainable_actions >= batch_size

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
            action_masks=self.action_masks[sampled_indices].copy(),
            actions=self.actions[sampled_indices].copy(),
            rewards=self.rewards[sampled_indices].copy(),
            next_agent_indices=self.next_agent_indices[sampled_indices].copy(),
            next_local_obs=self.next_local_obs[sampled_indices].copy(),
            next_global_states=self.next_global_states[sampled_indices].copy(),
            next_action_masks=self.next_action_masks[sampled_indices].copy(),
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
        self._position = 0
        self._size = 0
        self.total_added = 0
        self._num_trainable = 0
        self._trainable_indices.fill(-1)
        self._trainable_positions.fill(-1)
        self.agent_indices.fill(-1)
        self.local_obs.fill(0.0)
        self.global_states.fill(0.0)
        self.action_masks.fill(0)
        self.actions.fill(self.forced_action_value)
        self.rewards.fill(0.0)
        self.next_agent_indices.fill(-1)
        self.next_local_obs.fill(0.0)
        self.next_global_states.fill(0.0)
        self.next_action_masks.fill(0)
        self.terminated.fill(False)
        self.truncated.fill(False)
        self.done.fill(False)
        self.is_forced_action.fill(False)
        self.env_times.fill(0.0)
        self.next_env_times.fill(0.0)
        self.agent_ids = [None] * self.capacity
        self.job_ids = [None] * self.capacity
        self.next_agent_ids = [None] * self.capacity
        self.next_job_ids = [None] * self.capacity
        self.latest_job_transition_index.clear()





