import gymnasium as gym
from gymnasium import spaces
from pettingzoo import AECEnv
# from pettingzoo.utils import AgentSelector
import heapq
import numpy as np
import functools
import itertools
from enum import Enum
import math
import copy
from typing import Any, Dict, List, Optional, Tuple,Union
import networkx as nx
from environment.datacenter import (Host, DataCenter, hosts_generate, datacenters_generate)
from environment.job import (Job, JobList, jobs_generate)
from environment.env_generate import (EnvGenerator, UseOldEnv)

import config as conf

JOB_ARRIVAL = "JOB_ARRIVAL"
JOB_FINISH = "JOB_FINISH"
DROP_ACTION = -1

class CloudEdgeEnv(AECEnv):
    #  环境元信息
    metadata = {
        "name": "cloud_edge_masac_v0",
        "render_modes": [],
        "is_parallelizable": False,
    }

    JOB_FEAT_DIM = 4
    DC_FEAT_DIM = 8
    HOST_FEAT_DIM = 8

    def __init__(
        self,
        env_source: Optional[Union[EnvGenerator, UseOldEnv]] = None,
        old_env_path: Optional[str] = None,
        seed: Optional[int] = None,
        # invalid_action_mode: str = "raise",
    ):
        super().__init__()

        # 用于检查非法动作处理模式是否合法
        # if invalid_action_mode not in {"raise", "stay", "mask"}:
        #     raise ValueError("invalid_action_mode 必须是 'raise'、'stay' 或 'mask'。")

        # self.invalid_action_mode = invalid_action_mode

        # 随机数种子生成器用于任务分配
        self.rng = np.random.default_rng(seed)
        self.seed_value = seed

########################################### 环境准备与验证模块 ############################################################
        # 物理环境选择模块，判断是新生成环境还是用旧的环境
        if env_source is not None:
            self.env_source = env_source
        elif old_env_path is not None:
            self.env_source = UseOldEnv(old_env_path)
        else:
            self.env_source = EnvGenerator()
            self.env_source.generate_environment(
                lambda_rate=self.env_source.lambda_rate,
                job_dataset_path=self.env_source.job_dataset_path,
                cloud_latency_range=self.env_source.cloud_latency_range,
                edge_latency_range=self.env_source.edge_latency_range,
            )

        # 环境完备性检查
        required_attrs = ["wait_assign_jobs_list","global_dc_list","datacenter_graph",]
        for attr in required_attrs:
            if not hasattr(self.env_source, attr):
                raise AttributeError(
                    f"env_source 缺少必要属性 {attr}，"
                    f"请检查 EnvGenerator 或 UseOldEnv 是否正确生成/加载环境。"
                )
        if len(self.env_source.wait_assign_jobs_list) == 0:
            raise ValueError("环境中的 wait_assign_jobs_list 为空，无法进行训练。")
        if len(self.env_source.global_dc_list) == 0:
            raise ValueError("环境中的 global_dc_list 为空，无法进行训练。")
        if self.env_source.datacenter_graph is None:
            raise ValueError("环境中的 datacenter_graph 为空，无法计算节点间时延。")

        # 保存一份干净的基础环境副本,避免一个 episode 修改 host 队列、job 状态后污染下一个 episode
        self.base_jobs = copy.deepcopy(self.env_source.wait_assign_jobs_list)
        self.base_datacenters = copy.deepcopy(self.env_source.global_dc_list)
        self.base_graph = copy.deepcopy(self.env_source.datacenter_graph)

        # 云节点定义
        self.cloud_id = "cloud"
        dc_ids = [dc.dc_id for dc in self.base_datacenters]

        # 提取边缘dc id
        self.edge_dc_ids: List[str] = [
            dc.dc_id
            for dc in self.base_datacenters
            if dc.dc_id != self.cloud_id
        ]
        if len(self.edge_dc_ids) == 0:
            raise ValueError("环境中没有可作为智能体的边缘数据中心。")

        # 在训练开始时，为每个待调度任务随机指定一个到达的边缘数据中心,后续每个 episode 的 reset() 都会从 self.base_jobs 深拷贝
        # for job in self.base_jobs:
        #     arrived_dc_id = str(self.rng.choice(self.edge_dc_ids))
        #     job.set_target_datacenter(arrived_dc_id)


        # 边缘 DC 在前，cloud 放在最后
        self.all_dc_ids = self.edge_dc_ids + [self.cloud_id]
        # 记录数据中心数量
        self.num_edge_dc = len(self.edge_dc_ids)
        self.num_all_dc = len(self.all_dc_ids)
        # 计算边缘数据中心中最大的 host 数量，不同 DC 的 host 数可能不同。为了构造统一动作空间，需要用最大 host 数量对齐。
        self.max_host_num = max(
            len(dc.host_list)
            for dc in self.base_datacenters
            if dc.dc_id != self.cloud_id
        )

        ######## pettingzoo 要求这俩种获取智能体方法都得有，所以都得写
        # 自己定义的方法，获取边缘数据中心智能体总数
        # self.num_agents = len(self.edge_dc_ids)  pettingzoo里这个是只读的
        self.num_edge_agents = len(self.edge_dc_ids)

        # 获取环境中所有坑智能体数量
        self.possible_agents = self.edge_dc_ids[:]
        # 获取当前活跃智能体数量
        self.agents: List[str] = []
        self.agent_ids = self.possible_agents[:]

        # self.agent_id_to_dc_id = {
        #     agent_id: dc_id
        #     for agent_id, dc_id in zip(self.possible_agents, self.edge_dc_ids)
        # }
        # self.dc_id_to_agent_id = {
        #     dc_id: agent_id
        #     for agent_id, dc_id in self.agent_id_to_dc_id.items()
        # }
        self.agent_name_mapping = {
            agent: i
            for i, agent in enumerate(self.possible_agents)
        }
        # self.remote_edge_dc_ids = {
        #     agent_id: [
        #         dc_id
        #         for dc_id in self.edge_dc_ids
        #         if dc_id != agent_id
        #     ]
        #     for agent_id in self.possible_agents
        # }

        # 取代上面的动作编码方案（在选择datacenter时把自己剔除出去）这种方法会导致相同的动作id代表不同的动作选择
        # 新的方法把所有智能体的动作编码建模成一样的，这样会导致选择前面的host动作与选择本地datacenter是相同意义的动作，
        # 需要在后续第一步掩码时把选自己datacenter设置成非法的
        self.edge_action_start = self.max_host_num
        self.cloud_action_index = self.max_host_num + self.num_edge_dc
        self.edge_action_to_dc_id = {
            self.edge_action_start + idx: dc_id
            for idx, dc_id in enumerate(self.edge_dc_ids)
        }
        ########################################################################################################################

        # 动作空间维度与动作空间
        # self.action_dim = self.max_host_num + (self.num_edge_dc - 1) + 1

        # 新动作建模方法带来的维度改变（维度+1）
        self.action_dim = self.max_host_num + self.num_edge_dc  + 1
        self.action_spaces = {
            agent_id: spaces.Discrete(self.action_dim)
            for agent_id in self.agent_ids
        }

        # 保存单个智能体的动作空间后续可能用
        self.single_action_space = spaces.Discrete(self.action_dim)

        # 这个是为了兼容
        # self.action_space = spaces.Discrete(self.action_dim)

        # 事件队列
        # 队内元素固定为 (event_time, event_type, job_id)
        self.event_queue: List[Tuple[float, str, str]] = []

        # 当前时间
        self.current_time: float = 0.0

        # 当前需要决策的智能体、需要调度的job
        self.current_agent_id = None
        self.current_job_id = None
        self.current_dc_id = None
        self.running_job_location = {}

        # 丢弃机制
        self.drop_deadline_ratio = conf.DROP_DEADLINE_RATE
        self.drop_action = DROP_ACTION

        self.dropped_jobs_info: List[Dict[str, Any]] = []

        # 等待队列统计
        self.queued_jobs: int = 0
        self.started_from_waiting_jobs: int = 0
        self.waiting_timeout_drops: int = 0
        self.max_waiting_queue_length: int = 0

        # 先暂存每个智能体重做，然后再统一执行，我这里貌似不需要这样，我这里来一个任务，选个智能体做个决策，更新下奖励就行
        # self.pending_actions: Dict[str, Optional[int]] = {
        #     agent_id: None
        #     for agent_id in self.possible_agents
        # }


        # 局部观测空间
        # 任务特征维度4，两个资源需求，一个执行时间，一个等待时间
        self.job_feat_dim = self.JOB_FEAT_DIM
        # dc特征维度8，两个资源，两个负载，两个可用，一个等待队列一个执行队列
        self.dc_feat_dim = self.DC_FEAT_DIM
        # host特征维度8，和dc一样
        self.host_feat_dim = self.HOST_FEAT_DIM
        # 链路特征
        self.local_link_target_dc_ids = self.edge_dc_ids + [self.cloud_id]
        self.link_feat_dim = self.num_edge_dc + 1

        # 单个智能体局部观测维度
        self.local_obs_dim = (
                self.job_feat_dim
                + self.dc_feat_dim
                + self.max_host_num * self.host_feat_dim
                + self.link_feat_dim
        )

        # 每个智能体的局部观测空间
        self.local_observation_spaces = {
            agent_id: spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=(self.local_obs_dim,),
                dtype=np.float32,
            )
            for agent_id in self.possible_agents
        }

        # critic 的全局状态空间
        self.global_state_dim = (
                self.job_feat_dim
                +self.num_all_dc * self.dc_feat_dim
                + self.num_all_dc * self.max_host_num * self.host_feat_dim
                + self.num_all_dc * self.num_all_dc
        )
        self.state_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.global_state_dim,),
            dtype=np.float32,
        )
        self.global_state_space = self.state_space

        # 缓存当前状态，性能优化用的
        self._cached_global_state = np.zeros(
            self.global_state_dim,
            dtype=np.float32,
        )

        # 第一套动作掩码，修复 host 维度使用最大host数可能造成的问题
        self.action_mask_spaces = {
            agent_id: spaces.MultiBinary(self.action_dim)
            for agent_id in self.possible_agents
        }

        self.observation_spaces = {
            agent_id: spaces.Dict(
                {
                    "observation": self.local_observation_spaces[agent_id],
                    "action_mask": self.action_mask_spaces[agent_id],
                }
            )
            for agent_id in self.possible_agents
        }
        self.single_observation_space = self.observation_spaces[self.possible_agents[0]]

        self.rewards = {
            agent_id: 0.0
            for agent_id in self.possible_agents
        }

        self._cumulative_rewards = {
            agent_id: 0.0
            for agent_id in self.possible_agents
        }

        # pettingzoo要求记录智能体是否已经自然停止
        self.terminations = {
            agent_id: False
            for agent_id in self.possible_agents
        }

        # pettingzoo要求记录智能体是否因外部限制被强行结束
        self.truncations = {
            agent_id: False
            for agent_id in self.possible_agents
        }

        # 智能体保存的额外信息
        self.infos = {
            agent_id: {
                "agent_index": self.agent_name_mapping[agent_id],
                "dc_id": agent_id,
                "global_state": np.zeros(self.global_state_dim, dtype=np.float32),
                "action_mask": np.ones(self.action_dim, dtype=np.int8),
            }
            for agent_id in self.possible_agents
        }

        # 选择执行动作的智能体（任务到达位置）
        self.agent_selection = None

        # 为当前episode准备运行状态
        self.jobs: List[Job] = []
        self.datacenters: List[DataCenter] = []
        self.graph: Optional[nx.Graph] = None
        self.job_map: Dict[str, Job] = {}
        self.dc_map: Dict[str, DataCenter] = {}

        # 环境重置标记
        self.has_reset = False

        self.timeout_drop_penalty = 3.0
        self.resource_drop_penalty = 2.0

    # PettingZoo 标准接口，返回指定 agent 的观测空间。
    @functools.lru_cache(maxsize=None)
    def observation_space(self, agent):
        return self.observation_spaces[agent]

    # PettingZoo 标准接口，返回指定 agent 的动作空间。
    @functools.lru_cache(maxsize=None)
    def action_space(self, agent):
        return self.action_spaces[agent]

    # MASAC centralized critic 使用的全局状态接口
    # def state(self):
    #     if hasattr(self, "_get_global_state"):
    #         return self._get_global_state()
    #     return np.zeros(self.global_state_dim, dtype=np.float32)
    def state(self) -> np.ndarray:
        return self._cached_global_state.copy()

    # 每轮新训练重启环境
    def reset(self,seed: Optional[int] = None, options: Optional[dict] = None):

        # 如果传入新seed，更新随机数生成器
        if seed is not None:
            self.rng = np.random.default_rng(seed)
            self.seed_value = seed

        self.agents = self.possible_agents[:]

        # 从base_jobs模板恢复当前 episode 的任务、数据中心和拓扑图。
        self.jobs = copy.deepcopy(self.base_jobs)
        self.datacenters = copy.deepcopy(self.base_datacenters)
        self.graph = copy.deepcopy(self.base_graph)

        # 归一化统一不同特征的尺度
        self._init_normalization_stats()

        # 每次episode 开始时，重新随机指定每个 job 到达哪个边缘数据中心
        for job in self.jobs:
            arrived_dc_id = str(self.rng.choice(self.edge_dc_ids))
            job.set_target_datacenter(arrived_dc_id)

        # 构建快速查询映射。
        self.job_map = {
            str(job.job_id): job
            for job in self.jobs
        }
        self.dc_map = {
            dc.dc_id: dc
            for dc in self.datacenters
        }

        # 重置事件队列
        self.event_queue = []
        for job in self.jobs:
            heapq.heappush(
                self.event_queue,
                (
                    float(job.arrive_time),
                    JOB_ARRIVAL,
                    str(job.job_id),
                )
            )

        # 重置仿真时间
        self.current_time = 0.0

        # 重置当前决策相关状态
        self.current_agent_id = None
        self.current_job_id = None
        self.current_dc_id = None

        # 记录运行中任务的位置，reset 时清空
        self.running_job_location = {}
        self.dropped_jobs_info = []

        self.queued_jobs = 0
        self.started_from_waiting_jobs = 0
        self.waiting_timeout_drops = 0
        self.max_waiting_queue_length = 0

        # 重置 PettingZoo AEC 必需状态
        self.rewards = {
            agent_id: 0.0
            for agent_id in self.possible_agents
        }
        self._cumulative_rewards = {
            agent_id: 0.0
            for agent_id in self.possible_agents
        }
        self.terminations = {
            agent_id: False
            for agent_id in self.possible_agents
        }
        self.truncations = {
            agent_id: False
            for agent_id in self.possible_agents
        }

        # 从事件队列中找到第一个任务到达事件
        first_decision_found = False
        while self.event_queue:
            event_time, event_type, job_id = heapq.heappop(self.event_queue)
            self.current_time = event_time
            if event_type != JOB_ARRIVAL:
                continue
            job = self.job_map[job_id]

        # job.target_datacenter 应该已经在 __init__ 中固定，每个 episode 使用的是同一种
            arrived_dc_id = job.target_datacenter
            if arrived_dc_id is None:
                raise ValueError(
                    f"任务 {job.job_id} 没有预先分配 target_datacenter，"
                    f"请检查 __init__ 中是否已经完成 job 到边缘数据中心的固定随机分配。"
                )
            arrived_dc_id = str(arrived_dc_id)

            if arrived_dc_id not in self.possible_agents:
                raise ValueError(
                    f"任务 {job.job_id} 到达的数据中心 {arrived_dc_id} "
                    f"不是合法边缘智能体。"
                )

            self.current_job_id = job_id
            self.current_dc_id = arrived_dc_id
            self.current_agent_id = arrived_dc_id
            self.agent_selection = arrived_dc_id
            first_decision_found = True
            break

        # 如果没有找到任何任务到达事件，则直接结束 episode
        if not first_decision_found:
            for agent_id in self.possible_agents:
                self.terminations[agent_id] = True
            self.agent_selection = self.possible_agents[0]

        # global_state = self._get_global_state()
        global_state = np.asarray(self._get_global_state(), dtype=np.float32)
        self._cached_global_state = global_state.copy()

        self.infos = {
            agent_id: {
                "agent_index": self.agent_name_mapping[agent_id],
                "dc_id": agent_id,
                "global_state": self._cached_global_state,
                "action_mask": self._get_action_mask(agent_id),
            }
            for agent_id in self.possible_agents
        }

        self.has_reset = True

    # 局部观测
    def observe(self, agent: str) -> Dict[str, np.ndarray]:
        # 合法性检查，和_get_local_observation刚开始一样，检查agent合法性和是否reset
        if agent not in self.possible_agents:
            raise ValueError(
                f"非法 agent: {agent}。"
                f"合法智能体应为: {self.possible_agents}。"
            )
        if not self.has_reset:
            raise RuntimeError(
                "环境尚未 reset，不能调用 observe(agent)。"
                "请先调用 reset()。"
            )

        local_observation = self._get_local_observation(agent)
        action_mask = self._get_action_mask(agent)

        # 维度检查
        expected_obs_shape = self.local_observation_spaces[agent].shape
        if local_observation.shape != expected_obs_shape:
            raise ValueError(
                f"observe({agent}) 中 observation 维度错误，"
                f"期望 {expected_obs_shape}，实际 {local_observation.shape}。"
            )
        expected_mask_shape = self.action_mask_spaces[agent].shape
        if action_mask.shape != expected_mask_shape:
            raise ValueError(
                f"observe({agent}) 中 action_mask 维度错误，"
                f"期望 {expected_mask_shape}，实际 {action_mask.shape}。"
            )

        return {
            "observation": local_observation,
            "action_mask": action_mask,
        }

    #
    def step(self, action: Optional[int]) -> None:

        ############################ 基础状态检查 #################################
        if not self.has_reset:
            raise RuntimeError(
                "环境尚未 reset，不能调用 step()。请先调用 reset()。"
            )

        if self.agent_selection is None:
            raise RuntimeError(
                "agent_selection 为空，当前没有可执行动作的智能体。"
            )

        # pettingzoo有控制智能体死活的功能，必须把智能体注册为act状态才行
        acting_agent = str(self.agent_selection)
        if acting_agent not in self.agents:
            raise RuntimeError(
                f"当前 agent_selection={acting_agent} 不在活跃智能体列表 "
                f"agents={self.agents} 中。"
            )

        # PettingZoo AEC 约定：已经终止的智能体只能执行 step(None)，并交由 _was_dead_step() 从 agents 中依次清理。
        if (
                self.terminations.get(acting_agent, False)
                or self.truncations.get(acting_agent, False)
        ):
            self._was_dead_step(action)
            return

        if action is None:
            raise ValueError(
                f"智能体 {acting_agent} 尚未终止，action 不能为 None。"
            )

        if self.current_job_id is None:
            raise RuntimeError(
                "当前没有等待调度的任务，却调用了 step()。"
            )

        if self.current_agent_id != acting_agent:
            raise RuntimeError(
                "当前决策智能体状态不一致："
                f"agent_selection={acting_agent}，"
                f"current_agent_id={self.current_agent_id}。"
            )

        if self.current_dc_id != acting_agent:
            raise RuntimeError(
                "当前任务所在数据中心与决策智能体不一致："
                f"current_dc_id={self.current_dc_id}，"
                f"acting_agent={acting_agent}。"
            )

        acting_job_id = str(self.current_job_id)
        if acting_job_id not in self.job_map:
            raise KeyError(
                f"当前任务 {acting_job_id} 不存在于 job_map 中。"
            )

        acting_job = self.job_map[acting_job_id]

        # PettingZoo 的标准奖励清空处理
        self._cumulative_rewards[acting_agent] = 0.0
        self._clear_rewards()

        ##################### 处理当前任务动作 ######################################

        action_value = int(action)
        should_drop = self._should_drop_arrival_job(acting_job_id)
        action_reward: Optional[float] = None

        if should_drop:
            # if action_value != self.drop_action:
            #     raise ValueError(
            #         f"任务 {acting_job_id} 已满足超时丢弃条件，"
            #         f"此时只能传入 DROP_ACTION={self.drop_action}，"
            #         f"实际收到 action={action_value}。"
            #     )

            self._drop_arrival_job(
                job_id=acting_job_id,
                drop_reasion="等待超时",
            )

            action_reward = self._compute_action_reward(
                job_id=acting_job_id,
                action_type="drop",
                success=False,
                transfer_latency=0.0,
                failure_reason="等待超时",
            )

        else:
            # 解码动作
            decoded_action = self._decode_action(
                agent_id=acting_agent,
                action=action_value,
            )
            action_type = str(decoded_action["action_type"])

            if action_type == "local_host":
                host_idx = decoded_action["host_idx"]
                # success = self._execute_job_on_host(
                #     job_id=acting_job_id,
                #     dc_id=acting_agent,
                #     host_idx=int(host_idx),
                # )
                #
                # failure_reason = None if success else "资源不足"
                # action_reward = self._compute_action_reward(
                #     job_id=acting_job_id,
                #     action_type="local_host",
                #     success=success,
                #     transfer_latency=0.0,
                #     failure_reason=failure_reason,
                # )
                execution_result = self._execute_job_on_host(
                    job_id=acting_job_id,
                    dc_id=acting_agent,
                    host_idx=int(host_idx),
                )

                success = False
                failure_reason = None

                if execution_result == "started":
                    success = True
                    failure_reason = None

                elif execution_result == "queued":
                    success = True
                    failure_reason = None

                elif execution_result == "dropped":
                    # 真正丢弃时才视为本地调度失败。
                    success = False
                    failure_reason = "资源不足"

                # success = execution_result != "dropped"
                # failure_reason = (
                #     "资源不足"
                #     if execution_result == "dropped"
                #     else None
                # )

                action_reward = self._compute_action_reward(
                    job_id=acting_job_id,
                    action_type="local_host",
                    success=success,
                    transfer_latency=0.0,
                    failure_reason=failure_reason,
                )

            elif action_type in {"edge_dc", "cloud"}:
                target_dc_id = decoded_action["target_dc_id"]
                arrival_event_time = self._enqueue_transfer_arrival_event(
                    job=acting_job,
                    source_dc_id=acting_agent,
                    target_dc_id=str(target_dc_id),
                )
                transfer_latency = (
                        float(arrival_event_time)
                        - float(self.current_time)
                )
                action_reward = self._compute_action_reward(
                    job_id=acting_job_id,
                    action_type=action_type,
                    success=True,
                    transfer_latency=transfer_latency,
                    failure_reason=None,
                )

        # 完成动作的收尾
        self.rewards[acting_agent] = float(action_reward)
        self._clear_current_decision()

        # 推进事件队列
        next_decision_found = False
        while self.event_queue:
            current_event_time = float(self.event_queue[0][0])
            self.current_time = current_event_time
            pending_arrival_events = []
            while (
                    self.event_queue
                    and float(self.event_queue[0][0])
                    == current_event_time
            ):
                event_time, event_type, event_job_id = heapq.heappop(self.event_queue)

                event_time = float(event_time)
                event_job_id = str(event_job_id)

                if event_type == JOB_FINISH:
                    self._process_job_finish_event(
                        event_job_id
                    )
                    continue

                if event_type == JOB_ARRIVAL:
                    pending_arrival_events.append(
                        (
                            event_time,
                            event_type,
                            event_job_id,
                        )
                    )
                    continue

            arrival_index = 0

            while arrival_index < len(pending_arrival_events):
                while (
                        self.event_queue
                        and float(self.event_queue[0][0])
                        == current_event_time
                ):
                    new_event_time, new_event_type, new_event_job_id = (
                        heapq.heappop(self.event_queue)
                    )

                    new_event_time = float(new_event_time)
                    new_event_job_id = str(new_event_job_id)

                    if new_event_type == JOB_FINISH:
                        self._process_job_finish_event(
                            new_event_job_id
                        )
                        continue

                    if new_event_type == JOB_ARRIVAL:
                        pending_arrival_events.append(
                            (
                                new_event_time,
                                new_event_type,
                                new_event_job_id,
                            )
                        )
                        continue

                (
                    event_time,
                    event_type,
                    event_job_id,
                ) = pending_arrival_events[arrival_index]

                arrival_index += 1

                arrived_job = self.job_map[event_job_id]
                arrived_dc_id = str(
                    arrived_job.target_datacenter
                )

                if arrived_dc_id == self.cloud_id:
                    self._execute_job_on_host(
                        job_id=event_job_id,
                        dc_id=self.cloud_id,
                        host_idx=0,
                    )

                    # Cloud arrival 不需要暂停 AEC 环境，
                    # 继续处理同一时间点剩余事件。
                    continue

                if arrived_dc_id in self.possible_agents:
                    # 该任务需要交给对应边缘智能体做一次调度决策。
                    self.current_job_id = event_job_id
                    self.current_dc_id = arrived_dc_id
                    self.current_agent_id = arrived_dc_id
                    self.agent_selection = arrived_dc_id

                    next_decision_found = True

                    for remaining_event in (
                            pending_arrival_events[arrival_index:]
                    ):
                        heapq.heappush(
                            self.event_queue,
                            remaining_event,
                        )

                    break
            if next_decision_found:
                break

            #
            # event_time, event_type, event_job_id = heapq.heappop(self.event_queue)
            # event_time = float(event_time)
            # event_job_id = str(event_job_id)
            # self.current_time = event_time

            # if event_type == JOB_FINISH:
            #     self._process_job_finish_event(event_job_id)
            #     continue

            # if event_type == JOB_ARRIVAL:
            #     arrived_job = self.job_map[event_job_id]
            #     arrived_dc_id = str(arrived_job.target_datacenter)
            #
            #     # 任务到来事件来自云
            #     if arrived_dc_id == self.cloud_id:
            #         cloud_dc = self.dc_map[self.cloud_id]
            #         self._execute_job_on_host(
            #             job_id=event_job_id,
            #             dc_id=self.cloud_id,
            #             host_idx=0,
            #         )
            #         continue
            #
            #     # 任务到来事件来自边
            #     if arrived_dc_id in self.possible_agents:
            #         self.current_job_id = event_job_id
            #         self.current_dc_id = arrived_dc_id
            #         self.current_agent_id = arrived_dc_id
            #         self.agent_selection = arrived_dc_id
            #         next_decision_found = True
            #         break

        # episode 结束处理
        if not next_decision_found:
            if self._check_episode_finished():
                self._terminate_episode()

        # 更新 infos
        # global_state = self._get_global_state()
        global_state = np.asarray(self._get_global_state(), dtype=np.float32)
        self._cached_global_state = global_state.copy()

        for agent_id in self.possible_agents:
            # episode 结束后不应再提供普通合法动作，终止智能体 mask 全置零。
            if (
                    self.terminations.get(agent_id, False)
                    or self.truncations.get(agent_id, False)
            ):
                updated_action_mask = np.zeros(
                    self.action_dim,
                    dtype=np.int8,
                )
            else:
                updated_action_mask = self._get_action_mask(agent_id)

            self.infos[agent_id] = {
                "agent_index": self.agent_name_mapping[agent_id],
                "dc_id": agent_id,
                "global_state": self._cached_global_state,
                "action_mask": updated_action_mask,
            }

        # 将本次即时奖励累积到 PettingZoo 的 _cumulative_rewards 中。
        self._accumulate_rewards()

    ################################### 辅助函数部分 ####################################

    # 掩码掉因为要对齐host数量导致的部分host可能不合法行为
    def _get_action_mask(self, agent_id: str) -> np.ndarray:
        #默认所有动作合法
        mask = np.ones(self.action_dim, dtype=np.int8)

        local_dc = self.dc_map[agent_id]

        # 获取当前正在等待该智能体决策
        current_job = None
        if (
                self.current_job_id is not None
                and self.current_job_id in self.job_map
                and self.current_agent_id == agent_id
        ):
            current_job = self.job_map[self.current_job_id]

        # 屏蔽由于 max_host_num 对齐产生的、不存在的本地 host
        for host_idx in range(self.max_host_num):
            if host_idx >= len(local_dc.host_list):
                mask[host_idx] = 0
                continue
            host = local_dc.host_list[host_idx]
            if current_job is not None:
                if not host.can_ever_accommodate(current_job):
                    mask[host_idx] = 0
                    continue

        for edge_idx, target_dc_id in enumerate(self.edge_dc_ids):
            action_idx = self.edge_action_start + edge_idx
            # 屏蔽“通过边缘卸载槽位卸载到自己”的重复语义动作
            if target_dc_id == agent_id:
                mask[action_idx] = 0
                continue
            # 只要目标dc和链路都存在就合法
            if target_dc_id not in self.dc_map:
                mask[action_idx] = 0
                continue
            if self.graph is not None and not self.graph.has_edge(agent_id, target_dc_id):
                mask[action_idx] = 0

        # 最后一个动作是卸载到云
        cloud_action_idx = self.cloud_action_index

        if self.cloud_id not in self.dc_map:
            mask[cloud_action_idx] = 0
        elif self.graph is not None and not self.graph.has_edge(agent_id, self.cloud_id):
            mask[cloud_action_idx] = 0
        elif current_job is not None:
            cloud_dc = self.dc_map[self.cloud_id]
            if len(cloud_dc.host_list) == 0:
                mask[cloud_action_idx] = 0
            else:
                cloud_host = cloud_dc.host_list[0]
                if not cloud_host.can_ever_accommodate(current_job):
                    mask[cloud_action_idx] = 0
        return mask

    # 将一个 DataCenter 编码成长度为 self.dc_feat_dim 的特征向量
    def _encode_dc_features(self, dc: DataCenter) -> List[float]:

        # 先更新负载情况
        dc.calculate_dc_loads()

        # 计算总资源容量
        total_cpu = sum(host.cpu_num for host in dc.host_list)
        total_gpu = sum(host.gpu_capacity_num for host in dc.host_list)
        # 计算总负载情况
        running_cpu = sum(
            float(host.used_cpu)
            for host in dc.host_list
        )
        running_gpu = sum(
            float(host.used_gpu)
            for host in dc.host_list
        )
        # 计算资源剩余情况
        available_cpu = max(total_cpu - running_cpu, 0.0)
        available_gpu = max(total_gpu - running_gpu, 0.0)
        # 计算队列情况
        waiting_jobs = sum(len(host.waiting_queue) for host in dc.host_list)
        running_jobs = sum(len(host.running_queue) for host in dc.host_list)

        # return [
        #     float(total_cpu),
        #     float(total_gpu),
        #     float(dc.dc_cpu_load),
        #     float(dc.dc_gpu_load),
        #     float(available_cpu),
        #     float(available_gpu),
        #     float(waiting_jobs),
        #     float(running_jobs),
        # ]
        # 归一化return
        return [
            self._normalize(total_cpu, self.max_dc_cpu),
            self._normalize(total_gpu, self.max_dc_gpu),
            float(np.clip(dc.dc_cpu_load, 0.0, 1.0)),
            float(np.clip(dc.dc_gpu_load, 0.0, 1.0)),
            self._normalize(available_cpu, self.max_dc_cpu),
            self._normalize(available_gpu, self.max_dc_gpu),
            self._normalize(waiting_jobs, self.max_queue_len),
            self._normalize(running_jobs, self.max_queue_len),
        ]

    #  将一个 Host 编码成长度为 self.host_feat_dim 的特征向量
    #  逻辑与_encode_dc_features基本相同
    def _encode_host_features(self, host: Host) -> List[float]:
        host.calculate_load()
        running_cpu = float(host.used_cpu)
        running_gpu = float(host.used_gpu)
        available_cpu = max(host.cpu_num - running_cpu, 0.0)
        available_gpu = max(host.gpu_capacity_num - running_gpu, 0.0)
        waiting_jobs = len(host.waiting_queue)
        running_jobs = len(host.running_queue)
        # return [
        #     float(host.cpu_num),
        #     float(host.gpu_capacity_num),
        #     float(host.cpu_load),
        #     float(host.gpu_load),
        #     float(available_cpu),
        #     float(available_gpu),
        #     float(waiting_jobs),
        #     float(running_jobs),
        # ]
        return [
            self._normalize(host.cpu_num, self.max_host_cpu),
            self._normalize(host.gpu_capacity_num, self.max_host_gpu),
            float(np.clip(host.cpu_load, 0.0, 1.0)),
            float(np.clip(host.gpu_load, 0.0, 1.0)),
            self._normalize(available_cpu, self.max_host_cpu),
            self._normalize(available_gpu, self.max_host_gpu),
            self._normalize(waiting_jobs, self.max_queue_len),
            self._normalize(running_jobs, self.max_queue_len),
        ]

    # 把当前等待调度的任务编码成长度为 self.job_feat_dim 的特征向量
    def _encode_job_features(self, job: Optional[Job]) -> List[float]:
        if job is None:
            return [0.0] * self.job_feat_dim
        waiting_time = max(self.current_time - float(job.arrive_time), 0.0)
        # return [
        #     float(job.cpu_request),
        #     float(job.gpu_request),
        #     float(job.duration),
        #     float(waiting_time),
        # ]
        # 新的return是归一化结果
        return [
            self._normalize(job.cpu_request, self.max_job_cpu),
            self._normalize(job.gpu_request, self.max_job_gpu),
            self._normalize(job.duration, self.max_job_duration),
            self._normalize(waiting_time, self.max_arrive_time),
        ]

    def _encode_local_link_features(self, agent_id: str) -> List[float]:
        # 用于临时保存链路特征
        link_features: List[float] = []

        # 遍历固定的链路观测目标列表
        for target_dc_id in self.local_link_target_dc_ids:
            # 到自身时延是0
            if target_dc_id == agent_id:
                latency = 0.0
            elif self.graph is not None and self.graph.has_edge(agent_id, target_dc_id):
                latency = float(self.graph[agent_id][target_dc_id].get("weight", 0.0))
            else:
                latency = 0.0
            # 将链路时延归一化后加入局部观测
            link_features.append(self._normalize(latency, self.max_latency))

        if len(link_features) != self.link_feat_dim:
            raise ValueError(
                f"局部链路特征维度错误，期望 {self.link_feat_dim}，"
                f"实际 {len(link_features)}。"
            )
        return link_features

    # 将动作编码解码成动作，动作编码必须是int类型
    def _decode_action(self, agent_id: str, action: int) -> Dict[str, Any]:
        action = int(action)
        if action == self.drop_action:
            return {
                "action_type": "drop",
                "source_dc_id": agent_id,
                "target_dc_id": None,
                "host_idx": None,
            }
        if 0 <= action < self.max_host_num:
            return {
                "action_type": "local_host",
                "source_dc_id": agent_id,
                "target_dc_id": agent_id,
                "host_idx": action,
            }
        if self.edge_action_start <= action < self.cloud_action_index:
            target_dc_id = self.edge_action_to_dc_id[action]

            return {
                "action_type": "edge_dc",
                "source_dc_id": agent_id,
                "target_dc_id": target_dc_id,
                "host_idx": None,
            }
        if action == self.cloud_action_index:
            return {
                "action_type": "cloud",
                "source_dc_id": agent_id,
                "target_dc_id": self.cloud_id,
                "host_idx": None,
            }

        raise ValueError(
            f"无法解码动作 action={action}，"
            f"当前动作空间范围应为 [-1] 或 [0, {self.action_dim - 1}]。"
        )

    # 构造 MASAC centralized critic 使用的全局状态
    def _get_global_state(self) -> np.ndarray:
        state = []

        # 当前任务的特征
        if self.current_job_id is not None and self.current_job_id in self.job_map:
            current_job = self.job_map[self.current_job_id]
        else:
            current_job = None
        state.extend(self._encode_job_features(current_job))

        # 所有数据中心特征
        for dc_id in self.all_dc_ids:
            dc = self.dc_map[dc_id]
            state.extend(self._encode_dc_features(dc))

        # 所有 host 特征
        for dc_id in self.all_dc_ids:
            dc = self.dc_map[dc_id]
            for host_idx in range(self.max_host_num):
                if host_idx < len(dc.host_list):
                    host = dc.host_list[host_idx]
                    state.extend(self._encode_host_features(host))
                else:
                    # 不足 max_host_num 的 host 用 0 padding。
                    state.extend([0.0] * self.host_feat_dim)

        # 数据中心之间的时延矩阵
        for src_dc_id in self.all_dc_ids:
            for dst_dc_id in self.all_dc_ids:
                if src_dc_id == dst_dc_id:
                    latency = 0.0
                elif self.graph is not None and self.graph.has_edge(src_dc_id, dst_dc_id):
                    latency = float(self.graph[src_dc_id][dst_dc_id].get("weight", 0.0))
                else:
                    latency = 0.0

                state.append(self._normalize(latency, self.max_latency))

        state = np.asarray(state, dtype=np.float32)
        if state.shape != self.state_space.shape:
            raise ValueError(
                f"global_state 维度错误，期望 {self.state_space.shape}，"
                f"实际 {state.shape}。"
            )

        return state

    # 初始化 observation/state 归一化所需的尺度参数
    def _init_normalization_stats(self):
        # 避免出现0设定的极小正数
        eps = 1e-8
        self.norm_eps = eps

        # 统计任务相关特征的最大值，后续可以用这些最大值对 job 特征做 max-scale 归一化
        self.max_job_cpu = max(
            max(float(job.cpu_request) for job in self.base_jobs),
            eps,
        )
        self.max_job_gpu = max(
            max(float(job.gpu_request) for job in self.base_jobs),
            eps,
        )
        self.max_job_duration = max(
            max(float(job.duration) for job in self.base_jobs),
            eps,
        )
        self.max_arrive_time = max(
            max(float(job.arrive_time) for job in self.base_jobs),
            eps,
        )

        # 统计 host 级别和 datacenter 级别的资源容量最大值
        all_hosts = [
            host
            for dc in self.base_datacenters
            for host in dc.host_list
        ]
        self.max_host_cpu = max(
            max(float(host.cpu_num) for host in all_hosts),
            eps,
        )
        self.max_host_gpu = max(
            max(float(host.gpu_capacity_num) for host in all_hosts),
            eps,
        )
        self.max_dc_cpu = max(
            max(float(sum(host.cpu_num for host in dc.host_list)) for dc in self.base_datacenters),
            eps,
        )
        self.max_dc_gpu = max(
            max(float(sum(host.gpu_capacity_num for host in dc.host_list)) for dc in self.base_datacenters),
            eps,
        )

        # 队列长度上限暂时用任务总数代替，队列长度肯定是比这个短的，而且短的多，就怕任务总数太大了
        self.max_queue_len = max(float(len(self.base_jobs)), 1.0)

        # 统计拓扑图中最大的链路时延
        latencies = []

        # 如果基础拓扑图存在，就遍历图中的所有边。
        if self.base_graph is not None:
            for _, _, data in self.base_graph.edges(data=True):
                latencies.append(float(data.get("weight", 0.0)))
        self.max_latency = max(max(latencies) if latencies else 1.0, eps)

    # 安全除法，避免 scale 为 0
    def _safe_div(self, value: float, scale: float) -> float:
        return float(value) / max(float(scale), self.norm_eps)

    # 执行归一化
    def _normalize(self, value: float, scale: float) -> float:
        return float(np.clip(float(value) / max(float(scale), self.norm_eps), 0.0, 1.0))

    # 构造单个边缘智能体的局部观测
    def _get_local_observation(self, agent_id: str) -> np.ndarray:

        # 先合法性检查好，主要怕把cloud传进来
        if agent_id not in self.possible_agents:
            raise ValueError(
                f"非法 agent_id: {agent_id}。"
                f"合法智能体应为边缘数据中心: {self.possible_agents}。"
            )

        # 确保环境经过初始化
        if self.graph is None or len(self.dc_map) == 0:
            raise RuntimeError(
                "当前 episode 尚未初始化，无法构造局部观测。"
                "请先调用 reset()。"
            )

        # 确保向量归一化，其实reset里写过归一化，上一条通过了这个一定会过
        if not hasattr(self, "norm_eps") or not hasattr(self, "max_job_cpu"):
            self._init_normalization_stats()

        # 获取当前等待调度的任务
        if self.current_job_id is not None and self.current_job_id in self.job_map:
            current_job = self.job_map[self.current_job_id]
        else:
            current_job = None

        # 获取当前智能体对应的dc
        if agent_id not in self.dc_map:
            raise KeyError(
                f"agent_id={agent_id} 不在当前 episode 的 dc_map 中，"
                f"请检查 reset() 是否正确构建数据中心映射。"
            )
        local_dc = self.dc_map[agent_id]

        # 拼接xiangli
        obs: List[float] = []
        obs.extend(self._encode_job_features(current_job))
        obs.extend(self._encode_dc_features(local_dc))
        # host向量维度要向最大的看齐，不足的用0补全
        for host_idx in range(self.max_host_num):
            if host_idx < len(local_dc.host_list):
                obs.extend(self._encode_host_features(local_dc.host_list[host_idx]))
            else:
                obs.extend([0.0] * self.host_feat_dim)
        obs.extend(self._encode_local_link_features(agent_id))

        obs_array = np.asarray(obs, dtype=np.float32)

        expected_shape = self.local_observation_spaces[agent_id].shape
        if obs_array.shape != expected_shape:
            raise ValueError(
                f"local observation 维度错误，期望 {expected_shape}，"
                f"实际 {obs_array.shape}。"
            )

        return obs_array

    # 任务是否丢弃判断
    def _should_drop_arrival_job(self, job_id: str) -> bool:
        job_id = str(job_id)
        job = self.job_map[job_id]
        waiting_time = max(
            float(self.current_time) - float(job.arrive_time),
            0.0,
        )
        drop_threshold = self.drop_deadline_ratio * float(job.duration)
        return waiting_time > drop_threshold

    # 任务丢弃记录
    def _drop_arrival_job(self, job_id: str,drop_reasion: str) -> None:
        job_id = str(job_id)
        job = self.job_map[job_id]
        self.dropped_jobs_info.append(
            {
                "job_id": job_id,
                "drop_time": float(self.current_time),
                "drop_reasion": drop_reasion,
            }
        )

    # 打印丢弃任务信息
    def print_dropped_jobs(self) -> None:
        dropped_num = len(self.dropped_jobs_info)
        print("\n" + "=" * 60)
        print("任务丢弃统计")
        print("=" * 60)
        print(f"当前 episode 被丢弃的任务数量: {dropped_num}")

        if dropped_num == 0:
            print("当前 episode 暂无被丢弃任务。")
            print("=" * 60 + "\n")
            return

        print("\n被丢弃任务列表:")
        for idx, drop_info in enumerate(self.dropped_jobs_info, start=1):
            job_id = drop_info.get("job_id", "UNKNOWN")
            drop_time = drop_info.get("drop_time", None)
            drop_reasion = drop_info.get("drop_reasion", "UNKNOWN")
            print("-" * 60)
            print(f"{idx}. 任务 ID: {job_id}")
            print(f"丢弃原因：{drop_reasion}")
            if drop_time is not None:
                print(f"   丢弃时间 drop_time        : {drop_time:.4f}")

        print("=" * 60 + "\n")

    # 调度到其他地方计算的job，打包成新到达事件
    def _enqueue_transfer_arrival_event(self, job: Job, source_dc_id: str, target_dc_id: str,) -> float:
        job_id = str(job.job_id)
        source_dc_id = str(source_dc_id)
        target_dc_id = str(target_dc_id)

        latency = float(
            self.graph[source_dc_id][target_dc_id].get("weight", 0.0)
        )
        arrival_event_time = float(self.current_time) + latency
        job.set_target_datacenter(target_dc_id)

        heapq.heappush(
            self.event_queue,
            (
                arrival_event_time,
                JOB_ARRIVAL,
                job_id,
            )
        )
        return arrival_event_time

    # 真正启动一个任务到host上运行
    def _start_job_on_host(self, job_id: str, dc_id: str, host_idx: int,) -> bool:
        job_id = str(job_id)
        dc_id = str(dc_id)
        host_idx = int(host_idx)
        job = self.job_map[job_id]
        target_dc = self.dc_map[dc_id]
        target_host = target_dc.host_list[host_idx]

        started = target_host.add_to_running_queue(job=job, current_time=float(self.current_time),)

        if not started:
            return False

        # 添加任务完成事件到队列
        target_dc.calculate_dc_loads()
        self.running_job_location[job_id] = {"dc_id": dc_id, "host_idx": host_idx,}
        finish_time = (float(self.current_time) + float(job.duration))
        heapq.heappush(self.event_queue,(finish_time, JOB_FINISH, job_id,))

        return True

    # 调度动作执行给环境
    def _execute_job_on_host(self, job_id: str, dc_id: str, host_idx: int,) -> str:
        job_id = str(job_id)
        dc_id = str(dc_id)
        host_idx = int(host_idx)

        job = self.job_map[job_id]
        target_dc = self.dc_map[dc_id]
        target_host = target_dc.host_list[host_idx]

        drop_reasion_1 = "等待超时"
        drop_reasion_2 = "资源不足"

        # 任务已经经过了太久的调度，被扔掉了
        if self._should_drop_arrival_job(job_id):
            self._drop_arrival_job(job_id,drop_reasion_1)
            # self.running_job_location.pop(job_id, None)
            return "dropped"

        job.set_target_datacenter(dc_id)

        # host总资源不够
        if not target_host.can_ever_accommodate(job):
            self._drop_arrival_job(job_id, drop_reasion_2,)
            return "dropped"

        # 等待队列有人物，新来的也等待
        if not target_host.waiting_queue.is_empty():
            target_host.add_to_waiting_queue(job)
            self.queued_jobs += 1
            self.max_waiting_queue_length = max(
                self.max_waiting_queue_length,
                len(target_host.waiting_queue),
            )
            return "queued"

        # 可接受
        if target_host.can_accommodate(job):
            started = self._start_job_on_host(
                job_id=job_id,
                dc_id=dc_id,
                host_idx=host_idx,
            )
            if started:
                return "started"
            target_host.add_to_waiting_queue(job)
            self.queued_jobs += 1
            self.max_waiting_queue_length = max(
                self.max_waiting_queue_length,
                len(target_host.waiting_queue),
            )

            return "queued"

        target_host.add_to_waiting_queue(job)
        self.queued_jobs += 1
        self.max_waiting_queue_length = max(
            self.max_waiting_queue_length,
            len(target_host.waiting_queue),
        )
        return "queued"
        # started = self._start_job_on_host(
        #     job_id=job_id,
        #     dc_id=dc_id,
        #     host_idx=host_idx,
        # )
        # 因资源不足被丢弃
        # if not started:
        #     self._drop_arrival_job(job_id,drop_reasion_2)
        #     # self.running_job_location.pop(job_id, None)
        #     return  False

        # 成功卸载后更新dc负载
        # target_dc.calculate_dc_loads()
        # self.running_job_location[job_id] = {
        #     "dc_id": dc_id,
        #     "host_idx": host_idx,
        # }
        #
        # # 创建任务完成事件
        # finish_time = (float(self.current_time)+ float(job.duration))
        # heapq.heappush(self.event_queue,(finish_time,JOB_FINISH,job_id,))
        # return True

    # Host 释放资源后，严格按照 FCFS 尝试启动 waiting_queue 中的任务
    def _drain_host_waiting_queue(self, dc_id: str, host_idx: int,) -> None:
        dc_id = str(dc_id)
        host_idx = int(host_idx)

        target_dc = self.dc_map[dc_id]
        target_host = target_dc.host_list[host_idx]

        # 得用while，因为可能一次资源释放能满足多个等待队列中的任务同时上
        while not target_host.waiting_queue.is_empty():
            waiting_job = target_host.waiting_queue._queue[0]
            waiting_job_id = str(waiting_job.job_id)
            if self._should_drop_arrival_job(waiting_job_id):
                dropped_job = target_host.remove_from_waiting_queue()
                self._drop_arrival_job(
                    job_id=str(dropped_job.job_id),
                    drop_reasion="等待超时",
                )
                self.waiting_timeout_drops += 1
                continue

            if not target_host.can_accommodate(waiting_job):
                break

            started = self._start_job_on_host(
                job_id=waiting_job_id,
                dc_id=dc_id,
                host_idx=host_idx,
            )

            if not started:
                break

            removed_job = target_host.remove_from_waiting_queue()
            self.started_from_waiting_jobs += 1

    # 处理任务完成事件
    def _process_job_finish_event(self, job_id: str) -> Job:
        # 定位任务与执行位置
        job_id = str(job_id)
        location = self.running_job_location[job_id]
        dc_id = str(location["dc_id"])
        host_idx = int(location["host_idx"])
        target_dc = self.dc_map[dc_id]
        target_host = target_dc.host_list[host_idx]
        finished_job = target_host.remove_from_running_queue(job_id)

        # 放入完成队列
        target_host.add_to_completed_queue(
            job=finished_job,
            current_time=float(self.current_time),
        )

        # 更新负载
        target_dc.calculate_dc_loads()

        # 删除记录
        self.running_job_location.pop(job_id)

        self._drain_host_waiting_queue(
            dc_id=dc_id,
            host_idx=host_idx,
        )

        return finished_job

    # 清除调度决策执行时的临时变量
    def _clear_current_decision(self) -> None:
        self.current_job_id = None
        self.current_dc_id = None
        self.current_agent_id = None

    # 计算动作的即时奖励
    def _compute_action_reward(
            self,
            job_id: str,
            action_type: str,
            success: bool,
            transfer_latency: float = 0.0,
            failure_reason: Optional[str] = None,
    ) -> float:
        job_id = str(job_id)
        action_type = str(action_type)
        job = self.job_map[job_id]

        # 丢任务惩罚
        if action_type == "drop" or not success:

            if failure_reason == "等待超时":
                return -float(self.timeout_drop_penalty)

            if failure_reason == "资源不足":
                return -float(self.resource_drop_penalty)

        # 本地执行
        if action_type == "local_host":
            duration_cost = self._normalize(
                value=float(job.duration),
                scale=float(self.max_job_duration),
            )

            return -float(duration_cost)

        # 卸载到其他 edge 或 cloud
        if action_type in {"edge_dc", "cloud"}:
            transfer_latency = float(transfer_latency)
            latency_cost = self._normalize(
                value=transfer_latency,
                scale=float(self.max_latency),
            )
            return -float(latency_cost)

    # 检查好一个 episode 是否结束
    def _check_episode_finished(self) -> bool:

        # 当前还有任务等待边缘智能体决策
        if self.current_job_id is not None:
            return False
        if self.current_agent_id is not None:
            return False
        if self.current_dc_id is not None:
            return False

        # 事件队列中仍有任务到达事件或任务完成事件
        if len(self.event_queue) > 0:
            return False

        for dc in self.datacenters:
            for host in dc.host_list:
                if not host.running_queue.is_empty():
                    return False
                if not host.waiting_queue.is_empty():
                    return False

        return True

    # 将当前 episode 标记为自然结束
    def _terminate_episode(self) -> None:
        for agent_id in self.possible_agents:
            self.terminations[agent_id] = True
        self.current_job_id = None
        self.current_dc_id = None
        self.current_agent_id = None

