from __future__ import annotations

from typing import Any, Dict, List
import numpy as np


class RoutingObservationBuilder:

    # job特征，CPU请求、GPU请求、执行时长、已消耗SLA预算比例、已消耗丢弃预算比例
    JOB_FEAT_DIM = 5
    # 本地DC状态
    LOCAL_DC_BASE_FEAT_DIM = 8
    LOCAL_DC_ROUTING_FEAT_DIM = 5
    # 任务调度历史
    ROUTE_HISTORY_FEAT_DIM = 3
    # 任务执行反馈
    FEEDBACK_FEAT_PER_DC = 7

    def __init__(self, env: Any) -> None:

        self.env = env
        self.edge_dc_ids = list(env.edge_dc_ids)
        self.link_target_dc_ids = (list(env.edge_dc_ids) + [str(env.cloud_id)])
        self.job_edge_hop_counts: Dict[str, int] = {}
        self.job_feat_dim = self.JOB_FEAT_DIM
        self.local_dc_feat_dim = (self.LOCAL_DC_BASE_FEAT_DIM + self.LOCAL_DC_ROUTING_FEAT_DIM)
        self.route_history_feat_dim = (self.ROUTE_HISTORY_FEAT_DIM)
        self.link_feat_dim = len(self.link_target_dc_ids)
        self.feedback_feat_dim = (len(self.edge_dc_ids) * self.FEEDBACK_FEAT_PER_DC)
        self.obs_dim = (
            self.job_feat_dim
            + self.local_dc_feat_dim
            + self.route_history_feat_dim
            + self.link_feat_dim
            + self.feedback_feat_dim
        )

    # 重置单episode调度信息
    def reset_episode(self) -> None:
        self.job_edge_hop_counts.clear()

    # 记录调度
    def record_routing_action(self, job_id: str, action_type: str,) -> None:
        if str(action_type) != "edge_dc":
            return
        job_id = str(job_id)
        self.job_edge_hop_counts[job_id] = (self.job_edge_hop_counts.get(job_id, 0,) + 1)

    #
    def build(self, agent_id: str,) -> np.ndarray:

        agent_id = str(agent_id)

        job_id = str(self.env.current_job_id)
        job = self.env.job_map[job_id]
        local_dc = self.env.dc_map[agent_id]
        obs: List[float] = []

        # 当前 Job
        obs.extend(self._encode_job(job))

        # 当前 DC 聚合状态
        obs.extend(self._encode_local_dc(local_dc, job,))

        # 当前 Job 多跳 Routing 历史
        obs.extend(self._encode_route_history( job,))

        # 当前 DC 到其他 DC / Cloud 的链路
        obs.extend(self._encode_links(agent_id,))

        # Neighbor Historical Feedback 占位
        obs.extend([0.0] * self.feedback_feat_dim)

        obs_array = np.asarray(obs, dtype=np.float32,)

        return obs_array

