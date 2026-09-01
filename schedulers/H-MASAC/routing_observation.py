from __future__ import annotations

from typing import Any, Dict, List
import numpy as np
from typing import Any, Optional

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

    # 归一化辅助函数
    def _normalize(self, value: float,  scale: float,) -> float:

        eps = max(float(getattr(self.env, "norm_eps", 1e-8,)), 1e-12,)

        value = max(float(value), 0.0)
        scale = max(float(scale), eps)

        return float(np.clip(value / scale, 0.0, 1.0,))
    # 无上界归一化函数
    def _saturating_ratio(self, value: float, scale: float,) -> float:

        eps = max(float(getattr(self.env, "norm_eps", 1e-8,)), 1e-12,)

        value = max(float(value), 0.0)
        scale = max(float(scale), eps)

        return float(value / (value + scale))

    # 如果选择此host要等多久
    def _estimate_host_start_delay(self, host: Any, job: Any,) -> Optional[float]:

        # 资源不满足
        if not host.can_ever_accommodate(job):
            return None
        # 机器为空
        if (
                host.waiting_queue.is_empty()
                and host.can_accommodate(job)
        ):
            return 0.0


        max_running_remaining = 0.0
        running_jobs = getattr(host.running_queue, "_queue", [],)

        for running_job in running_jobs:
            if running_job.start_time is None:
                remaining_time = float(running_job.duration)

            else:
                remaining_time = max(
                    float(running_job.start_time)
                    + float(running_job.duration)
                    - float(self.env.current_time),
                    0.0,
                )

            max_running_remaining = max(max_running_remaining, remaining_time,)

        waiting_workload = float(host.waiting_queue.get_total_duration())

        return float(max_running_remaining + waiting_workload)

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
        local_dc_features = self._encode_local_dc(local_dc=local_dc, job=job,)
        obs.extend(local_dc_features)

        # 当前 Job 多跳 Routing 历史
        obs.extend(self._encode_route_history( job,))

        # 当前 DC 到其他 DC / Cloud 的链路
        obs.extend(self._encode_links(agent_id,))

        # Neighbor Historical Feedback 占位
        obs.extend([0.0] * self.feedback_feat_dim)

        obs_array = np.asarray(obs, dtype=np.float32,)

        return obs_array

    #
    def _encode_local_dc(self, local_dc: Any, job: Any,) -> List[float]:

        hosts = list(local_dc.host_list)
        host_count = len(hosts)

        # 更新所有 Host / DC 的当前负载。
        local_dc.calculate_dc_loads()

        total_cpu = sum(
            float(host.cpu_num)
            for host in hosts
        )
        total_gpu = sum(
            float(host.gpu_capacity_num)
            for host in hosts
        )
        used_cpu = sum(
            float(host.used_cpu)
            for host in hosts
        )
        used_gpu = sum(
            float(host.used_gpu)
            for host in hosts
        )

        available_cpu = max(total_cpu - used_cpu, 0.0,)
        available_gpu = max(total_gpu - used_gpu, 0.0,)

        waiting_jobs = sum(
            len(host.waiting_queue)
            for host in hosts
        )
        running_jobs = sum(
            len(host.running_queue)
            for host in hosts
        )
        waiting_workload = sum(
            float(host.waiting_queue.get_total_duration())
            for host in hosts
        )

        host_scale = max(host_count, 1,)
        dc_queue_length_scale = (float(self.env.queue_length_scale) * host_scale)
        dc_queue_workload_scale = (float(self.env.queue_workload_scale) * host_scale)
        immediate_feasible_count = 0
        ever_feasible_count = 0

        best_start_delay = None

        for host in hosts:
            # 物理总容量是否能够运行当前 Job。
            if host.can_ever_accommodate(job):
                ever_feasible_count += 1
                estimated_delay = (self._estimate_host_start_delay(host=host, job=job,))

                if estimated_delay is not None:
                    if (
                            best_start_delay is None
                            or estimated_delay < best_start_delay
                    ):
                        best_start_delay = (estimated_delay)

            if (
                    host.waiting_queue.is_empty()
                    and host.can_accommodate(job)
            ):
                immediate_feasible_count += 1

        host_count_float = float(max(host_count, 1))
        immediate_feasible_host_ratio = (float(immediate_feasible_count) / host_count_float)
        ever_feasible_host_ratio = (float(ever_feasible_count) / host_count_float)

        duration = max(float(job.duration),float(getattr(self.env, "norm_eps", 1e-8,)),)

        elapsed_time = max(float(self.env.current_time) - float(job.arrive_time), 0.0,)

        # 如果没有任何 Host 的物理容量能够执行 Job，
        # 直接把两个 local-risk 指标置为最高风险。
        if best_start_delay is None:
            best_local_start_delay_ratio = 1.0
            best_local_completion_ratio = 1.0

        else:
            local_wait_budget = max(
                (float(self.env.drop_deadline_ratio) - 1.0) * duration,float(getattr(self.env,"norm_eps", 1e-8,)),)
            best_local_start_delay_ratio = float(
                np.clip(
                    best_start_delay
                    / local_wait_budget,
                    0.0,
                    1.0,
                )
            )

            # ------------------------------------------------------
            # 从 Job 最初到达到预计执行完成的总时间：
            #
            # elapsed
            # + estimated local waiting
            # + execution duration
            # ------------------------------------------------------
            predicted_local_completion = (
                    elapsed_time
                    + float(best_start_delay)
                    + duration
            )

            drop_completion_limit = max(
                float(
                    self.env.drop_deadline_ratio
                )
                * duration,
                float(
                    getattr(
                        self.env,
                        "norm_eps",
                        1e-8,
                    )
                ),
            )

            best_local_completion_ratio = float(
                np.clip(
                    predicted_local_completion
                    / drop_completion_limit,
                    0.0,
                    1.0,
                )
            )

        # ==========================================================
        # 5. Final 13-dimensional Local DC Observation
        # ==========================================================

        features = [

            # ---------------- Base 8 ----------------

            self._normalize(
                total_cpu,
                self.env.max_dc_cpu,
            ),

            self._normalize(
                total_gpu,
                self.env.max_dc_gpu,
            ),

            float(
                np.clip(
                    local_dc.dc_cpu_load,
                    0.0,
                    1.0,
                )
            ),

            float(
                np.clip(
                    local_dc.dc_gpu_load,
                    0.0,
                    1.0,
                )
            ),

            self._normalize(
                available_cpu,
                self.env.max_dc_cpu,
            ),

            self._normalize(
                available_gpu,
                self.env.max_dc_gpu,
            ),

            self._saturating_ratio(
                value=float(waiting_jobs),
                scale=dc_queue_length_scale,
            ),

            self._saturating_ratio(
                value=float(running_jobs),
                scale=dc_queue_length_scale,
            ),

            # ------------- Routing-specific 5 -------------

            self._saturating_ratio(
                value=float(waiting_workload),
                scale=dc_queue_workload_scale,
            ),

            float(
                np.clip(
                    immediate_feasible_host_ratio,
                    0.0,
                    1.0,
                )
            ),

            float(
                np.clip(
                    ever_feasible_host_ratio,
                    0.0,
                    1.0,
                )
            ),

            float(
                np.clip(
                    best_local_start_delay_ratio,
                    0.0,
                    1.0,
                )
            ),

            float(
                np.clip(
                    best_local_completion_ratio,
                    0.0,
                    1.0,
                )
            ),
        ]

        # ==========================================================
        # 强制维度检查。
        #
        # Local DC Observation 一旦维度改变，
        # 必须显式修改常量，而不能静默改变 Actor 输入结构。
        # ==========================================================

        if len(features) != self.local_dc_feat_dim:
            raise ValueError(
                "Local DC Routing Observation 维度错误："
                f"expected={self.local_dc_feat_dim}, "
                f"actual={len(features)}"
            )

        return features

