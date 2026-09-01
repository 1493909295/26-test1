from __future__ import annotations

from typing import Any, Dict, List
from typing import Any, Optional
import numpy as np


class HostObservationBuilder:
    """
    H-MASAC Local Host Scheduling 专用 Observation Builder。

    Host Observation 只在 Routing Agent 选择 Self 后构造。

    Observation:

        Current Job
        +
        Current Local DC Context
        +
        Every Real Local Host State

    明确不包含：

        Remote DC state
        Routing topology
        Routing history
        Neighbor Historical Feedback
        Cloud
        Host padding
        Action mask

    Host 层不是 PettingZoo Agent。
    """

    JOB_FEAT_DIM = 5

    LOCAL_DC_FEAT_DIM = 5

    HOST_BASE_FEAT_DIM = 9

    HOST_JOB_EVAL_FEAT_DIM = 4

    HOST_FEAT_DIM = (
        HOST_BASE_FEAT_DIM
        + HOST_JOB_EVAL_FEAT_DIM
    )


    def __init__(
        self,
        env: Any,
    ) -> None:

        self.env = env

        self.edge_dc_ids = [
            str(dc_id)
            for dc_id in env.edge_dc_ids
        ]


        # ======================================================
        # Host 层只管理 Edge DC。
        #
        # Cloud 不进入 Local Host SAC；
        # Cloud 仍沿用环境中的自动执行逻辑。
        # ======================================================

        base_edge_dc_map = {
            str(dc.dc_id): dc
            for dc in env.base_datacenters
            if str(dc.dc_id) in self.edge_dc_ids
        }


        # ======================================================
        # 固定每个 DC 的 Host 顺序。
        #
        # 后续：
        #
        #     Host action 0
        #
        # 永远对应：
        #
        #     host_ids_by_dc[dc_id][0]
        #
        # Observation 中 Host block 顺序和 action 顺序
        # 必须严格一致。
        # ======================================================

        self.host_ids_by_dc: Dict[
            str,
            List[str],
        ] = {}


        self.host_count_by_dc: Dict[
            str,
            int,
        ] = {}


        for dc_id in self.edge_dc_ids:

            if dc_id not in base_edge_dc_map:
                raise KeyError(
                    "Host Observation 找不到 Edge DC："
                    f"{dc_id}"
                )

            dc = base_edge_dc_map[dc_id]

            host_ids = [
                str(host.host_id)
                for host in dc.host_list
            ]

            if len(host_ids) == 0:
                raise ValueError(
                    f"Edge DC {dc_id} 没有 Host。"
                )

            self.host_ids_by_dc[
                dc_id
            ] = host_ids

            self.host_count_by_dc[
                dc_id
            ] = len(host_ids)


        # ======================================================
        # 每个 DC 独立 Observation / Action dimension。
        #
        # 不使用 max_host_num。
        # 不进行 zero padding。
        # ======================================================

        self.obs_dim_by_dc: Dict[
            str,
            int,
        ] = {
            dc_id: (
                self.JOB_FEAT_DIM
                + self.LOCAL_DC_FEAT_DIM
                + self.HOST_FEAT_DIM
                * self.host_count_by_dc[dc_id]
            )
            for dc_id in self.edge_dc_ids
        }


        self.action_dim_by_dc: Dict[
            str,
            int,
        ] = {
            dc_id:
                self.host_count_by_dc[dc_id]
            for dc_id in self.edge_dc_ids
        }


    def get_obs_dim(
        self,
        dc_id: str,
    ) -> int:

        dc_id = str(dc_id)

        if dc_id not in self.obs_dim_by_dc:
            raise KeyError(
                f"未知 Edge DC：{dc_id}"
            )

        return int(
            self.obs_dim_by_dc[dc_id]
        )


    def get_action_dim(
        self,
        dc_id: str,
    ) -> int:

        dc_id = str(dc_id)

        if dc_id not in self.action_dim_by_dc:
            raise KeyError(
                f"未知 Edge DC：{dc_id}"
            )

        return int(
            self.action_dim_by_dc[dc_id]
        )


    def get_host_ids(
        self,
        dc_id: str,
    ) -> List[str]:

        dc_id = str(dc_id)

        if dc_id not in self.host_ids_by_dc:
            raise KeyError(
                f"未知 Edge DC：{dc_id}"
            )

        return list(
            self.host_ids_by_dc[dc_id]
        )

    def _normalize(
        self,
        value: float,
        scale: float,
    ) -> float:

        eps = max(
            float(
                getattr(
                    self.env,
                    "norm_eps",
                    1e-8,
                )
            ),
            1e-12,
        )

        value = max(
            float(value),
            0.0,
        )

        scale = max(
            float(scale),
            eps,
        )

        return float(
            np.clip(
                value / scale,
                0.0,
                1.0,
            )
        )

    def _saturating_ratio(
        self,
        value: float,
        scale: float,
    ) -> float:

        eps = max(
            float(
                getattr(
                    self.env,
                    "norm_eps",
                    1e-8,
                )
            ),
            1e-12,
        )

        value = max(
            float(value),
            0.0,
        )

        scale = max(
            float(scale),
            eps,
        )

        return float(
            value / (
                value + scale
            )
        )

    def _encode_job(
        self,
        job: Any,
    ) -> List[float]:

        eps = max(
            float(
                getattr(
                    self.env,
                    "norm_eps",
                    1e-8,
                )
            ),
            1e-12,
        )

        duration = max(
            float(job.duration),
            eps,
        )

        elapsed_time = max(
            float(self.env.current_time)
            - float(job.arrive_time),
            0.0,
        )


        sla_budget = max(
            float(
                self.env.sla_deadline_ratio
            )
            * duration,
            eps,
        )


        drop_budget = max(
            float(
                self.env.drop_deadline_ratio
            )
            * duration,
            eps,
        )


        features = [
            self._normalize(
                job.cpu_request,
                self.env.max_job_cpu,
            ),

            self._normalize(
                job.gpu_request,
                self.env.max_job_gpu,
            ),

            self._normalize(
                duration,
                self.env.max_job_duration,
            ),

            float(
                np.clip(
                    elapsed_time
                    / sla_budget,
                    0.0,
                    1.0,
                )
            ),

            float(
                np.clip(
                    elapsed_time
                    / drop_budget,
                    0.0,
                    1.0,
                )
            ),
        ]


        if len(features) != self.JOB_FEAT_DIM:
            raise ValueError(
                "Host Job feature dimension error："
                f"expected={self.JOB_FEAT_DIM}, "
                f"actual={len(features)}"
            )

        return features

    def _encode_local_dc(
        self,
        local_dc: Any,
    ) -> List[float]:

        hosts = list(
            local_dc.host_list
        )

        host_count = max(
            len(hosts),
            1,
        )


        local_dc.calculate_dc_loads()


        waiting_jobs = sum(
            len(host.waiting_queue)
            for host in hosts
        )

        running_jobs = sum(
            len(host.running_queue)
            for host in hosts
        )

        waiting_workload = sum(
            float(
                host.waiting_queue
                .get_total_duration()
            )
            for host in hosts
        )


        queue_length_scale = (
            float(
                self.env.queue_length_scale
            )
            * host_count
        )

        queue_workload_scale = (
            float(
                self.env.queue_workload_scale
            )
            * host_count
        )


        features = [
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

            self._saturating_ratio(
                waiting_jobs,
                queue_length_scale,
            ),

            self._saturating_ratio(
                running_jobs,
                queue_length_scale,
            ),

            self._saturating_ratio(
                waiting_workload,
                queue_workload_scale,
            ),
        ]


        if (
            len(features)
            != self.LOCAL_DC_FEAT_DIM
        ):
            raise ValueError(
                "Host Local DC feature dimension error："
                f"expected={self.LOCAL_DC_FEAT_DIM}, "
                f"actual={len(features)}"
            )

        return features

    def _estimate_host_start_delay(
        self,
        host: Any,
        job: Any,
    ) -> Optional[float]:

        # Host 的物理总容量永远无法执行该任务。
        if not host.can_ever_accommodate(
            job
        ):
            return None


        # Waiting Queue 为空且当前资源足够，
        # 根据现有环境语义可以立即执行。
        if (
            host.waiting_queue.is_empty()
            and host.can_accommodate(job)
        ):
            return 0.0


        max_running_remaining = 0.0


        for running_job in list(
            getattr(
                host.running_queue,
                "_queue",
                [],
            )
        ):

            if (
                running_job.start_time
                is None
            ):
                remaining_time = float(
                    running_job.duration
                )

            else:
                remaining_time = max(
                    float(
                        running_job.start_time
                    )
                    + float(
                        running_job.duration
                    )
                    - float(
                        self.env.current_time
                    ),
                    0.0,
                )

            max_running_remaining = max(
                max_running_remaining,
                remaining_time,
            )


        waiting_workload = float(
            host.waiting_queue
            .get_total_duration()
        )


        return float(
            max_running_remaining
            + waiting_workload
        )

    def _encode_host(
        self,
        host: Any,
        job: Any,
    ) -> List[float]:

        host.calculate_load()


        available_cpu = float(
            host.get_available_cpu()
        )

        available_gpu = float(
            host.get_available_gpu()
        )


        waiting_jobs = len(
            host.waiting_queue
        )

        running_jobs = len(
            host.running_queue
        )

        waiting_workload = float(
            host.waiting_queue
            .get_total_duration()
        )


        # ======================================================
        # 当前 Job 与 Host 的匹配关系。
        #
        # 注意：
        # 这是 Observation，不是 Action Mask。
        # ======================================================

        can_ever = bool(
            host.can_ever_accommodate(
                job
            )
        )


        can_start_now = bool(
            host.waiting_queue.is_empty()
            and host.can_accommodate(job)
        )


        estimated_start_delay = (
            self._estimate_host_start_delay(
                host=host,
                job=job,
            )
        )


        duration = max(
            float(job.duration),
            1e-8,
        )

        elapsed_time = max(
            float(self.env.current_time)
            - float(job.arrive_time),
            0.0,
        )


        if estimated_start_delay is None:

            start_delay_ratio = 1.0
            completion_ratio = 1.0

        else:

            wait_budget = max(
                (
                    float(
                        self.env
                        .drop_deadline_ratio
                    )
                    - 1.0
                )
                * duration,
                1e-8,
            )


            start_delay_ratio = float(
                np.clip(
                    estimated_start_delay
                    / wait_budget,
                    0.0,
                    1.0,
                )
            )


            predicted_completion = (
                elapsed_time
                + estimated_start_delay
                + duration
            )


            drop_limit = max(
                float(
                    self.env
                    .drop_deadline_ratio
                )
                * duration,
                1e-8,
            )


            completion_ratio = float(
                np.clip(
                    predicted_completion
                    / drop_limit,
                    0.0,
                    1.0,
                )
            )


        features = [

            # ==================================================
            # Base Host State：9维
            # ==================================================

            self._normalize(
                host.cpu_num,
                self.env.max_host_cpu,
            ),

            self._normalize(
                host.gpu_capacity_num,
                self.env.max_host_gpu,
            ),

            float(
                np.clip(
                    host.cpu_load,
                    0.0,
                    1.0,
                )
            ),

            float(
                np.clip(
                    host.gpu_load,
                    0.0,
                    1.0,
                )
            ),

            self._normalize(
                available_cpu,
                self.env.max_host_cpu,
            ),

            self._normalize(
                available_gpu,
                self.env.max_host_gpu,
            ),

            self._saturating_ratio(
                waiting_jobs,
                self.env.queue_length_scale,
            ),

            self._saturating_ratio(
                waiting_workload,
                self.env.queue_workload_scale,
            ),

            self._saturating_ratio(
                running_jobs,
                self.env.queue_length_scale,
            ),


            # ==================================================
            # Current Job ↔ Host Matching：4维
            # ==================================================

            1.0 if can_ever else 0.0,

            1.0 if can_start_now else 0.0,

            start_delay_ratio,

            completion_ratio,
        ]


        if len(features) != self.HOST_FEAT_DIM:

            raise ValueError(
                "Host feature dimension error："
                f"expected={self.HOST_FEAT_DIM}, "
                f"actual={len(features)}"
            )


        return features

    def build(
        self,
        dc_id: str,
        job_id: str,
    ) -> np.ndarray:

        dc_id = str(dc_id)
        job_id = str(job_id)


        # ======================================================
        # Host 层只允许 Edge DC。
        # ======================================================

        if dc_id not in self.edge_dc_ids:
            raise ValueError(
                "Host Observation 只能为 Edge DC 构造："
                f"dc_id={dc_id}"
            )


        if dc_id not in self.env.dc_map:
            raise KeyError(
                "当前 Episode 中不存在 DC："
                f"{dc_id}"
            )


        if job_id not in self.env.job_map:
            raise KeyError(
                "当前 Episode 中不存在 Job："
                f"{job_id}"
            )


        local_dc = self.env.dc_map[
            dc_id
        ]

        job = self.env.job_map[
            job_id
        ]


        # ======================================================
        # Host 数量和顺序属于网络结构的一部分。
        #
        # 每个 Episode 的硬件环境必须保持一致。
        # ======================================================

        runtime_host_ids = [
            str(host.host_id)
            for host in local_dc.host_list
        ]


        expected_host_ids = (
            self.host_ids_by_dc[
                dc_id
            ]
        )


        if (
            runtime_host_ids
            != expected_host_ids
        ):
            raise RuntimeError(
                "Host 列表发生变化，"
                "会破坏 Local Host SAC 的输入/动作映射："
                f"dc={dc_id}, "
                f"expected={expected_host_ids}, "
                f"actual={runtime_host_ids}"
            )


        obs: List[float] = []


        # 1. Current Job
        obs.extend(
            self._encode_job(job)
        )


        # 2. Local DC context
        obs.extend(
            self._encode_local_dc(
                local_dc
            )
        )


        # 3. Every real local Host
        #
        # 不 padding。
        # 不读取其他 DC。
        for host in local_dc.host_list:

            obs.extend(
                self._encode_host(
                    host=host,
                    job=job,
                )
            )


        obs_array = np.asarray(
            obs,
            dtype=np.float32,
        )


        expected_dim = (
            self.get_obs_dim(
                dc_id
            )
        )


        if obs_array.shape != (
            expected_dim,
        ):

            raise ValueError(
                "Host Observation dimension error："
                f"dc={dc_id}, "
                f"expected={(expected_dim,)}, "
                f"actual={obs_array.shape}"
            )


        if not np.all(
            np.isfinite(obs_array)
        ):

            raise ValueError(
                "Host Observation 出现 NaN/Inf："
                f"dc={dc_id}, "
                f"job={job_id}"
            )


        return obs_array

    def build_pending(
        self,
    ) -> np.ndarray:
        """
        为当前 Routing=Self 后等待 Host 决策的任务
        构造 Host Observation。
        """

        job_id = getattr(
            self.env,
            "pending_host_job_id",
            None,
        )

        dc_id = getattr(
            self.env,
            "pending_host_dc_id",
            None,
        )


        if job_id is None:
            raise RuntimeError(
                "当前不存在 pending Host Job。"
            )


        if dc_id is None:
            raise RuntimeError(
                "当前不存在 pending Host DC。"
            )


        dc_id = str(dc_id)
        job_id = str(job_id)


        # Host Decision 发生时 Routing AEC 应处于暂停状态。
        if (
            self.env.agent_selection
            is not None
        ):
            raise RuntimeError(
                "当前仍存在 PettingZoo Routing Agent，"
                "不能同时执行 Host Decision："
                f"agent_selection="
                f"{self.env.agent_selection}"
            )


        return self.build(
            dc_id=dc_id,
            job_id=job_id,
        )

