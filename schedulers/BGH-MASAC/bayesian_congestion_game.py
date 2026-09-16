"""BGH-MASAC 的运行时拥塞博弈计算边界。

第 1 步只建立模块职责和稳定的数据边界，不实现算法计算。

本模块未来负责：

* 按 directed source -> target Pair 维护 Bayesian Belief；
* 按最近 Routing 历史维护目标 DC 的竞争压力；
* 将 Benefit、Bayesian Risk 和 Congestion Cost 合成为 Utility；
* 把 Utility 转换成供 Routing Actor 使用的 action-level logit bias。

本模块明确不负责：

* 神经网络、ReplayBuffer、Reward 或 Environment 生命周期；
* 修改 Routing Observation；
* 读取远端实时 CPU、GPU、Queue、Host 或可用资源；
* 直接决定动作。最终动作仍由 Guided MASAC Policy 采样。

真正接入训练前，必须先完成第 2 步的 Evidence / Posterior 契约，
再实现本模块中的数学方法。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

__all__ = (
    "BayesianCongestionPairKey",
    "BayesianCongestionGame",
)


@dataclass(frozen=True)
class BayesianCongestionPairKey:
    """运行时历史状态使用的有向 Pair 标识。

    该类型只表达 source / target 身份，不携带任何远端实时状态。
    """

    source_dc_id: str
    target_dc_id: str


class BayesianCongestionGame:
    """Bayesian-Congestion Game 的状态化计算器骨架。

    注意：第 1 步不在这里实现 posterior、pressure、utility 或 bias。
    先保留明确的入口和职责边界，避免后续把计算逻辑重新塞回
    ``train_bgh_masac.py`` 或 ``bayesian_game.py``。
    """

    def __init__(self, game_definition: Any) -> None:
        # 只保存已经脱敏的静态 Game Definition。
        # 不保存完整 Environment，也不保存远端实时资源引用。
        self.game_definition = game_definition

    def update_belief(self, evidence: Any) -> None:
        """预留：由终止任务 Evidence 更新 Pair Posterior。"""

        raise NotImplementedError(
            "Step 2 才实现 Bayesian Evidence / Posterior 更新。"
        )

    def update_pressure(self, source_dc_id: str, target_dc_id: str) -> None:
        """预留：由已执行的历史 Edge-to-Edge 选择更新压力窗口。"""

        raise NotImplementedError(
            "Step 2 才实现历史 Routing Pressure 窗口。"
        )

    def get_action_bias(self, source_dc_id: str, job_context: Any) -> Any:
        """预留：根据历史状态生成 action-level logit bias。"""

        raise NotImplementedError(
            "Step 3 才实现 Utility 到 Actor logit bias 的转换。"
        )
