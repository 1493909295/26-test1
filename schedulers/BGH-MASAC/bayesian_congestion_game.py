"""BGH-MASAC 的运行时拥塞博弈计算边界。

第 2 步实现 Bayesian Belief 的统一语义和 Beta-Bernoulli 更新；
压力、拥塞成本、启发式效用和 Actor 偏置仍留给后续步骤。

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

本模块只接收已完成任务的历史 Evidence，不读取远端实时资源状态；
真正接入训练循环仍需在后续步骤完成生命周期和日志 wiring。
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Dict, Optional, Sequence, Tuple

from bayesian_game import BayesianHistoricalEvidence

__all__ = (
    "BayesianCongestionPairKey",
    "BetaBeliefState",
    "BayesianBeliefStore",
    "BayesianCongestionGame",
)


@dataclass(frozen=True)
class BayesianCongestionPairKey:
    """运行时历史状态使用的有向 Pair 标识。

    该类型只表达 source / target 身份，不携带任何远端实时状态。
    """

    source_dc_id: str
    target_dc_id: str


@dataclass
class BetaBeliefState:
    """一个有向 source -> target Pair 的 Beta-Bernoulli 信念状态。

    这里刻意使用 ``alpha_congested`` / ``beta_non_congested`` 全名，
    避免把传统的 alpha / beta 误读成成功率或失败率：

    ``p_congestion = alpha_congested / (alpha_congested + beta_non_congested)``。
    """

    alpha_congested: float = 1.0
    beta_non_congested: float = 1.0
    sample_weight: float = 0.0
    last_update_clock: Optional[int] = None

    @property
    def congestion_probability(self) -> float:
        """返回当前 Pair 的拥塞后验均值。"""

        denominator = self.alpha_congested + self.beta_non_congested
        return self.alpha_congested / denominator

    def confidence(self, confidence_scale: float) -> float:
        """将有效历史样本量映射到 [0, 1] 的可解释置信度。"""

        if confidence_scale <= 0.0:
            raise ValueError("confidence_scale 必须大于 0。")
        return min(
            1.0,
            self.sample_weight / (self.sample_weight + confidence_scale),
        )


class BayesianBeliefStore:
    """按有向 DC 对维护拥塞 Beta 信念。

    该 Store 是 Step 2 的最小运行时实现：

    * 初始先验固定为 Beta(1, 1)，也允许由配置显式传入；
    * ``Evidence.congestion_observed`` 为 True 时只增加
      ``alpha_congested``；否则只增加 ``beta_non_congested``；
    * 每个 Pair 独立更新，不把 source -> target 与 target -> source 混合；
    * 不在此处计算 pressure、utility 或 action bias。
    """

    def __init__(
            self,
            edge_dc_ids: Sequence[str],
            *,
            prior_alpha: float = 1.0,
            prior_beta: float = 1.0,
            confidence_scale: float = 20.0,
    ) -> None:
        if (
            not math.isfinite(prior_alpha)
            or not math.isfinite(prior_beta)
            or prior_alpha <= 0.0
            or prior_beta <= 0.0
        ):
            raise ValueError("Beta 先验参数 prior_alpha / prior_beta 必须大于 0。")
        if not math.isfinite(confidence_scale) or confidence_scale <= 0.0:
            raise ValueError("confidence_scale 必须大于 0。")

        normalized_ids = tuple(str(dc_id) for dc_id in edge_dc_ids)
        if len(set(normalized_ids)) != len(normalized_ids):
            raise ValueError("edge_dc_ids 不能包含重复 DC。")

        self.edge_dc_ids: Tuple[str, ...] = normalized_ids
        self.prior_alpha = float(prior_alpha)
        self.prior_beta = float(prior_beta)
        self.confidence_scale = float(confidence_scale)
        self._states: Dict[BayesianCongestionPairKey, BetaBeliefState] = {
            BayesianCongestionPairKey(source_dc_id, target_dc_id):
            BetaBeliefState(
                alpha_congested=self.prior_alpha,
                beta_non_congested=self.prior_beta,
            )
            for source_dc_id in self.edge_dc_ids
            for target_dc_id in self.edge_dc_ids
            if source_dc_id != target_dc_id
        }

    def _get_pair_state(
            self,
            source_dc_id: str,
            target_dc_id: str,
    ) -> BetaBeliefState:
        key = BayesianCongestionPairKey(
            str(source_dc_id),
            str(target_dc_id),
        )
        if key not in self._states:
            raise KeyError(
                f"未知或非法的 Bayesian DC Pair: "
                f"{key.source_dc_id} -> {key.target_dc_id}"
            )
        return self._states[key]

    def update(
            self,
            evidence: BayesianHistoricalEvidence,
            *,
            update_clock: Optional[int] = None,
    ) -> BetaBeliefState:
        """用一条已完成任务 Evidence 更新对应 Pair 的 Beta 信念。"""

        evidence.validate_information_boundary()
        state = self._get_pair_state(
            evidence.source_dc_id,
            evidence.target_dc_id,
        )

        # 统一语义：拥塞证据进 alpha，非拥塞证据进 beta。
        if evidence.congestion_observed:
            state.alpha_congested += evidence.evidence_weight
        else:
            state.beta_non_congested += evidence.evidence_weight

        state.sample_weight += evidence.evidence_weight
        state.last_update_clock = update_clock
        return state

    def get_state(
            self,
            source_dc_id: str,
            target_dc_id: str,
    ) -> BetaBeliefState:
        """读取 Pair 状态；返回对象仅供当前运行时查询。"""

        return self._get_pair_state(source_dc_id, target_dc_id)

    def get_congestion_probability(
            self,
            source_dc_id: str,
            target_dc_id: str,
    ) -> float:
        return self.get_state(
            source_dc_id,
            target_dc_id,
        ).congestion_probability

    def get_confidence(
            self,
            source_dc_id: str,
            target_dc_id: str,
    ) -> float:
        return self.get_state(
            source_dc_id,
            target_dc_id,
        ).confidence(self.confidence_scale)

    def snapshot(self) -> Dict[str, Dict[str, float]]:
        """导出可用于测试/日志的语义化快照，不改变训练状态。"""

        return {
            f"{key.source_dc_id}->{key.target_dc_id}": {
                "alpha_congested": state.alpha_congested,
                "beta_non_congested": state.beta_non_congested,
                "sample_weight": state.sample_weight,
                "congestion_probability": state.congestion_probability,
                "confidence": state.confidence(self.confidence_scale),
            }
            for key, state in self._states.items()
        }


class BayesianCongestionGame:
    """Bayesian-Congestion Game 的状态化计算器骨架。

    注意：第 2 步只实现 posterior；pressure、utility 或 bias 仍未实现。
    保留明确的入口和职责边界，避免后续把计算逻辑重新塞回
    ``train_bgh_masac.py`` 或 ``bayesian_game.py``。
    """

    def __init__(
            self,
            game_definition: Any,
            *,
            prior_alpha: float = 1.0,
            prior_beta: float = 1.0,
            confidence_scale: float = 20.0,
    ) -> None:
        # 只保存已经脱敏的静态 Game Definition。
        # 不保存完整 Environment，也不保存远端实时资源引用。
        self.game_definition = game_definition
        self.belief_store = BayesianBeliefStore(
            tuple(getattr(game_definition, "player_ids", ())),
            prior_alpha=prior_alpha,
            prior_beta=prior_beta,
            confidence_scale=confidence_scale,
        )

    def update_belief(
            self,
            evidence: BayesianHistoricalEvidence,
            *,
            update_clock: Optional[int] = None,
    ) -> BetaBeliefState:
        """由终止任务 Evidence 更新 Pair Posterior。"""

        return self.belief_store.update(
            evidence,
            update_clock=update_clock,
        )

    def get_congestion_probability(
            self,
            source_dc_id: str,
            target_dc_id: str,
    ) -> float:
        """读取 Pair 的拥塞后验均值。"""

        return self.belief_store.get_congestion_probability(
            source_dc_id,
            target_dc_id,
        )

    def update_pressure(self, source_dc_id: str, target_dc_id: str) -> None:
        """预留：由已执行的历史 Edge-to-Edge 选择更新压力窗口。"""

        raise NotImplementedError(
            "后续步骤才实现历史 Routing Pressure 窗口。"
        )

    def get_action_bias(self, source_dc_id: str, job_context: Any) -> Any:
        """预留：根据历史状态生成 action-level logit bias。"""

        raise NotImplementedError(
            "Step 3 才实现 Utility 到 Actor logit bias 的转换。"
        )
