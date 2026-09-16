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

from collections import deque
from dataclasses import dataclass
import math
from typing import Any, Deque, Dict, Mapping, Optional, Sequence, Tuple

from bayesian_game import BayesianHistoricalEvidence

__all__ = (
    "BayesianCongestionPairKey",
    "RoutingActionFeatures",
    "ActionUtilityBreakdown",
    "BetaBeliefState",
    "BayesianBeliefStore",
    "HistoricalPressureState",
    "HistoricalPressureStore",
    "BayesianCongestionGame",
)


@dataclass(frozen=True)
class BayesianCongestionPairKey:
    """运行时历史状态使用的有向 Pair 标识。

    该类型只表达 source / target 身份，不携带任何远端实时状态。
    """

    source_dc_id: str
    target_dc_id: str


@dataclass(frozen=True)
class RoutingActionFeatures:
    """一个候选 Routing Action 的历史/任务质量特征。

    三个分数都必须位于 [0, 1]，且只表示可供启发式层使用的质量：

    * success_score：历史成功质量；
    * sla_score：SLA 满足质量；
    * delay_score：延迟质量，越快越接近 1。

    这些字段不是远端实时资源，也不会写入 Routing Observation。
    """

    target_dc_id: str
    success_score: float = 0.0
    sla_score: float = 0.0
    delay_score: float = 0.0


@dataclass(frozen=True)
class ActionUtilityBreakdown:
    """保存一个候选动作的 Benefit / Risk / Utility / Bias 分解。"""

    target_dc_id: str
    benefit: float
    congestion_probability: float
    congestion_cost: float
    risk: float
    utility: float
    confidence: float
    bias: float


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


@dataclass
class HistoricalPressureState:
    """一个目标 DC 的历史竞争压力状态。"""

    target_dc_id: str
    recent_selection_count: int = 0
    total_selection_count: int = 0
    last_update_clock: Optional[int] = None


class HistoricalPressureStore:
    """按最近 Edge-to-Edge 选择窗口维护目标 DC 压力。

    压力定义为：

    ``x_j = count(recent_targets == j) / len(recent_targets)``。

    该定义只使用已经发生并完成归档的历史路由选择；窗口未满时，
    分母使用当前有效历史样本数，不人为填充未来或实时负载信息。
    """

    def __init__(
            self,
            edge_dc_ids: Sequence[str],
            *,
            window_size: int = 100,
            linear_cost_weight: float = 1.0,
            quadratic_cost_weight: float = 1.0,
    ) -> None:
        normalized_ids = tuple(str(dc_id) for dc_id in edge_dc_ids)
        if len(set(normalized_ids)) != len(normalized_ids):
            raise ValueError("edge_dc_ids 不能包含重复 DC。")
        if int(window_size) <= 0:
            raise ValueError("pressure window_size 必须大于 0。")
        if (
            not math.isfinite(linear_cost_weight)
            or not math.isfinite(quadratic_cost_weight)
            or linear_cost_weight < 0.0
            or quadratic_cost_weight < 0.0
        ):
            raise ValueError("拥塞成本权重必须是非负有限数。")

        self.edge_dc_ids: Tuple[str, ...] = normalized_ids
        self.window_size = int(window_size)
        self.linear_cost_weight = float(linear_cost_weight)
        self.quadratic_cost_weight = float(quadratic_cost_weight)
        self._recent_targets: Deque[str] = deque(
            maxlen=self.window_size
        )
        self._states: Dict[str, HistoricalPressureState] = {
            dc_id: HistoricalPressureState(target_dc_id=dc_id)
            for dc_id in self.edge_dc_ids
        }

    def record_selection(
            self,
            target_dc_id: str,
            *,
            update_clock: Optional[int] = None,
    ) -> float:
        """记录一次已完成归档的 Edge-to-Edge target 选择。"""

        target_dc_id = str(target_dc_id)
        if target_dc_id not in self._states:
            raise KeyError(f"未知的压力目标 DC：{target_dc_id}")

        if len(self._recent_targets) == self.window_size:
            evicted_target = self._recent_targets[0]
            self._states[evicted_target].recent_selection_count -= 1

        self._recent_targets.append(target_dc_id)
        state = self._states[target_dc_id]
        state.recent_selection_count += 1
        state.total_selection_count += 1
        state.last_update_clock = update_clock
        return self.get_pressure(target_dc_id)

    def get_pressure(self, target_dc_id: str) -> float:
        """返回目标 DC 在最近窗口中的选择占比 x_j。"""

        target_dc_id = str(target_dc_id)
        state = self._states.get(target_dc_id)
        if state is None:
            raise KeyError(f"未知的压力目标 DC：{target_dc_id}")
        denominator = max(1, len(self._recent_targets))
        return float(state.recent_selection_count / denominator)

    def get_congestion_cost(self, target_dc_id: str) -> float:
        """按一阶 + 二阶项计算目标 DC 的拥塞成本 C_j。"""

        pressure = self.get_pressure(target_dc_id)
        return float(
            self.linear_cost_weight * pressure
            + self.quadratic_cost_weight * pressure * pressure
        )

    def snapshot(self) -> Dict[str, Dict[str, float]]:
        """导出压力窗口快照，用于日志/调试，不包含实时资源字段。"""

        return {
            dc_id: {
                "pressure": self.get_pressure(dc_id),
                "congestion_cost": self.get_congestion_cost(dc_id),
                "recent_selection_count": float(
                    state.recent_selection_count
                ),
                "total_selection_count": float(
                    state.total_selection_count
                ),
            }
            for dc_id, state in self._states.items()
        }


class BayesianCongestionGame:
    """Bayesian-Congestion Game 的状态化计算器。

    第 2/4/5 步实现 posterior、历史 pressure 以及
    Benefit / Risk / Utility / Bias 数学层；不直接采样或决定动作。
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
            pressure_window_size: int = 100,
            linear_cost_weight: float = 1.0,
            quadratic_cost_weight: float = 1.0,
            benefit_success_weight: float = 0.4,
            benefit_sla_weight: float = 0.4,
            benefit_delay_weight: float = 0.2,
            risk_congestion_cost_weight: float = 1.0,
            guidance_scale: float = 0.3,
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
        self.pressure_store = HistoricalPressureStore(
            tuple(getattr(game_definition, "player_ids", ())),
            window_size=pressure_window_size,
            linear_cost_weight=linear_cost_weight,
            quadratic_cost_weight=quadratic_cost_weight,
        )
        benefit_weights = (
            benefit_success_weight,
            benefit_sla_weight,
            benefit_delay_weight,
        )
        if any(
            not math.isfinite(weight) or weight < 0.0
            for weight in benefit_weights
        ) or sum(benefit_weights) <= 0.0:
            raise ValueError(
                "Benefit 权重必须是非负有限数，且总和必须大于 0。"
            )
        if (
            not math.isfinite(risk_congestion_cost_weight)
            or risk_congestion_cost_weight < 0.0
        ):
            raise ValueError("risk_congestion_cost_weight 必须是非负有限数。")
        if not math.isfinite(guidance_scale) or guidance_scale < 0.0:
            raise ValueError("guidance_scale 必须是非负有限数。")

        self.benefit_success_weight = float(benefit_success_weight)
        self.benefit_sla_weight = float(benefit_sla_weight)
        self.benefit_delay_weight = float(benefit_delay_weight)
        self.risk_congestion_cost_weight = float(
            risk_congestion_cost_weight
        )
        self.guidance_scale = float(guidance_scale)

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

    def update_pressure(
            self,
            source_dc_id: str,
            target_dc_id: str,
            *,
            update_clock: Optional[int] = None,
    ) -> float:
        """由一次 source -> target 历史选择更新目标压力 x_j。"""

        self.belief_store.get_state(
            source_dc_id,
            target_dc_id,
        )
        return self.pressure_store.record_selection(
            target_dc_id,
            update_clock=update_clock,
        )

    def get_pressure(self, target_dc_id: str) -> float:
        """读取目标 DC 的历史竞争压力 x_j。"""

        return self.pressure_store.get_pressure(target_dc_id)

    def get_congestion_cost(self, target_dc_id: str) -> float:
        """读取目标 DC 的拥塞成本 C_j。"""

        return self.pressure_store.get_congestion_cost(target_dc_id)

    @staticmethod
    def _validate_quality_score(
            score_name: str,
            score: float,
    ) -> float:
        score = float(score)
        if not math.isfinite(score) or not 0.0 <= score <= 1.0:
            raise ValueError(
                f"{score_name} 必须是 [0, 1] 内的有限数。"
            )
        return score

    def calculate_benefit(
            self,
            action_features: RoutingActionFeatures,
    ) -> float:
        """按成功、SLA、延迟质量计算候选动作 Benefit。"""

        success_score = self._validate_quality_score(
            "success_score",
            action_features.success_score,
        )
        sla_score = self._validate_quality_score(
            "sla_score",
            action_features.sla_score,
        )
        delay_score = self._validate_quality_score(
            "delay_score",
            action_features.delay_score,
        )
        return float(
            self.benefit_success_weight * success_score
            + self.benefit_sla_weight * sla_score
            + self.benefit_delay_weight * delay_score
        )

    def calculate_risk(
            self,
            source_dc_id: str,
            target_dc_id: str,
    ) -> Tuple[float, float, float]:
        """返回 (Risk, congestion_probability, congestion_cost)。"""

        target_dc_id = str(target_dc_id)
        source_dc_id = str(source_dc_id)

        # Self / Cloud 不建立 Remote Edge Pair，因此没有 Bayesian
        # remote congestion risk；其 Benefit 仍可参与候选动作比较。
        if (
            target_dc_id not in self.pressure_store.edge_dc_ids
            or target_dc_id == source_dc_id
        ):
            return (0.0, 0.0, 0.0)

        congestion_probability = (
            self.get_congestion_probability(
                source_dc_id,
                target_dc_id,
            )
        )
        congestion_cost = self.get_congestion_cost(target_dc_id)
        risk = float(
            congestion_probability
            + self.risk_congestion_cost_weight * congestion_cost
        )
        return (
            risk,
            congestion_probability,
            congestion_cost,
        )

    @staticmethod
    def _coerce_action_features(
            target_dc_id: str,
            raw_features: Any,
    ) -> RoutingActionFeatures:
        if isinstance(raw_features, RoutingActionFeatures):
            if str(raw_features.target_dc_id) != str(target_dc_id):
                raise ValueError(
                    "Action features 的 target_dc_id 与候选动作键不一致。"
                )
            return raw_features

        if isinstance(raw_features, Mapping):
            features_target_dc_id = str(
                raw_features.get(
                    "target_dc_id",
                    target_dc_id,
                )
            )
            if features_target_dc_id != str(target_dc_id):
                raise ValueError(
                    "Action features 的 target_dc_id 与候选动作键不一致。"
                )
            return RoutingActionFeatures(
                target_dc_id=features_target_dc_id,
                success_score=float(
                    raw_features.get("success_score", 0.0)
                ),
                sla_score=float(
                    raw_features.get("sla_score", 0.0)
                ),
                delay_score=float(
                    raw_features.get("delay_score", 0.0)
                ),
            )

        raise TypeError(
            "job_context 必须是 RoutingActionFeatures 或字段映射。"
        )

    def evaluate_actions(
            self,
            source_dc_id: str,
            job_context: Mapping[str, Any],
    ) -> Tuple[ActionUtilityBreakdown, ...]:
        """计算候选动作的 Benefit、Risk、Utility 和中心化 Bias。"""

        if not isinstance(job_context, Mapping) or not job_context:
            raise ValueError(
                "job_context 必须是非空的 target_dc_id -> action features 映射。"
            )

        candidates = tuple(
            self._coerce_action_features(target_dc_id, raw_features)
            for target_dc_id, raw_features in job_context.items()
        )
        target_ids = tuple(
            str(action.target_dc_id)
            for action in candidates
        )
        if len(set(target_ids)) != len(target_ids):
            raise ValueError("候选动作不能包含重复 target_dc_id。")

        preliminary = []
        for action_features in candidates:
            benefit = self.calculate_benefit(action_features)
            (
                risk,
                congestion_probability,
                congestion_cost,
            ) = self.calculate_risk(
                source_dc_id,
                action_features.target_dc_id,
            )
            preliminary.append(
                (
                    action_features,
                    benefit,
                    risk,
                    congestion_probability,
                    congestion_cost,
                    benefit - risk,
                )
            )

        utility_mean = sum(
            item[-1]
            for item in preliminary
        ) / len(preliminary)
        confidence_by_target = {}
        raw_bias_by_target = {}
        for (
                action_features,
                _benefit,
                _risk,
                _congestion_probability,
                _congestion_cost,
                utility,
        ) in preliminary:
            is_remote_pair = (
                str(action_features.target_dc_id)
                in self.pressure_store.edge_dc_ids
                and str(action_features.target_dc_id)
                != str(source_dc_id)
            )
            confidence = (
                self.belief_store.get_confidence(
                    source_dc_id,
                    action_features.target_dc_id,
                )
                if is_remote_pair
                else 1.0
            )
            target_dc_id = str(action_features.target_dc_id)
            confidence_by_target[target_dc_id] = confidence
            raw_bias_by_target[target_dc_id] = (
                confidence * (utility - utility_mean)
            )

        raw_bias_mean = sum(
            raw_bias_by_target.values()
        ) / len(raw_bias_by_target)

        breakdowns = []
        for (
                action_features,
                benefit,
                risk,
                congestion_probability,
                congestion_cost,
                utility,
        ) in preliminary:
            target_dc_id = str(action_features.target_dc_id)
            confidence = confidence_by_target[target_dc_id]

            # 置信度先抑制不确定 Pair，再做一次中心化；共同平移不会改变 softmax。
            bias = float(
                self.guidance_scale
                * (raw_bias_by_target[target_dc_id] - raw_bias_mean)
            )
            breakdowns.append(
                ActionUtilityBreakdown(
                    target_dc_id=str(action_features.target_dc_id),
                    benefit=float(benefit),
                    congestion_probability=float(
                        congestion_probability
                    ),
                    congestion_cost=float(congestion_cost),
                    risk=float(risk),
                    utility=float(utility),
                    confidence=float(confidence),
                    bias=bias,
                )
            )

        return tuple(breakdowns)

    def get_action_bias(
            self,
            source_dc_id: str,
            job_context: Mapping[str, Any],
    ) -> Dict[str, float]:
        """返回 target_dc_id -> action-level logit bias。"""

        return {
            item.target_dc_id: item.bias
            for item in self.evaluate_actions(
                source_dc_id,
                job_context,
            )
        }
