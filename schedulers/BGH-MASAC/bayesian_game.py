from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, Optional, Tuple,List


# ==============================================================
# BGH-MASAC Bayesian Routing Game Definition
#
# Step 4:
#
# 本文件只负责正式定义：
#
#   1. 谁是 Bayesian Game Player；
#   2. Routing action 在 Game 中分别代表什么；
#   3. 哪些 action 存在隐藏 Remote Type；
#   4. Remote Type Space 是什么；
#   5. Belief 应该按什么粒度维护；
#   6. Bayesian Game 与 MASAC / Reward 的职责边界。
#
# 当前明确“不负责”：
#
#   - Bayesian Evidence；
#   - Bayesian Posterior；
#   - Dirichlet / Beta 更新；
#   - Heuristic Utility 数值；
#   - Actor Guidance；
#   - Bayesian Nash Equilibrium；
#   - Congestion Game。
#
# 因此本文件不会改变 H-MASAC-equivalent Zero-Diff 行为。
# ==============================================================


# BAYESIAN_GAME_DEFINITION_VERSION = 2

@dataclass(frozen=True)
class BayesianInformationPolicy:
    """
    Bayesian Game 的固定信息访问策略。

    注意：
        这不是可训练参数，也不应该放入 config.py。

    它用于明确保证：

        Remote Real-Time State
            X
            |
        Bayesian Belief
    """

    # ----------------------------------------------------------
    # Allowed information
    # ----------------------------------------------------------

    allow_static_routing_topology: bool = True

    allow_finalized_historical_outcomes: bool = True

    allow_neighbor_historical_feedback: bool = True

    allow_current_job_context: bool = True

    allow_source_local_information: bool = True

    # ----------------------------------------------------------
    # Forbidden Remote Real-Time information
    # ----------------------------------------------------------

    allow_remote_realtime_cpu: bool = False

    allow_remote_realtime_gpu: bool = False

    allow_remote_realtime_queue: bool = False

    allow_remote_realtime_host_state: bool = False

    allow_remote_available_resources: bool = False

    allow_remote_raw_observation: bool = False

    # Bayesian runtime core 不允许长期持有完整 Environment。
    allow_raw_environment_reference: bool = False

    def validate(self) -> None:
        """
        Fail-fast 验证 Bayesian Information Boundary。

        任何 Remote Real-Time 信息一旦被允许，
        都意味着当前算法已经破坏 Remote Edge
        Partial Observability 的研究假设。
        """

        forbidden_flags = {
            "allow_remote_realtime_cpu":
                self.allow_remote_realtime_cpu,

            "allow_remote_realtime_gpu":
                self.allow_remote_realtime_gpu,

            "allow_remote_realtime_queue":
                self.allow_remote_realtime_queue,

            "allow_remote_realtime_host_state":
                self.allow_remote_realtime_host_state,

            "allow_remote_available_resources":
                self.allow_remote_available_resources,

            "allow_remote_raw_observation":
                self.allow_remote_raw_observation,

            "allow_raw_environment_reference":
                self.allow_raw_environment_reference,
        }

        enabled_forbidden_flags = [
            flag_name
            for flag_name, enabled
            in forbidden_flags.items()
            if bool(enabled)
        ]

        if enabled_forbidden_flags:
            raise RuntimeError(
                "Bayesian Information Boundary 被破坏："
                "禁止 Bayesian Game 使用 Remote "
                "Real-Time State。"
                f" enabled={enabled_forbidden_flags}"
            )

    def to_metadata(
            self,
    ) -> Dict[str, bool]:
        """
        返回实验可记录的信息边界 metadata。
        """

        return {
            "allow_static_routing_topology":
                bool(
                    self.allow_static_routing_topology
                ),

            "allow_finalized_historical_outcomes":
                bool(
                    self.allow_finalized_historical_outcomes
                ),

            "allow_neighbor_historical_feedback":
                bool(
                    self.allow_neighbor_historical_feedback
                ),

            "allow_current_job_context":
                bool(
                    self.allow_current_job_context
                ),

            "allow_source_local_information":
                bool(
                    self.allow_source_local_information
                ),

            "allow_remote_realtime_cpu":
                bool(
                    self.allow_remote_realtime_cpu
                ),

            "allow_remote_realtime_gpu":
                bool(
                    self.allow_remote_realtime_gpu
                ),

            "allow_remote_realtime_queue":
                bool(
                    self.allow_remote_realtime_queue
                ),

            "allow_remote_realtime_host_state":
                bool(
                    self.allow_remote_realtime_host_state
                ),

            "allow_remote_available_resources":
                bool(
                    self.allow_remote_available_resources
                ),

            "allow_remote_raw_observation":
                bool(
                    self.allow_remote_raw_observation
                ),

            "allow_raw_environment_reference":
                bool(
                    self.allow_raw_environment_reference
                ),
        }

@dataclass(frozen=True)
class BayesianHistoricalEvidence:
    """
    Bayesian Evidence。

    表示：

        source DC
            |
            |
            v
        target DC

    历史调度结果对于 Remote Service Suitability
    的一次观测证据。


    注意：

    Evidence 不是 Remote State。

    禁止包含：

        CPU
        GPU
        Queue
        Host state
        utilization
        resource availability
        raw observation


    只允许包含：

        历史动作结果
        SLA outcome
        completion outcome
        latency outcome
        energy outcome
    """

    source_dc_id: str

    target_dc_id: str


    # --------------------------------------------------
    # 调度结果类别
    #
    # 用于未来映射：
    #
    # GOOD/NORMAL/RISKY
    #
    # --------------------------------------------------

    outcome_type: str


    # --------------------------------------------------
    # 是否成功完成
    # --------------------------------------------------

    success: bool


    # --------------------------------------------------
    # SLA 是否满足
    # --------------------------------------------------

    sla_satisfied: bool


    # --------------------------------------------------
    # 归一化后的历史质量指标
    #
    # 当前只保存结果，
    # 不参与 Bayesian 更新。
    #
    # --------------------------------------------------

    normalized_latency_score: float = 0.0

    normalized_energy_score: float = 0.0



    # --------------------------------------------------
    # 时间信息
    #
    # 用于未来 EWMA / decay
    # --------------------------------------------------

    timestamp: Optional[int] = None



    def validate_information_boundary(
        self,
    ) -> None:
        """
        防止 Evidence 演化成 Remote State。

        当前 Evidence 必须满足：

            Historical Outcome only

        """

        forbidden_fields = [

            "cpu",

            "gpu",

            "queue",

            "host_state",

            "available_resource",

            "raw_observation",

        ]


        for field_name in forbidden_fields:

            if hasattr(
                self,
                field_name,
            ):

                raise RuntimeError(
                    "Bayesian Evidence "
                    "禁止包含 Remote Real-Time State:"
                    f"{field_name}"
                )

def classify_historical_outcome(
        *,
        success: bool,
        sla_satisfied: bool,
        normalized_latency_score: float,
) -> str:
    """
    根据历史结果生成 Remote Hidden Type Evidence。

    注意：

    这里不是 Bayesian posterior。

    只是：

        outcome
          |
          v
        evidence label


    后续由 Bayesian Belief 模块学习：

        P(type | evidence)
    """

    if (
        success
        and sla_satisfied
        and normalized_latency_score >= 0.8
    ):
        return (
            BayesianRemoteType
            .GOOD
            .value
        )


    if (
        success
        and sla_satisfied
    ):
        return (
            BayesianRemoteType
            .NORMAL
            .value
        )


    return (
        BayesianRemoteType
        .RISKY
        .value
    )

class BayesianRemoteType(str, Enum):
    """
    Remote Edge DC 的隐藏服务类型。

    注意：
        这里不是 Remote DC 的真实 CPU/GPU/Queue 状态。

    它表示：

        对于 source DC -> target DC 这一有向调度关系，
        根据历史结果推断出的 Remote Service Suitability。

    后续 Bayesian Belief 将维护：

        P(
            theta_{source->target}
            |
            historical evidence
        )
    """

    GOOD = "good"
    NORMAL = "normal"
    RISKY = "risky"


class BayesianRoutingActionKind(str, Enum):
    """
    Routing Actor 动作在 Bayesian Game 中的语义类型。
    """

    SELF = "self"

    REMOTE_EDGE = "remote_edge"

    CLOUD = "cloud"


@dataclass(frozen=True)
class BayesianRoutingActionSemantic:
    """
    从某个 source DC 的视角描述一条 Routing action。

    只有 REMOTE_EDGE action 才对应未知 Remote Type。
    """

    action_index: int

    target_dc_id: str

    action_kind: BayesianRoutingActionKind

    has_hidden_remote_type: bool

@dataclass(frozen=True)
class BayesianStaticRoutingContext:
    """
    Bayesian Game 从 Environment 中允许获得的
    唯一静态 Routing Context。

    该对象刻意只保存 Routing identity / topology。

    因此这里明确不存在：

        CPU
        GPU
        Queue
        Host
        available resources
        utilization
        local observation
        global state

    完整 Environment 只允许在 whitelist adapter 中出现。
    Bayesian Game Core 后续只能接收本 Context。
    """

    edge_dc_ids: Tuple[str, ...]

    routing_action_target_dc_ids: Tuple[
        str,
        ...
    ]

    cloud_enabled: bool

    cloud_id: Optional[str]

def build_bayesian_static_routing_context(env: Any,) -> BayesianStaticRoutingContext:
    """
    从完整 Environment 中提取 Bayesian Game
    唯一允许使用的静态 Routing 信息。

    这是 Bayesian 子系统与完整 Environment 之间
    唯一允许存在的 direct adapter。

    白名单只允许读取：

        edge_dc_ids
        routing_action_target_dc_ids
        enable_cloud_action
        cloud_id

    明确禁止未来在本函数中加入：

        datacenters
        host_list
        CPU / GPU utilization
        waiting queue
        running queue
        available resources
        Remote observation
        Remote load

    如果以后 Bayesian 模块需要新的输入，
    必须先判断它是否违反 Remote Partial Observability。
    """

    edge_dc_ids = tuple(
        str(
            dc_id
        )
        for dc_id
        in env.edge_dc_ids
    )

    routing_action_target_dc_ids = tuple(
        str(
            dc_id
        )
        for dc_id
        in env.routing_action_target_dc_ids
    )

    cloud_enabled = bool(
        getattr(
            env,
            "enable_cloud_action",
            False,
        )
    )

    raw_cloud_id = getattr(
        env,
        "cloud_id",
        None,
    )

    cloud_id = (
        str(
            raw_cloud_id
        )
        if (
            cloud_enabled
            and raw_cloud_id is not None
        )
        else None
    )

    return BayesianStaticRoutingContext(
        edge_dc_ids=(
            edge_dc_ids
        ),

        routing_action_target_dc_ids=(
            routing_action_target_dc_ids
        ),

        cloud_enabled=(
            cloud_enabled
        ),

        cloud_id=(
            cloud_id
        ),
    )

@dataclass(frozen=True)
class BayesianRoutingGameDefinition:
    """
    BGH-MASAC Bayesian Routing Game 的静态定义。

    这是 Game Schema，不是 Bayesian State。

    因此：
        - 不维护 alpha / posterior；
        - 不读取 Neighbor Feedback；
        - 不访问 Remote realtime state；
        - 不改变 Routing Observation；
        - 不改变 MASAC policy；
        - 不改变 Reward。
    """

    # ----------------------------------------------------------
    # Players
    #
    # 每个 Edge DC 的 Routing Agent 是一个 Bayesian Player。
    # ----------------------------------------------------------

    player_ids: Tuple[str, ...]

    # ----------------------------------------------------------
    # Routing Action Domain
    #
    # 与 Environment 当前 Routing action mapping 完全一致。
    # ----------------------------------------------------------

    action_target_dc_ids: Tuple[str, ...]

    # Cloud 是否属于当前 Routing action space。
    cloud_enabled: bool

    cloud_id: Optional[str]

    # ----------------------------------------------------------
    # Hidden Type Space
    #
    # 只应用于：
    #
    #   source Edge -> remote Edge
    #
    # Self / Cloud 不建立 Remote Edge type。
    # ----------------------------------------------------------

    remote_type_space: Tuple[
        BayesianRemoteType,
        ...
    ]

    information_policy: (
        BayesianInformationPolicy
    )
    # ----------------------------------------------------------
    # 固定研究语义
    #
    # 这些不是 tunable hyperparameter。
    # ----------------------------------------------------------

    decision_layer: str = "routing_only"

    decision_process: str = (
        "asynchronous_repeated_routing"
    )

    game_objective: str = (
        "cooperative_system_scheduling"
    )

    belief_scope: str = (
        "directed_source_target_pair"
    )

    evidence_scope: str = (
        "finalized_historical_outcomes_only"
    )

    utility_semantics: str = (
        "cooperative_expected_routing_prior_utility"
    )

    # 本项目不使用独立 BNE / PBE solver。
    # MASAC 仍然是最终 policy learner。
    equilibrium_solver: str = "none"

    # Bayesian Game 禁止通过接口读取 Remote realtime state。
    # remote_realtime_state_allowed: bool = False

    # ----------------------------------------------------------
    # Future Congestion Game extension point
    #
    # 当前只预留语义接口，不实现任何 Congestion Cost。
    # ----------------------------------------------------------

    externality_extension: str = (
        "reserved_not_implemented"
    )

    def directed_remote_pairs(
            self,
    ) -> Tuple[
        Tuple[str, str],
        ...
    ]:
        """
        返回未来 Bayesian Belief Store 允许维护的全部：

            source Edge -> target Edge

        有向 pair。

        Self 和 Cloud 均不建立 Remote Type Belief。
        """

        return tuple(
            (
                source_dc_id,
                target_dc_id,
            )
            for source_dc_id
            in self.player_ids
            for target_dc_id
            in self.player_ids
            if target_dc_id != source_dc_id
        )

    def action_semantics_for(
            self,
            source_dc_id: str,
    ) -> Tuple[
        BayesianRoutingActionSemantic,
        ...
    ]:
        """
        返回一个 Routing Agent 对全部 action 的 Bayesian 语义。

        规则：

            source -> source
                Self

            source -> another Edge
                Remote Edge
                存在 Hidden Type

            source -> Cloud
                Cloud
                不建立 Remote Edge Type
        """

        source_dc_id = str(
            source_dc_id
        )

        if source_dc_id not in self.player_ids:
            raise ValueError(
                "Bayesian Routing Game 中不存在 player："
                f"{source_dc_id}"
            )

        action_semantics = []

        for action_index, target_dc_id in enumerate(
                self.action_target_dc_ids
        ):

            target_dc_id = str(
                target_dc_id
            )

            if target_dc_id == source_dc_id:

                action_kind = (
                    BayesianRoutingActionKind.SELF
                )

            elif target_dc_id in self.player_ids:

                action_kind = (
                    BayesianRoutingActionKind
                    .REMOTE_EDGE
                )

            elif (
                self.cloud_enabled
                and self.cloud_id is not None
                and target_dc_id == self.cloud_id
            ):

                action_kind = (
                    BayesianRoutingActionKind.CLOUD
                )

            else:

                raise RuntimeError(
                    "发现未知 Routing target："
                    f"source={source_dc_id}, "
                    f"target={target_dc_id}, "
                    f"action={action_index}"
                )

            action_semantics.append(
                BayesianRoutingActionSemantic(
                    action_index=int(
                        action_index
                    ),

                    target_dc_id=(
                        target_dc_id
                    ),

                    action_kind=(
                        action_kind
                    ),

                    has_hidden_remote_type=(
                        action_kind
                        == BayesianRoutingActionKind
                        .REMOTE_EDGE
                    ),
                )
            )

        return tuple(
            action_semantics
        )

    def to_metadata(
            self,
    ) -> Dict[str, Any]:
        """
        返回可日志化的静态 Game Definition。

        注意：
            这里不包含任何 Bayesian Posterior。
        """

        return {
            # "definition_version":
            #     int(
            #         BAYESIAN_GAME_DEFINITION_VERSION
            #     ),

            "player_ids":
                list(
                    self.player_ids
                ),

            "action_target_dc_ids":
                list(
                    self.action_target_dc_ids
                ),

            "cloud_enabled":
                bool(
                    self.cloud_enabled
                ),

            "cloud_id":
                self.cloud_id,

            "remote_type_space": [
                remote_type.value
                for remote_type
                in self.remote_type_space
            ],

            "decision_layer":
                self.decision_layer,

            "decision_process":
                self.decision_process,

            "game_objective":
                self.game_objective,

            "belief_scope":
                self.belief_scope,

            "evidence_scope":
                self.evidence_scope,

            "utility_semantics":
                self.utility_semantics,

            "equilibrium_solver":
                self.equilibrium_solver,

            # "remote_realtime_state_allowed":
            #     bool(
            #         self.remote_realtime_state_allowed
            #     ),
            "information_policy":
                self.information_policy.to_metadata(),

            "externality_extension":
                self.externality_extension,

            "directed_remote_pair_count":
                len(
                    self.directed_remote_pairs()
                ),
        }


def build_bayesian_routing_game_definition(
        static_context:
        BayesianStaticRoutingContext,
) -> BayesianRoutingGameDefinition:
    """
    根据已经脱敏的 BayesianStaticRoutingContext
    构造 Bayesian Routing Game Definition。

    重要：

        本函数从 Step 5 开始不再接收完整 Environment。

    因此本函数在接口层面无法读取：

        Remote CPU
        Remote GPU
        Remote Queue
        Remote Host
        Remote available resources
        Remote raw observation

    本函数只处理：

        Edge DC identity
        Routing action target mapping
        Cloud ON/OFF
        Cloud ID

    它：

        不调用随机数；
        不维护 Bayesian posterior；
        不读取 Historical Feedback；
        不修改 Observation；
        不修改 Reward；
        不修改 Replay；
        不修改 Actor / Critic。
    """

    # ==========================================================
    # 0. Bayesian Information Boundary
    #
    # 研究约束固定在代码中，而不是由 config 动态开启。
    # ==========================================================

    information_policy = (
        BayesianInformationPolicy()
    )

    information_policy.validate()

    # ==========================================================
    # 1. Players = Edge DC Routing Agents
    # ==========================================================

    player_ids = tuple(
        str(
            dc_id
        )
        for dc_id
        in static_context.edge_dc_ids
    )

    if not player_ids:
        raise RuntimeError(
            "Bayesian Routing Game 至少需要 "
            "一个 Edge DC player。"
        )

    if (
        len(
            set(
                player_ids
            )
        )
        != len(
            player_ids
        )
    ):
        raise RuntimeError(
            "Bayesian Routing Game 中存在重复 Edge DC ID。"
        )

    # ==========================================================
    # 2. Action targets
    #
    # 完全继承当前 Environment 的 Routing action mapping。
    # ==========================================================

    action_target_dc_ids = tuple(
        str(
            dc_id
        )
        for dc_id
        in static_context.routing_action_target_dc_ids
    )

    if not action_target_dc_ids:
        raise RuntimeError(
            "Routing action target 不能为空。"
        )

    if (
        len(
            set(
                action_target_dc_ids
            )
        )
        != len(
            action_target_dc_ids
        )
    ):
        raise RuntimeError(
            "Routing action target 存在重复。"
        )

    player_set = set(
        player_ids
    )

    target_set = set(
        action_target_dc_ids
    )

    # 每个 player 必须保留自己的 Self action。
    missing_self_targets = (
        player_set
        - target_set
    )

    if missing_self_targets:

        raise RuntimeError(
            "Bayesian Routing Game 缺少 Self action："
            f"{sorted(missing_self_targets)}"
        )

    # ==========================================================
    # 3. Cloud semantics
    # ==========================================================

    cloud_enabled = bool(
        getattr(
            static_context,
            "enable_cloud_action",
            False,
        )
    )

    raw_cloud_id = getattr(
        static_context,
        "cloud_id",
        None,
    )

    cloud_id = (
        str(
            raw_cloud_id
        )
        if (
            cloud_enabled
            and raw_cloud_id is not None
        )
        else None
    )

    non_edge_targets = (
        target_set
        - player_set
    )

    if cloud_enabled:

        if cloud_id is None:

            raise RuntimeError(
                "Cloud action 已开启，"
                "但 Environment 缺少 cloud_id。"
            )

        if non_edge_targets != {
            cloud_id
        }:

            raise RuntimeError(
                "非 Edge Routing target "
                "必须且只能是 Cloud："
                f"actual={sorted(non_edge_targets)}, "
                f"cloud={cloud_id}"
            )

    elif non_edge_targets:

        raise RuntimeError(
            "Cloud action 已关闭，"
            "但 Routing action 中仍存在非 Edge target："
            f"{sorted(non_edge_targets)}"
        )

    # ==========================================================
    # 4. Formal Bayesian Game Definition
    # ==========================================================

    definition = (
        BayesianRoutingGameDefinition(
            player_ids=(
                player_ids
            ),

            action_target_dc_ids=(
                action_target_dc_ids
            ),

            cloud_enabled=(
                cloud_enabled
            ),

            cloud_id=(
                cloud_id
            ),

            remote_type_space=(
                BayesianRemoteType.GOOD,
                BayesianRemoteType.NORMAL,
                BayesianRemoteType.RISKY,
            ),
            information_policy=(
                information_policy
            ),
        )
    )

    # ==========================================================
    # 5. 对每个 Player 做一次 action semantic validation
    #
    # 这里只校验静态结构，不产生任何 Bayesian state。
    # ==========================================================

    for player_id in (
        definition.player_ids
    ):

        definition.action_semantics_for(
            player_id
        )

    return definition