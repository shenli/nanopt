"""Strict Pydantic models for NanoPT configuration profiles.

These classes are intentionally declarative: each class mirrors one conceptual block in a YAML
profile. Keeping the schema visible in ordinary Python makes the allowed configuration surface
easy to inspect without learning a separate configuration framework.
"""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

Scalar = str | int | float | bool | None


class StrictModel(BaseModel):
    """Base model that rejects undocumented configuration keys."""

    model_config = ConfigDict(extra="forbid", strict=True)


# Hardware profiles describe capabilities and evidence, not runtime policy guesses.
class PlatformConfig(StrictModel):
    os: str
    architecture: str


class AcceleratorConfig(StrictModel):
    vendor: Literal["nvidia"]
    count: int = Field(ge=1)
    name_regex: str
    nominal_total_vram_gib: float = Field(gt=0)
    expected_compute_capability: str


class PrecisionConfig(StrictModel):
    preferred_compute_dtype: Literal["bfloat16", "float16", "float32"]
    require_bf16_runtime_check: bool
    allow_tf32: bool


class MemoryBudgetConfig(StrictModel):
    minimum_free_vram_before_start_gib: float = Field(ge=0)
    soft_peak_reserved_gib: float = Field(gt=0)
    hard_peak_reserved_gib: float = Field(gt=0)

    @model_validator(mode="after")
    def check_budget_order(self) -> MemoryBudgetConfig:
        if self.soft_peak_reserved_gib > self.hard_peak_reserved_gib:
            raise ValueError("soft VRAM budget must not exceed hard VRAM budget")
        return self


class RuntimeDefaultsConfig(StrictModel):
    attention_backend: Literal["sdpa", "eager"]
    torch_compile: bool
    gradient_checkpointing: bool
    dataloader_pin_memory: bool
    quantization: Literal["none"]


class HardwareValidationConfig(StrictModel):
    evidence_manifest: str | None
    validated_commit: str | None
    validated_at: str | None


class HardwareProfile(StrictModel):
    schema_version: Literal[1]
    id: str
    support_status: Literal["proposed_unvalidated", "smoke_tested", "validated"]
    description: str
    platform: PlatformConfig
    accelerator: AcceleratorConfig
    precision: PrecisionConfig
    memory_budget: MemoryBudgetConfig
    runtime_defaults: RuntimeDefaultsConfig
    validation: HardwareValidationConfig
    notes: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def check_validation_evidence(self) -> HardwareProfile:
        if self.support_status == "validated" and not self.validation.evidence_manifest:
            raise ValueError("validated hardware requires an evidence manifest")
        return self


# Model profiles pin loading and adapter behavior independently of an experiment.
class ModelSourceConfig(StrictModel):
    provider: Literal["huggingface"]
    model_id: str
    revision: str | None
    tokenizer_revision: str | None
    trust_remote_code: Literal[False]


class ModelLoadingConfig(StrictModel):
    dtype: Literal["bfloat16", "float16", "float32"]
    low_cpu_mem_usage: bool
    use_safetensors: bool


class RendererConfig(StrictModel):
    type: Literal["tokenizer_chat_template"]
    enable_thinking: bool
    add_generation_prompt: bool


class LoraConfig(StrictModel):
    method: Literal["lora"]
    rank: int = Field(gt=0)
    alpha: int = Field(gt=0)
    dropout: float = Field(ge=0, lt=1)
    bias: Literal["none", "all", "lora_only"]
    target_modules: list[str] = Field(min_length=1)


class ModelChecksConfig(StrictModel):
    min_transformers_version: str
    assert_target_modules_exist: bool
    record_chat_template_hash: bool


class ModelUsageConfig(StrictModel):
    official_pipeline_start: bool
    allowed_for: list[str]


class ModelProfile(StrictModel):
    schema_version: Literal[1]
    id: str
    source: ModelSourceConfig
    loading: ModelLoadingConfig
    renderer: RendererConfig
    adapter: LoraConfig | None = None
    checks: ModelChecksConfig | None = None
    usage: ModelUsageConfig | None = None
    notes: list[str] = Field(default_factory=list)


# Evaluation profiles separate deterministic scoring from sampled qualitative output.
class BaseEvalDataConfig(StrictModel):
    dataset: str
    splits: list[str]
    max_prompt_length: int = Field(gt=0)


class DeterministicGenerationConfig(StrictModel):
    do_sample: Literal[False]
    max_new_tokens: int = Field(gt=0)


class SampledGenerationConfig(StrictModel):
    do_sample: Literal[True]
    temperature: float = Field(gt=0)
    top_p: float = Field(gt=0, le=1)
    num_samples_per_prompt: int = Field(gt=0)
    max_new_tokens: int = Field(gt=0)


class GenerationConfig(StrictModel):
    deterministic: DeterministicGenerationConfig
    sampled: SampledGenerationConfig


class BaseEvaluationConfig(StrictModel):
    parser: Literal["strict_answer_v1"]
    verifier: Literal["exact_answer_v1"]
    confidence_interval: Literal["wilson"]
    confidence_level: float = Field(gt=0, lt=1)

    @model_validator(mode="after")
    def require_implemented_confidence_level(self) -> BaseEvaluationConfig:
        if self.confidence_level != 0.95:
            raise ValueError("M3 implements only a 0.95 confidence level")
        return self


class BaseArtifactsConfig(StrictModel):
    save_token_ids: Literal[True]
    save_logprobs: Literal[True]
    save_all_examples: Literal[True]


class BaseEvalExperiment(StrictModel):
    schema_version: Literal[1]
    id: str
    stage: Literal["evaluation"]
    seed: int
    data: BaseEvalDataConfig
    generation: GenerationConfig
    evaluation: BaseEvaluationConfig
    artifacts: BaseArtifactsConfig


# SFT profiles define completion-only data and optimizer behavior.
class TrainDataConfig(StrictModel):
    dataset: str
    train_split: str
    validation_split: str
    max_sequence_length: int = Field(gt=0)
    completion_only: Literal[True]


class OptimizerConfig(StrictModel):
    micro_batch_size: int = Field(gt=0)
    gradient_accumulation_steps: int = Field(gt=0)
    max_steps: int | None = Field(default=None, gt=0)
    epochs: int = Field(gt=0)
    learning_rate: float = Field(gt=0)
    weight_decay: float = Field(ge=0)
    warmup_ratio: float = Field(ge=0, lt=1)
    scheduler: Literal["cosine"]
    max_grad_norm: float = Field(gt=0)
    gradient_checkpointing: bool
    compute_dtype: Literal["bfloat16", "float16", "float32"]
    optimizer: Literal["adamw"]
    log_every_optimizer_steps: int = Field(gt=0)
    eval_every_optimizer_steps: int = Field(gt=0)
    save_every_optimizer_steps: int = Field(gt=0)


class TrainAdapterConfig(StrictModel):
    inherit_from: str | None
    name: str
    trainable: Literal[True]


class LinkedEvaluationConfig(StrictModel):
    experiment: str


class SftArtifactsConfig(StrictModel):
    save_adapter_only: bool
    save_optimizer_state: bool
    save_samples: bool


class SftExperiment(StrictModel):
    schema_version: Literal[1]
    id: str
    stage: Literal["sft"]
    seed: int
    data: TrainDataConfig
    training: OptimizerConfig
    adapter: TrainAdapterConfig
    evaluation: LinkedEvaluationConfig
    artifacts: SftArtifactsConfig
    status: Literal["proposed_unvalidated", "smoke_tested", "validated"]


# DPO profiles make policy/reference roles and sequence reduction explicit.
class DpoDataConfig(StrictModel):
    dataset: str
    train_split: str
    validation_split: str
    max_prompt_length: int = Field(gt=0)
    max_completion_length: int = Field(gt=0)
    sequence_logprob_reduction: Literal["sum", "mean"]


class DpoReferenceConfig(StrictModel):
    checkpoint_stage: str
    mode: Literal["precomputed"]
    cache_dtype: Literal["float32"]
    cache_validation_sample_size: int = Field(gt=0)


class DpoPolicyConfig(StrictModel):
    initialize_from_stage: str
    adapter_name: str


class DpoTrainingConfig(StrictModel):
    pair_micro_batch_size: int = Field(gt=0)
    gradient_accumulation_steps: int = Field(gt=0)
    epochs: int = Field(gt=0)
    learning_rate: float = Field(gt=0)
    beta: float = Field(gt=0)
    weight_decay: float = Field(ge=0)
    warmup_ratio: float = Field(ge=0, lt=1)
    scheduler: Literal["cosine"]
    max_grad_norm: float = Field(gt=0)
    gradient_checkpointing: bool
    compute_dtype: Literal["bfloat16", "float16", "float32"]
    optimizer: Literal["adamw"]
    concatenate_chosen_rejected: bool
    log_every_optimizer_steps: int = Field(gt=0)
    eval_every_optimizer_steps: int = Field(gt=0)
    save_every_optimizer_steps: int = Field(gt=0)


class DpoEvaluationConfig(LinkedEvaluationConfig):
    preference_breakdown: bool


class DpoArtifactsConfig(StrictModel):
    save_adapter_only: bool
    save_reference_cache_manifest: bool


class DpoExperiment(StrictModel):
    schema_version: Literal[1]
    id: str
    stage: Literal["dpo"]
    seed: int
    data: DpoDataConfig
    reference: DpoReferenceConfig
    policy: DpoPolicyConfig
    training: DpoTrainingConfig
    evaluation: DpoEvaluationConfig
    artifacts: DpoArtifactsConfig
    status: Literal["proposed_unvalidated", "smoke_tested", "validated"]


# GRPO profiles expose rollout, reward, advantage, and optimization choices separately.
class GrpoDataConfig(StrictModel):
    dataset: str
    prompt_pool_split: str
    validation_split: str
    max_prompt_length: int = Field(gt=0)


class GrpoPolicyConfig(StrictModel):
    initialize_from_stage: str
    policy_adapter_name: str
    reference_adapter_stage: str


class RolloutConfig(StrictModel):
    prompt_batch_size: int = Field(gt=0)
    group_size: int = Field(ge=2)
    max_completion_length: int = Field(gt=0)
    temperature: float = Field(gt=0)
    top_p: float = Field(gt=0, le=1)
    top_k: int | None = Field(default=None, gt=0)
    stop_on_eos: bool
    store_old_logprobs_dtype: Literal["float32"]


class RewardComponentsConfig(StrictModel):
    correctness: float
    format: float
    length_penalty: float


class RewardConfig(StrictModel):
    parser: str
    verifier: str
    components: RewardComponentsConfig


class AdvantageConfig(StrictModel):
    mode: Literal["group_centered", "group_zscore"]
    population_std: Literal[True]
    epsilon: float = Field(gt=0)


class GrpoOptimizationConfig(StrictModel):
    iterations: int = Field(gt=0)
    update_epochs: int = Field(gt=0)
    minibatch_completions: int = Field(gt=0)
    gradient_accumulation_steps: int = Field(gt=0)
    learning_rate: float = Field(gt=0)
    weight_decay: float = Field(ge=0)
    warmup_ratio: float = Field(ge=0, lt=1)
    scheduler: Literal["cosine"]
    clip_epsilon: float = Field(gt=0, lt=1)
    loss_normalization: Literal["token_mean", "sequence_mean"]
    kl_beta: float = Field(ge=0)
    kl_estimator: Literal["direct", "k3"]
    max_grad_norm: float = Field(gt=0)
    gradient_checkpointing: bool
    compute_dtype: Literal["bfloat16", "float16", "float32"]
    optimizer: Literal["adamw"]
    log_every_iterations: int = Field(gt=0)
    eval_every_iterations: int = Field(gt=0)
    save_every_iterations: int = Field(gt=0)


class GrpoEvaluationConfig(LinkedEvaluationConfig):
    reward_hacking_suite: bool


class GrpoArtifactsConfig(StrictModel):
    save_adapter_only: bool
    save_trajectories: bool
    save_reward_examples: bool


class GrpoExperiment(StrictModel):
    schema_version: Literal[1]
    id: str
    stage: Literal["grpo"]
    seed: int
    data: GrpoDataConfig
    policy: GrpoPolicyConfig
    rollout: RolloutConfig
    reward: RewardConfig
    advantage: AdvantageConfig
    optimization: GrpoOptimizationConfig
    evaluation: GrpoEvaluationConfig
    artifacts: GrpoArtifactsConfig
    status: Literal["proposed_unvalidated", "smoke_tested", "validated"]
    notes: list[str] = Field(default_factory=list)


# The toy PPO lab is deliberately isolated from the reference Qwen training pipeline.
class ToyEnvironmentConfig(StrictModel):
    type: Literal["tiny_sequence_environment"]
    horizon: int = Field(gt=0)


class ToyPolicyConfig(StrictModel):
    type: Literal["tiny_causal_model"]
    hidden_size: int = Field(gt=0)
    layers: int = Field(gt=0)


class ToyAlgorithmConfig(StrictModel):
    gamma: float
    gae_lambda: float
    clip_epsilon: float = Field(gt=0)
    update_epochs: int = Field(gt=0)
    minibatch_size: int = Field(gt=0)
    value_loss_coefficient: float = Field(ge=0)
    entropy_coefficient: float = Field(ge=0)
    max_grad_norm: float = Field(gt=0)


class ToyScopeConfig(StrictModel):
    part_of_reference_qwen_pipeline: Literal[False]
    cpu_supported: bool


class TeachingLabExperiment(StrictModel):
    schema_version: Literal[1]
    id: str
    stage: Literal["teaching_lab"]
    seed: int
    environment: ToyEnvironmentConfig
    policy: ToyPolicyConfig
    algorithm: ToyAlgorithmConfig
    scope: ToyScopeConfig


# Agent evaluation profiles encode sandbox limits as data that can be audited.
class AgentTasksConfig(StrictModel):
    suite: str
    split: str


class AgentPolicyConfig(StrictModel):
    checkpoint: str | None
    max_new_tokens_per_turn: int = Field(gt=0)
    temperature: float = Field(gt=0)
    top_p: float = Field(gt=0, le=1)
    max_turns: int = Field(gt=0)


class AgentEnvironmentConfig(StrictModel):
    backend: Literal["docker", "fake"]
    image: str
    network: Literal["none"]
    run_as_non_root: Literal[True]
    expose_gpu: Literal[False]
    tool_budget: int = Field(gt=0)
    test_run_budget: int = Field(gt=0)
    wall_clock_timeout_seconds: int = Field(gt=0)
    memory_limit_mib: int = Field(gt=0)
    pids_limit: int = Field(gt=0)
    cpu_limit: float = Field(gt=0)


class AgentVerificationConfig(StrictModel):
    public_tests: bool
    hidden_tests: bool
    separate_workspace: Literal[True]


class AgentArtifactsConfig(StrictModel):
    save_trajectory: bool
    save_final_patch: bool
    save_hidden_summary_only: bool


class AgentEvaluationExperiment(StrictModel):
    schema_version: Literal[1]
    id: str
    stage: Literal["agent_evaluation"]
    seed: int
    tasks: AgentTasksConfig
    policy: AgentPolicyConfig
    environment: AgentEnvironmentConfig
    tools: list[Literal["list_files", "read_file", "search", "apply_patch", "run_tests", "finish"]]
    verification: AgentVerificationConfig
    artifacts: AgentArtifactsConfig
    status: Literal["proposed_unvalidated", "smoke_tested", "validated"]


ExperimentProfile = Annotated[
    BaseEvalExperiment
    | SftExperiment
    | DpoExperiment
    | GrpoExperiment
    | TeachingLabExperiment
    | AgentEvaluationExperiment,
    Field(discriminator="stage"),
]


# Recipes compose named experiments; they do not hide their stages in trainer callbacks.
class RecipeStage(StrictModel):
    id: str
    command: Literal[
        "calibrate_load",
        "calibrate_eval",
        "calibrate_sft",
        "calibrate_dpo",
        "calibrate_grpo",
        "eval",
        "data_preferences",
        "train_sft",
        "train_dpo",
        "train_grpo",
        "report_build",
    ]
    experiment: str | None = None
    input_checkpoint: str | None = None
    compare: list[str] | None = None
    overrides: dict[str, Scalar] = Field(default_factory=dict)


class StagePolicy(StrictModel):
    independently_resumable: bool
    fail_fast: bool
    require_calibration_before_training: bool
    protected_test_evaluation_after_recipe_freeze: bool


class RecipeProfile(StrictModel):
    schema_version: Literal[1]
    id: str
    description: str
    hardware: str
    model: str
    stages: list[RecipeStage]
    stage_policy: StagePolicy
    status: Literal["proposed_unvalidated", "smoke_tested", "validated"]


class ResolvedConfig(StrictModel):
    """A resolved profile bundle with non-colliding configuration namespaces."""

    schema_version: Literal[1] = 1
    hardware: HardwareProfile
    model: ModelProfile
    experiment: ExperimentProfile
    recipe: RecipeProfile | None = None
    recipe_stage: str | None = None
