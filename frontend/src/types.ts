export type Health = {
  progress: number;
  quality: number;
  schedule: number;
  reliability: number;
};

export type Treatment = {
  incentive_pressure: number;
  attention_budget: number;
};

export type Agent = {
  id: string;
  role: string;
  department: string;
  manager_id?: string;
};

export type TimelineEvent = {
  sequence: number;
  kind: string;
  tick: number;
  actor_id: string | null;
  causes: number[];
  payload: Record<string, unknown>;
};

export type EvidenceNode = {
  sequence: number;
  kind: string;
  tick: number;
  actor_id: string | null;
  department: string;
  depth: number | null;
  causes: number[];
  evidence_refs: string[];
  event: TimelineEvent;
};

export type DecisionReport = {
  agent_id: string;
  department: string;
  depth: number;
  tick: number;
  scope: string[];
  health: Health;
  confidence: number;
  escalate: boolean;
  resource_request: number;
  explanation: string;
};

export type DecisionNode = {
  sequence: number;
  agent_id: string;
  tick: number;
  policy: string;
  context_hash: string;
  report: DecisionReport;
  actions: Array<Record<string, unknown>>;
  provider_metadata: Record<string, unknown>;
};

export type IndexedRun = {
  status?: "completed";
  seed: number;
  treatment: string;
  run_id: string;
  run_directory: string;
  executive_optimism_bias: number;
  incident_regret: number | null;
  incentive_pressure: number;
  attention_budget: number;
};

export type AnalysisSpecification = {
  seed: number;
  missingness: "complete_case" | "fail_if_missing";
  contrasts: Array<{
    id: string;
    baseline: string;
    intervention: string;
    outcome: string;
    direction: "increase" | "decrease" | "two_sided";
    family: string;
    status: "confirmatory" | "exploratory";
  }>;
  sensitivities?: Array<Record<string, unknown>>;
};

export type ExperimentRequest = {
  name: string;
  seeds: number[];
  scenario: Record<string, unknown>;
  organization: { agents: Agent[] };
  treatments: Record<string, Treatment>;
  analysis: AnalysisSpecification;
  max_concurrency: number;
};

export type Experiment = {
  name: string;
  request: ExperimentRequest | null;
  failures: Array<{ seed: number; treatment: string | null; error: string }>;
  analysis: {
    baseline: string;
    unit_of_analysis: string;
    design_diagnostics: { complete_pairs: number };
    comparisons: Record<string, {
      status?: string;
      n_pairs: number;
      mean_difference?: number;
      p_value?: number;
      p_value_method?: "exact_sign_flip" | "monte_carlo_sign_flip";
      ci_low?: number;
      ci_high?: number;
    }>;
  };
  runs: IndexedRun[];
};

export type RunDetail = {
  manifest: { run_id: string; seed: number; event_hash: string; policy: string };
  request: {
    scenario: Record<string, unknown>;
    treatment: Treatment;
    organization: { agents: Agent[] };
  };
  metrics: {
    distortion: Array<{
      agent_id: string;
      department: string;
      depth: number;
      tick: number;
      optimism_bias: number;
      absolute_error: number;
    }>;
  };
};

export type JobStatus = {
  job_id: string;
  experiment: string;
  status: "queued" | "running" | "completed" | "failed";
  completed_runs: number;
  failed_runs: number;
  total_runs: number;
  error: string | null;
};

// ---------------------------------------------------------------------------
// Builder input types — mirror the backend ExperimentRequest schema
// (src/groundline/experiments/runner.py, world/models.py, organization/models.py,
//  experiments/analysis.py, metrics/outcomes.py). Server is source of truth;
// these are for authoring + light client-side validation.
// ---------------------------------------------------------------------------

export type AgentRole = "contributor" | "manager" | "executive";
export type MissingnessPolicy = "complete_case" | "fail_if_missing";
export type ExpectedDirection = "increase" | "decrease" | "two_sided";
export type AnalysisStatus = "confirmatory" | "exploratory";
export type ReportingSpan = "narrow" | "wide";

export const ANALYSIS_OUTCOMES = [
  "upward_amplification",
  "pre_release_upward_amplification",
  "executive_optimism_bias_adverse_mean",
  "executive_optimism_bias_adverse_median",
  "executive_optimism_bias_all_mean",
  "executive_optimism_bias_all_median",
  "executive_absolute_error_adverse_mean",
  "executive_absolute_error_all_mean",
  "executive_vector_loss_adverse_mean",
  "executive_vector_loss_all_mean",
  "executive_equal_weight_vector_loss_adverse_mean",
  "executive_equal_weight_vector_loss_all_mean",
  "executive_progress_error_adverse_mean",
  "executive_progress_error_all_mean",
  "executive_quality_error_adverse_mean",
  "executive_quality_error_all_mean",
  "executive_schedule_error_adverse_mean",
  "executive_schedule_error_all_mean",
  "executive_reliability_error_adverse_mean",
  "executive_reliability_error_all_mean",
  "executive_optimism_bias_pre_release_adverse_mean",
  "executive_optimism_bias_pre_release_adverse_median",
  "executive_optimism_bias_pre_release_all_mean",
  "executive_optimism_bias_pre_release_all_median",
  "executive_absolute_error_pre_release_adverse_mean",
  "executive_absolute_error_pre_release_all_mean",
  "executive_vector_loss_pre_release_adverse_mean",
  "executive_vector_loss_pre_release_all_mean",
  "edge_transformation",
  "pre_release_edge_transformation",
  "calibration_brier_score",
  "operational_harm",
  "oracle_regret",
  "escalation_delay_mean",
] as const;
export type AnalysisOutcome = (typeof ANALYSIS_OUTCOMES)[number];

export const SENSITIVITY_KINDS = [
  "adverse_vs_all_ticks",
  "mean_vs_median",
  "equal_health_dimension_weights",
  "progress_dimension_only",
  "quality_dimension_only",
  "schedule_dimension_only",
  "reliability_dimension_only",
  "exclude_post_release_ticks",
  "exclude_invalid_or_refused",
] as const;

export type BuilderAgent = {
  id: string;
  manager_id: string | null;
  role: AgentRole;
  department: string;
  skills: Record<string, number>;
  traits: Record<string, number>;
  utility_weights: Record<string, number>;
};

export type WorkItemConfig = {
  id: string;
  department: string;
  business_value: number;
  effort: number;
  deadline_tick: number;
  dependencies: string[];
};

export type OperationalHarmMaxima = {
  release_delay: number;
  escaped_defects: number;
  incident: number;
  remediation: number;
  scope_loss: number;
};

export type ScenarioConfig = {
  max_ticks: number;
  shock_tick: number;
  shock_item_id: string;
  shock_severity: number;
  work_items: WorkItemConfig[];
  harm_maxima: OperationalHarmMaxima;
};

export type TreatmentConfig = {
  incentive_pressure: number;
  attention_budget: number;
  reporting_span?: ReportingSpan;
};

export type ContrastSpecification = {
  id: string;
  baseline: string;
  intervention: string;
  outcome: AnalysisOutcome;
  direction: ExpectedDirection;
  family: string;
  status: AnalysisStatus;
};

export type SensitivitySpecification = {
  id: string;
  contrast_id: string;
  kind: string;
  missingness: MissingnessPolicy | null;
  threshold: number | null;
};

export type OutcomeSpecification = {
  adverse_health_threshold: number;
  release_health_threshold: number;
  escalation_severity_threshold: number;
  escalation_sensitivity_thresholds: number[];
};

export type BuilderAnalysisSpecification = {
  seed: number;
  missingness: MissingnessPolicy;
  contrasts: ContrastSpecification[];
  sensitivities: SensitivitySpecification[];
};

export type BuilderExperimentRequest = {
  name: string;
  seeds: number[];
  scenario: ScenarioConfig;
  organization: { agents: BuilderAgent[] };
  treatments: Record<string, TreatmentConfig>;
  outcome_specification: OutcomeSpecification;
  analysis: BuilderAnalysisSpecification;
  max_concurrency: number;
};

