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
