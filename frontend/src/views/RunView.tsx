import { useEffect, useState } from "react";
import { CaretDown, CheckCircle, Flask, Warning } from "@phosphor-icons/react";
import { loadEvidence } from "../api";
import { DecisionInspector, EvidenceInspector, InterventionControls, type InterventionValues } from "../components";
import { DistortionLadder, SignalChart } from "../charts";
import type { DecisionNode, EvidenceNode, Experiment, JobStatus, RunDetail, TimelineEvent } from "../types";

export type ExperimentStatus = "loading" | "loaded" | "empty" | "error";
export type SelectedRunState =
  | { status: "idle" }
  | { status: "loading"; runId: string }
  | { status: "error"; runId: string; message: string }
  | {
    status: "loaded";
    runId: string;
    detail: RunDetail;
    timeline: TimelineEvent[];
    evidence: EvidenceNode[];
    evidenceKey: string;
    evidenceStatus: "loaded" | "loading" | "error";
    evidenceError: string;
    decisions: DecisionNode[];
    decisionsStatus: "loaded" | "loading" | "error";
    decisionsError: string;
  };

const mean = (values: number[]) => values.reduce((sum, value) => sum + value, 0) / Math.max(values.length, 1);

export function RunView({
  experiment,
  experimentStatus,
  experimentError,
  onReloadExperiment,
  selectedRun,
  selectedRunState,
  onChooseRun,
  onSetRunReload,
  job,
  launchError,
  launchPolicy,
  onRunIntervention,
}: {
  experiment: Experiment | null;
  experimentStatus: ExperimentStatus;
  experimentError: string;
  onReloadExperiment: () => void;
  selectedRun: string;
  selectedRunState: SelectedRunState;
  onChooseRun: (runId: string) => void;
  onSetRunReload: (fn: (value: number) => number) => void;
  job: JobStatus | null;
  launchError: string;
  launchPolicy: "fixture" | "record" | "locked";
  onRunIntervention: (values: InterventionValues) => void;
}) {
  const [departmentFilter, setDepartmentFilter] = useState("");
  const [depthFilter, setDepthFilter] = useState("");
  const [filteredEvidence, setFilteredEvidence] = useState<EvidenceNode[] | null>(null);
  const filterKey = JSON.stringify([departmentFilter, depthFilter]);
  const loadedRunId = selectedRunState.status === "loaded" ? selectedRunState.runId : "";
  const loadedEvidenceKey = selectedRunState.status === "loaded" ? selectedRunState.evidenceKey : "";

  useEffect(() => {
    if (!loadedRunId || loadedEvidenceKey === filterKey) return;
    const controller = new AbortController();
    loadEvidence(loadedRunId, {
      department: departmentFilter || undefined,
      depth: depthFilter === "" ? undefined : Number(depthFilter),
    }, controller.signal).then((chain) => {
      if (!controller.signal.aborted) setFilteredEvidence(chain.nodes);
    }).catch(() => {});
    return () => controller.abort();
  }, [departmentFilter, depthFilter, filterKey, loadedEvidenceKey, loadedRunId]);

  if (experimentStatus === "error") {
    return <RunState icon={<Warning size={28} />} title="Artifact read failed" body={experimentError} action={onReloadExperiment} code="uv run groundline experiment --config configs/demo.yaml" />;
  }

  if (experimentStatus === "empty" || (experimentStatus === "loaded" && !experiment?.runs.length)) {
    return <RunState icon={<Flask size={28} />} title="No runs yet" body="Generate a paired experiment to populate the control room." action={onReloadExperiment} />;
  }

  if (experimentStatus === "loading" || !experiment) return <RunLoading />;

  if (selectedRunState.status === "error") {
    return (
      <section className="run-view view-enter">
        <div className="state-page">
          <Warning size={28} />
          <div><h1>Run read failed</h1><p>{selectedRunState.message}</p></div>
          <RunSelect id="failed-run" experiment={experiment} selectedRun={selectedRun} onChooseRun={onChooseRun} />
          <button type="button" className="md3-button md3-button--filled" onClick={() => onSetRunReload((value) => value + 1)}>RETRY</button>
        </div>
      </section>
    );
  }

  if (selectedRunState.status !== "loaded") {
    return (
      <section className="run-view view-enter">
        <div className="loading-page">
          <div className="skeleton" />
          <p>Loading selected run artifacts.</p>
          <RunSelect id="loading-run" experiment={experiment} selectedRun={selectedRun} onChooseRun={onChooseRun} />
        </div>
      </section>
    );
  }

  const detail = selectedRunState.detail;
  const timeline = selectedRunState.timeline;
  const executive = detail.request.organization.agents.find((agent) => agent.role === "executive")?.id;
  const truth = timeline.filter((event) => event.kind === "truth_snapshot").map((event) => ({ tick: event.tick, value: event.payload.health_score as number }));
  const belief = timeline.filter((event) => event.kind === "report" && event.actor_id === executive).map((event) => ({ tick: event.tick, value: event.payload.health_score as number }));
  const lastTick = Math.max(...detail.metrics.distortion.map((metric) => metric.tick));
  const levels = [...new Set(detail.metrics.distortion.map((metric) => metric.depth))].map((depth) => ({
    depth,
    value: mean(detail.metrics.distortion.filter((metric) => metric.depth === depth && metric.tick === lastTick).map((metric) => metric.optimism_bias)),
  }));
  const finalTruth = truth.at(-1)?.value ?? 0;
  const finalBelief = belief.at(-1)?.value ?? finalTruth;
  const finalGap = finalBelief - finalTruth;
  const selected = experiment.runs.find((run) => run.run_id === selectedRun) ?? experiment.runs[0];
  const groupedRuns = experiment.runs.reduce<Record<string, typeof experiment.runs>>(
    (groups, run) => ({ ...groups, [run.treatment]: [...(groups[run.treatment] ?? []), run] }),
    {},
  );
  const treatmentMeans = Object.entries(groupedRuns).map(([name, runs]) => ({
    name,
    value: mean(runs.map((run) => run.executive_optimism_bias)),
    regret: mean(runs.flatMap((run) => run.incident_regret === null ? [] : [run.incident_regret])),
  }));
  const maxTreatment = Math.max(...treatmentMeans.map((item) => Math.abs(item.value)), 0.01);
  const departments = [...new Set(detail.request.organization.agents.map((agent) => agent.department))].sort();
  const depths = [...new Set(detail.metrics.distortion.map((metric) => metric.depth))].sort((a, b) => a - b);
  const comparison = experiment.analysis.comparisons[Object.keys(experiment.analysis.comparisons)[0]];

  return (
    <section className="run-view view-enter" aria-labelledby="run-title">
      <header className="run-header">
        <div>
          <p className="section-label">FIRMWORLD / CAUSAL RUN</p>
          <h1 id="run-title">The Groundline</h1>
        </div>
        <div className="run-badge"><CheckCircle size={17} weight="fill" /><span>LEDGER VERIFIED</span><code>{detail.manifest.event_hash.slice(0, 12)}</code></div>
      </header>

      <section className="control-bar" aria-label="Run controls">
        <RunSelect id="run" experiment={experiment} selectedRun={selectedRun} onChooseRun={onChooseRun} />
        <div className="stat-group">
          <RunStat label="INCENTIVE" value={detail.request.treatment.incentive_pressure.toFixed(2)} />
          <RunStat label="MANAGER AUDITS" value={String(detail.request.treatment.attention_budget)} />
          <RunStat label="POLICY" value={detail.manifest.policy} />
          <RunStat label="SEED" value={String(selected.seed)} />
        </div>
      </section>

      <details className="intervention-section" open>
        <summary>
          <span><strong id="intervention-title">CAUSAL INTERVENTION</strong><small>Launch a fresh 2x2 matrix while holding every seed fixed across cells.</small></span>
          <CaretDown size={18} />
        </summary>
        <div className="intervention-body">
          <InterventionControls
            key={selectedRun}
            initialIncentive={detail.request.treatment.incentive_pressure === 0 ? 0.9 : 0}
            initialAttention={detail.request.treatment.attention_budget === 0 ? 2 : 0}
            busy={job?.status === "queued" || job?.status === "running"}
            onLaunch={(values) => { void onRunIntervention(values); }}
          />
          {job && <div className={`job-progress ${job.status}`}><strong>{job.status.toUpperCase()}</strong><span>{job.completed_runs} / {job.total_runs} RUNS FINALIZED</span><span className="job-policy">POLICY: {launchPolicy.toUpperCase()}</span></div>}
          {launchError && <p className="inline-error" role="alert">{launchError}</p>}
        </div>
      </details>

      <div className="analysis-grid">
        <article className="chart-panel">
          <div className="section-header">
            <div><h2 className="section-title">WORLD TRUTH / EXECUTIVE BELIEF</h2><p>Operational health, computed independently from every agent report.</p></div>
            <div className="gap-display"><span>FINAL GAP</span><strong className="gap-value">{finalGap >= 0 ? "+" : ""}{(finalGap * 100).toFixed(1)}</strong></div>
          </div>
          <div className="chart-legend"><span className="truth-key">WORLD TRUTH</span><span className="belief-key">EXECUTIVE BELIEF</span></div>
          <SignalChart truth={truth} belief={belief} />
        </article>

        <aside className="side-panel">
          <div><h2 className="section-title">DISTORTION CLIMBS</h2><p>Mean optimism bias at the final tick.</p></div>
          <DistortionLadder levels={levels} />
          <div className="mechanism-note"><span>MECHANISM</span><p>Private incentives shape reports. Manager attention buys audits of subordinate claims.</p></div>
        </aside>

        <section className="comparison-section">
          <div className="comparison-heading"><h2 className="section-title">Same seeds. Different organization.</h2><p>Executive optimism by intervention cell. Each value averages paired seed outcomes.</p></div>
          <div className="bar-chart">
            {treatmentMeans.map((item) => (
              <div className="bar-item" key={item.name}>
                <span className="bar-label">{item.name.replaceAll("_", " ")}</span>
                <div className="bar-track"><div className="bar-fill" style={{ width: `${Math.max(4, Math.abs(item.value) / maxTreatment * 100)}%` }} /></div>
                <span className="bar-value">{item.value >= 0 ? "+" : ""}{(item.value * 100).toFixed(1)}</span>
                <small className="bar-regret">REGRET {item.regret.toFixed(1)}</small>
              </div>
            ))}
          </div>
        </section>
      </div>

      <div className="inspector-grid">
        <div className="inspector-pane">
          {selectedRunState.evidenceStatus === "error" && <p className="inline-error" role="alert">{selectedRunState.evidenceError}</p>}
          <EvidenceInspector nodes={filteredEvidence ?? selectedRunState.evidence} departments={departments} depths={depths} department={departmentFilter} depth={depthFilter} onDepartment={(value) => { setDepartmentFilter(value); setFilteredEvidence(null); }} onDepth={(value) => { setDepthFilter(value); setFilteredEvidence(null); }} />
        </div>
        <div className="inspector-pane inspector-pane--wide">
          {selectedRunState.decisionsStatus === "error" && <p className="inline-error" role="alert">{selectedRunState.decisionsError}</p>}
          <DecisionInspector nodes={selectedRunState.decisions} departments={departments} policy={detail.manifest.policy} />
        </div>
      </div>

      <footer className="app-footer">
        <span>{experiment.analysis.unit_of_analysis.toUpperCase()} IS THE UNIT OF ANALYSIS</span>
        <span>{comparison?.n_pairs ?? experiment.analysis.design_diagnostics.complete_pairs} PAIRED REPLICATES</span>
        <span>MODEL OUTPUT NEVER SETS WORLD STATE</span>
      </footer>
    </section>
  );
}

function RunSelect({ id, experiment, selectedRun, onChooseRun }: { id: string; experiment: Experiment; selectedRun: string; onChooseRun: (runId: string) => void }) {
  return (
    <label className="field-label run-select-label" htmlFor={id}>Treatment and seed
      <select className="md3-select" id={id} value={selectedRun} onChange={(event) => onChooseRun(event.target.value)}>
        {experiment.runs.map((run) => <option key={run.run_id} value={run.run_id}>{run.treatment} / seed {run.seed}</option>)}
      </select>
    </label>
  );
}

function RunStat({ label, value }: { label: string; value: string }) {
  return <div className="stat-item"><span className="stat-label">{label}</span><span className="stat-value">{value}</span></div>;
}

function RunState({ icon, title, body, action, code }: { icon: React.ReactNode; title: string; body: string; action: () => void; code?: string }) {
  return (
    <section className="run-view view-enter">
      <div className="state-page">{icon}<div><h1>{title}</h1><p>{body}</p></div><button type="button" className="md3-button md3-button--filled" onClick={action}>RETRY</button>{code && <code>{code}</code>}</div>
    </section>
  );
}

function RunLoading() {
  return <section className="run-view view-enter"><div className="loading-page"><div className="skeleton" /><div className="skeleton" /></div></section>;
}
