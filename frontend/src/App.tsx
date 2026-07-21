import { useCallback, useEffect, useState } from "react";
import {
  House,
  Flask,
  ChartLine,
  List,
  Sun,
  Moon,
  Warning,
  CheckCircle,
} from "@phosphor-icons/react";
import {
  checkBackend,
  launchExperiment,
  loadDecisions,
  loadEvidence,
  loadExperiment,
  loadFirstExperiment,
  loadRun,
  waitForJob,
} from "./api";
import { DecisionInspector, EvidenceInspector, InterventionControls, type InterventionValues } from "./components";
import { DistortionLadder, SignalChart } from "./charts";
import type { DecisionNode, EvidenceNode, Experiment, JobStatus, RunDetail, TimelineEvent } from "./types";

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

const mean = (values: number[]) => values.reduce((sum, value) => sum + value, 0) / Math.max(values.length, 1);
const isAbortError = (caught: unknown) => caught instanceof DOMException && caught.name === "AbortError";

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

type View = "dashboard" | "experiments" | "run";
type BackendStatus = "checking" | "online" | "offline";
type ExperimentStatus = "loading" | "loaded" | "empty" | "error";

type SelectedRunState =
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

// ---------------------------------------------------------------------------
// Sidebar
// ---------------------------------------------------------------------------

function Sidebar({
  view,
  onNavigate,
  backendStatus,
  theme,
  onToggleTheme,
  collapsed,
  onToggleCollapse,
  mobileOpen,
  onCloseMobile,
}: {
  view: View;
  onNavigate: (v: View) => void;
  backendStatus: BackendStatus;
  theme: string;
  onToggleTheme: () => void;
  collapsed: boolean;
  onToggleCollapse: () => void;
  mobileOpen: boolean;
  onCloseMobile: () => void;
}) {
  const statusLabel = backendStatus === "online" ? "Connected" : backendStatus === "offline" ? "Offline" : "Checking...";

  return (
    <>
      <div
        className={`sidebar-backdrop ${mobileOpen ? "visible" : ""}`}
        onClick={onCloseMobile}
        onKeyDown={(e) => { if (e.key === "Escape") onCloseMobile(); }}
        role="button"
        tabIndex={-1}
        aria-label="Close sidebar"
      />
      <nav className={`sidebar ${collapsed ? "collapsed" : ""} ${mobileOpen ? "mobile-open" : ""}`} aria-label="Main navigation">
        <div className="sidebar-header">
          <span className="sidebar-title">Groundline</span>
          <button
            type="button"
            className="sidebar-toggle"
            onClick={onToggleCollapse}
            aria-label={collapsed ? "Expand sidebar" : "Collapse sidebar"}
          >
            <List size={18} weight="bold" />
          </button>
        </div>

        <div className="sidebar-nav">
          <SidebarItem
            icon={<House size={20} weight={view === "dashboard" ? "fill" : "regular"} />}
            label="Dashboard"
            active={view === "dashboard"}
            onClick={() => { onNavigate("dashboard"); onCloseMobile(); }}
          />
          <SidebarItem
            icon={<Flask size={20} weight={view === "experiments" ? "fill" : "regular"} />}
            label="Experiments"
            active={view === "experiments"}
            onClick={() => { onNavigate("experiments"); onCloseMobile(); }}
          />
          <SidebarItem
            icon={<ChartLine size={20} weight={view === "run" ? "fill" : "regular"} />}
            label="Run Viewer"
            active={view === "run"}
            onClick={() => { onNavigate("run"); onCloseMobile(); }}
          />
        </div>

        <div className="sidebar-footer">
          <div className="sidebar-status">
            <span className={`status-dot ${backendStatus}`} />
            <span>{statusLabel}</span>
          </div>
          <button
            type="button"
            className="sidebar-nav-item"
            onClick={onToggleTheme}
            aria-label={`Switch to ${theme === "light" ? "dark" : "light"} mode`}
          >
            <span className="nav-icon">
              {theme === "light" ? <Sun size={20} /> : <Moon size={20} />}
            </span>
            <span className="nav-label">{theme === "light" ? "Light" : "Dark"}</span>
          </button>
        </div>
      </nav>
    </>
  );
}

function SidebarItem({
  icon,
  label,
  active,
  onClick,
}: {
  icon: React.ReactNode;
  label: string;
  active: boolean;
  onClick: () => void;
}) {
  return (
    <button
      type="button"
      className={`sidebar-nav-item ${active ? "active" : ""}`}
      onClick={onClick}
      aria-current={active ? "page" : undefined}
    >
      <span className="nav-icon">{icon}</span>
      <span className="nav-label">{label}</span>
    </button>
  );
}

// ---------------------------------------------------------------------------
// Dashboard View
// ---------------------------------------------------------------------------

function DashboardView({
  backendStatus,
  experimentCount,
  latestRun,
}: {
  backendStatus: BackendStatus;
  experimentCount: number;
  latestRun: string;
}) {
  if (backendStatus === "checking") {
    return (
      <div className="dashboard-view">
        <div className="dashboard-skeleton">
          <div className="skeleton skeleton-card" />
          <div className="skeleton skeleton-card" />
          <div className="skeleton skeleton-card" />
        </div>
      </div>
    );
  }

  if (backendStatus === "offline") {
    return (
      <div className="dashboard-view">
        <h1>Groundline</h1>
        <p className="dashboard-subtitle">Causal evaluation environment for hierarchical agent organizations.</p>
        <div className="dashboard-empty">
          <div className="empty-icon"><Warning size={48} /></div>
          <h2>Backend not running</h2>
          <p>Start the API server to begin exploring experiments.</p>
          <code>uv run python tui.py</code>
          <p style={{ marginTop: 12, fontSize: 13 }}>Then select option 1: Launch Web UI</p>
        </div>
      </div>
    );
  }

  if (experimentCount === 0) {
    return (
      <div className="dashboard-view">
        <h1>Groundline</h1>
        <p className="dashboard-subtitle">Causal evaluation environment for hierarchical agent organizations.</p>
        <div className="dashboard-empty">
          <div className="empty-icon"><Flask size={48} /></div>
          <h2>No experiments yet</h2>
          <p>Run the offline demo or create an experiment from the command line.</p>
          <code>uv run groundline experiment --config configs/demo.yaml</code>
        </div>
      </div>
    );
  }

  return (
    <div className="dashboard-view">
      <h1>Groundline</h1>
      <p className="dashboard-subtitle">Causal evaluation environment for hierarchical agent organizations.</p>
      <div className="dashboard-cards">
        <div className="dashboard-card">
          <p className="card-label">Experiments</p>
          <p className="card-value">{experimentCount}</p>
        </div>
        <div className="dashboard-card">
          <p className="card-label">Latest Experiment</p>
          <p className="card-value" style={{ fontSize: 18 }}>{latestRun || "None"}</p>
        </div>
      </div>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Experiments View
// ---------------------------------------------------------------------------

function ExperimentsView({
  onSelectExperiment,
}: {
  onSelectExperiment: (name: string) => void;
}) {
  const [experiments, setExperiments] = useState<Array<{ name: string }>>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");

  useEffect(() => {
    const controller = new AbortController();
    setLoading(true);
    fetch("/api/experiments", { signal: controller.signal })
      .then((r) => {
        if (!r.ok) throw new Error(`${r.status} ${r.statusText}`);
        return r.json();
      })
      .then((data: Array<{ name: string }>) => {
        if (!controller.signal.aborted) {
          setExperiments(data);
          setLoading(false);
        }
      })
      .catch((err: unknown) => {
        if (!controller.signal.aborted && !isAbortError(err)) {
          setError(err instanceof Error ? err.message : "Failed to load experiments");
          setLoading(false);
        }
      });
    return () => controller.abort();
  }, []);

  if (loading) {
    return (
      <div className="experiments-view">
        <h1>Experiments</h1>
        <div className="experiment-list">
          {Array.from({ length: 3 }).map((_, i) => (
            <div key={i} className="skeleton" style={{ height: 56, borderRadius: "var(--md3-shape-md)" }} />
          ))}
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="experiments-view">
        <h1>Experiments</h1>
        <div className="dashboard-empty">
          <div className="empty-icon"><Warning size={48} /></div>
          <h2>Failed to load</h2>
          <p>{error}</p>
        </div>
      </div>
    );
  }

  if (experiments.length === 0) {
    return (
      <div className="experiments-view">
        <h1>Experiments</h1>
        <div className="dashboard-empty">
          <div className="empty-icon"><Flask size={48} /></div>
          <h2>No experiments found</h2>
          <p>Create an experiment from the command line to get started.</p>
        </div>
      </div>
    );
  }

  return (
    <div className="experiments-view">
      <h1>Experiments</h1>
      <div className="experiment-list">
        {experiments.map((exp) => (
          <button
            key={exp.name}
            type="button"
            className="experiment-row"
            onClick={() => onSelectExperiment(exp.name)}
          >
            <Flask size={20} />
            <span className="experiment-name">{exp.name}</span>
            <span className="experiment-meta">View runs</span>
          </button>
        ))}
      </div>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Run View (the existing run detail, extracted)
// ---------------------------------------------------------------------------

function RunView({
  experiment,
  experimentStatus,
  experimentError,
  onReloadExperiment,
  selectedRun,
  selectedRunState,
  onChooseRun,
  runReload,
  onSetRunReload,
  job,
  launchError,
  launchPolicy,
  onRunIntervention,
  theme,
}: {
  experiment: Experiment | null;
  experimentStatus: ExperimentStatus;
  experimentError: string;
  onReloadExperiment: () => void;
  selectedRun: string;
  selectedRunState: SelectedRunState;
  onChooseRun: (runId: string) => void;
  runReload: number;
  onSetRunReload: (fn: (v: number) => number) => void;
  job: JobStatus | null;
  launchError: string;
  launchPolicy: "fixture" | "record" | "locked";
  onRunIntervention: (values: InterventionValues) => void;
  theme: string;
}) {
  const [departmentFilter, setDepartmentFilter] = useState("");
  const [depthFilter, setDepthFilter] = useState("");

  // Evidence re-fetch on filter change
  const filterKey = JSON.stringify([departmentFilter, depthFilter]);
  const loadedRunId = selectedRunState.status === "loaded" ? selectedRunState.runId : "";
  const loadedEvidenceKey = selectedRunState.status === "loaded" ? selectedRunState.evidenceKey : "";

  useEffect(() => {
    if (!loadedRunId || loadedEvidenceKey === filterKey) return;
    const controller = new AbortController();
    const runId = loadedRunId;
    loadEvidence(runId, {
      department: departmentFilter || undefined,
      depth: depthFilter === "" ? undefined : Number(depthFilter),
    }, controller.signal).then((chain) => {
      if (controller.signal.aborted) return;
      // Update will happen via parent re-render
    }).catch(() => {});
    return () => controller.abort();
  }, [departmentFilter, depthFilter, filterKey, loadedEvidenceKey, loadedRunId]);

  if (experimentStatus === "error") {
    return (
      <div className="run-view">
        <div className="state-page">
          <Warning size={28} />
          <h1>Artifact read failed</h1>
          <p>{experimentError}</p>
          <button type="button" className="md3-button md3-button--filled" onClick={onReloadExperiment}>RETRY</button>
          <code>uv run groundline experiment --config configs/demo.yaml</code>
        </div>
      </div>
    );
  }

  if (experimentStatus === "empty" || (experimentStatus === "loaded" && !experiment?.runs.length)) {
    return (
      <div className="run-view">
        <div className="state-page">
          <Flask size={28} />
          <h1>No runs yet</h1>
          <p>Generate a paired experiment to populate the control room.</p>
          <button type="button" className="md3-button md3-button--filled" onClick={onReloadExperiment}>RETRY</button>
        </div>
      </div>
    );
  }

  if (experimentStatus === "loading" || !experiment) {
    return (
      <div className="run-view">
        <div className="loading-page">
          <div className="skeleton" style={{ height: "2rem", width: "42%" }} />
          <div className="skeleton" style={{ height: "14rem" }} />
        </div>
      </div>
    );
  }

  if (selectedRunState.status === "error") {
    return (
      <div className="run-view">
        <div className="state-page">
          <Warning size={28} />
          <h1>Run read failed</h1>
          <p>{selectedRunState.message}</p>
          <label className="field-label" htmlFor="failed-run">Treatment and seed</label>
          <select className="md3-select" id="failed-run" value={selectedRun} onChange={(event) => onChooseRun(event.target.value)}>
            {experiment.runs.map((run) => <option key={run.run_id} value={run.run_id}>{run.treatment} / seed {run.seed}</option>)}
          </select>
          <button type="button" className="md3-button md3-button--filled" onClick={() => onSetRunReload((v) => v + 1)}>RETRY</button>
        </div>
      </div>
    );
  }

  if (selectedRunState.status !== "loaded" || !selectedRunState || !selectedRunState.detail) {
    return (
      <div className="run-view">
        <div className="loading-page">
          <div className="skeleton" style={{ height: "2rem", width: "42%" }} />
          <p>Loading selected run artifacts.</p>
          <label className="field-label" htmlFor="loading-run">Treatment and seed</label>
          <select className="md3-select" id="loading-run" value={selectedRun} onChange={(event) => onChooseRun(event.target.value)}>
            {experiment.runs.map((run) => <option key={run.run_id} value={run.run_id}>{run.treatment} / seed {run.seed}</option>)}
          </select>
        </div>
      </div>
    );
  }

  const detail = selectedRunState.detail;
  const timeline = selectedRunState.timeline;

  const derived = (() => {
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
    return { executive, truth, belief, levels, finalGap: finalBelief - finalTruth };
  })();

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
    <div className="run-view">
      <main className="app-shell">
        <header className="app-header">
          <div>
            <p className="section-label">FIRMWORLD / CAUSAL RUN</p>
            <h1 className="app-title">The Groundline</h1>
          </div>
          <div>
            <div className="run-badge">
              <CheckCircle size={18} />
              <span>LEDGER VERIFIED</span>
              <code>{detail.manifest.event_hash.slice(0, 12)}</code>
            </div>
          </div>
        </header>

        <section className="control-bar" aria-label="Run controls">
          <label className="field-label" htmlFor="run">Treatment and seed</label>
          <select className="md3-select" id="run" value={selectedRun} onChange={(event) => onChooseRun(event.target.value)}>
            {experiment.runs.map((run) => <option key={run.run_id} value={run.run_id}>{run.treatment} / seed {run.seed}</option>)}
          </select>
          <div className="stat-group">
            <div className="stat-item"><span className="stat-label">INCENTIVE</span><span className="stat-value">{detail.request.treatment.incentive_pressure.toFixed(2)}</span></div>
            <div className="stat-item"><span className="stat-label">MANAGER AUDITS</span><span className="stat-value">{detail.request.treatment.attention_budget}</span></div>
            <div className="stat-item"><span className="stat-label">POLICY</span><span className="stat-value">{detail.manifest.policy}</span></div>
            <div className="stat-item"><span className="stat-label">SEED</span><span className="stat-value">{selected.seed}</span></div>
          </div>
        </section>

        <section className="intervention-section" aria-labelledby="intervention-title">
          <div>
            <h2 id="intervention-title" className="section-title">CAUSAL INTERVENTION</h2>
            <p>Launch a fresh 2x2 matrix while holding every seed fixed across cells.</p>
          </div>
          <div className="intervention-form">
            <InterventionControls
              key={selectedRun}
              initialIncentive={detail.request.treatment.incentive_pressure === 0 ? 0.9 : 0}
              initialAttention={detail.request.treatment.attention_budget === 0 ? 2 : 0}
              busy={job?.status === "queued" || job?.status === "running"}
              onLaunch={(values) => { void onRunIntervention(values); }}
            />
          </div>
          {job && <div className={`job-progress ${job.status}`}><strong>{job.status.toUpperCase()}</strong><span>{job.completed_runs} / {job.total_runs} RUNS FINALIZED</span><span className="job-policy">POLICY: {launchPolicy.toUpperCase()}</span></div>}
          {launchError && <p className="inline-error" role="alert">{launchError}</p>}
        </section>

        <section className="chart-grid">
          <article className="chart-panel">
            <div className="section-header">
              <div>
                <h2 className="section-title">WORLD TRUTH / EXECUTIVE BELIEF</h2>
                <p>Operational health, computed independently from every agent report.</p>
              </div>
              <div className="gap-display">
                <span>FINAL GAP</span>
                <span className="gap-value">{derived.finalGap >= 0 ? "+" : ""}{(derived.finalGap * 100).toFixed(1)}</span>
              </div>
            </div>
            <div className="chart-legend"><span className="truth-key">WORLD TRUTH</span><span className="belief-key">EXECUTIVE BELIEF</span></div>
            <SignalChart truth={derived.truth} belief={derived.belief} />
          </article>

          <aside className="side-panel">
            <h2 className="section-title">DISTORTION CLIMBS</h2>
            <p>Mean optimism bias at the final tick.</p>
            <DistortionLadder levels={derived.levels} />
            <div className="mechanism-note"><span>MECHANISM</span><p>Private incentives shape reports. Manager attention buys audits of subordinate claims.</p></div>
          </aside>
        </section>

        <section className="comparison-section">
          <div>
            <h2 className="section-title">Same seeds. Different organization.</h2>
            <p>Executive optimism by intervention cell. Each value averages paired seed outcomes.</p>
          </div>
          <div className="bar-chart">
            {treatmentMeans.map((item) => (
              <div className="bar-item" key={item.name}>
                <div className="bar-track">
                  <div className="bar-fill" style={{ height: `${Math.max(4, Math.abs(item.value) / maxTreatment * 100)}%` }} />
                </div>
                <span className="bar-value">{item.value >= 0 ? "+" : ""}{(item.value * 100).toFixed(1)}</span>
                <span className="bar-label">
                  {item.name.replaceAll("_", " ")}
                  <small className="bar-regret">INCIDENT REGRET {item.regret.toFixed(1)}</small>
                </span>
              </div>
            ))}
          </div>
        </section>

        {selectedRunState.evidenceStatus === "error" && <p className="inline-error" role="alert">{selectedRunState.evidenceError}</p>}
        <EvidenceInspector
          nodes={selectedRunState.evidence}
          departments={departments}
          depths={depths}
          department={departmentFilter}
          depth={depthFilter}
          onDepartment={setDepartmentFilter}
          onDepth={setDepthFilter}
        />

        {selectedRunState.decisionsStatus === "error" && <p className="inline-error" role="alert">{selectedRunState.decisionsError}</p>}
        <DecisionInspector
          nodes={selectedRunState.decisions}
          departments={departments}
          policy={detail.manifest.policy}
        />

        <footer className="app-footer">
          <span>{experiment.analysis.unit_of_analysis.toUpperCase()} IS THE UNIT OF ANALYSIS</span>
          <span>{comparison?.n_pairs ?? experiment.analysis.design_diagnostics.complete_pairs} PAIRED REPLICATES</span>
          <span>MODEL OUTPUT NEVER SETS WORLD STATE</span>
        </footer>
      </main>
    </div>
  );
}

// ---------------------------------------------------------------------------
// App
// ---------------------------------------------------------------------------

export function App() {
  const [view, setView] = useState<View>("dashboard");
  const [backendStatus, setBackendStatus] = useState<BackendStatus>("checking");
  const [theme, setTheme] = useState(() => localStorage.getItem("theme") ?? "light");
  const [sidebarCollapsed, setSidebarCollapsed] = useState(false);
  const [mobileOpen, setMobileOpen] = useState(false);

  // Experiment state
  const [experiment, setExperiment] = useState<Experiment | null>(null);
  const [experimentStatus, setExperimentStatus] = useState<ExperimentStatus>("loading");
  const [experimentError, setExperimentError] = useState("");
  const [experimentReload, setExperimentReload] = useState(0);
  const [selectedRun, setSelectedRun] = useState("");
  const [selectedRunState, setSelectedRunState] = useState<SelectedRunState>({ status: "idle" });
  const [runReload, setRunReload] = useState(0);
  const [job, setJob] = useState<JobStatus | null>(null);
  const [launchError, setLaunchError] = useState("");
  const [launchPolicy, setLaunchPolicy] = useState<"fixture" | "record" | "locked">("fixture");

  // Theme
  useEffect(() => {
    document.documentElement.dataset.theme = theme;
    localStorage.setItem("theme", theme);
  }, [theme]);

  // Backend health check
  useEffect(() => {
    const controller = new AbortController();
    setBackendStatus("checking");
    checkBackend(controller.signal).then((ok) => {
      if (!controller.signal.aborted) {
        setBackendStatus(ok ? "online" : "offline");
      }
    });
    return () => controller.abort();
  }, []);

  // Load the latest experiment for dashboard summaries and the run viewer.
  useEffect(() => {
    const controller = new AbortController();
    setExperimentStatus("loading");
    setExperimentError("");
    loadFirstExperiment(controller.signal).then((data) => {
      if (controller.signal.aborted) return;
      if (data === null || !data.runs.length) {
        setExperiment(data);
        setSelectedRun("");
        setSelectedRunState({ status: "idle" });
        setExperimentStatus("empty");
        return;
      }
      setExperiment(data);
      setSelectedRun(data.runs[0].run_id);
      setExperimentStatus("loaded");
    }).catch((caught: unknown) => {
      if (controller.signal.aborted || isAbortError(caught)) return;
      setExperimentStatus("error");
      setExperimentError(caught instanceof Error ? caught.message : "experiment load failed");
    });
    return () => controller.abort();
  }, [experimentReload]);

  // Load selected run
  useEffect(() => {
    if (!selectedRun) {
      setSelectedRunState({ status: "idle" });
      return;
    }
    const controller = new AbortController();
    const runId = selectedRun;
    setSelectedRunState({ status: "loading", runId });
    Promise.all([
      loadRun(runId, controller.signal),
      loadEvidence(runId, {}, controller.signal),
      loadDecisions(runId, controller.signal),
    ]).then(([[detail, timeline], chain, decisions]) => {
      if (controller.signal.aborted) return;
      if (detail.manifest.run_id !== runId) {
        setSelectedRunState({ status: "error", runId, message: `run identity mismatch: expected ${runId}` });
        return;
      }
      setSelectedRunState({
        status: "loaded", runId, detail, timeline,
        evidence: chain.nodes, evidenceKey: "", evidenceStatus: "loaded", evidenceError: "",
        decisions: decisions.nodes, decisionsStatus: "loaded", decisionsError: "",
      });
    }).catch((caught: unknown) => {
      if (controller.signal.aborted || isAbortError(caught)) return;
      setSelectedRunState({ status: "error", runId, message: caught instanceof Error ? caught.message : "run load failed" });
    });
    return () => controller.abort();
  }, [runReload, selectedRun]);

  const chooseRun = useCallback((runId: string) => {
    setLaunchError("");
    setSelectedRun(runId);
  }, []);

  const handleNavigate = useCallback((v: View) => {
    setView(v);
    if (v === "run" && experiment && experiment.runs.length && !selectedRun) {
      setSelectedRun(experiment.runs[0].run_id);
    }
  }, [experiment, selectedRun]);

  const handleSelectExperiment = useCallback((name: string) => {
    const controller = new AbortController();
    loadExperiment(name, controller.signal).then((data) => {
      if (!controller.signal.aborted) {
        setExperiment(data);
        setExperimentStatus("loaded");
        if (data.runs.length) {
          setSelectedRun(data.runs[0].run_id);
        }
        setView("run");
      }
    });
  }, []);

  const runIntervention = useCallback(async (values: InterventionValues) => {
    if (selectedRunState.status !== "loaded" || !experiment) return;
    const sourceRunId = selectedRunState.runId;
    const sourceDetail = selectedRunState.detail;
    if (sourceRunId !== selectedRun || sourceDetail.manifest.run_id !== sourceRunId) {
      setLaunchError("selected run data is not ready");
      return;
    }
    setLaunchError("");
    const reference = sourceDetail.request.treatment;
    const lowIncentive = Math.min(reference.incentive_pressure, values.incentive);
    const highIncentive = Math.max(reference.incentive_pressure, values.incentive);
    const lowAttention = Math.min(reference.attention_budget, values.attention);
    const highAttention = Math.max(reference.attention_budget, values.attention);
    const selectedSeed = experiment.runs.find((run) => run.run_id === sourceRunId)?.seed ?? sourceDetail.manifest.seed;
    const name = `ui-${selectedSeed}-i${Math.round(values.incentive * 100)}-a${values.attention}-n${values.seedCount}`;
    const seeds = Array.from({ length: values.seedCount }, (_, index) => selectedSeed + index * 104729);
    try {
      setLaunchPolicy(values.policy);
      const launched = await launchExperiment(
        {
          name, seeds,
          scenario: sourceDetail.request.scenario,
          organization: sourceDetail.request.organization,
          treatments: {
            low_incentive_low_attention: { incentive_pressure: lowIncentive, attention_budget: lowAttention },
            low_incentive_high_attention: { incentive_pressure: lowIncentive, attention_budget: highAttention },
            high_incentive_low_attention: { incentive_pressure: highIncentive, attention_budget: lowAttention },
            high_incentive_high_attention: { incentive_pressure: highIncentive, attention_budget: highAttention },
          },
          analysis: {
            seed: selectedSeed, missingness: "fail_if_missing",
            contrasts: [{
              id: "pressure-at-low-attention", baseline: "low_incentive_low_attention",
              intervention: "high_incentive_low_attention", outcome: "upward_amplification",
              direction: "increase", family: "primary", status: "confirmatory",
            }],
          },
          max_concurrency: 4,
        },
        values.policy,
        values.model,
      );
      setJob(launched);
      await waitForJob(launched.job_id, setJob);
      const completed = await loadExperiment(name);
      setExperiment(completed);
      chooseRun(
        completed.runs.find((run) => run.treatment === "high_incentive_high_attention")?.run_id
          ?? completed.runs[0]?.run_id ?? "",
      );
    } catch (caught) {
      setLaunchError(caught instanceof Error ? caught.message : "intervention failed");
    }
  }, [selectedRunState, experiment, selectedRun, chooseRun]);

  // Compute dashboard summary
  const experimentCount = experiment && experimentStatus === "loaded" ? 1 : 0;
  const latestRunName = experiment?.name ?? "";

  return (
    <div className="app-layout">
      <Sidebar
        view={view}
        onNavigate={handleNavigate}
        backendStatus={backendStatus}
        theme={theme}
        onToggleTheme={() => setTheme((prev) => (prev === "light" ? "dark" : "light"))}
        collapsed={sidebarCollapsed}
        onToggleCollapse={() => setSidebarCollapsed((prev) => !prev)}
        mobileOpen={mobileOpen}
        onCloseMobile={() => setMobileOpen(false)}
      />

      <div className={`main-content ${sidebarCollapsed ? "sidebar-collapsed" : ""}`}>
        {/* Mobile menu button */}
        <button
          type="button"
          className="mobile-menu-btn"
          onClick={() => setMobileOpen(true)}
          aria-label="Open navigation menu"
        >
          <List size={20} weight="bold" />
        </button>

        {view === "dashboard" && (
          <DashboardView
            backendStatus={backendStatus}
            experimentCount={experimentCount}
            latestRun={latestRunName}
          />
        )}

        {view === "experiments" && (
          <ExperimentsView onSelectExperiment={handleSelectExperiment} />
        )}

        {view === "run" && (
          <RunView
            experiment={experiment}
            experimentStatus={experimentStatus}
            experimentError={experimentError}
            onReloadExperiment={() => setExperimentReload((v) => v + 1)}
            selectedRun={selectedRun}
            selectedRunState={selectedRunState}
            onChooseRun={chooseRun}
            runReload={runReload}
            onSetRunReload={setRunReload}
            job={job}
            launchError={launchError}
            launchPolicy={launchPolicy}
            onRunIntervention={runIntervention}
            theme={theme}
          />
        )}
      </div>
    </div>
  );
}
