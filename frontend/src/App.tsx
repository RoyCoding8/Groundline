import { useCallback, useEffect, useState } from "react";
import { List } from "@phosphor-icons/react";
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
import { type InterventionValues } from "./components";
import { ExperimentBuilder } from "./builder/ExperimentBuilder";
import { DashboardView } from "./dashboard/DashboardView";
import { NavigationRail, type BackendStatus, type ThemeChoice, type View } from "./shell/NavigationRail";
import { ExperimentsView } from "./views/ExperimentsView";
import { RunView, type ExperimentStatus, type SelectedRunState } from "./views/RunView";
import type { Experiment, JobStatus } from "./types";

const isAbortError = (caught: unknown) => caught instanceof DOMException && caught.name === "AbortError";

function usePrefersDarkScheme(): boolean {
  const [dark, setDark] = useState(() =>
    typeof window !== "undefined" && window.matchMedia
      ? window.matchMedia("(prefers-color-scheme: dark)").matches
      : false,
  );
  useEffect(() => {
    if (typeof window === "undefined" || !window.matchMedia) return;
    const media = window.matchMedia("(prefers-color-scheme: dark)");
    const onChange = () => setDark(media.matches);
    media.addEventListener("change", onChange);
    return () => media.removeEventListener("change", onChange);
  }, []);
  return dark;
}

export function App() {
  const [view, setView] = useState<View>("dashboard");
  const [backendStatus, setBackendStatus] = useState<BackendStatus>("checking");
  const [theme, setTheme] = useState<ThemeChoice>(() => {
    const stored = localStorage.getItem("theme");
    return stored === "light" || stored === "dark" ? stored : "";
  });
  const [sidebarCollapsed, setSidebarCollapsed] = useState(true);
  const [mobileOpen, setMobileOpen] = useState(false);
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
  const prefersDark = usePrefersDarkScheme();

  useEffect(() => {
    if (theme === "") {
      delete document.documentElement.dataset.theme;
      localStorage.removeItem("theme");
    } else {
      document.documentElement.dataset.theme = theme;
      localStorage.setItem("theme", theme);
    }
  }, [theme]);

  const resolvedDark = theme === "" ? prefersDark : theme === "dark";
  const cycleTheme = useCallback(() => {
    setTheme((previous) => (previous === "light" ? "dark" : previous === "dark" ? "" : "light"));
  }, []);

  useEffect(() => {
    const controller = new AbortController();
    setBackendStatus("checking");
    checkBackend(controller.signal).then((ok) => {
      if (!controller.signal.aborted) setBackendStatus(ok ? "online" : "offline");
    });
    return () => controller.abort();
  }, []);

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
        status: "loaded",
        runId,
        detail,
        timeline,
        evidence: chain.nodes,
        evidenceKey: "",
        evidenceStatus: "loaded",
        evidenceError: "",
        decisions: decisions.nodes,
        decisionsStatus: "loaded",
        decisionsError: "",
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

  const handleNavigate = useCallback((nextView: View) => {
    const changeView = () => setView(nextView);
    const transition = (document as Document & { startViewTransition?: (callback: () => void) => void }).startViewTransition;
    if (transition && !window.matchMedia("(prefers-reduced-motion: reduce)").matches) transition.call(document, changeView);
    else changeView();
    if (nextView === "run" && experiment?.runs.length && !selectedRun) setSelectedRun(experiment.runs[0].run_id);
  }, [experiment, selectedRun]);

  const handleSelectExperiment = useCallback((name: string) => {
    const controller = new AbortController();
    loadExperiment(name, controller.signal).then((data) => {
      if (controller.signal.aborted) return;
      setExperiment(data);
      setExperimentStatus("loaded");
      if (data.runs.length) setSelectedRun(data.runs[0].run_id);
      setView("run");
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
      const launched = await launchExperiment({
        name,
        seeds,
        scenario: sourceDetail.request.scenario,
        organization: sourceDetail.request.organization,
        treatments: {
          low_incentive_low_attention: { incentive_pressure: lowIncentive, attention_budget: lowAttention },
          low_incentive_high_attention: { incentive_pressure: lowIncentive, attention_budget: highAttention },
          high_incentive_low_attention: { incentive_pressure: highIncentive, attention_budget: lowAttention },
          high_incentive_high_attention: { incentive_pressure: highIncentive, attention_budget: highAttention },
        },
        analysis: {
          seed: selectedSeed,
          missingness: "fail_if_missing",
          contrasts: [{
            id: "pressure-at-low-attention",
            baseline: "low_incentive_low_attention",
            intervention: "high_incentive_low_attention",
            outcome: "upward_amplification",
            direction: "increase",
            family: "primary",
            status: "confirmatory",
          }],
        },
        max_concurrency: 4,
      }, values.policy, values.model);
      setJob(launched);
      await waitForJob(launched.job_id, setJob);
      const completed = await loadExperiment(name);
      setExperiment(completed);
      chooseRun(completed.runs.find((run) => run.treatment === "high_incentive_high_attention")?.run_id ?? completed.runs[0]?.run_id ?? "");
    } catch (caught) {
      setLaunchError(caught instanceof Error ? caught.message : "intervention failed");
    }
  }, [selectedRunState, experiment, selectedRun, chooseRun]);

  const experimentCount = experiment && experimentStatus === "loaded" ? 1 : 0;
  const latestRunName = experiment?.name ?? "";

  return (
    <div className={`app-layout ${sidebarCollapsed ? "rail-collapsed" : ""}`}>
      <a className="skip-link" href="#main-workspace">Skip to content</a>
      <NavigationRail
        view={view}
        onNavigate={handleNavigate}
        backendStatus={backendStatus}
        theme={theme}
        resolvedDark={resolvedDark}
        onToggleTheme={cycleTheme}
        collapsed={sidebarCollapsed}
        onToggleCollapse={() => setSidebarCollapsed((previous) => !previous)}
        mobileOpen={mobileOpen}
        onCloseMobile={() => setMobileOpen(false)}
      />

      <main id="main-workspace" className="main-content">
        <button type="button" className="mobile-menu-btn" onClick={() => setMobileOpen(true)} aria-label="Open navigation menu"><List size={20} weight="bold" /></button>
        {view === "dashboard" && <DashboardView backendStatus={backendStatus} experimentCount={experimentCount} latestRun={latestRunName} onNavigate={handleNavigate} />}
        {view === "experiments" && <ExperimentsView onSelectExperiment={handleSelectExperiment} />}
        {view === "builder" && (
          <ExperimentBuilder
            onLaunched={(completed, selectRunId) => { setExperiment(completed); chooseRun(selectRunId); setView("run"); }}
            setLaunchError={setLaunchError}
            setJob={setJob}
            setLaunchPolicy={setLaunchPolicy}
            initialPolicy={launchPolicy}
          />
        )}
        {view === "run" && (
          <RunView
            experiment={experiment}
            experimentStatus={experimentStatus}
            experimentError={experimentError}
            onReloadExperiment={() => setExperimentReload((value) => value + 1)}
            selectedRun={selectedRun}
            selectedRunState={selectedRunState}
            onChooseRun={chooseRun}
            onSetRunReload={setRunReload}
            job={job}
            launchError={launchError}
            launchPolicy={launchPolicy}
            onRunIntervention={runIntervention}
          />
        )}
      </main>
    </div>
  );
}
