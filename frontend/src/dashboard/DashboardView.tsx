import { ArrowRight, ChartLine, CheckCircle, Flask, Pulse, Warning } from "@phosphor-icons/react";

type BackendStatus = "checking" | "online" | "offline";

type DashboardViewProps = {
  backendStatus: BackendStatus;
  experimentCount: number;
  latestRun: string;
  onNavigate?: (view: "experiments" | "run") => void;
};

export function DashboardView({ backendStatus, experimentCount, latestRun, onNavigate }: DashboardViewProps) {
  const ready = backendStatus === "online" && experimentCount > 0;

  return (
    <section className="dashboard-view view-enter" aria-labelledby="dashboard-title">
      <header className="dashboard-hero">
        <div className="dashboard-hero-copy">
          <span className="dashboard-signal" aria-hidden="true"><Pulse size={22} weight="duotone" /></span>
          <div>
            <p className="dashboard-hero-eyebrow">Causal evaluation environment</p>
            <h1 id="dashboard-title" className="dashboard-hero-title">The Groundline</h1>
            <p className="dashboard-hero-sub">Computed company truth, compared with what leadership believes.</p>
          </div>
        </div>
        <div className={`dashboard-live-state ${backendStatus}`}>
          {backendStatus === "online" ? <CheckCircle size={18} weight="fill" /> : backendStatus === "offline" ? <Warning size={18} weight="fill" /> : <Pulse size={18} />}
          <span>{backendStatus === "online" ? "System ready" : backendStatus === "offline" ? "Backend offline" : "Checking system"}</span>
        </div>
      </header>

      {backendStatus === "checking" && (
        <div className="dashboard-overview dashboard-overview--loading">
          <div className="skeleton dashboard-skeleton dashboard-skeleton--wide" />
          <div className="skeleton dashboard-skeleton" />
        </div>
      )}

      {backendStatus === "offline" && (
        <div className="dashboard-state-panel">
          <Warning size={30} weight="duotone" />
          <div><h2>Backend not running</h2><p>Start the API server to begin exploring experiments.</p></div>
          <code className="dashboard-code">uv run python tui.py</code>
          <p className="dashboard-state-note">Then select option 1: Launch Web UI</p>
        </div>
      )}

      {backendStatus === "online" && experimentCount === 0 && (
        <div className="dashboard-state-panel">
          <Flask size={30} weight="duotone" />
          <div><h2>No experiments yet</h2><p>Run the offline demo or create an experiment from the command line.</p></div>
          <code className="dashboard-code">uv run groundline experiment --config configs/demo.yaml</code>
        </div>
      )}

      {ready && (
        <>
          <div className="dashboard-overview">
            <button type="button" className="dashboard-latest" aria-label="Open latest experiment" onClick={() => onNavigate?.("run")}>
              <span className="dashboard-panel-kicker">Latest experiment</span>
              <strong>{latestRun || "None"}</strong>
              <span className="dashboard-panel-action">Open analysis <ArrowRight size={17} /></span>
            </button>
            <div className="dashboard-count-panel">
              <span className="dashboard-panel-kicker">Experiments</span>
              <strong>{experimentCount}</strong>
              <span>In the artifacts ledger</span>
            </div>
          </div>

          <nav className="dashboard-quicknav" aria-label="Dashboard quick actions">
            <button type="button" className="dashboard-quicknav-item" onClick={() => onNavigate?.("experiments")}>
              <span className="dashboard-quicknav-icon"><Flask size={19} weight="duotone" /></span>
              <span className="dashboard-quicknav-text"><span className="dashboard-quicknav-label">Browse experiments</span><span className="dashboard-quicknav-hint">Every run on disk</span></span>
              <ArrowRight size={17} />
            </button>
            <button type="button" className="dashboard-quicknav-item" aria-label="Inspect latest run" onClick={() => onNavigate?.("run")}>
              <span className="dashboard-quicknav-icon"><ChartLine size={19} weight="duotone" /></span>
              <span className="dashboard-quicknav-text"><span className="dashboard-quicknav-label">Open Run Viewer</span><span className="dashboard-quicknav-hint">Truth vs belief, per run</span></span>
              <ArrowRight size={17} />
            </button>
          </nav>
        </>
      )}
    </section>
  );
}
