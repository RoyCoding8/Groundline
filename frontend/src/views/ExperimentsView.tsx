import { useEffect, useState } from "react";
import { ArrowUpRight, Flask, Warning } from "@phosphor-icons/react";

const isAbortError = (caught: unknown) => caught instanceof DOMException && caught.name === "AbortError";

export function ExperimentsView({ onSelectExperiment }: { onSelectExperiment: (name: string) => void }) {
  const [experiments, setExperiments] = useState<Array<{ name: string }>>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");

  useEffect(() => {
    const controller = new AbortController();
    setLoading(true);
    fetch("/api/experiments", { signal: controller.signal })
      .then((response) => {
        if (!response.ok) throw new Error(`${response.status} ${response.statusText}`);
        return response.json();
      })
      .then((data: Array<{ name: string }>) => {
        if (!controller.signal.aborted) {
          setExperiments(data);
          setLoading(false);
        }
      })
      .catch((caught: unknown) => {
        if (!controller.signal.aborted && !isAbortError(caught)) {
          setError(caught instanceof Error ? caught.message : "Failed to load experiments");
          setLoading(false);
        }
      });
    return () => controller.abort();
  }, []);

  return (
    <section className="experiments-view view-enter" aria-labelledby="experiments-title">
      <header className="workspace-header experiments-header">
        <div>
          <p className="section-label">Artifact index</p>
          <h1 id="experiments-title">Experiments</h1>
          <p>Open a causal evaluation and inspect its paired runs.</p>
        </div>
        {!loading && !error && <span className="workspace-count">{experiments.length.toString().padStart(2, "0")}</span>}
      </header>

      {loading && (
        <div className="experiment-list experiment-list--loading" aria-label="Loading experiments">
          {Array.from({ length: 6 }).map((_, index) => <div key={index} className="skeleton experiment-skeleton" />)}
        </div>
      )}

      {error && (
        <div className="workspace-state workspace-state--error">
          <Warning size={24} weight="duotone" />
          <div><h2>Failed to load</h2><p>{error}</p></div>
        </div>
      )}

      {!loading && !error && experiments.length === 0 && (
        <div className="workspace-state">
          <Flask size={24} weight="duotone" />
          <div><h2>No experiments found</h2><p>Create an experiment from the command line to get started.</p></div>
        </div>
      )}

      {!loading && !error && experiments.length > 0 && (
        <div className="experiment-list">
          {experiments.map((experiment, index) => (
            <button
              key={experiment.name}
              type="button"
              className="experiment-row"
              onClick={() => onSelectExperiment(experiment.name)}
              style={{ "--item-index": index } as React.CSSProperties}
            >
              <span className="experiment-index">{String(index + 1).padStart(2, "0")}</span>
              <span className="experiment-name">{experiment.name}</span>
              <span className="experiment-meta">View runs <ArrowUpRight size={16} /></span>
            </button>
          ))}
        </div>
      )}
    </section>
  );
}
