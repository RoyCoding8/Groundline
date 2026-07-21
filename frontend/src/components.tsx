import type { EvidenceNode } from "./types";

export type InterventionValues = {
  incentive: number;
  attention: number;
  seedCount: number;
};

export function InterventionControls({
  initialIncentive,
  initialAttention,
  busy,
  onLaunch,
}: {
  initialIncentive: number;
  initialAttention: number;
  busy: boolean;
  onLaunch: (values: InterventionValues) => void;
}) {
  return (
    <form
      className="intervention-form"
      onSubmit={(event) => {
        event.preventDefault();
        const values = new FormData(event.currentTarget);
        onLaunch({
          incentive: Number(values.get("incentive")),
          attention: Number(values.get("attention")),
          seedCount: Number(values.get("seedCount")),
        });
      }}
    >
      <label className="field-label">INCENTIVE PRESSURE<input className="md3-input" name="incentive" type="number" min="0" max="1" step="0.05" defaultValue={initialIncentive} /></label>
      <label className="field-label">MANAGER ATTENTION<input className="md3-input" name="attention" type="number" min="0" max="20" step="1" defaultValue={initialAttention} /></label>
      <label className="field-label">PAIRED SEEDS<input className="md3-input" name="seedCount" type="number" min="7" max="64" step="1" defaultValue="12" /></label>
      <button className="md3-button md3-button--filled" type="submit" disabled={busy}>{busy ? "RUNNING MATRIX" : "RUN INTERVENTION"}</button>
    </form>
  );
}

export function EvidenceInspector({
  nodes,
  departments,
  depths,
  department,
  depth,
  onDepartment,
  onDepth,
}: {
  nodes: EvidenceNode[];
  departments: string[];
  depths: number[];
  department: string;
  depth: string;
  onDepartment: (value: string) => void;
  onDepth: (value: string) => void;
}) {
  return (
    <section className="evidence-section" aria-labelledby="evidence-title">
      <div className="evidence-header">
        <div><h2 id="evidence-title">EVIDENCE CHAIN</h2><p>Every claim traces backward through explicit ledger parents.</p></div>
        <div className="evidence-filters">
          <label className="field-label">DEPARTMENT<select className="md3-select" aria-label="Evidence department" value={department} onChange={(event) => onDepartment(event.target.value)}><option value="">ALL</option>{departments.map((value) => <option key={value} value={value}>{value.toUpperCase()}</option>)}</select></label>
          <label className="field-label">DEPTH<select className="md3-select" aria-label="Evidence depth" value={depth} onChange={(event) => onDepth(event.target.value)}><option value="">ALL</option>{depths.map((value) => <option key={value} value={value}>{value}</option>)}</select></label>
        </div>
      </div>
      <ol className="evidence-list">
        {nodes.slice(-24).reverse().map((node) => (
          <li className="evidence-row" key={node.sequence}>
            <span className="sequence-num">#{String(node.sequence).padStart(3, "0")}</span>
            <strong>{node.kind.replaceAll("_", " ")}</strong>
            <span>{node.actor_id ?? "world engine"} / T{node.tick}</span>
            <code>{node.causes.length ? `FROM ${node.causes.map((value) => `#${value}`).join(" · ")}` : "ROOT FACT"}</code>
          </li>
        ))}
      </ol>
      {!nodes.length && <p className="empty-evidence">No ledger nodes match these filters.</p>}
    </section>
  );
}