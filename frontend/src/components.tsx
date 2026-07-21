import { useState } from "react";
import type { DecisionNode, EvidenceNode } from "./types";

const POLICY_LABEL: Record<string, string> = {
  fixture: "FIXTURE",
  record: "RECORD",
  locked: "LOCKED",
};

export type InterventionValues = {
  incentive: number;
  attention: number;
  seedCount: number;
  policy: "fixture" | "record" | "locked";
  model: string;
};

const POLICY_HELP: Record<InterventionValues["policy"], string> = {
  fixture: "deterministic — no network call",
  record: "live LLM — decisions captured to cache",
  locked: "replay only — no network, fails on unseen context",
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
  const [policy, setPolicy] = useState<InterventionValues["policy"]>("fixture");
  const livePolicy = policy !== "fixture";
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
          policy,
          model: String(values.get("model") ?? ""),
        });
      }}
    >
      <label className="field-label">INCENTIVE PRESSURE<input className="md3-input" name="incentive" type="number" min="0" max="1" step="0.05" defaultValue={initialIncentive} /></label>
      <label className="field-label">MANAGER ATTENTION<input className="md3-input" name="attention" type="number" min="0" max="20" step="1" defaultValue={initialAttention} /></label>
      <label className="field-label">PAIRED SEEDS<input className="md3-input" name="seedCount" type="number" min="7" max="64" step="1" defaultValue="12" /></label>
      <label className="field-label">
        POLICY
        <select className="md3-select" name="policy" value={policy} onChange={(event) => setPolicy(event.target.value as InterventionValues["policy"])}>
          <option value="fixture">fixture — deterministic</option>
          <option value="record">record — live LLM</option>
          <option value="locked">locked — replay only</option>
        </select>
      </label>
      <label className="field-label">
        MODEL (optional)
        <input className="md3-input" name="model" type="text" maxLength={100} disabled={!livePolicy} placeholder={livePolicy ? "uses GROUNDLINE_MODEL from .env if blank" : "only used for record / locked"} />
      </label>
      <span className="field-help" aria-live="polite">{POLICY_HELP[policy]}</span>
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

export function DecisionInspector({
  nodes,
  departments,
  policy,
}: {
  nodes: DecisionNode[];
  departments: string[];
  policy: string;
}) {
  const [department, setDepartment] = useState("");
  const filtered = department ? nodes.filter((node) => node.report.department === department) : nodes;
  return (
    <section className="decisions-section" aria-labelledby="decisions-title">
      <div className="decisions-header">
        <div>
          <h2 id="decisions-title">AGENT DECISIONS</h2>
          <p>
            What each agent actually reported — policy <code>{POLICY_LABEL[policy] ?? policy.toUpperCase()}</code>.
            {policy === "record" ? " Live LLM reasoning captured to cache." : policy === "locked" ? " Replayed from a locked cache." : " Deterministic fixture reasoning."}
          </p>
        </div>
        <div className="decisions-filters">
          <label className="field-label">DEPARTMENT
            <select className="md3-select" aria-label="Decisions department" value={department} onChange={(event) => setDepartment(event.target.value)}>
              <option value="">ALL</option>
              {departments.map((value) => <option key={value} value={value}>{value.toUpperCase()}</option>)}
            </select>
          </label>
        </div>
      </div>
      <ol className="decisions-list">
        {filtered.slice(-24).reverse().map((node) => (
          <li className="decision-card" key={node.sequence} data-policy={node.policy} data-escalate={node.report.escalate ? "true" : "false"}>
            <div className="decision-meta">
              <span className="sequence-num">#{String(node.sequence).padStart(3, "0")}</span>
              <strong>{node.report.agent_id}</strong>
              <span>{node.report.department.toUpperCase()} · DEPTH {node.report.depth} · T{node.report.tick}</span>
              <span className={`policy-badge policy-${node.policy}`}>{POLICY_LABEL[node.policy] ?? node.policy.toUpperCase()}</span>
              {node.report.escalate && <span className="escalate-flag">ESCALATE</span>}
            </div>
            <div className="decision-stats">
              <span>CONFIDENCE {(node.report.confidence * 100).toFixed(0)}%</span>
              <span>RESOURCE REQUEST {node.report.resource_request}</span>
              <span>HEALTH P{node.report.health.progress.toFixed(2)} · Q{node.report.health.quality.toFixed(2)} · S{node.report.health.schedule.toFixed(2)} · R{node.report.health.reliability.toFixed(2)}</span>
            </div>
            <blockquote className="decision-explanation">{node.report.explanation || "no explanation recorded"}</blockquote>
          </li>
        ))}
      </ol>
      {!filtered.length && <p className="empty-evidence">No agent decisions for this filter.</p>}
    </section>
  );
}