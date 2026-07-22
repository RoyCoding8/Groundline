import { useCallback, useEffect, useMemo, useState } from "react";
import { Warning } from "@phosphor-icons/react";
import { launchExperiment, loadExperiment, waitForJob } from "../api";
import type { LaunchPolicy } from "../api";
import type { JobStatus, Experiment } from "../types";
import {
  ANALYSIS_OUTCOMES,
  SENSITIVITY_KINDS,
  type AnalysisOutcome,
  type AnalysisStatus,
  type AgentRole,
  type BuilderAgent,
  type BuilderExperimentRequest,
  type ContrastSpecification,
  type ExpectedDirection,
  type MissingnessPolicy,
  type OutcomeSpecification,
  type ReportingSpan,
  type ScenarioConfig,
  type SensitivitySpecification,
  type TreatmentConfig,
  type WorkItemConfig,
} from "../types";

// ---------------------------------------------------------------------------
// Defaults + presets
// ---------------------------------------------------------------------------

const ID_PATTERN = /^[A-Za-z0-9][A-Za-z0-9_-]*$/;

const DEFAULT_OUTCOME_SPEC: OutcomeSpecification = {
  adverse_health_threshold: 0.95,
  release_health_threshold: 0.95,
  escalation_severity_threshold: 1,
  escalation_sensitivity_thresholds: [],
};

const DEFAULT_HARM_MAXIMA = {
  release_delay: 100,
  escaped_defects: 10,
  incident: 100,
  remediation: 100,
  scope_loss: 1,
};

function emptyAgent(): BuilderAgent {
  return { id: "", manager_id: null, role: "contributor", department: "", skills: {}, traits: {}, utility_weights: {} };
}

function emptyWorkItem(department: string): WorkItemConfig {
  return { id: "", department, business_value: 1, effort: 1, deadline_tick: 1, dependencies: [] };
}

/** The exact 2x2 request the Run Viewer's runIntervention builds, generalised
 *  so the builder can reproduce it without a source run. Uses the demo
 *  scenario + org from configs/demo.yaml as the seed world. */
function presetDemo2x2(): BuilderExperimentRequest {
  return {
    name: "builder-2x2",
    seeds: [7, 13, 29, 47, 71, 101, 127],
    scenario: {
      max_ticks: 6,
      shock_tick: 3,
      shock_item_id: "api",
      shock_severity: 7,
      work_items: [
        { id: "launch-spec", department: "Product", business_value: 0.8, effort: 5, deadline_tick: 3, dependencies: [] },
        { id: "api", department: "Engineering", business_value: 1, effort: 8, deadline_tick: 5, dependencies: ["launch-spec"] },
        { id: "release-gate", department: "QA", business_value: 1, effort: 5, deadline_tick: 6, dependencies: ["api"] },
      ],
      harm_maxima: { ...DEFAULT_HARM_MAXIMA },
    },
    organization: { agents: [
      { id: "ceo", manager_id: null, role: "executive", department: "Executive", skills: {}, traits: { honesty: 0.55, blame_sensitivity: 0.65 }, utility_weights: { delivery: 0.9, reputation: 0.8, quality: 0.6 } },
      { id: "engineering-director", manager_id: "ceo", role: "manager", department: "Engineering", skills: {}, traits: { honesty: 0.52, blame_sensitivity: 0.82 }, utility_weights: { delivery: 0.95, reputation: 0.9, quality: 0.55 } },
      { id: "qa-director", manager_id: "ceo", role: "manager", department: "QA", skills: {}, traits: { honesty: 0.78, blame_sensitivity: 0.55 }, utility_weights: { delivery: 0.45, reputation: 0.65, quality: 1.0 } },
      { id: "backend-engineer", manager_id: "engineering-director", role: "contributor", department: "Engineering", skills: { backend: 0.95 }, traits: { honesty: 0.6, blame_sensitivity: 0.7 }, utility_weights: { delivery: 0.9, reputation: 0.72, quality: 0.6 } },
      { id: "qa-analyst", manager_id: "qa-director", role: "contributor", department: "QA", skills: { defect_detection: 0.95, testing: 0.9 }, traits: { honesty: 0.86, blame_sensitivity: 0.45 }, utility_weights: { delivery: 0.4, reputation: 0.65, quality: 1.0 } },
    ] },
    treatments: {
      low_incentive_low_attention: { incentive_pressure: 0, attention_budget: 0 },
      low_incentive_high_attention: { incentive_pressure: 0, attention_budget: 3 },
      high_incentive_low_attention: { incentive_pressure: 0.9, attention_budget: 0 },
      high_incentive_high_attention: { incentive_pressure: 0.9, attention_budget: 3 },
    },
    outcome_specification: { ...DEFAULT_OUTCOME_SPEC, escalation_sensitivity_thresholds: [0.5, 2.0] },
    analysis: {
      seed: 20260716,
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
      sensitivities: [],
    },
    max_concurrency: 4,
  };
}

function presetFlatTeam(): BuilderExperimentRequest {
  const base = presetDemo2x2();
  return {
    ...base,
    name: "builder-flat-team",
    organization: { agents: [
      { id: "exec", manager_id: null, role: "executive", department: "Executive", skills: {}, traits: { honesty: 0.6, blame_sensitivity: 0.6 }, utility_weights: { delivery: 0.8, reputation: 0.7, quality: 0.6 } },
      { id: "lead", manager_id: "exec", role: "manager", department: "Engineering", skills: {}, traits: { honesty: 0.65, blame_sensitivity: 0.6 }, utility_weights: { delivery: 0.85, reputation: 0.8, quality: 0.65 } },
      { id: "ic1", manager_id: "lead", role: "contributor", department: "Engineering", skills: { backend: 0.9 }, traits: { honesty: 0.7, blame_sensitivity: 0.5 }, utility_weights: { delivery: 0.8, reputation: 0.7, quality: 0.6 } },
      { id: "ic2", manager_id: "lead", role: "contributor", department: "Engineering", skills: { frontend: 0.9 }, traits: { honesty: 0.5, blame_sensitivity: 0.75 }, utility_weights: { delivery: 0.9, reputation: 0.85, quality: 0.5 } },
    ] },
  };
}

function presetDeepHierarchy(): BuilderExperimentRequest {
  const base = presetDemo2x2();
  return {
    ...base,
    name: "builder-deep-hierarchy",
    scenario: { ...base.scenario, max_ticks: 8, shock_tick: 4 },
    organization: { agents: [
      { id: "ceo", manager_id: null, role: "executive", department: "Executive", skills: {}, traits: { honesty: 0.55, blame_sensitivity: 0.65 }, utility_weights: { delivery: 0.9, reputation: 0.8, quality: 0.6 } },
      { id: "vp-eng", manager_id: "ceo", role: "manager", department: "Engineering", skills: {}, traits: { honesty: 0.52, blame_sensitivity: 0.82 }, utility_weights: { delivery: 0.95, reputation: 0.9, quality: 0.55 } },
      { id: "vp-qa", manager_id: "ceo", role: "manager", department: "QA", skills: {}, traits: { honesty: 0.78, blame_sensitivity: 0.55 }, utility_weights: { delivery: 0.45, reputation: 0.65, quality: 1.0 } },
      { id: "eng-director", manager_id: "vp-eng", role: "manager", department: "Engineering", skills: {}, traits: { honesty: 0.6, blame_sensitivity: 0.7 }, utility_weights: { delivery: 0.9, reputation: 0.75, quality: 0.6 } },
      { id: "backend", manager_id: "eng-director", role: "contributor", department: "Engineering", skills: { backend: 0.95 }, traits: { honesty: 0.6, blame_sensitivity: 0.7 }, utility_weights: { delivery: 0.9, reputation: 0.72, quality: 0.6 } },
      { id: "frontend", manager_id: "eng-director", role: "contributor", department: "Engineering", skills: { frontend: 0.94 }, traits: { honesty: 0.5, blame_sensitivity: 0.8 }, utility_weights: { delivery: 0.92, reputation: 0.88, quality: 0.52 } },
      { id: "qa-lead", manager_id: "vp-qa", role: "manager", department: "QA", skills: {}, traits: { honesty: 0.7, blame_sensitivity: 0.6 }, utility_weights: { delivery: 0.5, reputation: 0.7, quality: 0.9 } },
      { id: "tester", manager_id: "qa-lead", role: "contributor", department: "QA", skills: { defect_detection: 0.95, testing: 0.9 }, traits: { honesty: 0.86, blame_sensitivity: 0.45 }, utility_weights: { delivery: 0.4, reputation: 0.65, quality: 1.0 } },
    ] },
  };
}

const PRESETS: Array<{ id: string; label: string; build: () => BuilderExperimentRequest }> = [
  { id: "demo-2x2", label: "2×2 Pressure × Attention (demo)", build: presetDemo2x2 },
  { id: "flat-team", label: "Flat team", build: presetFlatTeam },
  { id: "deep-hierarchy", label: "Deep hierarchy", build: presetDeepHierarchy },
];

const DRAFT_KEY = "groundline:builder-draft";

// ---------------------------------------------------------------------------
// Client-side validation (mirrors backend Pydantic; server is source of truth)
// ---------------------------------------------------------------------------

function validate(req: BuilderExperimentRequest): string[] {
  const errors: string[] = [];
  if (!ID_PATTERN.test(req.name)) errors.push("experiment name: must match ^[A-Za-z0-9][A-Za-z0-9_-]{0,63}$");
  if (!req.seeds.length) errors.push("seeds: at least one paired seed is required");
  const agentIds = req.organization.agents.map((a) => a.id);
  if (new Set(agentIds).size !== agentIds.length) errors.push("organization: agent ids must be unique");
  const execCount = req.organization.agents.filter((a) => a.role === "executive").length;
  if (execCount !== 1) errors.push("organization: exactly one executive is required");
  for (const a of req.organization.agents) {
    if (!ID_PATTERN.test(a.id)) errors.push(`agent id "${a.id}": must match ^[A-Za-z0-9][A-Za-z0-9_-]*$`);
    if (!a.department) errors.push(`agent "${a.id}": department is required`);
    if (a.manager_id && !agentIds.includes(a.manager_id)) errors.push(`agent "${a.id}": manager_id "${a.manager_id}" does not exist`);
  }
  for (const w of req.scenario.work_items) {
    if (!ID_PATTERN.test(w.id)) errors.push(`work item id "${w.id}": must match ^[A-Za-z0-9][A-Za-z0-9_-]*$`);
    if (!(w.business_value > 0)) errors.push(`work item "${w.id}": business_value must be > 0`);
    if (!(w.effort >= 1)) errors.push(`work item "${w.id}": effort must be >= 1`);
    if (w.deadline_tick < 1) errors.push(`work item "${w.id}": deadline_tick must be >= 1`);
    for (const dep of w.dependencies) if (!req.scenario.work_items.some((x) => x.id === dep)) errors.push(`work item "${w.id}": dependency "${dep}" does not exist`);
  }
  if (req.scenario.work_items.length && !req.scenario.work_items.some((w) => w.id === req.scenario.shock_item_id)) {
    errors.push("scenario: shock_item_id must reference an existing work item");
  }
  const treatmentNames = Object.keys(req.treatments);
  if (!treatmentNames.length) errors.push("treatments: at least one treatment cell is required");
  for (const [name, t] of Object.entries(req.treatments)) {
    if (!ID_PATTERN.test(name)) errors.push(`treatment "${name}": name must match ^[A-Za-z0-9][A-Za-z0-9_-]*$`);
    if (t.incentive_pressure < 0 || t.incentive_pressure > 1) errors.push(`treatment "${name}": incentive_pressure must be in [0,1]`);
    if (t.attention_budget < 0) errors.push(`treatment "${name}": attention_budget must be >= 0`);
  }
  if (!req.analysis.contrasts.length) errors.push("analysis: at least one contrast is required");
  for (const c of req.analysis.contrasts) {
    if (!ID_PATTERN.test(c.id)) errors.push(`contrast "${c.id}": id must match ^[A-Za-z0-9][A-Za-z0-9_-]*$`);
    if (!treatmentNames.includes(c.baseline)) errors.push(`contrast "${c.id}": baseline "${c.baseline}" is not a treatment`);
    if (!treatmentNames.includes(c.intervention)) errors.push(`contrast "${c.id}": intervention "${c.intervention}" is not a treatment`);
    if (!ID_PATTERN.test(c.family)) errors.push(`contrast "${c.id}": family must match ^[A-Za-z0-9][A-Za-z0-9_-]*$`);
  }
  if (req.max_concurrency < 1 || req.max_concurrency > 16) errors.push("max_concurrency: must be in [1,16]");
  return errors;
}

// ---------------------------------------------------------------------------
// Small reusable inputs
// ---------------------------------------------------------------------------

function NumberField({ label, value, onChange, min, max, step }: {
  label: string; value: number; onChange: (v: number) => void; min?: number; max?: number; step?: number;
}) {
  return (
    <label className="field-label">{label}
      <input className="md3-input" type="number" value={Number.isFinite(value) ? value : ""} min={min} max={max} step={step}
        onChange={(e) => onChange(e.target.value === "" ? Number.NaN : Number(e.target.value))} />
    </label>
  );
}

function TextField({ label, value, onChange, placeholder }: {
  label: string; value: string; onChange: (v: string) => void; placeholder?: string;
}) {
  return (
    <label className="field-label">{label}
      <input className="md3-input" type="text" value={value} placeholder={placeholder} onChange={(e) => onChange(e.target.value)} />
    </label>
  );
}

function RemoveButton({ onClick }: { onClick: () => void }) {
  return <button type="button" className="builder-remove" aria-label="Remove row" onClick={onClick}>×</button>;
}

function MapField({ label, value, onChange }: { label: string; value: Record<string, number>; onChange: (v: Record<string, number>) => void }) {
  const [draft, setDraft] = useState("");
  const entries = Object.entries(value);
  return (
    <div className="builder-map">
      <span className="field-label">{label}</span>
      {entries.map(([k, v]) => (
        <div className="builder-map-row" key={k}>
          <code>{k}</code>
          <input className="md3-input" type="number" value={v} step="0.01"
            onChange={(e) => onChange({ ...value, [k]: Number(e.target.value) })} />
          <RemoveButton onClick={() => { const { [k]: _drop, ...rest } = value; onChange(rest); }} />
        </div>
      ))}
      <div className="builder-map-add">
        <input className="md3-input" type="text" placeholder="key" value={draft} onChange={(e) => setDraft(e.target.value)} />
        <button type="button" className="md3-button md3-button--tonal" disabled={!draft.trim()}
          onClick={() => { if (draft.trim() && !(draft.trim() in value)) { onChange({ ...value, [draft.trim()]: 0.5 }); setDraft(""); } }}>ADD</button>
      </div>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Section: Organization
// ---------------------------------------------------------------------------

function OrganizationSection({ agents, onChange }: { agents: BuilderAgent[]; onChange: (a: BuilderAgent[]) => void }) {
  const update = (i: number, patch: Partial<BuilderAgent>) => onChange(agents.map((a, idx) => (idx === i ? { ...a, ...patch } : a)));
  const departments = [...new Set(agents.map((a) => a.department).filter(Boolean))].sort();
  return (
    <section className="builder-section">
      <header><h2>ORGANIZATION</h2><p>Hierarchy of persistent employees. Exactly one executive; each agent optionally reports to a manager.</p></header>
      <div className="builder-rows">
        {agents.map((a, i) => (
          <div className="builder-row builder-row-grid" key={i}>
            <TextField label="ID" value={a.id} onChange={(v) => update(i, { id: v })} />
            <label className="field-label">MANAGER
              <select className="md3-select" value={a.manager_id ?? ""} onChange={(e) => update(i, { manager_id: e.target.value || null })}>
                <option value="">(none / executive)</option>
                {agents.filter((x) => x.id && x.id !== a.id).map((x) => <option key={x.id} value={x.id}>{x.id}</option>)}
              </select>
            </label>
            <label className="field-label">ROLE
              <select className="md3-select" value={a.role} onChange={(e) => update(i, { role: e.target.value as AgentRole })}>
                <option value="contributor">contributor</option>
                <option value="manager">manager</option>
                <option value="executive">executive</option>
              </select>
            </label>
            <label className="field-label">DEPARTMENT
              <input className="md3-input" type="text" value={a.department} list="builder-depts" onChange={(e) => update(i, { department: e.target.value })} />
              <datalist id="builder-depts">{departments.map((d) => <option key={d} value={d} />)}</datalist>
            </label>
            <RemoveButton onClick={() => onChange(agents.filter((_, idx) => idx !== i))} />
            <details className="builder-submap">
              <summary>skills / traits / utility</summary>
              <MapField label="SKILLS" value={a.skills} onChange={(v) => update(i, { skills: v })} />
              <MapField label="TRAITS" value={a.traits} onChange={(v) => update(i, { traits: v })} />
              <MapField label="UTILITY WEIGHTS" value={a.utility_weights} onChange={(v) => update(i, { utility_weights: v })} />
            </details>
          </div>
        ))}
      </div>
      <button type="button" className="md3-button md3-button--tonal" onClick={() => onChange([...agents, emptyAgent()])}>ADD AGENT</button>
    </section>
  );
}

// ---------------------------------------------------------------------------
// Section: Scenario
// ---------------------------------------------------------------------------

function ScenarioSection({ scenario, departments, onChange }: {
  scenario: ScenarioConfig; departments: string[]; onChange: (s: ScenarioConfig) => void;
}) {
  const update = (patch: Partial<ScenarioConfig>) => onChange({ ...scenario, ...patch });
  const updateItem = (i: number, patch: Partial<WorkItemConfig>) =>
    update({ work_items: scenario.work_items.map((w, idx) => (idx === i ? { ...w, ...patch } : w)) });
  const itemIds = scenario.work_items.map((w) => w.id);
  return (
    <section className="builder-section">
      <header><h2>SCENARIO</h2><p>The deterministic world: work items, the shock, and harm maxima.</p></header>
      <div className="builder-inline">
        <NumberField label="MAX TICKS" value={scenario.max_ticks} min={1} step={1} onChange={(v) => update({ max_ticks: v })} />
        <NumberField label="SHOCK TICK" value={scenario.shock_tick} min={1} step={1} onChange={(v) => update({ shock_tick: v })} />
        <label className="field-label">SHOCK ITEM
          <select className="md3-select" value={scenario.shock_item_id} onChange={(e) => update({ shock_item_id: e.target.value })}>
            {itemIds.map((id) => <option key={id} value={id}>{id}</option>)}
          </select>
        </label>
        <NumberField label="SHOCK SEVERITY" value={scenario.shock_severity} min={0} max={10} step={0.1} onChange={(v) => update({ shock_severity: v })} />
      </div>
      <div className="builder-rows">
        {scenario.work_items.map((w, i) => (
          <div className="builder-row builder-row-grid" key={i}>
            <TextField label="ID" value={w.id} onChange={(v) => updateItem(i, { id: v })} />
            <label className="field-label">DEPARTMENT
              <input className="md3-input" type="text" value={w.department} list="builder-depts" onChange={(e) => updateItem(i, { department: e.target.value })} />
            </label>
            <NumberField label="BUSINESS VALUE" value={w.business_value} min={0.01} step={0.05} onChange={(v) => updateItem(i, { business_value: v })} />
            <NumberField label="EFFORT" value={w.effort} min={1} step={0.5} onChange={(v) => updateItem(i, { effort: v })} />
            <NumberField label="DEADLINE TICK" value={w.deadline_tick} min={1} step={1} onChange={(v) => updateItem(i, { deadline_tick: v })} />
            <TextField label="DEPENDENCIES (comma)" value={w.dependencies.join(", ")} onChange={(v) => updateItem(i, { dependencies: v.split(",").map((s) => s.trim()).filter(Boolean) })} />
            <RemoveButton onClick={() => update({ work_items: scenario.work_items.filter((_, idx) => idx !== i) })} />
          </div>
        ))}
      </div>
      <button type="button" className="md3-button md3-button--tonal"
        onClick={() => onChange({ ...scenario, work_items: [...scenario.work_items, emptyWorkItem(departments[0] ?? "Engineering")] })}>ADD WORK ITEM</button>
      <details className="builder-submap">
        <summary>harm maxima (advanced)</summary>
        <div className="builder-inline">
          <NumberField label="RELEASE DELAY" value={scenario.harm_maxima.release_delay} min={0.01} step={1} onChange={(v) => update({ harm_maxima: { ...scenario.harm_maxima, release_delay: v } })} />
          <NumberField label="ESCAPED DEFECTS" value={scenario.harm_maxima.escaped_defects} min={0.01} step={1} onChange={(v) => update({ harm_maxima: { ...scenario.harm_maxima, escaped_defects: v } })} />
          <NumberField label="INCIDENT" value={scenario.harm_maxima.incident} min={0.01} step={1} onChange={(v) => update({ harm_maxima: { ...scenario.harm_maxima, incident: v } })} />
          <NumberField label="REMEDIATION" value={scenario.harm_maxima.remediation} min={0.01} step={1} onChange={(v) => update({ harm_maxima: { ...scenario.harm_maxima, remediation: v } })} />
          <NumberField label="SCOPE LOSS" value={scenario.harm_maxima.scope_loss} min={0.01} step={0.1} onChange={(v) => update({ harm_maxima: { ...scenario.harm_maxima, scope_loss: v } })} />
        </div>
      </details>
    </section>
  );
}

// ---------------------------------------------------------------------------
// Section: Treatments
// ---------------------------------------------------------------------------

function TreatmentsSection({ treatments, onChange }: { treatments: Record<string, TreatmentConfig>; onChange: (t: Record<string, TreatmentConfig>) => void }) {
  const [newName, setNewName] = useState("");
  const entries = Object.entries(treatments);
  const update = (name: string, patch: Partial<TreatmentConfig>) => onChange({ ...treatments, [name]: { ...treatments[name], ...patch } });
  const rename = (old: string, next: string) => {
    if (!next || next in treatments) return;
    const { [old]: _drop, ...rest } = treatments;
    onChange({ ...rest, [next]: treatments[old] });
  };
  return (
    <section className="builder-section">
      <header><h2>TREATMENTS</h2><p>Named intervention cells. At least one is required; contrasts reference them by name.</p></header>
      <div className="builder-rows">
        {entries.map(([name, t]) => (
          <div className="builder-row builder-row-grid" key={name}>
            <TextField label="NAME" value={name} onChange={(v) => rename(name, v)} />
            <NumberField label="INCENTIVE PRESSURE" value={t.incentive_pressure} min={0} max={1} step={0.05} onChange={(v) => update(name, { incentive_pressure: v })} />
            <NumberField label="ATTENTION BUDGET" value={t.attention_budget} min={0} step={1} onChange={(v) => update(name, { attention_budget: v })} />
            <label className="field-label">REPORTING SPAN (adv)
              <select className="md3-select" value={t.reporting_span ?? ""} onChange={(e) => update(name, { reporting_span: (e.target.value || undefined) as ReportingSpan | undefined })}>
                <option value="">(default)</option>
                <option value="narrow">narrow</option>
                <option value="wide">wide</option>
              </select>
            </label>
            <RemoveButton onClick={() => { const { [name]: _drop, ...rest } = treatments; onChange(rest); }} />
          </div>
        ))}
      </div>
      <div className="builder-map-add">
        <input className="md3-input" type="text" placeholder="new treatment name" value={newName} onChange={(e) => setNewName(e.target.value)} />
        <button type="button" className="md3-button md3-button--tonal" disabled={!ID_PATTERN.test(newName) || newName in treatments}
          onClick={() => { onChange({ ...treatments, [newName]: { incentive_pressure: 0, attention_budget: 0 } }); setNewName(""); }}>ADD TREATMENT</button>
      </div>
    </section>
  );
}

// ---------------------------------------------------------------------------
// Section: Analysis
// ---------------------------------------------------------------------------

function AnalysisSection({ req, onChange }: { req: BuilderExperimentRequest; onChange: (r: BuilderExperimentRequest) => void }) {
  const analysis = req.analysis;
  const update = (patch: Partial<typeof analysis>) => onChange({ ...req, analysis: { ...analysis, ...patch } });
  const treatmentNames = Object.keys(req.treatments);
  const updateContrast = (i: number, patch: Partial<ContrastSpecification>) =>
    update({ contrasts: analysis.contrasts.map((c, idx) => (idx === i ? { ...c, ...patch } : c)) });
  const updateSensitivity = (i: number, patch: Partial<SensitivitySpecification>) =>
    update({ sensitivities: analysis.sensitivities.map((s, idx) => (idx === i ? { ...s, ...patch } : s)) });
  return (
    <section className="builder-section">
      <header><h2>ANALYSIS</h2><p>Paired contrasts across treatments. The seed is the analysis RNG; missingness governs incomplete pairs.</p></header>
      <div className="builder-inline">
        <NumberField label="ANALYSIS SEED" value={analysis.seed} step={1} onChange={(v) => update({ seed: v })} />
        <label className="field-label">MISSINGNESS
          <select className="md3-select" value={analysis.missingness} onChange={(e) => update({ missingness: e.target.value as MissingnessPolicy })}>
            <option value="complete_case">complete_case</option>
            <option value="fail_if_missing">fail_if_missing</option>
          </select>
        </label>
        <TextField label="SEEDS (comma)" value={req.seeds.join(", ")} onChange={(v) => onChange({ ...req, seeds: v.split(",").map((s) => Number(s.trim())).filter((n) => Number.isFinite(n)) })} />
        <NumberField label="MAX CONCURRENCY" value={req.max_concurrency} min={1} max={16} step={1} onChange={(v) => onChange({ ...req, max_concurrency: v })} />
      </div>
      <h3>CONTRASTS</h3>
      <div className="builder-rows">
        {analysis.contrasts.map((c, i) => (
          <div className="builder-row builder-row-grid" key={i}>
            <TextField label="ID" value={c.id} onChange={(v) => updateContrast(i, { id: v })} />
            <label className="field-label">BASELINE
              <select className="md3-select" value={c.baseline} onChange={(e) => updateContrast(i, { baseline: e.target.value })}>
                {treatmentNames.map((n) => <option key={n} value={n}>{n}</option>)}
              </select>
            </label>
            <label className="field-label">INTERVENTION
              <select className="md3-select" value={c.intervention} onChange={(e) => updateContrast(i, { intervention: e.target.value })}>
                {treatmentNames.map((n) => <option key={n} value={n}>{n}</option>)}
              </select>
            </label>
            <label className="field-label">OUTCOME
              <select className="md3-select" value={c.outcome} onChange={(e) => updateContrast(i, { outcome: e.target.value as AnalysisOutcome })}>
                {ANALYSIS_OUTCOMES.map((o) => <option key={o} value={o}>{o}</option>)}
              </select>
            </label>
            <label className="field-label">DIRECTION
              <select className="md3-select" value={c.direction} onChange={(e) => updateContrast(i, { direction: e.target.value as ExpectedDirection })}>
                <option value="increase">increase</option>
                <option value="decrease">decrease</option>
                <option value="two_sided">two_sided</option>
              </select>
            </label>
            <TextField label="FAMILY" value={c.family} onChange={(v) => updateContrast(i, { family: v })} />
            <label className="field-label">STATUS
              <select className="md3-select" value={c.status} onChange={(e) => updateContrast(i, { status: e.target.value as AnalysisStatus })}>
                <option value="confirmatory">confirmatory</option>
                <option value="exploratory">exploratory</option>
              </select>
            </label>
            <RemoveButton onClick={() => update({ contrasts: analysis.contrasts.filter((_, idx) => idx !== i) })} />
          </div>
        ))}
      </div>
      <button type="button" className="md3-button md3-button--tonal"
        onClick={() => update({ contrasts: [...analysis.contrasts, { id: "", baseline: treatmentNames[0] ?? "", intervention: treatmentNames[1] ?? treatmentNames[0] ?? "", outcome: "upward_amplification", direction: "increase", family: "primary", status: "confirmatory" }] })}>ADD CONTRAST</button>
      <details className="builder-submap">
        <summary>sensitivities (advanced)</summary>
        <div className="builder-rows">
          {analysis.sensitivities.map((s, i) => (
            <div className="builder-row builder-row-grid" key={i}>
              <TextField label="ID" value={s.id} onChange={(v) => updateSensitivity(i, { id: v })} />
              <label className="field-label">CONTRAST
                <select className="md3-select" value={s.contrast_id} onChange={(e) => updateSensitivity(i, { contrast_id: e.target.value })}>
                  {analysis.contrasts.map((c) => <option key={c.id} value={c.id}>{c.id}</option>)}
                </select>
              </label>
              <label className="field-label">KIND
                <select className="md3-select" value={s.kind} onChange={(e) => updateSensitivity(i, { kind: e.target.value })}>
                  {SENSITIVITY_KINDS.map((k) => <option key={k} value={k}>{k}</option>)}
                </select>
              </label>
              <NumberField label="THRESHOLD" value={s.threshold ?? 0} min={0} step={0.1} onChange={(v) => updateSensitivity(i, { threshold: v })} />
              <RemoveButton onClick={() => update({ sensitivities: analysis.sensitivities.filter((_, idx) => idx !== i) })} />
            </div>
          ))}
        </div>
        <button type="button" className="md3-button md3-button--tonal"
          onClick={() => update({ sensitivities: [...analysis.sensitivities, { id: "", contrast_id: analysis.contrasts[0]?.id ?? "", kind: "adverse_vs_all_ticks", missingness: null, threshold: null }] })}>ADD SENSITIVITY</button>
      </details>
      <details className="builder-submap">
        <summary>outcome specification (advanced)</summary>
        <div className="builder-inline">
          <NumberField label="ADVERSE HEALTH THRESHOLD" value={req.outcome_specification.adverse_health_threshold} min={0} max={1} step={0.01} onChange={(v) => onChange({ ...req, outcome_specification: { ...req.outcome_specification, adverse_health_threshold: v } })} />
          <NumberField label="RELEASE HEALTH THRESHOLD" value={req.outcome_specification.release_health_threshold} min={0} max={1} step={0.01} onChange={(v) => onChange({ ...req, outcome_specification: { ...req.outcome_specification, release_health_threshold: v } })} />
          <NumberField label="ESCALATION SEVERITY THRESHOLD" value={req.outcome_specification.escalation_severity_threshold} min={0} step={0.1} onChange={(v) => onChange({ ...req, outcome_specification: { ...req.outcome_specification, escalation_severity_threshold: v } })} />
          <TextField label="ESCALATION SENSITIVITY (comma)" value={req.outcome_specification.escalation_sensitivity_thresholds.join(", ")} onChange={(v) => onChange({ ...req, outcome_specification: { ...req.outcome_specification, escalation_sensitivity_thresholds: v.split(",").map((s) => Number(s.trim())).filter((n) => Number.isFinite(n)) } })} />
        </div>
      </details>
    </section>
  );
}

// ---------------------------------------------------------------------------
// Main builder
// ---------------------------------------------------------------------------

export function ExperimentBuilder({
  onLaunched,
  setLaunchError,
  setJob,
  setLaunchPolicy,
  initialPolicy,
}: {
  onLaunched: (experiment: Experiment, selectRunId: string) => void;
  setLaunchError: (message: string) => void;
  setJob: (job: JobStatus | null) => void;
  setLaunchPolicy: (policy: LaunchPolicy) => void;
  initialPolicy: LaunchPolicy;
}) {
  const [req, setReq] = useState<BuilderExperimentRequest>(() => {
    try {
      const saved = localStorage.getItem(DRAFT_KEY);
      if (saved) return JSON.parse(saved) as BuilderExperimentRequest;
    } catch { /* ignore corrupt draft */ }
    return presetDemo2x2();
  });
  const [policy, setPolicy] = useState<LaunchPolicy>(initialPolicy);
  const [model, setModel] = useState("");
  const [submitting, setSubmitting] = useState(false);
  const [activePreset, setActivePreset] = useState("demo-2x2");

  useEffect(() => {
    try { localStorage.setItem(DRAFT_KEY, JSON.stringify(req)); } catch { /* quota / private mode */ }
  }, [req]);

  const departments = useMemo(() => [...new Set(req.organization.agents.map((a) => a.department).filter(Boolean))].sort(), [req.organization.agents]);
  const errors = useMemo(() => validate(req), [req]);
  const livePolicy = policy !== "fixture";

  const loadPreset = useCallback((id: string) => {
    const preset = PRESETS.find((p) => p.id === id);
    if (!preset) return;
    setReq(preset.build());
    setActivePreset(id);
  }, []);

  const submit = useCallback(async () => {
    setLaunchError("");
    if (errors.length) {
      setLaunchError(errors.join("; "));
      return;
    }
    setSubmitting(true);
    setLaunchPolicy(policy);
    try {
      const launched = await launchExperiment(req, policy, model);
      setJob(launched);
      await waitForJob(launched.job_id, setJob);
      const completed = await loadExperiment(req.name);
      const selectRunId = completed.runs[0]?.run_id ?? "";
      onLaunched(completed, selectRunId);
    } catch (caught) {
      setLaunchError(caught instanceof Error ? caught.message : "launch failed");
    } finally {
      setSubmitting(false);
    }
  }, [errors, model, onLaunched, policy, req, setJob, setLaunchError, setLaunchPolicy]);

  return (
    <div className="builder-view">
      <main className="app-shell">
        <header className="app-header">
          <div>
            <p className="section-label">FIRMWORLD / EXPERIMENT BUILDER</p>
            <h1 className="app-title">Build a custom run</h1>
          </div>
          <div className="builder-presets">
            <label className="field-label">PRESET
              <select className="md3-select" value={activePreset} onChange={(e) => loadPreset(e.target.value)}>
                {PRESETS.map((p) => <option key={p.id} value={p.id}>{p.label}</option>)}
              </select>
            </label>
          </div>
        </header>

        <div className="builder-workspace">
          <div className="builder-canvas">
            <div className="builder-name-field"><TextField label="EXPERIMENT NAME" value={req.name} onChange={(v) => setReq({ ...req, name: v })} /></div>
            <OrganizationSection agents={req.organization.agents} onChange={(agents) => setReq({ ...req, organization: { agents } })} />
            <ScenarioSection scenario={req.scenario} departments={departments} onChange={(scenario) => setReq({ ...req, scenario })} />
            <TreatmentsSection treatments={req.treatments} onChange={(treatments) => setReq({ ...req, treatments })} />
            <AnalysisSection req={req} onChange={setReq} />
          </div>

          <aside className="builder-launch-panel">
            <section className="builder-section builder-launch">
              <header><h2>LAUNCH</h2><p>The world is deterministic; agent output never sets world state.</p></header>
              <div className="builder-launch-fields">
                <label className="field-label">POLICY
                  <select className="md3-select" value={policy} onChange={(e) => setPolicy(e.target.value as LaunchPolicy)}>
                    <option value="fixture">fixture - deterministic</option>
                    <option value="record">record - live LLM</option>
                    <option value="locked">locked - replay only</option>
                  </select>
                </label>
                <label className="field-label">MODEL (optional)
                  <input className="md3-input" type="text" maxLength={100} disabled={!livePolicy} placeholder={livePolicy ? "uses GROUNDLINE_MODEL from .env if blank" : "only used for record / locked"} value={model} onChange={(e) => setModel(e.target.value)} />
                </label>
              </div>
              <div className="builder-launch-summary">
                <span><strong>{req.organization.agents.length}</strong> agents</span>
                <span><strong>{Object.keys(req.treatments).length}</strong> treatments</span>
                <span><strong>{req.seeds.length}</strong> seeds</span>
              </div>
              {errors.length > 0 && (
                <p className="inline-error" role="alert"><Warning size={16} /> {errors[0]}{errors.length > 1 ? ` (and ${errors.length - 1} more)` : ""}</p>
              )}
              <button type="button" className="md3-button md3-button--filled" disabled={submitting || errors.length > 0} onClick={() => { void submit(); }}>
                {submitting ? "RUNNING" : "LAUNCH EXPERIMENT"}
              </button>
            </section>
          </aside>
        </div>
      </main>
    </div>
  );
}
