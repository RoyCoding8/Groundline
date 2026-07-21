import { expect, test } from "@playwright/test";

const experiment = {
  name: "sample",
  request: null,
  failures: [],
  analysis: {
    baseline: "control",
    unit_of_analysis: "seed",
    design_diagnostics: { complete_pairs: 12 },
    comparisons: { intervention_minus_control: { n_pairs: 12 } },
  },
  runs: [
    { seed: 7, treatment: "control", run_id: "run-control", run_directory: "", executive_optimism_bias: 0.05, incident_regret: 0, incentive_pressure: 0, attention_budget: 2 },
    { seed: 7, treatment: "intervention", run_id: "run-intervention", run_directory: "", executive_optimism_bias: 0.30, incident_regret: 2, incentive_pressure: 0.9, attention_budget: 0 },
  ],
};

const detail = (pressure: number) => ({
  manifest: { run_id: pressure ? "run-intervention" : "run-control", seed: 7, event_hash: "1234567890abcdef", policy: "fixture" },
  request: {
    scenario: { max_ticks: 1, shock_tick: 1, shock_item_id: "api", shock_severity: 0.6, work_items: [] },
    treatment: { incentive_pressure: pressure, attention_budget: pressure ? 0 : 2 },
    organization: { agents: [{ id: "worker", role: "contributor", department: "Engineering" }, { id: "exec", role: "executive", department: "Executive" }] },
  },
  metrics: { distortion: [{ agent_id: "worker", department: "Engineering", depth: 1, tick: 1, optimism_bias: 0.1, absolute_error: 0.1 }, { agent_id: "exec", department: "Executive", depth: 0, tick: 1, optimism_bias: pressure ? 0.3 : 0.05, absolute_error: pressure ? 0.3 : 0.05 }] },
});

const timeline = (belief: number) => [
  { sequence: 0, kind: "truth_snapshot", tick: 1, actor_id: null, causes: [], payload: { health_score: 0.4 } },
  { sequence: 1, kind: "report", tick: 1, actor_id: "exec", causes: [0], payload: { health_score: belief } },
];

test("switches treatments, filters evidence, and launches a fresh intervention", async ({ page }) => {
  await page.route("**/api/**", async (route) => {
    const url = new URL(route.request().url());
    if (url.pathname === "/api/health") return route.fulfill({ json: { status: "ok" } });
    if (url.pathname === "/api/experiments" && route.request().method() === "GET") return route.fulfill({ json: [{ name: "sample" }] });
    if (url.pathname === "/api/experiments" && route.request().method() === "POST") return route.fulfill({ status: 202, json: { job_id: "job-1", experiment: "new", status: "queued", completed_runs: 0, failed_runs: 0, total_runs: 60, error: null } });
    if (url.pathname === "/api/experiments/sample") return route.fulfill({ json: experiment });
    if (url.pathname === "/api/experiments/ui-7-i0-a2-n12") return route.fulfill({ json: experiment });
    if (url.pathname === "/api/jobs/job-1") return route.fulfill({ json: { job_id: "job-1", experiment: "new", status: "completed", completed_runs: 60, failed_runs: 0, total_runs: 60, error: null } });
    if (url.pathname.endsWith("/evidence")) return route.fulfill({ json: { nodes: [{ sequence: 1, kind: "report", tick: 1, actor_id: "exec", department: "Executive", depth: 0, causes: [0], evidence_refs: [], event: timeline(0.45)[1] }] } });
    if (url.pathname.endsWith("/decisions")) return route.fulfill({ json: { run_id: "run-control", nodes: [{ sequence: 0, agent_id: "worker", tick: 1, policy: "fixture", context_hash: "abc", report: { agent_id: "worker", department: "Engineering", depth: 1, tick: 1, scope: ["ops"], health: { progress: 0.5, quality: 0.5, schedule: 0.5, reliability: 0.5 }, confidence: 0.7, escalate: false, resource_request: 1, explanation: "everything looks on track from the floor" }, actions: [], provider_metadata: {} }] } });
    if (url.pathname.endsWith("/timeline")) return route.fulfill({ json: timeline(url.pathname.includes("intervention") ? 0.7 : 0.45) });
    if (url.pathname.includes("run-intervention")) return route.fulfill({ json: detail(0.9) });
    if (url.pathname.includes("run-control")) return route.fulfill({ json: detail(0) });
    return route.fulfill({ status: 404, json: { detail: "unhandled test route" } });
  });

  await page.goto("/");
  await page.getByRole("button", { name: "Run Viewer" }).click();
  await expect(page.getByText("WORLD TRUTH / EXECUTIVE BELIEF")).toBeVisible();
  await page.getByLabel("Treatment and seed").selectOption("run-intervention");
  await expect(page.getByText("0.90")).toBeVisible();
  await page.getByLabel("Evidence department").selectOption("Executive");
  await expect(page.getByText("FROM #0")).toBeVisible();
  await expect(page.getByText("AGENT DECISIONS")).toBeVisible();
  await expect(page.getByText("everything looks on track from the floor")).toBeVisible();
  await page.getByRole("button", { name: "RUN INTERVENTION" }).click();
  await expect(page.getByText("COMPLETED")).toBeVisible();
});
