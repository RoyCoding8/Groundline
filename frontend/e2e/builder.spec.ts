import { expect, test } from "@playwright/test";

const experiment = {
  name: "builder-2x2",
  request: null,
  failures: [],
  analysis: {
    baseline: "low_incentive_low_attention",
    unit_of_analysis: "seed",
    design_diagnostics: { complete_pairs: 28 },
    comparisons: { "pressure-at-low-attention": { n_pairs: 28 } },
  },
  runs: [
    { seed: 7, treatment: "low_incentive_low_attention", run_id: "run-low-low", run_directory: "", executive_optimism_bias: 0.05, incident_regret: 0, incentive_pressure: 0, attention_budget: 0 },
    { seed: 7, treatment: "high_incentive_low_attention", run_id: "run-high-low", run_directory: "", executive_optimism_bias: 0.30, incident_regret: 2, incentive_pressure: 0.9, attention_budget: 0 },
  ],
};

const detail = (runId: string) => ({
  manifest: { run_id: runId, seed: 7, event_hash: "1234567890abcdef", policy: "fixture" },
  request: {
    scenario: { max_ticks: 6, shock_tick: 3, shock_item_id: "api", shock_severity: 7, work_items: [] },
    treatment: { incentive_pressure: 0, attention_budget: 0 },
    organization: { agents: [{ id: "worker", role: "contributor", department: "Engineering" }, { id: "exec", role: "executive", department: "Executive" }] },
  },
  metrics: { distortion: [{ agent_id: "exec", department: "Executive", depth: 0, tick: 1, optimism_bias: 0.1, absolute_error: 0.1 }] },
});

const timeline = [{ sequence: 0, kind: "truth_snapshot", tick: 1, actor_id: null, causes: [], payload: { health_score: 0.4 } }];

test("builds an experiment from the 2x2 preset and launches through the real API", async ({ page }) => {
  let postedBody: unknown = null;
  await page.route("**/api/**", async (route) => {
    const url = new URL(route.request().url());
    if (url.pathname === "/api/health") return route.fulfill({ json: { status: "ok" } });
    if (url.pathname === "/api/experiments" && route.request().method() === "GET") return route.fulfill({ json: [{ name: "builder-2x2" }] });
    if (url.pathname === "/api/experiments" && route.request().method() === "POST") {
      postedBody = route.request().postDataJSON();
      return route.fulfill({ status: 202, json: { job_id: "builder-job", experiment: "builder-2x2", status: "queued", completed_runs: 0, failed_runs: 0, total_runs: 28, error: null } });
    }
    if (url.pathname === "/api/experiments/builder-2x2") return route.fulfill({ json: experiment });
    if (url.pathname === "/api/jobs/builder-job") return route.fulfill({ json: { job_id: "builder-job", experiment: "builder-2x2", status: "completed", completed_runs: 28, failed_runs: 0, total_runs: 28, error: null } });
    if (url.pathname.endsWith("/evidence")) return route.fulfill({ json: { nodes: [] } });
    if (url.pathname.endsWith("/decisions")) return route.fulfill({ json: { nodes: [] } });
    if (url.pathname.endsWith("/timeline")) return route.fulfill({ json: timeline });
    if (url.pathname.includes("run-low-low")) return route.fulfill({ json: detail("run-low-low") });
    if (url.pathname.includes("run-high-low")) return route.fulfill({ json: detail("run-high-low") });
    return route.fulfill({ status: 404, json: { detail: "unhandled builder route" } });
  });

  await page.goto("/");
  await page.getByRole("button", { name: "Builder" }).click();

  await expect(page.getByText("Build a custom run")).toBeVisible();
  await expect(page.getByLabel("EXPERIMENT NAME")).toHaveValue("builder-2x2");
  await page.getByRole("button", { name: "LAUNCH EXPERIMENT" }).click();

  // The job bar should appear once the launch returns and the run completes.
  await expect(page.getByText("COMPLETED")).toBeVisible({ timeout: 20_000 });

  // The submitted body must carry the full ExperimentRequest shape.
  expect(postedBody).toBeTruthy();
  const body = postedBody as { experiment: { name: string; treatments: Record<string, unknown> } };
  expect(body.experiment.name).toBe("builder-2x2");
  expect(Object.keys(body.experiment.treatments).sort()).toEqual([
    "high_incentive_high_attention",
    "high_incentive_low_attention",
    "low_incentive_high_attention",
    "low_incentive_low_attention",
  ]);

  // After completion the user is handed off to the Run Viewer.
  await expect(page.getByText("WORLD TRUTH / EXECUTIVE BELIEF")).toBeVisible();
});
