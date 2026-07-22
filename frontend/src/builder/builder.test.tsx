import { act, cleanup, render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { ExperimentBuilder } from "./ExperimentBuilder";
import * as api from "../api";

vi.mock("../api", () => ({
  launchExperiment: vi.fn(),
  loadExperiment: vi.fn(),
  waitForJob: vi.fn(),
}));

const mockedApi = vi.mocked(api);

afterEach(() => {
  cleanup();
  localStorage.clear();
  vi.clearAllMocks();
});

beforeEach(() => {
  mockedApi.launchExperiment.mockResolvedValue({
    job_id: "builder-job",
    experiment: "builder-2x2",
    status: "queued",
    completed_runs: 0,
    failed_runs: 0,
    total_runs: 28,
    error: null,
  });
  mockedApi.loadExperiment.mockResolvedValue({
    name: "builder-2x2",
    request: null,
    failures: [],
    analysis: {
      baseline: "low_incentive_low_attention",
      unit_of_analysis: "seed",
      design_diagnostics: { complete_pairs: 28 },
      comparisons: { "pressure-at-low-attention": { n_pairs: 28 } },
    },
    runs: [{ seed: 7, treatment: "low_incentive_low_attention", run_id: "run-first", run_directory: "", executive_optimism_bias: 0, incident_regret: 0, incentive_pressure: 0, attention_budget: 0 }],
  });
});

describe("ExperimentBuilder", () => {
  it("defaults to the 2x2 preset with the four treatment cells", () => {
    render(
      <ExperimentBuilder
        onLaunched={vi.fn()}
        setLaunchError={vi.fn()}
        setJob={vi.fn()}
        setLaunchPolicy={vi.fn()}
        initialPolicy="fixture"
      />,
    );

    expect(screen.getByDisplayValue("builder-2x2")).toBeInTheDocument();
    // The four 2x2 treatment cells are present by name (also referenced by
    // contrast selects, so use getAll and assert count > 0).
    expect(screen.getAllByDisplayValue("low_incentive_low_attention").length).toBeGreaterThan(0);
    expect(screen.getAllByDisplayValue("low_incentive_high_attention").length).toBeGreaterThan(0);
    expect(screen.getAllByDisplayValue("high_incentive_low_attention").length).toBeGreaterThan(0);
    expect(screen.getAllByDisplayValue("high_incentive_high_attention").length).toBeGreaterThan(0);
  });

  it("submits the full ExperimentRequest to launchExperiment and hands off to Run Viewer", async () => {
    const user = userEvent.setup();
    const onLaunched = vi.fn();
    const setJob = vi.fn();
    const setLaunchPolicy = vi.fn();
    render(
      <ExperimentBuilder
        onLaunched={onLaunched}
        setLaunchError={vi.fn()}
        setJob={setJob}
        setLaunchPolicy={setLaunchPolicy}
        initialPolicy="fixture"
      />,
    );

    await user.click(screen.getByRole("button", { name: "LAUNCH EXPERIMENT" }));

    await waitFor(() => expect(mockedApi.launchExperiment).toHaveBeenCalledTimes(1));
    const [experiment, policy] = mockedApi.launchExperiment.mock.calls[0] as [
      import("../types").BuilderExperimentRequest,
      string,
      string,
    ];
    expect(policy).toBe("fixture");
    expect(experiment.name).toBe("builder-2x2");
    expect(experiment.treatments.low_incentive_low_attention).toBeDefined();
    expect(experiment.treatments.high_incentive_high_attention).toEqual({ incentive_pressure: 0.9, attention_budget: 3 });
    expect(experiment.analysis.contrasts[0]).toEqual(
      expect.objectContaining({
        baseline: "low_incentive_low_attention",
        intervention: "high_incentive_low_attention",
        outcome: "upward_amplification",
        direction: "increase",
      }),
    );
    expect(experiment.scenario.work_items.map((w) => w.id)).toEqual(["launch-spec", "api", "release-gate"]);

    await waitFor(() => expect(mockedApi.loadExperiment).toHaveBeenCalledWith("builder-2x2"));
    await waitFor(() => expect(onLaunched).toHaveBeenCalledTimes(1));
    expect(setLaunchPolicy).toHaveBeenCalledWith("fixture");
    expect(setJob).toHaveBeenCalled();
    expect(onLaunched.mock.calls[0][1]).toBe("run-first");
  });

  it("switching to the deep-hierarchy preset loads the deeper org", async () => {
    const user = userEvent.setup();
    render(
      <ExperimentBuilder
        onLaunched={vi.fn()}
        setLaunchError={vi.fn()}
        setJob={vi.fn()}
        setLaunchPolicy={vi.fn()}
        initialPolicy="fixture"
      />,
    );

    await user.selectOptions(screen.getByLabelText("PRESET"), "deep-hierarchy");

    expect(screen.getByDisplayValue("builder-deep-hierarchy")).toBeInTheDocument();
    // Deep hierarchy has a 4-level chain: ceo -> vp-eng -> eng-director -> backend.
    expect(screen.getAllByDisplayValue("vp-eng").length).toBeGreaterThan(0);
  });
});
