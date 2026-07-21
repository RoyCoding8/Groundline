import { cleanup, render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { App } from "./App";
import * as api from "./api";
import type { Experiment, RunDetail, TimelineEvent } from "./types";

vi.mock("./api", () => ({
  checkBackend: vi.fn().mockResolvedValue(true),
  launchExperiment: vi.fn(),
  loadDecisions: vi.fn(),
  loadEvidence: vi.fn(),
  loadExperiment: vi.fn(),
  loadFirstExperiment: vi.fn(),
  loadJob: vi.fn().mockResolvedValue(null),
  loadRun: vi.fn(),
  waitForJob: vi.fn(),
}));

vi.mock("./charts", () => ({
  DistortionLadder: () => <div>ladder</div>,
  SignalChart: () => <div>signal</div>,
}));

const mockedApi = vi.mocked(api);

const timeline: TimelineEvent[] = [
  {
    sequence: 0,
    kind: "truth_snapshot",
    tick: 1,
    actor_id: null,
    causes: [],
    payload: { health_score: 0.4 },
  },
  {
    sequence: 1,
    kind: "report",
    tick: 1,
    actor_id: "exec",
    causes: [0],
    payload: { health_score: 0.5 },
  },
];

const experiment: Experiment = {
  name: "sample",
  request: null,
  failures: [],
  analysis: {
    baseline: "control",
    unit_of_analysis: "seed",
    design_diagnostics: { complete_pairs: 1 },
    comparisons: { intervention_minus_control: { n_pairs: 1 } },
  },
  runs: [
    {
      seed: 7,
      treatment: "control",
      run_id: "run-a",
      run_directory: "",
      executive_optimism_bias: 0.1,
      incident_regret: 0,
      incentive_pressure: 0,
      attention_budget: 1,
    },
    {
      seed: 11,
      treatment: "intervention",
      run_id: "run-b",
      run_directory: "",
      executive_optimism_bias: 0.2,
      incident_regret: 1,
      incentive_pressure: 0.9,
      attention_budget: 0,
    },
  ],
};

function detail(runId: string, policy: string, marker: string): RunDetail {
  return {
    manifest: { run_id: runId, seed: runId === "run-a" ? 7 : 11, event_hash: "1234567890abcdef", policy },
    request: {
      scenario: { marker },
      treatment: {
        incentive_pressure: runId === "run-a" ? 0 : 0.9,
        attention_budget: runId === "run-a" ? 1 : 0,
      },
      organization: {
        agents: [
          { id: "worker", role: "contributor", department: "Engineering" },
          { id: "exec", role: "executive", department: "Executive" },
        ],
      },
    },
    metrics: {
      distortion: [
        {
          agent_id: "exec",
          department: "Executive",
          depth: 0,
          tick: 1,
          optimism_bias: 0.1,
          absolute_error: 0.1,
        },
      ],
    },
  };
}

function deferred<T>() {
  let resolve!: (value: T) => void;
  let reject!: (reason: unknown) => void;
  const promise = new Promise<T>((resolvePromise, rejectPromise) => {
    resolve = resolvePromise;
    reject = rejectPromise;
  });
  return { promise, resolve, reject };
}

function mockSuccessfulReads() {
  mockedApi.loadFirstExperiment.mockResolvedValue(experiment);
  mockedApi.loadRun.mockImplementation(async (runId) => [
    detail(runId, runId === "run-a" ? "policy-a" : "policy-b", runId),
    timeline,
  ]);
  mockedApi.loadEvidence.mockResolvedValue({ nodes: [] });
  mockedApi.loadDecisions.mockResolvedValue({
    nodes: [
      {
        sequence: 0,
        agent_id: "worker",
        tick: 1,
        policy: "policy-a",
        context_hash: "abc",
        report: {
          agent_id: "worker",
          department: "Engineering",
          depth: 2,
          tick: 1,
          scope: ["ops"],
          health: { progress: 0.5, quality: 0.5, schedule: 0.5, reliability: 0.5 },
          confidence: 0.7,
          escalate: false,
          resource_request: 1,
          explanation: "everything looks on track from the floor",
        },
        actions: [],
        provider_metadata: {},
      },
    ],
  });
}

/** Navigate into the Run Viewer, which triggers loadFirstExperiment and
 *  auto-selects the first run so run details (e.g. "policy-a") appear. */
async function openRunViewer(user: ReturnType<typeof userEvent.setup>) {
  await user.click(screen.getByRole("button", { name: "Run Viewer" }));
}

afterEach(cleanup);

beforeEach(() => {
  vi.clearAllMocks();
  mockSuccessfulReads();
});

describe("App run identity and states", () => {
  it("loads the latest experiment for the dashboard summary", async () => {
    render(<App />);

    expect(await screen.findByText("sample")).toBeInTheDocument();
    expect(screen.queryByText("No experiments yet")).not.toBeInTheDocument();
  });

  it("ignores stale success responses after the selected run changes", async () => {
    const user = userEvent.setup();
    const staleRun = deferred<[RunDetail, TimelineEvent[]]>();
    const staleEvidence = deferred<{ nodes: [] }>();
    mockedApi.loadRun.mockImplementation((runId) => runId === "run-b"
      ? staleRun.promise
      : Promise.resolve([detail("run-a", "policy-a", "run-a"), timeline]));
    mockedApi.loadEvidence.mockImplementation((runId) => runId === "run-b"
      ? staleEvidence.promise
      : Promise.resolve({ nodes: [] }));

    render(<App />);
    await openRunViewer(user);
    expect(await screen.findByText("policy-a")).toBeInTheDocument();
    await user.selectOptions(screen.getByLabelText("Treatment and seed"), "run-b");
    await user.selectOptions(screen.getByLabelText("Treatment and seed"), "run-a");
    expect(await screen.findByText("policy-a")).toBeInTheDocument();

    staleRun.resolve([detail("run-b", "policy-b", "run-b"), timeline]);
    staleEvidence.resolve({ nodes: [] });
    await Promise.resolve();

    expect(screen.getByText("policy-a")).toBeInTheDocument();
    expect(screen.queryByText("policy-b")).not.toBeInTheDocument();
  });

  it("ignores stale request failures after the selected run changes", async () => {
    const user = userEvent.setup();
    const staleRun = deferred<[RunDetail, TimelineEvent[]]>();
    mockedApi.loadRun.mockImplementation((runId) => runId === "run-b"
      ? staleRun.promise
      : Promise.resolve([detail("run-a", "policy-a", "run-a"), timeline]));

    render(<App />);
    await openRunViewer(user);
    await screen.findByText("policy-a");
    await user.selectOptions(screen.getByLabelText("Treatment and seed"), "run-b");
    await user.selectOptions(screen.getByLabelText("Treatment and seed"), "run-a");
    await screen.findByText("policy-a");

    staleRun.reject(new Error("stale failure"));
    await Promise.resolve();

    expect(screen.queryByText("Run read failed")).not.toBeInTheDocument();
    expect(screen.queryByText("stale failure")).not.toBeInTheDocument();
  });

  it("surfaces per-agent decision explanations in the run viewer", async () => {
    const user = userEvent.setup();
    render(<App />);
    await openRunViewer(user);
    expect(await screen.findByText("AGENT DECISIONS")).toBeInTheDocument();
    expect(screen.getByText("everything looks on track from the floor")).toBeInTheDocument();
  });

  it("launches from the identity-matched selected run", async () => {
    const user = userEvent.setup();
    mockedApi.launchExperiment.mockResolvedValue({
      job_id: "job-1",
      experiment: "new",
      status: "queued",
      completed_runs: 0,
      failed_runs: 0,
      total_runs: 1,
      error: null,
    });
    mockedApi.waitForJob.mockImplementation(() => new Promise(() => undefined));

    render(<App />);
    await openRunViewer(user);
    await screen.findByText("policy-a");
    await user.selectOptions(screen.getByLabelText("Treatment and seed"), "run-b");
    await screen.findByText("policy-b");
    await user.click(screen.getByRole("button", { name: "RUN INTERVENTION" }));

    await waitFor(() => expect(mockedApi.launchExperiment).toHaveBeenCalled());
    expect(mockedApi.launchExperiment.mock.calls[0][0]).toEqual(
      expect.objectContaining({
        seeds: expect.arrayContaining([11]),
        scenario: { marker: "run-b" },
        analysis: expect.objectContaining({
          missingness: "fail_if_missing",
          contrasts: [expect.objectContaining({
            baseline: "low_incentive_low_attention",
            intervention: "high_incentive_low_attention",
            outcome: "upward_amplification",
          })],
        }),
      }),
    );
  });

  it("renders an explicit empty state", async () => {
    const user = userEvent.setup();
    mockedApi.loadFirstExperiment.mockResolvedValue(null);

    render(<App />);
    await openRunViewer(user);

    expect(await screen.findByText("No runs yet")).toBeInTheDocument();
    expect(screen.queryByText("Loading selected run artifacts.")).not.toBeInTheDocument();
  });

  it("renders and retries an initial load failure", async () => {
    const user = userEvent.setup();
    mockedApi.loadFirstExperiment.mockRejectedValueOnce(new Error("offline"));

    render(<App />);
    await openRunViewer(user);
    expect(await screen.findByText("Artifact read failed")).toBeInTheDocument();
    expect(screen.getByText("offline")).toBeInTheDocument();
    await user.click(screen.getByRole("button", { name: "RETRY" }));

    expect(await screen.findByText("policy-a")).toBeInTheDocument();
  });
});
