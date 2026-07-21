import { afterEach, describe, expect, it, vi } from "vitest";
import { checkBackend, launchExperiment, loadDecisions, loadEvidence } from "./api";

afterEach(() => vi.unstubAllGlobals());

describe("artifact API client", () => {
  it("uses the dedicated health endpoint for backend availability", async () => {
    const fetchMock = vi.fn().mockResolvedValue(new Response(JSON.stringify({ status: "ok" }), { status: 200 }));
    vi.stubGlobal("fetch", fetchMock);

    await expect(checkBackend()).resolves.toBe(true);

    expect(fetchMock).toHaveBeenCalledWith("/api/health", { signal: undefined });
  });

  it("posts a new intervention experiment", async () => {
    const fetchMock = vi.fn().mockResolvedValue(new Response(JSON.stringify({ job_id: "job-1", status: "queued" }), { status: 202 }));
    vi.stubGlobal("fetch", fetchMock);
    const experiment = {
      name: "new-matrix",
      seeds: [1, 2, 3, 4, 5, 6, 7],
      scenario: {},
      organization: { agents: [] },
      treatments: {
        control: { incentive_pressure: 0, attention_budget: 1 },
        intervention: { incentive_pressure: 1, attention_budget: 0 },
      },
      analysis: {
        seed: 17,
        missingness: "fail_if_missing" as const,
        contrasts: [{
          id: "intervention-minus-control",
          baseline: "control",
          intervention: "intervention",
          outcome: "upward_amplification",
          direction: "increase" as const,
          family: "primary",
          status: "confirmatory" as const,
        }],
      },
      max_concurrency: 2,
    };

    await launchExperiment(experiment);

    expect(fetchMock).toHaveBeenCalledWith("/api/experiments", expect.objectContaining({ method: "POST" }));
    expect(JSON.parse(fetchMock.mock.calls[0][1].body)).toEqual({ experiment, policy: "fixture", model: "" });
  });

  it("threads policy and model into the launch body", async () => {
    const fetchMock = vi.fn().mockResolvedValue(new Response(JSON.stringify({ job_id: "job-2", status: "queued" }), { status: 202 }));
    vi.stubGlobal("fetch", fetchMock);
    const experiment = {
      name: "live-matrix",
      seeds: [1, 2, 3, 4, 5, 6, 7],
      scenario: {},
      organization: { agents: [] },
      treatments: { control: { incentive_pressure: 0, attention_budget: 1 } },
      analysis: { seed: 17, missingness: "fail_if_missing" as const, contrasts: [] },
      max_concurrency: 2,
    };

    await launchExperiment(experiment, "record", "claude-sonnet-5");

    expect(JSON.parse(fetchMock.mock.calls[0][1].body)).toEqual({ experiment, policy: "record", model: "claude-sonnet-5" });
  });

  it("loads agent decisions for a run", async () => {
    const fetchMock = vi.fn().mockResolvedValue(
      new Response(JSON.stringify({ nodes: [{ sequence: 1, agent_id: "a1" }] }), { status: 200 }),
    );
    vi.stubGlobal("fetch", fetchMock);

    await expect(loadDecisions("run/a")).resolves.toEqual({ nodes: [{ sequence: 1, agent_id: "a1" }] });

    expect(fetchMock).toHaveBeenCalledWith("/api/runs/run%2Fa/decisions", { signal: undefined });
  });

  it("encodes evidence filters in the query", async () => {
    const fetchMock = vi.fn().mockResolvedValue(new Response(JSON.stringify({ nodes: [] }), { status: 200 }));
    vi.stubGlobal("fetch", fetchMock);

    await loadEvidence("run/a", { department: "Quality Assurance", depth: 2 });

    expect(fetchMock.mock.calls[0][0]).toBe("/api/runs/run%2Fa/evidence?department=Quality+Assurance&depth=2");
  });

  it.each([
    [{ detail: "run not found" }, "run not found"],
    [
      { detail: [{ loc: ["body", "experiment", "name"], msg: "Field required" }] },
      "body.experiment.name: Field required",
    ],
    [
      { detail: { code: "invalid_structure", artifact: "metrics.json", message: "artifact is malformed" } },
      "artifact is malformed",
    ],
  ])("normalizes API error details", async (body, expected) => {
    vi.stubGlobal(
      "fetch",
      vi.fn().mockResolvedValue(
        new Response(JSON.stringify(body), { status: 422, statusText: "Unprocessable Entity" }),
      ),
    );

    await expect(loadEvidence("run-a", {})).rejects.toThrow(expected);
  });
});
