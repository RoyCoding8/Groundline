import type {
  EvidenceNode,
  Experiment,
  ExperimentRequest,
  JobStatus,
  RunDetail,
  TimelineEvent,
} from "./types";

function errorDetail(value: unknown): string | null {
  if (typeof value === "string") return value;
  if (Array.isArray(value)) {
    const messages = value.map(errorDetail).filter((message): message is string => Boolean(message));
    return messages.length ? messages.join("; ") : null;
  }
  if (value && typeof value === "object") {
    const detail = value as Record<string, unknown>;
    const message = typeof detail.message === "string"
      ? detail.message
      : typeof detail.msg === "string"
        ? detail.msg
        : null;
    if (message) {
      const location = Array.isArray(detail.loc) ? detail.loc.map(String).join(".") : "";
      return location ? `${location}: ${message}` : message;
    }
    try {
      return JSON.stringify(value);
    } catch {
      return null;
    }
  }
  return null;
}

async function requestJson<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(path, init);
  if (!response.ok) {
    let detail = `${response.status} ${response.statusText}`;
    try {
      const body = await response.json() as { detail?: unknown };
      detail = errorDetail(body.detail) ?? detail;
    } catch {
      // Preserve the HTTP status when an upstream proxy does not return JSON.
    }
    throw new Error(detail);
  }
  return response.json() as Promise<T>;
}

export async function checkBackend(signal?: AbortSignal): Promise<boolean> {
  try {
    const response = await fetch("/api/health", { signal });
    return response.ok;
  } catch {
    return false;
  }
}

export async function loadFirstExperiment(signal?: AbortSignal): Promise<Experiment | null> {
  const summaries = await requestJson<Array<{ name: string }>>("/api/experiments", { signal });
  if (!summaries.length) return null;
  return loadExperiment(summaries.at(-1)!.name, signal);
}

export function loadExperiment(name: string, signal?: AbortSignal): Promise<Experiment> {
  return requestJson<Experiment>(`/api/experiments/${encodeURIComponent(name)}`, { signal });
}

export function loadRun(runId: string, signal?: AbortSignal): Promise<[RunDetail, TimelineEvent[]]> {
  return Promise.all([
    requestJson<RunDetail>(`/api/runs/${encodeURIComponent(runId)}`, { signal }),
    requestJson<TimelineEvent[]>(`/api/runs/${encodeURIComponent(runId)}/timeline`, { signal }),
  ]);
}

export function loadEvidence(
  runId: string,
  filters: { department?: string; depth?: number; kind?: string },
  signal?: AbortSignal,
): Promise<{ nodes: EvidenceNode[] }> {
  const query = new URLSearchParams();
  if (filters.department) query.set("department", filters.department);
  if (filters.depth !== undefined) query.set("depth", String(filters.depth));
  if (filters.kind) query.set("kind", filters.kind);
  return requestJson(`/api/runs/${encodeURIComponent(runId)}/evidence?${query}`, { signal });
}

export function launchExperiment(experiment: ExperimentRequest): Promise<JobStatus> {
  return requestJson<JobStatus>("/api/experiments", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ experiment, policy: "fixture" }),
  });
}

export function loadJob(jobId: string): Promise<JobStatus> {
  return requestJson<JobStatus>(`/api/jobs/${encodeURIComponent(jobId)}`);
}

export async function waitForJob(
  jobId: string,
  onUpdate: (job: JobStatus) => void,
  delayMs = 250,
): Promise<JobStatus> {
  for (let attempt = 0; attempt < 1200; attempt += 1) {
    const job = await loadJob(jobId);
    onUpdate(job);
    if (job.status === "completed") return job;
    if (job.status === "failed") throw new Error(job.error ?? "experiment failed");
    await new Promise((resolve) => window.setTimeout(resolve, delayMs));
  }
  throw new Error("experiment did not finish before the polling deadline");
}
