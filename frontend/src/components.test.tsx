import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it, vi } from "vitest";
import { EvidenceInspector, InterventionControls } from "./components";
import type { EvidenceNode } from "./types";

describe("InterventionControls", () => {
  it("submits operator-selected knobs and paired seed count", async () => {
    const user = userEvent.setup();
    const launch = vi.fn();
    render(<InterventionControls initialIncentive={0.2} initialAttention={1} busy={false} onLaunch={launch} />);

    const incentive = screen.getByLabelText("INCENTIVE PRESSURE");
    await user.clear(incentive);
    await user.type(incentive, "0.9");
    const attention = screen.getByLabelText("MANAGER ATTENTION");
    await user.clear(attention);
    await user.type(attention, "3");
    await user.click(screen.getByRole("button", { name: "RUN INTERVENTION" }));

    expect(launch).toHaveBeenCalledWith({ incentive: 0.9, attention: 3, seedCount: 12, policy: "fixture", model: "" });
  });
});

describe("EvidenceInspector", () => {
  it("renders causal parents and forwards hierarchy filters", async () => {
    const user = userEvent.setup();
    const onDepartment = vi.fn();
    const onDepth = vi.fn();
    const node: EvidenceNode = {
      sequence: 8,
      kind: "report",
      tick: 2,
      actor_id: "manager",
      department: "Engineering",
      depth: 1,
      causes: [7],
      evidence_refs: [],
      event: { sequence: 8, kind: "report", tick: 2, actor_id: "manager", causes: [7], payload: {} },
    };
    render(
      <EvidenceInspector
        nodes={[node]}
        departments={["Engineering", "QA"]}
        depths={[0, 1, 2]}
        department=""
        depth=""
        onDepartment={onDepartment}
        onDepth={onDepth}
      />,
    );

    expect(screen.getByText("FROM #7")).toBeInTheDocument();
    await user.selectOptions(screen.getByLabelText("Evidence department"), "QA");
    await user.selectOptions(screen.getByLabelText("Evidence depth"), "2");
    expect(onDepartment).toHaveBeenCalledWith("QA");
    expect(onDepth).toHaveBeenCalledWith("2");
  });
});
