import { describe, expect, it } from "vitest";
import {
  formatReportStatusLabel,
  REPORT_EXPORT_SECTIONS,
  reportStatusTone,
} from "@/lib/report-export";

describe("report export status helpers", () => {
  it("maps capital statuses to the expected badge tones", () => {
    expect(reportStatusTone("ok")).toBe("success");
    expect(reportStatusTone("warn")).toBe("warning");
    expect(reportStatusTone("degraded")).toBe("warning");
    expect(reportStatusTone("breach")).toBe("danger");
    expect(reportStatusTone("failed")).toBe("danger");
    expect(reportStatusTone("pending")).toBe("neutral");
    expect(reportStatusTone(null)).toBe("neutral");
  });

  it("normalizes report status labels for display", () => {
    expect(formatReportStatusLabel("warn")).toBe("WARN");
    expect(formatReportStatusLabel("in_progress")).toBe("IN PROGRESS");
    expect(formatReportStatusLabel("")).toBe("PENDING");
  });

  it("keeps report export sections ordered and aligned", () => {
    expect(REPORT_EXPORT_SECTIONS).toEqual([
      { key: "executive", number: "01", title: "Executive Summary" },
      { key: "analytics", number: "02", title: "Analytics" },
      { key: "capital", number: "03", title: "Capital" },
      { key: "governance", number: "04", title: "Governance" },
      { key: "narrative", number: "05", title: "Desk Narrative" },
      { key: "audit", number: "06", title: "Audit Trail" },
    ]);
  });
});
