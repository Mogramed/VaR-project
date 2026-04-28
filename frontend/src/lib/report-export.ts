export type ReportStatusTone = "neutral" | "success" | "warning" | "danger";

export const REPORT_EXPORT_SECTIONS = [
  { key: "executive", number: "01", title: "Executive Summary" },
  { key: "analytics", number: "02", title: "Analytics" },
  { key: "capital", number: "03", title: "Capital" },
  { key: "governance", number: "04", title: "Governance" },
  { key: "narrative", number: "05", title: "Desk Narrative" },
  { key: "audit", number: "06", title: "Audit Trail" },
] as const;

export type ReportExportSectionKey = (typeof REPORT_EXPORT_SECTIONS)[number]["key"];

export function getReportExportSection(key: ReportExportSectionKey) {
  return REPORT_EXPORT_SECTIONS.find((section) => section.key === key)!;
}

export function reportStatusTone(status?: string | null): ReportStatusTone {
  const normalized = String(status ?? "").trim().toLowerCase();
  if (["ok", "healthy", "success", "pass", "matched"].includes(normalized)) {
    return "success";
  }
  if (["warn", "warning", "degraded", "stale"].includes(normalized)) {
    return "warning";
  }
  if (["breach", "critical", "fail", "failed", "error"].includes(normalized)) {
    return "danger";
  }
  return "neutral";
}

export function formatReportStatusLabel(status?: string | null, fallback = "pending"): string {
  const normalized = String(status ?? "").trim();
  const value = normalized || fallback;
  return value.replace(/[_-]+/g, " ").toUpperCase();
}
