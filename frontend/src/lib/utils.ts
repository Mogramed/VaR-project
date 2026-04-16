import { clsx, type ClassValue } from "clsx";
import { twMerge } from "tailwind-merge";

const STABLE_TIMESTAMP_FORMATTER = new Intl.DateTimeFormat("en-GB", {
  timeZone: "UTC",
  year: "numeric",
  month: "short",
  day: "2-digit",
  hour: "2-digit",
  minute: "2-digit",
  hourCycle: "h23",
});

export function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs));
}

export function formatCurrency(value: number | null | undefined, digits = 0) {
  if (value == null || Number.isNaN(value)) {
    return "n/a";
  }

  return new Intl.NumberFormat("en-US", {
    maximumFractionDigits: digits,
    minimumFractionDigits: digits,
  }).format(value);
}

export function formatSignedCurrency(value: number | null | undefined, digits = 0) {
  if (value == null || Number.isNaN(value)) {
    return "n/a";
  }

  const sign = value > 0 ? "+" : "";
  return `${sign}${formatCurrency(value, digits)}`;
}

export function formatPercent(value: number | null | undefined, digits = 1) {
  if (value == null || Number.isNaN(value)) {
    return "n/a";
  }

  return `${(value * 100).toFixed(digits)}%`;
}

export function formatTimestamp(value: string | null | undefined) {
  if (!value) {
    return "n/a";
  }

  const parsed = new Date(value);
  if (Number.isNaN(parsed.valueOf()) || parsed.getUTCFullYear() < 2000) {
    return "n/a";
  }

  return `${STABLE_TIMESTAMP_FORMATTER.format(parsed)} UTC`;
}

const SOURCE_LABELS: Record<string, string> = {
  auto: "Auto source",
  api: "API",
  broker: "Broker",
  historical: "Historical snapshot",
  mt5_agent_bridge: "MT5 agent bridge",
  mt5_live: "MT5 live",
  mt5_live_bridge: "MT5 live bridge",
  snapshot: "Persisted snapshot",
  target: "Desk target",
  tick: "Tick feed",
};

const OPERATIONAL_TRUTH_LABELS: Record<string, string> = {
  broker: "Broker truth",
  broker_delayed: "Broker delayed (last reference)",
  broker_unavailable: "Broker unavailable",
  snapshot: "Persisted snapshot",
  target: "Desk target",
  target_fallback: "Target fallback",
};

export function joinLabelParts(...values: Array<string | null | undefined>) {
  return values
    .map((value) => value?.trim())
    .filter((value): value is string => Boolean(value))
    .join(" | ");
}

export function formatSourceLabel(value: string | null | undefined) {
  if (!value) {
    return "n/a";
  }

  const normalized = value.trim().toLowerCase();
  const mapped = SOURCE_LABELS[normalized];
  if (mapped) {
    return mapped;
  }

  return value
    .replace(/[_-]+/g, " ")
    .replace(/\bmt5\b/gi, "MT5")
    .replace(/\bapi\b/gi, "API")
    .replace(/\bpnl\b/gi, "PnL")
    .replace(/\bfx\b/gi, "FX")
    .replace(/\btick\b/gi, "Tick")
    .replace(/\b\w/g, (match) => match.toUpperCase());
}

export function formatOperationalTruth(value: string | null | undefined) {
  if (!value) {
    return "n/a";
  }
  return OPERATIONAL_TRUTH_LABELS[value.trim().toLowerCase()] ?? formatSourceLabel(value);
}

export function formatTimestampWithSource({
  timestamp,
  source,
  fallback = "n/a",
}: {
  timestamp?: string | null;
  source?: string | null;
  fallback?: string;
}) {
  const label = joinLabelParts(
    source ? formatSourceLabel(source) : null,
    timestamp ? formatTimestamp(timestamp) : null,
  );
  return label || fallback;
}

export function titleCase(value: string) {
  return value
    .replace(/[_-]+/g, " ")
    .replace(/\b\w/g, (match) => match.toUpperCase());
}

export function humanizeIdentifier(value: string | null | undefined) {
  if (!value) {
    return "n/a";
  }

  return value.replace(/[_-]+/g, " ").trim();
}

export function slugifyHeading(value: string) {
  return value
    .toLowerCase()
    .trim()
    .replace(/[^\w\s-]/g, "")
    .replace(/\s+/g, "-")
    .replace(/-+/g, "-");
}

export function formatRelativeTime(value: string | null | undefined): string {
  if (!value) return "n/a";
  const parsed = new Date(value);
  if (Number.isNaN(parsed.valueOf())) return "n/a";
  const diffMs = Date.now() - parsed.getTime();
  if (diffMs < 0) return "just now";
  const seconds = Math.floor(diffMs / 1000);
  if (seconds < 5) return "just now";
  if (seconds < 60) return `${seconds}s ago`;
  const minutes = Math.floor(seconds / 60);
  if (minutes < 60) return `${minutes}m ago`;
  const hours = Math.floor(minutes / 60);
  if (hours < 24) return `${hours}h ago`;
  return `${Math.floor(hours / 24)}d ago`;
}
