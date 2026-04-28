import type {
  DecisionBacktestTrajectoryResponse,
  DecisionForecastResponse,
  DecisionIntelligenceResponse,
  DecisionPortfolioForecastResponse,
  DecisionReplayResponse,
  RiskDecisionResponse,
} from "@/lib/api/types";

export type DecisionAlphaFeatureRow = {
  key: string;
  label: string;
  value: number | null;
  contribution: number | null;
  available: boolean;
};

export type DecisionAlphaProjectionSeries = {
  labels: string[];
  actual: Array<number | null>;
  predicted: Array<number | null>;
  historyCount: number;
  forecastCount: number;
  splitLabel: string | null;
};

const FEATURE_LABELS: Record<string, string> = {
  momentum_short_term: "Momentum (short)",
  volatility_recent: "Recent volatility",
  headroom_delta: "Headroom delta",
  risk_delta: "Risk delta",
  validation_confidence: "Validation confidence",
  exception_pressure: "Exception pressure",
  spread_cost_norm: "Spread cost (norm)",
  slippage_points: "Slippage (pts)",
};

function toFinite(value: unknown, fallback = 0): number {
  const numeric = Number(value);
  return Number.isFinite(numeric) ? numeric : fallback;
}

function toFiniteOrNull(value: unknown): number | null {
  const numeric = Number(value);
  return Number.isFinite(numeric) ? numeric : null;
}

function toTimestampMillis(value: string | null | undefined): number | null {
  if (!value) {
    return null;
  }
  const parsed = Date.parse(String(value));
  return Number.isFinite(parsed) ? parsed : null;
}

function flagFromUnknown(value: unknown): boolean | null {
  if (typeof value === "boolean") {
    return value;
  }
  const numeric = Number(value);
  if (Number.isFinite(numeric)) {
    return numeric >= 0.5;
  }
  return null;
}

function byAbsDescending(left: DecisionAlphaFeatureRow, right: DecisionAlphaFeatureRow) {
  return Math.abs(right.contribution ?? 0) - Math.abs(left.contribution ?? 0);
}

function featureAvailable(
  intelligence: DecisionIntelligenceResponse,
  key: string,
): boolean {
  const explicitAvailability = (
    intelligence as unknown as { feature_availability?: Record<string, unknown> }
  ).feature_availability;
  const explicit = flagFromUnknown(explicitAvailability?.[key]);
  if (explicit != null) {
    return explicit;
  }
  const fallbackFlag = flagFromUnknown(intelligence.calculations?.[`feature_available_${key}`]);
  if (fallbackFlag != null) {
    return fallbackFlag;
  }
  return true;
}

export function extractDecisionIntelligence(
  payload: Pick<RiskDecisionResponse, "decision_intelligence"> | null | undefined,
): DecisionIntelligenceResponse | null {
  return payload?.decision_intelligence ?? null;
}

export function decisionAlphaFeatureRows(
  intelligence: DecisionIntelligenceResponse | null | undefined,
): DecisionAlphaFeatureRow[] {
  if (!intelligence) {
    return [];
  }
  const featureEntries = Object.entries(intelligence.features ?? {});
  return featureEntries
    .map(([key, rawValue]) => {
      const available = featureAvailable(intelligence, key);
      return {
        key,
        label: FEATURE_LABELS[key] ?? key.replace(/[_-]+/g, " "),
        value: available ? toFiniteOrNull(rawValue) : null,
        contribution:
          !available || intelligence.feature_contributions?.[key] == null
            ? null
            : toFinite(intelligence.feature_contributions[key], 0),
        available,
      };
    })
    .sort(byAbsDescending);
}

export function replayChartSeries(
  replay: DecisionReplayResponse | null | undefined,
): {
  labels: string[];
  predicted: number[];
  realized: number[];
  cumulative: number[];
} {
  const points = replay?.predicted_vs_realized ?? [];
  return {
    labels: points.map((point) => point.timestamp),
    predicted: points.map((point) => toFinite(point.predicted_score)),
    realized: points.map((point) => toFinite(point.realized_pnl)),
    cumulative: points.map((point) => toFinite(point.cum_pnl)),
  };
}

export function forecastChartSeries(
  forecast: DecisionForecastResponse | null | undefined,
): {
  labels: string[];
  bear: number[];
  base: number[];
  bull: number[];
} {
  const scenarioByName = new Map(
    (forecast?.scenarios ?? []).map((scenario) => [scenario.name, scenario]),
  );
  const labels = (scenarioByName.get("base")?.path ?? scenarioByName.get("bear")?.path ?? [])
    .map((point) => point.date);

  const pricesFor = (name: string) =>
    (scenarioByName.get(name)?.path ?? []).map((point) => toFinite(point.price));

  return {
    labels,
    bear: pricesFor("bear"),
    base: pricesFor("base"),
    bull: pricesFor("bull"),
  };
}

export function trajectoryChartSeries(
  trajectory: DecisionBacktestTrajectoryResponse | null | undefined,
): {
  labels: string[];
  predicted: number[];
  actual: number[];
} {
  const points = trajectory?.predicted_vs_actual ?? [];
  return {
    labels: points.map((point) => point.timestamp),
    predicted: points.map((point) => toFinite(point.predicted_price)),
    actual: points.map((point) => toFinite(point.actual_price)),
  };
}

export function projectionHistoryForecastSeries(
  trajectory: DecisionBacktestTrajectoryResponse | null | undefined,
  forecast: DecisionForecastResponse | null | undefined,
  options?: { historyWindowDays?: number },
): DecisionAlphaProjectionSeries {
  const historyWindowDays = Math.max(Number(options?.historyWindowDays ?? 30), 1);
  const historySource = (trajectory?.predicted_vs_actual ?? [])
    .map((point) => ({
      label: String(point.timestamp ?? "").trim(),
      actual: toFiniteOrNull(point.actual_price),
      predicted: toFiniteOrNull(point.predicted_price),
      ts: toTimestampMillis(point.timestamp),
    }))
    .filter((point) => point.label && (point.actual != null || point.predicted != null));

  let history = historySource.slice();
  const datedHistory = history.filter((point) => point.ts != null) as Array<{
    label: string;
    actual: number | null;
    predicted: number | null;
    ts: number;
  }>;
  if (datedHistory.length > 0) {
    const latestTs = Math.max(...datedHistory.map((point) => point.ts));
    const cutoff = latestTs - historyWindowDays * 24 * 60 * 60 * 1000;
    const filtered = datedHistory.filter((point) => point.ts >= cutoff);
    history = (filtered.length > 0 ? filtered : datedHistory).sort((left, right) => left.ts - right.ts);
  } else if (history.length > 200) {
    history = history.slice(-200);
  }

  const scenarioByName = new Map((forecast?.scenarios ?? []).map((scenario) => [scenario.name, scenario]));
  const forecastPath = (
    scenarioByName.get("base")?.path
    ?? scenarioByName.get("bull")?.path
    ?? scenarioByName.get("bear")?.path
    ?? []
  )
    .map((point) => ({
      label: String(point.date ?? "").trim(),
      predicted: toFiniteOrNull(point.price),
      ts: toTimestampMillis(point.date),
    }))
    .filter((point) => point.label && point.predicted != null);

  let forecastStartIndex = 0;
  if (history.length > 0 && forecastPath.length > 0) {
    const lastHistory = history[history.length - 1];
    const firstForecast = forecastPath[0];
    if (lastHistory.label === firstForecast.label) {
      forecastStartIndex = 1;
    } else if (lastHistory.ts != null && firstForecast.ts != null) {
      const hoursGap = Math.abs(firstForecast.ts - lastHistory.ts) / (60 * 60 * 1000);
      if (hoursGap <= 36) {
        forecastStartIndex = 1;
      }
    }
  }
  const projection = forecastPath.slice(forecastStartIndex);
  const lastHistoryActual =
    history.length > 0
      ? history[history.length - 1].actual
      : null;
  const firstProjectionPredicted =
    projection.length > 0
      ? projection[0].predicted
      : null;
  if (
    lastHistoryActual != null
    && firstProjectionPredicted != null
    && Math.abs(firstProjectionPredicted) > 1e-9
  ) {
    const jump = Math.abs(firstProjectionPredicted - lastHistoryActual) / Math.abs(lastHistoryActual);
    // Defensive continuity guard for stale forecast anchors: preserve the
    // relative path while rebasing it to the latest observed price.
    if (jump > 0.04) {
      const rebaseRatio = lastHistoryActual / firstProjectionPredicted;
      for (const point of projection) {
        point.predicted = toFiniteOrNull(point.predicted == null ? null : point.predicted * rebaseRatio);
      }
    }
  }

  const labels = [
    ...history.map((point) => point.label),
    ...projection.map((point) => point.label),
  ];
  const actual = [
    ...history.map((point) => point.actual),
    ...projection.map(() => null),
  ];
  const predicted = [
    ...history.map((point) => point.predicted),
    ...projection.map((point) => point.predicted),
  ];
  return {
    labels,
    actual,
    predicted,
    historyCount: history.length,
    forecastCount: projection.length,
    splitLabel: history.length > 0 ? history[history.length - 1].label : null,
  };
}

export function portfolioPnlScenarioSeries(
  portfolioForecast: DecisionPortfolioForecastResponse | null | undefined,
): {
  labels: string[];
  bear: number[];
  base: number[];
  bull: number[];
} {
  const scenarioByName = new Map(
    (portfolioForecast?.pnl_scenarios ?? []).map((scenario) => [scenario.name, scenario]),
  );
  const labels = (scenarioByName.get("base")?.path ?? scenarioByName.get("bear")?.path ?? [])
    .map((point) => point.date);
  const pnlFor = (name: string) =>
    (scenarioByName.get(name)?.path ?? []).map((point) => toFinite(point.pnl));
  return {
    labels,
    bear: pnlFor("bear"),
    base: pnlFor("base"),
    bull: pnlFor("bull"),
  };
}
