import type { BacktestSeriesPoint, TimeSeriesPoint } from "@/lib/view-models";

export type ChartMode = "sparse" | "standard" | "dense" | "comparison" | "trace";

type ChartConfig = {
  mode?: ChartMode;
};

type BarChartConfig = ChartConfig & {
  color?: string;
  negativeColor?: string;
};

type GroupedBarConfig = ChartConfig;

/* ------------------------------------------------------------------ */
/*  Palette — Binance-inspired dark trading terminal                  */
/* ------------------------------------------------------------------ */

export const CHART_PALETTE = {
  gold: "#f0b90b",
  green: "#0ecb81",
  blue: "#47b7f8",
  red: "#f6465d",
  teal: "#2ee5db",
  purple: "#a78bfa",
  cyan: "#00d2ff",
  amber: "#d89b49",
} as const;

const textPrimary = "rgba(234,236,239,0.87)";
const textSecondary = "rgba(234,236,239,0.55)";
const gridLine = "rgba(255,255,255,0.05)";
const axisStroke = "rgba(255,255,255,0.08)";

/* ------------------------------------------------------------------ */
/*  Color helpers                                                     */
/* ------------------------------------------------------------------ */

function hexToRgb(hex: string): [number, number, number] | null {
  const h = hex.replace("#", "");
  if (h.length !== 6) return null;
  return [parseInt(h.slice(0, 2), 16), parseInt(h.slice(2, 4), 16), parseInt(h.slice(4, 6), 16)];
}

function lighten(hex: string, amount = 0.15): string {
  const rgb = hexToRgb(hex);
  if (!rgb) return hex;
  const [r, g, b] = rgb.map((c) => Math.min(255, Math.round(c + (255 - c) * amount)));
  return `rgb(${r},${g},${b})`;
}

function rgba(hex: string, alpha: number): string {
  const rgb = hexToRgb(hex);
  if (!rgb) return hex;
  return `rgba(${rgb[0]},${rgb[1]},${rgb[2]},${alpha})`;
}

function verticalGradient(topColor: string, bottomColor: string) {
  return {
    type: "linear" as const,
    x: 0,
    y: 0,
    x2: 0,
    y2: 1,
    colorStops: [
      { offset: 0, color: topColor },
      { offset: 1, color: bottomColor },
    ],
  };
}

/** Rich multi-stop area gradient — glows near the line, fades to transparent */
function areaGradient(hex: string) {
  const rgb = hexToRgb(hex);
  if (!rgb) return `${hex}18`;
  const [r, g, b] = rgb;
  return {
    type: "linear" as const,
    x: 0,
    y: 0,
    x2: 0,
    y2: 1,
    colorStops: [
      { offset: 0, color: `rgba(${r},${g},${b},0.32)` },
      { offset: 0.4, color: `rgba(${r},${g},${b},0.12)` },
      { offset: 0.85, color: `rgba(${r},${g},${b},0.03)` },
      { offset: 1, color: `rgba(${r},${g},${b},0)` },
    ],
  };
}

/** Bar gradient — glossy top highlight fading to base */
function barGradient(hex: string) {
  const rgb = hexToRgb(hex);
  if (!rgb) return verticalGradient(lighten(hex, 0.18), hex);
  const [r, g, b] = rgb;
  return {
    type: "linear" as const,
    x: 0,
    y: 0,
    x2: 0,
    y2: 1,
    colorStops: [
      { offset: 0, color: lighten(hex, 0.28) },
      { offset: 0.3, color: `rgba(${r},${g},${b},0.95)` },
      { offset: 1, color: `rgba(${r},${g},${b},0.78)` },
    ],
  };
}

/* ------------------------------------------------------------------ */
/*  Formatting helpers                                                */
/* ------------------------------------------------------------------ */

const SHORT_MONTHS = [
  "Jan", "Feb", "Mar", "Apr", "May", "Jun",
  "Jul", "Aug", "Sep", "Oct", "Nov", "Dec",
];

function formatTimestampLabel(raw: string): string {
  const ts = timestampFromLabel(raw);
  if (ts == null) {
    return raw;
  }
  const d = new Date(ts);
  if (Number.isNaN(d.getTime())) {
    return raw;
  }
  const day = d.getUTCDate();
  const mon = SHORT_MONTHS[d.getUTCMonth()];
  return `${day} ${mon}`;
}

function formatCategoryLabel(value: string, mode: ChartMode) {
  const normalized = String(value ?? "").replace(/[_-]+/g, " ").trim();
  const formatted = formatTimestampLabel(normalized);

  if (mode !== "sparse" && mode !== "comparison") {
    return formatted;
  }

  const words = formatted.split(/\s+/).filter(Boolean);
  if (words.length <= 2) {
    return words.join("\n");
  }

  const lines: string[] = [];
  let current = "";

  for (const word of words) {
    const next = current ? `${current} ${word}` : word;
    if (next.length > 12 && current) {
      lines.push(current);
      current = word;
    } else {
      current = next;
    }
  }

  if (current) {
    lines.push(current);
  }

  return lines.slice(0, 3).join("\n");
}

function roundValue(value: number | null | undefined) {
  const numeric = Number(value ?? 0);
  if (!Number.isFinite(numeric)) {
    return 0;
  }
  const absolute = Math.abs(numeric);
  if (absolute >= 1000) {
    return Math.round(numeric);
  }
  if (absolute >= 100) {
    return Number(numeric.toFixed(1));
  }
  if (absolute >= 1) {
    return Number(numeric.toFixed(2));
  }
  return Number(numeric.toFixed(3));
}

function toFiniteNumber(value: unknown): number | null {
  if (value == null || typeof value === "boolean") {
    return null;
  }
  if (typeof value === "string" && value.trim() === "") {
    return null;
  }
  const numeric = Number(value);
  if (!Number.isFinite(numeric)) {
    return null;
  }
  return numeric;
}

function normalizeEpochLikeMillis(value: number): number | null {
  if (!Number.isFinite(value)) {
    return null;
  }
  const abs = Math.abs(value);
  const millis =
    abs < 1e11
      ? value * 1000
      : abs < 1e14
        ? value
        : abs < 1e17
          ? value / 1000
          : value / 1_000_000;
  if (!Number.isFinite(millis)) {
    return null;
  }
  return millis;
}

function isValidTimestampMillis(millis: number): boolean {
  if (!Number.isFinite(millis)) {
    return false;
  }
  const date = new Date(millis);
  if (Number.isNaN(date.getTime())) {
    return false;
  }
  return date.getUTCFullYear() >= 2000;
}

function timestampFromLabel(label: string): number | null {
  const trimmed = label.trim();
  if (!trimmed) {
    return null;
  }
  if (/^-?\d+(?:\.\d+)?$/.test(trimmed)) {
    const normalized = normalizeEpochLikeMillis(Number(trimmed));
    return normalized != null && isValidTimestampMillis(normalized) ? normalized : null;
  }
  const parsed = Date.parse(trimmed);
  if (!Number.isFinite(parsed) || !isValidTimestampMillis(parsed)) {
    return null;
  }
  return parsed;
}

function normalizeLabel(raw: unknown, index: number): string {
  const text = String(raw ?? "").trim();
  if (!text) {
    return String(index + 1);
  }
  return text;
}

function resolveTimeOrder<T extends { sortKey: number | null; sourceIndex: number }>(rows: T[]): T[] {
  if (rows.length <= 1) {
    return rows;
  }

  const withSortKey = rows.filter((row): row is T & { sortKey: number } => row.sortKey != null);
  const withoutSortKey = rows.filter((row) => row.sortKey == null);
  const sorted = withSortKey.slice().sort((left, right) => {
    if (left.sortKey === right.sortKey) {
      return left.sourceIndex - right.sourceIndex;
    }
    return left.sortKey - right.sortKey;
  });
  return [...sorted, ...withoutSortKey];
}

function computeAxisDomain(
  values: Array<number | null | undefined>,
  options?: { includeZero?: boolean },
) {
  const finiteValues = values.filter((value): value is number => Number.isFinite(value));
  if (finiteValues.length === 0) {
    return { min: 0, max: 1 };
  }
  const allNonNegative = finiteValues.every((value) => value >= 0);
  const allNonPositive = finiteValues.every((value) => value <= 0);
  const finite = finiteValues.slice();
  if (options?.includeZero) {
    finite.push(0);
  }

  let min = Math.min(...finite);
  let max = Math.max(...finite);

  if (min === max) {
    if (options?.includeZero && min === 0) {
      return { min: 0, max: 1 };
    }
    const pad = Math.max(Math.abs(min) * 0.12, 1);
    min -= pad;
    max += pad;
  } else {
    const range = max - min;
    const pad = Math.max(range * 0.08, 0.5);
    min -= pad;
    max += pad;
  }

  if (options?.includeZero) {
    if (allNonNegative) {
      min = 0;
    }
    if (allNonPositive) {
      max = 0;
    }
  }

  return { min, max };
}

function yAxisForValues(
  values: Array<number | null | undefined>,
  options?: { includeZero?: boolean },
) {
  const domain = computeAxisDomain(values, options);
  return {
    ...yAxisBase(),
    min: domain.min,
    max: domain.max,
  };
}

type NormalizedSeriesPoint = {
  label: string;
  value: number;
};

function normalizeSeriesPoints(
  points: TimeSeriesPoint[],
  options?: { sortByTime?: boolean },
): NormalizedSeriesPoint[] {
  const normalized = points
    .map((point, index) => {
      const value = toFiniteNumber(point?.value);
      if (value == null) {
        return null;
      }
      const rawLabel = normalizeLabel(point?.label, index);
      const parsedLabelTs = options?.sortByTime ? timestampFromLabel(rawLabel) : null;
      const label =
        options?.sortByTime && parsedLabelTs == null ? String(index + 1) : rawLabel;
      return {
        label,
        value,
        sortKey: parsedLabelTs,
        sourceIndex: index,
      };
    })
    .filter(
      (
        point,
      ): point is { label: string; value: number; sortKey: number | null; sourceIndex: number } =>
        point != null,
    );

  const ordered = options?.sortByTime ? resolveTimeOrder(normalized) : normalized;
  return ordered.map(({ label, value }) => ({ label, value }));
}

type NormalizedBacktestPoint = {
  label: string;
  pnl: number;
  var_hist: number | null;
  var_garch: number | null;
  var_fhs: number | null;
  var_alpha: number | null;
};

function normalizeBacktestPoints(points: BacktestSeriesPoint[]): NormalizedBacktestPoint[] {
  const normalized = points
    .map((point, index) => {
      const pnl = toFiniteNumber(point?.pnl);
      if (pnl == null) {
        return null;
      }
      const rawLabel = normalizeLabel(point?.label, index);
      const parsedLabelTs = timestampFromLabel(rawLabel);
      const label = parsedLabelTs == null ? String(index + 1) : rawLabel;
      return {
        label,
        pnl,
        var_hist: toFiniteNumber(point?.var_hist),
        var_garch: toFiniteNumber(point?.var_garch),
        var_fhs: toFiniteNumber(point?.var_fhs),
        var_alpha: toFiniteNumber(point?.var_alpha),
        sortKey: parsedLabelTs,
        sourceIndex: index,
      };
    })
    .filter(
      (
        point,
      ): point is {
        label: string;
        pnl: number;
        var_hist: number | null;
        var_garch: number | null;
        var_fhs: number | null;
        var_alpha: number | null;
        sortKey: number | null;
        sourceIndex: number;
      } => point != null,
    );

  const ordered = resolveTimeOrder(normalized);
  return ordered.map(({ label, pnl, var_hist, var_garch, var_fhs, var_alpha }) => ({
    label,
    pnl,
    var_hist,
    var_garch,
    var_fhs,
    var_alpha,
  }));
}

type GroupedSeriesInput = {
  name: string;
  data: Array<number | null | undefined>;
  color: string;
};

type NormalizedGroupedSeries = {
  name: string;
  data: Array<number | null>;
  color: string;
};

function normalizeGroupedSeries(
  labels: string[],
  series: GroupedSeriesInput[],
): {
  labels: string[];
  series: NormalizedGroupedSeries[];
  values: number[];
} {
  const maxDataLength = series.reduce((max, item) => Math.max(max, item.data.length), 0);
  const length = Math.max(labels.length, maxDataLength);
  if (length === 0) {
    return { labels: [], series: [], values: [] };
  }

  const normalizedLabels = Array.from({ length }, (_, index) => normalizeLabel(labels[index], index));
  const normalizedSeries = series.map((item) => ({
    name: item.name,
    color: item.color,
    data: Array.from({ length }, (_, index) => toFiniteNumber(item.data[index])),
  }));

  const validIndexes = normalizedLabels
    .map((_, index) => index)
    .filter((index) => normalizedSeries.some((item) => item.data[index] != null));

  const filteredLabels = validIndexes.map((index) => normalizedLabels[index]);
  const filteredSeries = normalizedSeries.map((item) => ({
    name: item.name,
    color: item.color,
    data: validIndexes.map((index) => item.data[index]),
  }));
  const values = filteredSeries.flatMap((item) =>
    item.data.filter((value): value is number => value != null),
  );

  return {
    labels: filteredLabels,
    series: filteredSeries,
    values,
  };
}

function resolveMode(count: number, preferred?: ChartMode): ChartMode {
  if (preferred) {
    return preferred;
  }
  if (count <= 5) {
    return "sparse";
  }
  if (count >= 90) {
    return "dense";
  }
  return "standard";
}

/* ------------------------------------------------------------------ */
/*  Shared building blocks                                            */
/* ------------------------------------------------------------------ */

/** Binance-style crosshair tooltip with backdrop blur */
function sharedTooltip() {
  return {
    trigger: "axis" as const,
    confine: true,
    axisPointer: {
      type: "cross" as const,
      crossStyle: { color: "rgba(234,236,239,0.2)", width: 1, type: "dashed" as const },
      lineStyle: { color: "rgba(240,185,11,0.35)", width: 1, type: "dashed" as const },
      label: {
        show: true,
        backgroundColor: "rgba(24,29,39,0.92)",
        borderColor: "rgba(240,185,11,0.25)",
        borderWidth: 1,
        color: "#eaecef",
        fontSize: 10,
        fontFamily: "IBM Plex Mono, monospace",
        padding: [4, 8],
      },
    },
    backgroundColor: "rgba(14,18,23,0.94)",
    borderColor: "rgba(255,255,255,0.08)",
    borderWidth: 1,
    textStyle: { color: "#eaecef", fontSize: 11, fontFamily: "'Inter', sans-serif" },
    extraCssText:
      "backdrop-filter: blur(20px); -webkit-backdrop-filter: blur(20px); box-shadow: 0 8px 32px rgba(0,0,0,0.55), 0 0 1px rgba(255,255,255,0.1); border-radius: 8px; padding: 10px 14px;",
  };
}

function gridForMode(mode: ChartMode) {
  switch (mode) {
    case "sparse":
      return { left: 12, right: 12, top: 16, bottom: 20, containLabel: true };
    case "dense":
    case "trace":
      return { left: 16, right: 20, top: 24, bottom: 58, containLabel: true };
    case "comparison":
      return { left: 16, right: 16, top: 34, bottom: 42, containLabel: true };
    default:
      return { left: 14, right: 14, top: 20, bottom: 36, containLabel: true };
  }
}

function xAxisForMode(count: number, mode: ChartMode, labels: string[]) {
  const rotate =
    mode === "dense" || mode === "trace"
      ? count > 24
        ? 28
        : 0
      : count > 16
        ? 20
        : 0;

  return {
    type: "category" as const,
    data: labels,
    axisLabel: {
      color: textSecondary,
      fontSize: mode === "sparse" ? 11 : 10,
      fontFamily: "IBM Plex Mono, monospace",
      hideOverlap: true,
      interval: mode === "sparse" ? 0 : count > 40 ? "auto" : 0,
      rotate,
      margin: 12,
    },
    axisTick: { show: false },
    axisLine: { lineStyle: { color: axisStroke } },
    splitLine: { show: false },
  };
}

function formatAxisValue(value: number | string): string {
  const num = Number(value);
  if (!Number.isFinite(num)) {
    return String(value);
  }
  const abs = Math.abs(num);
  if (abs >= 1_000_000) {
    return `${(num / 1_000_000).toFixed(1)}M`;
  }
  if (abs >= 1_000) {
    return `${(num / 1_000).toFixed(1)}k`;
  }
  if (abs >= 100) {
    return num.toFixed(1);
  }
  if (abs >= 1) {
    return num.toFixed(2);
  }
  return num.toFixed(3);
}

function yAxisBase() {
  return {
    type: "value" as const,
    splitNumber: 4,
    axisLabel: {
      color: textSecondary,
      fontSize: 10,
      fontFamily: "IBM Plex Mono, monospace",
      margin: 12,
      formatter: formatAxisValue,
    },
    splitLine: {
      lineStyle: { color: gridLine, type: "dashed" as const, width: 1 },
    },
    axisLine: { show: false },
    axisTick: { show: false },
  };
}

function zoomForMode(count: number, mode: ChartMode) {
  if (mode === "sparse" || count <= 12) {
    return [];
  }

  return [
    {
      type: "inside",
      zoomLock: mode !== "dense",
      moveOnMouseMove: true,
      throttle: 50,
    },
    {
      type: "slider",
      height: 20,
      bottom: 2,
      borderColor: "transparent",
      backgroundColor: "rgba(255,255,255,0.02)",
      fillerColor: "rgba(240,185,11,0.12)",
      borderRadius: 4,
      handleIcon: "path://M10.7,11.9v-1.3H9.3v1.3c-4.9,0.3-8.8,4.4-8.8,9.4c0,5,3.9,9.1,8.8,9.4v1.3h1.3v-1.3c4.9-0.3,8.8-4.4,8.8-9.4C19.5,16.3,15.6,12.2,10.7,11.9z",
      handleSize: "60%",
      handleStyle: {
        color: "#f0b90b",
        borderColor: "rgba(240,185,11,0.4)",
        borderWidth: 1,
        shadowBlur: 6,
        shadowColor: "rgba(240,185,11,0.2)",
      },
      dataBackground: {
        lineStyle: { color: "rgba(255,255,255,0.06)", width: 1 },
        areaStyle: { color: "rgba(255,255,255,0.02)" },
      },
      selectedDataBackground: {
        lineStyle: { color: "rgba(240,185,11,0.3)", width: 1 },
        areaStyle: { color: "rgba(240,185,11,0.06)" },
      },
      textStyle: { color: textSecondary, fontSize: 9, fontFamily: "IBM Plex Mono, monospace" },
      brushSelect: false,
      showDetail: false,
    },
  ];
}

function resolveBarConfig(
  colorOrConfig: string | BarChartConfig | undefined,
  config?: BarChartConfig,
) {
  if (typeof colorOrConfig === "string" || colorOrConfig == null) {
    return {
      color: colorOrConfig ?? CHART_PALETTE.gold,
      negativeColor: config?.negativeColor,
      mode: config?.mode,
    };
  }

  return {
    color: colorOrConfig.color ?? CHART_PALETTE.gold,
    negativeColor: colorOrConfig.negativeColor,
    mode: colorOrConfig.mode,
  };
}

/** Shared legend style — Binance compact pills */
function legendStyle(position: "top" | "bottom" = "bottom", extraTop?: number) {
  return {
    ...(position === "bottom" ? { bottom: 22 } : { top: extraTop ?? 2 }),
    textStyle: { color: textSecondary, fontSize: 10, fontFamily: "'Inter', sans-serif" },
    itemGap: 18,
    itemWidth: 16,
    itemHeight: 3,
    icon: "roundRect",
    inactiveColor: "rgba(255,255,255,0.15)",
    inactiveBorderWidth: 0,
  };
}

/** Glow shadow for line series */
function lineGlow(hex: string, intensity = 0.3) {
  return {
    shadowBlur: 10,
    shadowColor: rgba(hex, intensity),
    shadowOffsetY: 4,
  };
}

/* ------------------------------------------------------------------ */
/*  Chart option builders                                             */
/* ------------------------------------------------------------------ */

export function makeBarOption(
  points: TimeSeriesPoint[],
  colorOrConfig?: string | BarChartConfig,
  maybeConfig?: BarChartConfig,
) {
  const normalizedPoints = normalizeSeriesPoints(points);
  const count = normalizedPoints.length;
  const config = resolveBarConfig(colorOrConfig, maybeConfig);
  const mode = resolveMode(count, config.mode);
  const labels = normalizedPoints.map((point) => formatCategoryLabel(point.label, mode));
  const palette = {
    positive: config.color,
    negative: config.negativeColor ?? CHART_PALETTE.red,
  };
  const yAxis = yAxisForValues(
    normalizedPoints.map((point) => point.value),
    { includeZero: true },
  );

  return {
    animationDuration: 600,
    animationDurationUpdate: 400,
    animationEasing: "cubicOut",
    grid: gridForMode(mode),
    tooltip: sharedTooltip(),
    xAxis: xAxisForMode(count, mode, labels),
    yAxis,
    dataZoom: zoomForMode(count, mode),
    series: [
      {
        type: "bar",
        data: normalizedPoints.map((point) => ({
          value: roundValue(point.value),
          itemStyle: {
            color: point.value < 0 ? barGradient(palette.negative) : barGradient(palette.positive),
            borderColor: point.value < 0 ? rgba(palette.negative, 0.5) : rgba(palette.positive, 0.5),
            borderWidth: 1,
          },
        })),
        barMaxWidth: mode === "sparse" ? 56 : 36,
        barMinHeight: mode === "sparse" ? 6 : 0,
        barCategoryGap: mode === "sparse" ? "36%" : "28%",
        emphasis: {
          focus: "series",
          itemStyle: {
            shadowBlur: 12,
            shadowColor: rgba(palette.positive, 0.3),
          },
        },
        showBackground: true,
        backgroundStyle: {
          color: "rgba(255,255,255,0.015)",
          borderRadius: [4, 4, 0, 0],
        },
        label:
          mode === "sparse"
            ? {
                show: true,
                position: "top",
                distance: 8,
                color: textPrimary,
                fontFamily: "IBM Plex Mono, monospace",
                fontSize: 11,
                fontWeight: 600,
                formatter: "{c}",
              }
            : undefined,
        itemStyle: {
          borderRadius: [4, 4, 0, 0],
        },
      },
    ],
  };
}

export function makeBacktestOption(points: BacktestSeriesPoint[]) {
  const normalizedPoints = normalizeBacktestPoints(points);
  const count = normalizedPoints.length;
  const mode = resolveMode(count, "trace");
  const labels = normalizedPoints.map((point) => formatCategoryLabel(point.label, mode));
  const yAxis = yAxisForValues(
    normalizedPoints.flatMap((point) => [
      point.pnl,
      point.var_hist,
      point.var_garch,
      point.var_fhs,
      point.var_alpha,
    ]),
    { includeZero: true },
  );

  const alphaSeriesData = normalizedPoints.map((point) =>
    point.var_alpha == null ? null : roundValue(point.var_alpha),
  );
  const hasAlphaSeries = alphaSeriesData.some((value) => value != null);

  const series: Array<Record<string, unknown>> = [
    {
      name: "PnL",
      type: "line",
      smooth: false,
      symbol: "none",
      lineStyle: { width: 2, color: CHART_PALETTE.blue, ...lineGlow(CHART_PALETTE.blue, 0.25) },
      areaStyle: { color: areaGradient(CHART_PALETTE.blue) },
      data: normalizedPoints.map((point) => roundValue(point.pnl)),
      z: 4,
      markLine: {
        silent: true,
        symbol: "none",
        lineStyle: { color: "rgba(255,255,255,0.12)", type: "dashed", width: 1 },
        data: [{ yAxis: 0 }],
        label: { show: false },
      },
    },
    {
      name: "Hist VaR",
      type: "line",
      smooth: 0.3,
      symbol: "none",
      lineStyle: { width: 1.5, color: CHART_PALETTE.gold, ...lineGlow(CHART_PALETTE.gold, 0.2) },
      data: normalizedPoints.map((point) =>
        point.var_hist == null ? null : roundValue(point.var_hist),
      ),
      z: 3,
    },
    {
      name: "GARCH VaR",
      type: "line",
      smooth: 0.3,
      symbol: "none",
      lineStyle: { width: 1.5, color: CHART_PALETTE.green, ...lineGlow(CHART_PALETTE.green, 0.2) },
      data: normalizedPoints.map((point) =>
        point.var_garch == null ? null : roundValue(point.var_garch),
      ),
      z: 2,
    },
    {
      name: "FHS VaR",
      type: "line",
      smooth: 0.3,
      symbol: "none",
      lineStyle: { width: 1.5, color: CHART_PALETTE.red, ...lineGlow(CHART_PALETTE.red, 0.2) },
      data: normalizedPoints.map((point) =>
        point.var_fhs == null ? null : roundValue(point.var_fhs),
      ),
      z: 1,
    },
  ];
  if (hasAlphaSeries) {
    series.push({
      name: "Alpha VaR",
      type: "line",
      smooth: 0.35,
      symbol: "none",
      lineStyle: { width: 1.5, color: CHART_PALETTE.teal, ...lineGlow(CHART_PALETTE.teal, 0.22) },
      data: alphaSeriesData,
      z: 2,
    });
  }

  return {
    animationDuration: 800,
    animationDurationUpdate: 400,
    animationEasing: "cubicOut",
    grid: gridForMode(mode),
    tooltip: sharedTooltip(),
    legend: legendStyle("bottom"),
    xAxis: xAxisForMode(count, mode, labels),
    yAxis,
    dataZoom: zoomForMode(count, mode),
    series,
  };
}

export function makeLineOption(
  points: TimeSeriesPoint[],
  color: string = CHART_PALETTE.gold,
  config?: ChartConfig,
) {
  const normalizedPoints = normalizeSeriesPoints(points, { sortByTime: true });
  const count = normalizedPoints.length;
  const mode = resolveMode(count, config?.mode);
  const labels = normalizedPoints.map((point) => formatCategoryLabel(point.label, mode));
  const yAxis = yAxisForValues(normalizedPoints.map((point) => point.value));

  return {
    animationDuration: 700,
    animationDurationUpdate: 400,
    animationEasing: "cubicOut",
    grid: gridForMode(mode),
    tooltip: sharedTooltip(),
    xAxis: xAxisForMode(count, mode, labels),
    yAxis,
    dataZoom: zoomForMode(count, mode),
    series: [
      {
        type: "line",
        smooth: mode !== "trace" ? 0.35 : false,
        symbol: "none",
        sampling: "lttb",
        lineStyle: {
          width: mode === "sparse" ? 2.5 : 2,
          color,
          ...lineGlow(color, 0.3),
        },
        areaStyle: { color: areaGradient(color) },
        data: normalizedPoints.map((point) => roundValue(point.value)),
        markLine: {
          silent: true,
          symbol: "none",
          lineStyle: { color: rgba(color, 0.28), type: "dashed", width: 1 },
          data: [{ type: "average", name: "Avg" }],
          label: {
            color: textSecondary,
            fontSize: 9,
            fontFamily: "IBM Plex Mono, monospace",
            formatter: "avg: {c}",
          },
        },
      },
    ],
  };
}

export function makeGroupedBarOption(
  labels: string[],
  series: Array<{ name: string; data: Array<number | null | undefined>; color: string }>,
  config?: GroupedBarConfig,
) {
  const normalized = normalizeGroupedSeries(labels, series);
  const count = normalized.labels.length;
  const mode = resolveMode(count, config?.mode ?? "comparison");
  const formattedLabels = normalized.labels.map((label) => formatCategoryLabel(label, mode));
  const yAxis = yAxisForValues(normalized.values, { includeZero: true });

  return {
    animationDuration: 650,
    animationDurationUpdate: 400,
    animationEasing: "cubicOut",
    grid: gridForMode(mode),
    tooltip: sharedTooltip(),
    legend: legendStyle("top"),
    xAxis: xAxisForMode(count, mode, formattedLabels),
    yAxis,
    dataZoom: zoomForMode(count, mode),
    series: normalized.series.map((item) => ({
      name: item.name,
      type: "bar",
      data: item.data.map((value) => (value == null ? null : roundValue(value))),
      barMaxWidth: mode === "sparse" ? 44 : 30,
      barCategoryGap: mode === "sparse" ? "38%" : "32%",
      emphasis: {
        focus: "series",
        itemStyle: {
          shadowBlur: 10,
          shadowColor: rgba(item.color, 0.3),
        },
      },
      showBackground: true,
      backgroundStyle: {
        color: "rgba(255,255,255,0.012)",
        borderRadius: [3, 3, 0, 0],
      },
      itemStyle: {
        color: barGradient(item.color),
        borderColor: rgba(item.color, 0.4),
        borderWidth: 1,
        borderRadius: [4, 4, 0, 0],
      },
    })),
  };
}
