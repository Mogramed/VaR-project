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

function formatCategoryLabel(value: string, mode: ChartMode) {
  const normalized = String(value ?? "").replace(/[_-]+/g, " ").trim();

  if (mode !== "sparse" && mode !== "comparison") {
    return normalized;
  }

  const words = normalized.split(/\s+/).filter(Boolean);
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

function yAxisBase() {
  return {
    type: "value" as const,
    splitNumber: 4,
    axisLabel: {
      color: textSecondary,
      fontSize: 10,
      fontFamily: "IBM Plex Mono, monospace",
      margin: 12,
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
  const count = points.length;
  const config = resolveBarConfig(colorOrConfig, maybeConfig);
  const mode = resolveMode(count, config.mode);
  const labels = points.map((point) => formatCategoryLabel(point.label, mode));
  const palette = {
    positive: config.color,
    negative: config.negativeColor ?? CHART_PALETTE.red,
  };

  return {
    animationDuration: 600,
    animationDurationUpdate: 400,
    animationEasing: "cubicOut",
    grid: gridForMode(mode),
    tooltip: sharedTooltip(),
    xAxis: xAxisForMode(count, mode, labels),
    yAxis: yAxisBase(),
    dataZoom: zoomForMode(count, mode),
    series: [
      {
        type: "bar",
        data: points.map((point) => ({
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
  const count = points.length;
  const mode = resolveMode(count, "trace");
  const labels = points.map((point) => formatCategoryLabel(point.label, mode));

  return {
    animationDuration: 800,
    animationDurationUpdate: 400,
    animationEasing: "cubicOut",
    grid: gridForMode(mode),
    tooltip: sharedTooltip(),
    legend: legendStyle("bottom"),
    xAxis: xAxisForMode(count, mode, labels),
    yAxis: yAxisBase(),
    dataZoom: zoomForMode(count, mode),
    series: [
      {
        name: "PnL",
        type: "line",
        smooth: false,
        symbol: "none",
        lineStyle: { width: 2, color: CHART_PALETTE.blue, ...lineGlow(CHART_PALETTE.blue, 0.25) },
        areaStyle: { color: areaGradient(CHART_PALETTE.blue) },
        data: points.map((point) => roundValue(point.pnl)),
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
        data: points.map((point) =>
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
        data: points.map((point) =>
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
        data: points.map((point) =>
          point.var_fhs == null ? null : roundValue(point.var_fhs),
        ),
        z: 1,
      },
    ],
  };
}

export function makeLineOption(
  points: TimeSeriesPoint[],
  color: string = CHART_PALETTE.gold,
  config?: ChartConfig,
) {
  const count = points.length;
  const mode = resolveMode(count, config?.mode);
  const labels = points.map((point) => formatCategoryLabel(point.label, mode));

  return {
    animationDuration: 700,
    animationDurationUpdate: 400,
    animationEasing: "cubicOut",
    grid: gridForMode(mode),
    tooltip: sharedTooltip(),
    xAxis: xAxisForMode(count, mode, labels),
    yAxis: yAxisBase(),
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
        data: points.map((point) => roundValue(point.value)),
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
  series: Array<{ name: string; data: number[]; color: string }>,
  config?: GroupedBarConfig,
) {
  const count = labels.length;
  const mode = resolveMode(count, config?.mode ?? "comparison");
  const formattedLabels = labels.map((label) => formatCategoryLabel(label, mode));

  return {
    animationDuration: 650,
    animationDurationUpdate: 400,
    animationEasing: "cubicOut",
    grid: gridForMode(mode),
    tooltip: sharedTooltip(),
    legend: legendStyle("top"),
    xAxis: xAxisForMode(count, mode, formattedLabels),
    yAxis: yAxisBase(),
    dataZoom: zoomForMode(count, mode),
    series: series.map((item) => ({
      name: item.name,
      type: "bar",
      data: item.data.map((value) => roundValue(value)),
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
