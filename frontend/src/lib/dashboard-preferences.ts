"use client";

import { useCallback, useEffect, useMemo, useState } from "react";

export const OVERVIEW_WIDGET_IDS = [
  "risk-strip",
  "account-ticker",
  "capital-chart",
  "alert-posture",
  "desk-alignment",
  "data-integrity",
  "portfolio-slices",
] as const;

export type OverviewWidgetId = (typeof OVERVIEW_WIDGET_IDS)[number];

export const OVERVIEW_WIDGET_LABELS: Record<OverviewWidgetId, string> = {
  "risk-strip": "Risk metrics (VaR / ES / Stress)",
  "account-ticker": "Account ticker (Balance / Equity / Profit)",
  "capital-chart": "Capital by portfolio chart",
  "alert-posture": "Alert posture",
  "desk-alignment": "Desk alignment and live posture",
  "data-integrity": "Data integrity checks",
  "portfolio-slices": "Portfolio slices",
};

export const PAGE_IDS = [
  "overview",
  "live",
  "incidents",
  "universe",
  "models",
  "attribution",
  "capital",
  "decisions",
  "execution",
  "stress",
  "blotter",
  "reports",
] as const;

export type PageId = (typeof PAGE_IDS)[number];

export const PAGE_LABELS: Record<PageId, string> = {
  overview: "Overview",
  live: "MT5 Ops",
  incidents: "Incidents",
  universe: "Universe",
  models: "Models",
  attribution: "Attribution",
  capital: "Capital",
  decisions: "Decisions",
  execution: "Execution",
  stress: "Stress",
  blotter: "Blotter",
  reports: "Reports",
};

export const HORIZON_OPTIONS = ["1d", "5d", "10d"] as const;
export type Horizon = (typeof HORIZON_OPTIONS)[number];

export const MODEL_OPTIONS = ["auto", "hist", "param", "mc", "ewma", "garch", "fhs"] as const;
export type ModelOption = (typeof MODEL_OPTIONS)[number];

export const MODEL_LABELS: Record<ModelOption, string> = {
  auto: "Auto (champion)",
  hist: "Historical",
  param: "Parametric",
  mc: "Monte Carlo",
  ewma: "EWMA",
  garch: "GARCH",
  fhs: "FHS",
};

export const PRESET_NAMES = ["trading", "risk-monitoring", "minimal"] as const;
export type PresetName = (typeof PRESET_NAMES)[number];
export type ActivePresetName = PresetName | "custom";

export interface DashboardPreferences {
  visibleWidgets: OverviewWidgetId[];
  visiblePages: PageId[];
  symbolFilter: string;
  horizon: Horizon;
  model: ModelOption;
  activePreset: ActivePresetName;
}

type DashboardPreferencesWithoutPreset = Omit<DashboardPreferences, "activePreset">;
type DashboardPreferencesStoragePayload = {
  version?: number;
  data?: Partial<DashboardPreferences>;
};

const DASHBOARD_PREFERENCES_STORAGE_VERSION = 2;
export const DASHBOARD_PREFERENCES_STORAGE_KEY = "desk:dashboard-preferences";

const MODEL_ALIASES: Record<string, ModelOption> = {
  auto: "auto",
  hist: "hist",
  historical: "hist",
  param: "param",
  parametric: "param",
  mc: "mc",
  monte_carlo: "mc",
  "monte-carlo": "mc",
  ewma: "ewma",
  garch: "garch",
  fhs: "fhs",
  cornish_fisher: "fhs",
  "cornish-fisher": "fhs",
};

const PRESET_BASE: Record<PresetName, DashboardPreferencesWithoutPreset> = {
  trading: {
    visibleWidgets: [...OVERVIEW_WIDGET_IDS],
    visiblePages: [...PAGE_IDS],
    symbolFilter: "",
    horizon: "1d",
    model: "auto",
  },
  "risk-monitoring": {
    visibleWidgets: [
      "risk-strip",
      "capital-chart",
      "alert-posture",
      "desk-alignment",
      "data-integrity",
    ],
    visiblePages: [
      "overview",
      "incidents",
      "models",
      "attribution",
      "capital",
      "stress",
      "reports",
    ],
    symbolFilter: "",
    horizon: "10d",
    model: "auto",
  },
  minimal: {
    visibleWidgets: ["risk-strip", "account-ticker", "capital-chart"],
    visiblePages: ["overview", "live", "capital", "execution", "blotter"],
    symbolFilter: "",
    horizon: "1d",
    model: "auto",
  },
};

function cloneWithoutPreset(
  value: DashboardPreferencesWithoutPreset,
): DashboardPreferencesWithoutPreset {
  return {
    visibleWidgets: [...value.visibleWidgets],
    visiblePages: [...value.visiblePages],
    symbolFilter: value.symbolFilter,
    horizon: value.horizon,
    model: value.model,
  };
}

function buildPreset(name: PresetName): DashboardPreferences {
  return {
    ...cloneWithoutPreset(PRESET_BASE[name]),
    activePreset: name,
  };
}

export const PRESETS: Record<PresetName, DashboardPreferences> = {
  trading: buildPreset("trading"),
  "risk-monitoring": buildPreset("risk-monitoring"),
  minimal: buildPreset("minimal"),
};

export const DEFAULT_PREFERENCES: DashboardPreferences = PRESETS.trading;

function orderedUnique<T extends string>(values: T[], order: readonly T[]): T[] {
  const incoming = new Set(values);
  const result: T[] = [];
  for (const item of order) {
    if (incoming.has(item)) {
      result.push(item);
    }
  }
  return result;
}

function normalizeWidgetIds(values: unknown): OverviewWidgetId[] {
  const requested = Array.isArray(values)
    ? values
      .map((item) => String(item))
      .filter((item): item is OverviewWidgetId =>
        OVERVIEW_WIDGET_IDS.includes(item as OverviewWidgetId),
      )
    : [];
  const ordered = orderedUnique(requested, OVERVIEW_WIDGET_IDS);
  return ordered.length > 0
    ? ordered
    : [...DEFAULT_PREFERENCES.visibleWidgets];
}

function normalizePageIds(values: unknown): PageId[] {
  const requested = Array.isArray(values)
    ? values
      .map((item) => String(item))
      .filter((item): item is PageId => PAGE_IDS.includes(item as PageId))
    : [];
  const ordered = orderedUnique(requested, PAGE_IDS);
  const withOverview: PageId[] = ordered.includes("overview")
    ? ordered
    : ["overview", ...ordered];
  return withOverview.length > 0
    ? withOverview
    : [...DEFAULT_PREFERENCES.visiblePages];
}

function normalizeHorizon(value: unknown): Horizon {
  if (typeof value !== "string") {
    return DEFAULT_PREFERENCES.horizon;
  }
  const normalized = value.trim().toLowerCase();
  return HORIZON_OPTIONS.includes(normalized as Horizon)
    ? (normalized as Horizon)
    : DEFAULT_PREFERENCES.horizon;
}

function normalizeModel(value: unknown): ModelOption {
  if (typeof value !== "string") {
    return DEFAULT_PREFERENCES.model;
  }
  const normalized = value.trim().toLowerCase();
  return MODEL_ALIASES[normalized] ?? DEFAULT_PREFERENCES.model;
}

export function normalizeSymbolFilter(value: unknown): string {
  if (typeof value !== "string") {
    return "";
  }
  return value
    .trim()
    .toUpperCase()
    .replace(/\s*,\s*/g, ",")
    .replace(/\s+/g, " ");
}

export function symbolFilterTokens(filter: string): string[] {
  return normalizeSymbolFilter(filter)
    .split(/[,\s]+/)
    .map((item) => item.trim())
    .filter(Boolean);
}

export function symbolMatchesFilter(symbol: string | null | undefined, filter: string): boolean {
  const tokens = symbolFilterTokens(filter);
  if (tokens.length === 0) {
    return true;
  }
  if (symbol == null || String(symbol).trim() === "") {
    return false;
  }
  const normalized = String(symbol).trim().toUpperCase();
  return tokens.some((token) => normalized.includes(token));
}

function sameArray<T extends string>(left: readonly T[], right: readonly T[]): boolean {
  if (left.length !== right.length) {
    return false;
  }
  for (let index = 0; index < left.length; index += 1) {
    if (left[index] !== right[index]) {
      return false;
    }
  }
  return true;
}

function detectPreset(
  prefs: DashboardPreferencesWithoutPreset,
): ActivePresetName {
  for (const name of PRESET_NAMES) {
    const preset = PRESET_BASE[name];
    if (
      sameArray(prefs.visibleWidgets, preset.visibleWidgets)
      && sameArray(prefs.visiblePages, preset.visiblePages)
      && prefs.symbolFilter === preset.symbolFilter
      && prefs.horizon === preset.horizon
      && prefs.model === preset.model
    ) {
      return name;
    }
  }
  return "custom";
}

export function resolveModelPreference(
  model: ModelOption,
  fallbackModel: string | null | undefined,
): string | null {
  if (model !== "auto") {
    return model;
  }
  if (fallbackModel == null || String(fallbackModel).trim() === "") {
    return null;
  }
  return MODEL_ALIASES[String(fallbackModel).trim().toLowerCase()] ?? String(fallbackModel).trim().toLowerCase();
}

export function horizonDays(horizon: Horizon): number {
  if (horizon === "5d") {
    return 5;
  }
  if (horizon === "10d") {
    return 10;
  }
  return 1;
}

export function normalizeDashboardPreferences(
  value: Partial<DashboardPreferences> | null | undefined,
): DashboardPreferences {
  const normalized: DashboardPreferencesWithoutPreset = {
    visibleWidgets: normalizeWidgetIds(value?.visibleWidgets),
    visiblePages: normalizePageIds(value?.visiblePages),
    symbolFilter: normalizeSymbolFilter(value?.symbolFilter ?? ""),
    horizon: normalizeHorizon(value?.horizon),
    model: normalizeModel(value?.model),
  };
  return {
    ...normalized,
    activePreset: detectPreset(normalized),
  };
}

function loadPreferencesFromStorage(): DashboardPreferences {
  if (typeof window === "undefined") {
    return {
      ...cloneWithoutPreset(DEFAULT_PREFERENCES),
      activePreset: DEFAULT_PREFERENCES.activePreset,
    };
  }
  const raw = window.localStorage.getItem(DASHBOARD_PREFERENCES_STORAGE_KEY);
  if (!raw) {
    return {
      ...cloneWithoutPreset(DEFAULT_PREFERENCES),
      activePreset: DEFAULT_PREFERENCES.activePreset,
    };
  }
  try {
    const parsed = JSON.parse(raw) as DashboardPreferencesStoragePayload | Partial<DashboardPreferences>;
    const payload = (
      typeof parsed === "object"
      && parsed != null
      && "data" in parsed
      && parsed.data
      && typeof parsed.data === "object"
    )
      ? parsed.data
      : (parsed as Partial<DashboardPreferences>);
    return normalizeDashboardPreferences(payload);
  } catch {
    return {
      ...cloneWithoutPreset(DEFAULT_PREFERENCES),
      activePreset: DEFAULT_PREFERENCES.activePreset,
    };
  }
}

function savePreferencesToStorage(prefs: DashboardPreferences): void {
  if (typeof window === "undefined") {
    return;
  }
  const payload: DashboardPreferencesStoragePayload = {
    version: DASHBOARD_PREFERENCES_STORAGE_VERSION,
    data: prefs,
  };
  try {
    window.localStorage.setItem(DASHBOARD_PREFERENCES_STORAGE_KEY, JSON.stringify(payload));
  } catch {
    // Ignore quota errors.
  }
}

export function useDashboardPreferences() {
  const [prefs, setPrefsState] = useState<DashboardPreferences>(loadPreferencesFromStorage);

  useEffect(() => {
    savePreferencesToStorage(prefs);
  }, [prefs]);

  const setPrefs = useCallback((next: DashboardPreferences) => {
    setPrefsState(normalizeDashboardPreferences(next));
  }, []);

  const toggleWidget = useCallback((id: OverviewWidgetId) => {
    setPrefsState((prev) => {
      const nextWidgets = prev.visibleWidgets.includes(id)
        ? prev.visibleWidgets.filter((widgetId) => widgetId !== id)
        : [...prev.visibleWidgets, id];
      const safeWidgets = nextWidgets.length > 0 ? nextWidgets : [id];
      return normalizeDashboardPreferences({
        ...prev,
        visibleWidgets: safeWidgets,
      });
    });
  }, []);

  const togglePage = useCallback((id: PageId) => {
    setPrefsState((prev) => {
      if (id === "overview") {
        return prev;
      }
      const nextPages = prev.visiblePages.includes(id)
        ? prev.visiblePages.filter((pageId) => pageId !== id)
        : [...prev.visiblePages, id];
      return normalizeDashboardPreferences({
        ...prev,
        visiblePages: nextPages,
      });
    });
  }, []);

  const setSymbolFilter = useCallback((value: string) => {
    setPrefsState((prev) => normalizeDashboardPreferences({
      ...prev,
      symbolFilter: value,
    }));
  }, []);

  const setHorizon = useCallback((value: Horizon) => {
    setPrefsState((prev) => normalizeDashboardPreferences({
      ...prev,
      horizon: value,
    }));
  }, []);

  const setModel = useCallback((value: ModelOption) => {
    setPrefsState((prev) => normalizeDashboardPreferences({
      ...prev,
      model: value,
    }));
  }, []);

  const applyPreset = useCallback((name: PresetName) => {
    setPrefsState(buildPreset(name));
  }, []);

  const resetToDefault = useCallback(() => {
    setPrefsState(buildPreset("trading"));
  }, []);

  const isWidgetVisible = useCallback(
    (id: OverviewWidgetId) => prefs.visibleWidgets.includes(id),
    [prefs.visibleWidgets],
  );

  const isPageVisible = useCallback(
    (id: PageId) => prefs.visiblePages.includes(id),
    [prefs.visiblePages],
  );

  const matchesSymbol = useCallback(
    (symbol: string | null | undefined) => symbolMatchesFilter(symbol, prefs.symbolFilter),
    [prefs.symbolFilter],
  );

  const hasSymbolFilter = useMemo(
    () => symbolFilterTokens(prefs.symbolFilter).length > 0,
    [prefs.symbolFilter],
  );

  const preferredHorizonDays = useMemo(
    () => horizonDays(prefs.horizon),
    [prefs.horizon],
  );

  const resolvePreferredModel = useCallback(
    (fallbackModel: string | null | undefined) => resolveModelPreference(prefs.model, fallbackModel),
    [prefs.model],
  );

  return useMemo(
    () => ({
      prefs,
      setPrefs,
      toggleWidget,
      togglePage,
      setSymbolFilter,
      setHorizon,
      setModel,
      applyPreset,
      resetToDefault,
      isWidgetVisible,
      isPageVisible,
      matchesSymbol,
      hasSymbolFilter,
      preferredHorizonDays,
      resolvePreferredModel,
    }),
    [
      prefs,
      setPrefs,
      toggleWidget,
      togglePage,
      setSymbolFilter,
      setHorizon,
      setModel,
      applyPreset,
      resetToDefault,
      isWidgetVisible,
      isPageVisible,
      matchesSymbol,
      hasSymbolFilter,
      preferredHorizonDays,
      resolvePreferredModel,
    ],
  );
}

export type DashboardPreferencesAPI = ReturnType<typeof useDashboardPreferences>;
