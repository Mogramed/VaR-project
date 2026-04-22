"use client";

import { Check, ChevronDown, HelpCircle, RotateCcw, X } from "lucide-react";
import { useEffect, useMemo, useState } from "react";

import { cn } from "@/lib/utils";
import {
  HORIZON_OPTIONS,
  MODEL_LABELS,
  MODEL_OPTIONS,
  OVERVIEW_WIDGET_IDS,
  OVERVIEW_WIDGET_LABELS,
  PAGE_IDS,
  PAGE_LABELS,
  PRESET_NAMES,
  symbolFilterTokens,
  type DashboardPreferencesAPI,
  type Horizon,
  type ModelOption,
  type PresetName,
} from "@/lib/dashboard-preferences";

const PRESET_DESCRIPTIONS: Record<PresetName, string> = {
  trading: "All widgets and pages for active desk operations.",
  "risk-monitoring": "Risk-first view with long-horizon focus.",
  minimal: "Compact view with only critical widgets and pages.",
};

export function DashboardConfigPanel({
  open,
  onClose,
  api,
}: {
  open: boolean;
  onClose: () => void;
  api: DashboardPreferencesAPI;
}) {
  const { prefs } = api;
  const [helpOpen, setHelpOpen] = useState(false);
  const activeSymbolTokens = useMemo(
    () => symbolFilterTokens(prefs.symbolFilter),
    [prefs.symbolFilter],
  );

  useEffect(() => {
    if (!open) {
      return;
    }
    const onKeyDown = (event: KeyboardEvent) => {
      if (event.key === "Escape") {
        onClose();
      }
    };
    window.addEventListener("keydown", onKeyDown);
    return () => window.removeEventListener("keydown", onKeyDown);
  }, [open, onClose]);

  return (
    <>
      <div
        className={cn(
          "fixed inset-0 z-40 bg-black/40 transition-opacity",
          open ? "pointer-events-auto opacity-100" : "pointer-events-none opacity-0",
        )}
        onClick={onClose}
      />

      <aside
        className={cn(
          "fixed inset-y-0 right-0 z-50 flex w-[340px] max-w-[90vw] flex-col border-l border-[var(--color-border)] bg-[var(--color-bg)] transition-transform duration-200",
          open ? "translate-x-0" : "translate-x-full",
        )}
        aria-label="Configure dashboard view"
      >
        <div className="flex h-11 items-center justify-between border-b border-[var(--color-border)] px-4">
          <span className="text-xs font-medium text-[var(--color-text-soft)]">
            Configure my view
          </span>
          <div className="flex items-center gap-1">
            <button
              type="button"
              onClick={() => setHelpOpen((prev) => !prev)}
              title="Help"
              className="flex size-6 items-center justify-center rounded-[var(--radius-sm)] text-[var(--color-text-muted)] transition-colors hover:bg-[var(--color-surface)] hover:text-[var(--color-text)]"
            >
              <HelpCircle className="size-3.5" />
            </button>
            <button
              type="button"
              onClick={() => api.resetToDefault()}
              title="Reset to default"
              className="flex size-6 items-center justify-center rounded-[var(--radius-sm)] text-[var(--color-text-muted)] transition-colors hover:bg-[var(--color-surface)] hover:text-[var(--color-text)]"
            >
              <RotateCcw className="size-3.5" />
            </button>
            <button
              type="button"
              onClick={onClose}
              title="Close"
              className="flex size-6 items-center justify-center rounded-[var(--radius-sm)] text-[var(--color-text-muted)] transition-colors hover:bg-[var(--color-surface)] hover:text-[var(--color-text)]"
            >
              <X className="size-3.5" />
            </button>
          </div>
        </div>

        <div className="flex-1 space-y-5 overflow-y-auto p-4">
          {helpOpen ? (
            <div className="rounded-[var(--radius-md)] border border-[var(--color-accent)]/20 bg-[var(--color-accent-soft)] p-3 text-[11px] leading-relaxed text-[var(--color-text-soft)]">
              <p className="mb-1.5 font-semibold text-[var(--color-accent)]">Quick guide</p>
              <ul className="list-disc space-y-1 pl-3.5">
                <li>Toggle overview widgets on or off.</li>
                <li>Hide pages you do not use from the sidebar.</li>
                <li>Set global filters (symbol, horizon, model).</li>
                <li>Start from a preset and fine tune if needed.</li>
                <li>Preferences are auto-saved and restored after refresh.</li>
              </ul>
            </div>
          ) : null}

          <ConfigSection title="Presets">
            <div className="space-y-1.5">
              {PRESET_NAMES.map((name) => (
                <button
                  key={name}
                  type="button"
                  onClick={() => api.applyPreset(name)}
                  className={cn(
                    "flex w-full items-start gap-2.5 rounded-[var(--radius-md)] border px-3 py-2 text-left transition-colors",
                    prefs.activePreset === name
                      ? "border-[var(--color-accent)] bg-[var(--color-accent-soft)]"
                      : "border-[var(--color-border)] bg-[var(--color-surface)] hover:border-[var(--color-border-strong)]",
                  )}
                >
                  <div
                    className={cn(
                      "mt-0.5 flex size-4 shrink-0 items-center justify-center rounded-full border",
                      prefs.activePreset === name
                        ? "border-[var(--color-accent)] bg-[var(--color-accent)] text-[var(--color-bg)]"
                        : "border-[var(--color-border-strong)] bg-transparent",
                    )}
                  >
                    {prefs.activePreset === name ? <Check className="size-2.5" /> : null}
                  </div>
                  <div className="min-w-0">
                    <div className="text-xs font-medium capitalize text-[var(--color-text)]">
                      {name.replace(/-/g, " ")}
                    </div>
                    <div className="text-[10px] text-[var(--color-text-muted)]">
                      {PRESET_DESCRIPTIONS[name]}
                    </div>
                  </div>
                </button>
              ))}
              {prefs.activePreset === "custom" ? (
                <div className="flex items-center gap-2 rounded-[var(--radius-md)] border border-dashed border-[var(--color-border-strong)] bg-[var(--color-surface)] px-3 py-2">
                  <div className="flex size-4 shrink-0 items-center justify-center rounded-full border border-[var(--color-accent)] bg-[var(--color-accent)] text-[var(--color-bg)]">
                    <Check className="size-2.5" />
                  </div>
                  <span className="text-xs font-medium text-[var(--color-text)]">
                    Custom configuration
                  </span>
                </div>
              ) : null}
            </div>
          </ConfigSection>

          <ConfigSection title="Global filters">
            <label className="block">
              <span className="text-[10px] font-medium uppercase tracking-wider text-[var(--color-text-muted)]">
                Symbol filter
              </span>
              <input
                type="text"
                value={prefs.symbolFilter}
                onChange={(event) => api.setSymbolFilter(event.target.value)}
                placeholder="All symbols (example: EURUSD,USDJPY)"
                className="mt-1 block w-full rounded-[var(--radius-sm)] border border-[var(--color-border)] bg-[var(--color-surface)] px-2.5 py-1.5 text-xs text-[var(--color-text)] placeholder-[var(--color-text-muted)] outline-none transition-colors focus:border-[var(--color-accent)]"
              />
              {activeSymbolTokens.length > 0 ? (
                <p className="mt-1 text-[10px] text-[var(--color-text-muted)]">
                  Active: {activeSymbolTokens.join(" | ")}
                </p>
              ) : null}
            </label>

            <label className="block">
              <span className="text-[10px] font-medium uppercase tracking-wider text-[var(--color-text-muted)]">
                Risk horizon
              </span>
              <SelectField
                value={prefs.horizon}
                options={HORIZON_OPTIONS.map((horizon) => ({ value: horizon, label: horizon }))}
                onChange={(value) => api.setHorizon(value as Horizon)}
              />
            </label>

            <label className="block">
              <span className="text-[10px] font-medium uppercase tracking-wider text-[var(--color-text-muted)]">
                VaR model
              </span>
              <SelectField
                value={prefs.model}
                options={MODEL_OPTIONS.map((model) => ({ value: model, label: MODEL_LABELS[model] }))}
                onChange={(value) => api.setModel(value as ModelOption)}
              />
            </label>
          </ConfigSection>

          <ConfigSection title="Overview widgets">
            <div className="space-y-1">
              {OVERVIEW_WIDGET_IDS.map((id) => (
                <ToggleRow
                  key={id}
                  label={OVERVIEW_WIDGET_LABELS[id]}
                  checked={prefs.visibleWidgets.includes(id)}
                  onChange={() => api.toggleWidget(id)}
                />
              ))}
            </div>
          </ConfigSection>

          <ConfigSection title="Sidebar pages">
            <div className="space-y-1">
              {PAGE_IDS.map((id) => (
                <ToggleRow
                  key={id}
                  label={PAGE_LABELS[id]}
                  checked={prefs.visiblePages.includes(id)}
                  onChange={() => api.togglePage(id)}
                  disabled={id === "overview"}
                />
              ))}
            </div>
          </ConfigSection>
        </div>

        <div className="border-t border-[var(--color-border)] px-4 py-2.5">
          <p className="text-[10px] text-[var(--color-text-muted)]">
            Preferences saved automatically.
            {prefs.activePreset === "custom"
              ? " Custom configuration active."
              : ` Preset: ${prefs.activePreset}.`}
          </p>
        </div>
      </aside>
    </>
  );
}

function ConfigSection({ title, children }: { title: string; children: React.ReactNode }) {
  return (
    <section>
      <h4 className="mb-2 text-[10px] font-semibold uppercase tracking-wider text-[var(--color-text-muted)]">
        {title}
      </h4>
      <div className="space-y-2.5">{children}</div>
    </section>
  );
}

function ToggleRow({
  label,
  checked,
  onChange,
  disabled,
}: {
  label: string;
  checked: boolean;
  onChange: () => void;
  disabled?: boolean;
}) {
  return (
    <button
      type="button"
      onClick={disabled ? undefined : onChange}
      disabled={disabled}
      className={cn(
        "flex w-full items-center gap-2.5 rounded-[var(--radius-sm)] px-2 py-1.5 text-left transition-colors",
        disabled
          ? "cursor-not-allowed opacity-50"
          : "hover:bg-[var(--color-surface-hover)]",
      )}
    >
      <div
        className={cn(
          "flex size-4 shrink-0 items-center justify-center rounded-[3px] border transition-colors",
          checked
            ? "border-[var(--color-accent)] bg-[var(--color-accent)] text-[var(--color-bg)]"
            : "border-[var(--color-border-strong)] bg-transparent",
        )}
      >
        {checked ? <Check className="size-2.5" /> : null}
      </div>
      <span className="text-xs text-[var(--color-text-soft)]">{label}</span>
    </button>
  );
}

function SelectField({
  value,
  options,
  onChange,
}: {
  value: string;
  options: Array<{ value: string; label: string }>;
  onChange: (value: string) => void;
}) {
  return (
    <div className="relative mt-1">
      <select
        value={value}
        onChange={(event) => onChange(event.target.value)}
        className="block w-full appearance-none rounded-[var(--radius-sm)] border border-[var(--color-border)] bg-[var(--color-surface)] px-2.5 py-1.5 pr-7 text-xs text-[var(--color-text)] outline-none transition-colors focus:border-[var(--color-accent)]"
      >
        {options.map((option) => (
          <option key={option.value} value={option.value}>
            {option.label}
          </option>
        ))}
      </select>
      <ChevronDown className="pointer-events-none absolute right-2 top-1/2 size-3 -translate-y-1/2 text-[var(--color-text-muted)]" />
    </div>
  );
}
