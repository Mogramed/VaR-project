"use client";

import { startTransition, useEffect, useState } from "react";

import { LiveOperatorAlerts } from "@/components/app-shell/live-operator-alerts";
import { PageHeader } from "@/components/app-shell/page-header";
import { ChartSurface } from "@/components/charts/chart-surface";
import { AttributionTable } from "@/components/data/risk-tables";
import { MetricBlock } from "@/components/ui/metric-block";
import { StatusBadge } from "@/components/ui/primitives";
import { api } from "@/lib/api/client";
import type {
  ModelComparisonResponse,
  MT5LiveStateResponse,
  RiskAttributionResponse,
} from "@/lib/api/types";
import { makeBarOption } from "@/lib/chart-options";
import { useMt5LiveState } from "@/lib/use-mt5-live-state";
import { formatCurrency, formatPercent, formatTimestamp } from "@/lib/utils";
import { flattenAttribution } from "@/lib/view-models";

function preferredAttributionSource(liveState: MT5LiveStateResponse | null) {
  return liveState?.risk_budget ? "mt5_live_bridge" : "historical";
}

export function AttributionLiveSurface({
  portfolioSlug,
  preferredModel,
  initialLiveState,
  initialAttribution,
  initialComparison,
}: {
  portfolioSlug: string;
  preferredModel?: string;
  initialLiveState: MT5LiveStateResponse | null;
  initialAttribution: RiskAttributionResponse | null;
  initialComparison: ModelComparisonResponse | null;
}) {
  const { liveState, transport } = useMt5LiveState(portfolioSlug, initialLiveState);
  const [attribution, setAttribution] = useState<RiskAttributionResponse | null>(initialAttribution);
  const [comparison, setComparison] = useState<ModelComparisonResponse | null>(initialComparison);
  const attributionSource = preferredAttributionSource(liveState);

  useEffect(() => {
    let cancelled = false;

    const refreshAttribution = async () => {
      try {
        const nextAttribution = await api.latestAttribution(portfolioSlug, attributionSource);
        if (cancelled) {
          return;
        }
        startTransition(() => {
          setAttribution(nextAttribution);
        });
      } catch {
        if (attributionSource === "historical") {
          return;
        }
        try {
          const fallback = await api.latestAttribution(portfolioSlug, "historical");
          if (cancelled) {
            return;
          }
          startTransition(() => {
            setAttribution(fallback);
          });
        } catch {
          // Keep the current attribution snapshot on transient failures.
        }
      }
    };

    void refreshAttribution();
    return () => {
      cancelled = true;
    };
  }, [attributionSource, portfolioSlug, liveState?.sequence]);

  useEffect(() => {
    let cancelled = false;
    const timer = window.setInterval(() => {
      void (async () => {
        try {
          const nextComparison = await api.latestModelComparison(portfolioSlug);
          if (cancelled) {
            return;
          }
          startTransition(() => {
            setComparison(nextComparison);
          });
        } catch {
          // Keep the current comparison summary on transient failures.
        }
      })();
    }, 30000);
    return () => {
      cancelled = true;
      window.clearInterval(timer);
    };
  }, [portfolioSlug]);

  const selectedModel =
    preferredModel ??
    liveState?.risk_budget?.preferred_model ??
    liveState?.risk_summary?.reference_model ??
    comparison?.champion_model ??
    (attribution ? Object.keys(attribution.models)[0] : undefined) ??
    "hist";
  const rows = attribution ? flattenAttribution(attribution, selectedModel) : [];
  const primaryContributor = rows[0];
  const diversifier =
    rows
      .slice()
      .sort((left, right) => left.componentVar - right.componentVar)
      .find((row) => row.componentVar < 0) ?? null;
  const operatorAlerts = liveState?.operator_alerts ?? [];
  const riskBudgetModel = liveState?.risk_budget?.models?.[selectedModel] ?? null;

  return (
    <div className="desk-page space-y-8">
      <PageHeader
        eyebrow="Risk Attribution"
        title="Per-symbol contribution, marginality and live portfolio pressure."
        description="Attribution now follows the MT5 bridge persistence path: live holdings drive the selected model view, while the table falls back to historical snapshots only when live attribution is not available yet."
        aside={
          <div className="flex flex-wrap items-center gap-2">
            <StatusBadge label={selectedModel.toUpperCase()} tone="accent" />
            <StatusBadge
              label={(attribution?.snapshot_source ?? "historical").replaceAll("_", " ")}
              tone={attribution?.snapshot_source === "mt5_live_bridge" ? "success" : "neutral"}
            />
            <StatusBadge
              label={transport === "stream" ? "stream" : transport === "polling" ? "polling" : "connecting"}
              tone={transport === "stream" ? "success" : transport === "polling" ? "warning" : "neutral"}
            />
          </div>
        }
      />

      <section className="surface rounded-[1.7rem] p-6">
        <div className="flex flex-wrap gap-2">
          {attribution ? (
            Object.keys(attribution.models).map((model) => {
              const active = model === selectedModel;
              const href = `/desk/attribution?portfolio=${encodeURIComponent(portfolioSlug)}&model=${model}`;
              return (
                <a
                  key={model}
                  href={href}
                  className={`rounded-full border px-3 py-2 text-xs uppercase tracking-[0.24em] transition ${
                    active
                      ? "border-[var(--color-border-strong)] bg-[var(--color-accent-soft)] text-white"
                      : "border-white/8 text-[var(--color-text-soft)] hover:border-white/16 hover:text-white"
                  }`}
                >
                  {model}
                </a>
              );
            })
          ) : (
            <div className="text-sm text-[var(--color-text-muted)]">
              No attribution snapshot available yet.
            </div>
          )}
        </div>
        <div className="mt-4 text-sm leading-7 text-[var(--color-text-soft)]">
          {attribution?.snapshot_timestamp
            ? `Current attribution snapshot ${formatTimestamp(attribution.snapshot_timestamp)}.`
            : "No persisted attribution timestamp available yet."}{" "}
          {liveState?.generated_at
            ? `Bridge state refreshed ${formatTimestamp(liveState.generated_at)}.`
            : "Bridge refresh pending."}
        </div>
      </section>

      <LiveOperatorAlerts
        alerts={operatorAlerts}
        title="Attribution watchlist"
        copy="Budget pressure, bridge health and reconciliation incidents are visible here while you read contribution and diversification effects."
      />

      <ChartSurface
        option={makeBarOption(
          rows.map((row) => ({
            label: row.symbol,
            value: row.componentVar,
          })),
          {
            color: "#d89b49",
            negativeColor: "#5fd4a6",
            mode: rows.length <= 5 ? "sparse" : "standard",
          },
        )}
        mode={rows.length <= 5 ? "sparse" : "standard"}
        dataCount={rows.length}
        eyebrow="Component VaR by symbol"
        title="Contribution profile"
        description="Live holdings now feed the selected attribution model whenever the MT5 bridge has a persisted snapshot for this portfolio."
        insight={
          rows.length ? (
            <div className="grid gap-4 md:grid-cols-2">
              <MetricBlock
                label="Primary contributor"
                value={primaryContributor?.symbol ?? "n/a"}
                hint={
                  primaryContributor
                    ? `${formatCurrency(primaryContributor.componentVar)} component VaR`
                    : "No contribution available"
                }
                tone="warning"
                className="h-full bg-transparent"
              />
              <MetricBlock
                label="Diversifier"
                value={diversifier?.symbol ?? "none"}
                hint={
                  diversifier
                    ? `${formatCurrency(diversifier.componentVar)} component VaR`
                    : "No negative component contribution"
                }
                tone="success"
                className="h-full bg-transparent"
              />
              <div className="rounded-[1.35rem] border border-white/8 bg-black/18 p-4 md:col-span-2">
                <div className="mono text-[11px] uppercase tracking-[0.24em] text-[var(--color-text-muted)]">
                  Reading note
                </div>
                <div className="mt-3 text-sm leading-7 text-[var(--color-text-soft)]">
                  {primaryContributor
                    ? `${primaryContributor.symbol} drives ${formatPercent(
                        primaryContributor.contributionPctVar,
                      )} of the selected model VaR.`
                    : "No attribution rows are available."}{" "}
                  {diversifier
                    ? `${diversifier.symbol} partially offsets the stack with a negative component contribution.`
                    : "No symbol is currently acting as a visible diversifier."}{" "}
                  {riskBudgetModel
                    ? `Budget utilization for ${selectedModel.toUpperCase()} sits at ${formatPercent(
                        riskBudgetModel.utilization_var ?? 0,
                      )}.`
                    : "No live risk-budget overlay is available yet."}
                </div>
              </div>
            </div>
          ) : undefined
        }
        emptyState={
          <div className="rounded-[1.3rem] border border-white/8 bg-black/16 px-4 py-4 text-sm text-[var(--color-text-muted)]">
            No attribution rows are available yet.
          </div>
        }
      />

      <section className="space-y-3">
        <div className="mono text-[11px] uppercase tracking-[0.28em] text-[var(--color-text-muted)]">
          Attribution table
        </div>
        <AttributionTable rows={rows} />
      </section>
    </div>
  );
}
