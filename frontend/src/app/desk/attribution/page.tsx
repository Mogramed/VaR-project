import { PageHeader } from "@/components/app-shell/page-header";
import { ChartSurface } from "@/components/charts/chart-surface";
import { AttributionTable } from "@/components/data/risk-tables";
import { MetricBlock } from "@/components/ui/metric-block";
import { StatusBadge } from "@/components/ui/primitives";
import { api } from "@/lib/api/client";
import { makeBarOption } from "@/lib/chart-options";
import { formatCurrency, formatPercent } from "@/lib/utils";
import { flattenAttribution } from "@/lib/view-models";

export default async function DeskAttributionPage({
  searchParams,
}: {
  searchParams: Promise<Record<string, string | string[] | undefined>>;
}) {
  const query = await searchParams;
  const portfolioSlug =
    typeof query.portfolio === "string" ? query.portfolio : undefined;
  const preferredModel = typeof query.model === "string" ? query.model : undefined;

  const [attribution, comparison] = await Promise.all([
    api.latestAttribution(portfolioSlug).catch(() => null),
    api.latestModelComparison(portfolioSlug).catch(() => null),
  ]);

  const selectedModel =
    preferredModel ??
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

  return (
    <div className="desk-page space-y-8">
      <PageHeader
        eyebrow="Risk Attribution"
        title="Per-pair contribution, marginality and portfolio pressure."
        description="The attribution layer now reads as a proper decision surface: dominant contributors, diversifiers and component pressure stay visible without leaving empty space around a tiny chart."
        aside={<StatusBadge label={selectedModel.toUpperCase()} tone="accent" />}
      />

      <section className="surface rounded-[1.7rem] p-6">
        <div className="flex flex-wrap gap-2">
          {attribution ? (
            Object.keys(attribution.models).map((model) => {
              const active = model === selectedModel;
              const href = portfolioSlug
                ? `/desk/attribution?portfolio=${encodeURIComponent(portfolioSlug)}&model=${model}`
                : `/desk/attribution?model=${model}`;
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
      </section>

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
        description="Sparse attribution states now expand into narrative insight instead of leaving a large blank pane beside a single bar."
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
                    : "No pair is currently acting as a visible diversifier."}
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
