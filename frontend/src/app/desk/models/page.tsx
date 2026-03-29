import { PageHeader } from "@/components/app-shell/page-header";
import { ChartSurface } from "@/components/charts/chart-surface";
import { ModelRankingTable } from "@/components/data/risk-tables";
import { MetricBlock } from "@/components/ui/metric-block";
import { StatusBadge } from "@/components/ui/primitives";
import { api } from "@/lib/api/client";
import { makeBacktestOption, makeBarOption } from "@/lib/chart-options";
import { formatCurrency, formatPercent } from "@/lib/utils";
import { buildBacktestSeries, buildModelScoreSeries } from "@/lib/view-models";

export default async function DeskModelsPage({
  searchParams,
}: {
  searchParams: Promise<Record<string, string | string[] | undefined>>;
}) {
  const query = await searchParams;
  const portfolioSlug =
    typeof query.portfolio === "string" ? query.portfolio : undefined;

  const [comparison, validation, frame] = await Promise.all([
    api.latestModelComparison(portfolioSlug).catch(() => null),
    api.latestValidation(portfolioSlug).catch(() => null),
    api.latestBacktestFrame(portfolioSlug, 240).catch(() => null),
  ]);

  const ranking = comparison?.ranking ?? [];
  const backtestSeries = frame ? buildBacktestSeries(frame) : [];
  const scoreSeries = comparison ? buildModelScoreSeries(comparison) : [];

  return (
    <div className="desk-page space-y-8">
      <PageHeader
        eyebrow="Model Monitor"
        title="Champion, challenger and statistical credibility."
        description="The model layer stays dense and operational: ranking, exceptions, score gap and backtest trace read as one continuous diagnostic surface."
        aside={
          comparison?.champion_model ? (
            <StatusBadge label={comparison.champion_model.toUpperCase()} tone="accent" />
          ) : null
        }
      />

      <section className="grid gap-4 xl:grid-cols-4">
        <MetricBlock
          label="Champion"
          value={(comparison?.champion_model ?? "n/a").toUpperCase()}
          hint="Current best performer"
        />
        <MetricBlock
          label="Challenger"
          value={(comparison?.challenger_model ?? "n/a").toUpperCase()}
          hint="Second-ranked model"
        />
        <MetricBlock
          label="Expected exceptions"
          value={validation ? formatPercent(validation.expected_rate) : "n/a"}
          hint="Target hit-rate"
          tone="warning"
        />
        <MetricBlock
          label="Score gap"
          value={comparison?.score_gap != null ? comparison.score_gap.toFixed(1) : "n/a"}
          hint="Champion vs challenger"
          tone="accent"
        />
      </section>

      <section className="grid gap-6 xl:grid-cols-[minmax(0,1.15fr)_340px]">
        <ChartSurface
          option={makeBacktestOption(backtestSeries)}
          mode="trace"
          dataCount={backtestSeries.length}
          eyebrow="Backtest trace"
          title="Portfolio PnL versus selected VaR series"
          description="This stays central on the page so exception posture is not visually downgraded behind ranking widgets."
          meta={backtestSeries.length ? `${backtestSeries.length} rows` : "No backtest frame"}
          emptyState={
            <div className="rounded-[1.3rem] border border-white/8 bg-black/16 px-4 py-4 text-sm text-[var(--color-text-muted)]">
              No backtest frame available yet.
            </div>
          }
        />

        <div className="space-y-6">
          <div className="surface rounded-[1.7rem] p-6">
            <div className="mono text-[11px] uppercase tracking-[0.28em] text-[var(--color-text-muted)]">
              Validation pulse
            </div>
            <div className="mt-5 grid gap-4">
              <MetricBlock
                label="Best model"
                value={(validation?.best_model ?? "n/a").toUpperCase()}
                hint="Latest persisted validation"
                className="bg-transparent"
              />
              <MetricBlock
                label="Alpha"
                value={validation ? formatPercent(validation.alpha, 0) : "n/a"}
                hint="Confidence level"
                tone="accent"
                className="bg-transparent"
              />
            </div>
            {ranking.length > 0 ? (
              <div className="mt-5 space-y-3">
                {ranking.slice(0, 3).map((row) => (
                  <div
                    key={row.model}
                    className="rounded-[1.3rem] border border-white/8 bg-black/18 p-4"
                  >
                    <div className="flex items-center justify-between gap-3">
                      <div className="text-base font-semibold text-white">
                        {row.model.toUpperCase()}
                      </div>
                      <div className="mono text-sm text-[var(--color-accent)]">
                        {row.score.toFixed(1)}
                      </div>
                    </div>
                    <div className="mt-3 flex items-center justify-between text-sm text-[var(--color-text-soft)]">
                      <span>Actual rate</span>
                      <span>{formatPercent(row.actual_rate)}</span>
                    </div>
                    <div className="mt-2 flex items-center justify-between text-sm text-[var(--color-text-soft)]">
                      <span>Current VaR</span>
                      <span>{formatCurrency(row.current_var)}</span>
                    </div>
                  </div>
                ))}
              </div>
            ) : null}
          </div>
        </div>
      </section>

      <section className="grid gap-6 xl:grid-cols-[minmax(320px,0.88fr)_minmax(0,1.12fr)]">
        <ChartSurface
          option={makeBarOption(scoreSeries, {
            color: "#d89b49",
            negativeColor: "#5fd4a6",
            mode: ranking.length <= 5 ? "sparse" : "standard",
          })}
          mode={ranking.length <= 5 ? "sparse" : "standard"}
          dataCount={ranking.length}
          eyebrow="Ranking surface"
          title="Model score spread"
          description="When the model set is small, the chart compresses and hands space back to interpretation instead of stretching a few bars."
          insight={
            ranking.length > 0 ? (
              <div className="grid gap-4">
                <div className="rounded-[1.35rem] border border-white/8 bg-black/18 p-4">
                  <div className="text-sm text-[var(--color-text-soft)]">Champion edge</div>
                  <div className="mt-2 text-2xl font-semibold text-white">
                    {comparison?.score_gap != null ? comparison.score_gap.toFixed(1) : "n/a"}
                  </div>
                  <div className="mt-2 text-sm leading-6 text-[var(--color-text-soft)]">
                    Gap between champion and challenger under the latest validation snapshot.
                  </div>
                </div>
                <div className="rounded-[1.35rem] border border-white/8 bg-black/18 p-4">
                  <div className="text-sm text-[var(--color-text-soft)]">Expected rate</div>
                  <div className="mt-2 text-2xl font-semibold text-white">
                    {validation ? formatPercent(validation.expected_rate) : "n/a"}
                  </div>
                  <div className="mt-2 text-sm leading-6 text-[var(--color-text-soft)]">
                    Traffic-light and statistical diagnostics remain aligned with this rate.
                  </div>
                </div>
              </div>
            ) : undefined
          }
          emptyState={
            <div className="rounded-[1.3rem] border border-white/8 bg-black/16 px-4 py-4 text-sm text-[var(--color-text-muted)]">
              No ranking snapshot is available yet.
            </div>
          }
        />

        <div className="space-y-3">
          <div className="mono text-[11px] uppercase tracking-[0.28em] text-[var(--color-text-muted)]">
            Ranking table
          </div>
          <ModelRankingTable rows={ranking} />
        </div>
      </section>
    </div>
  );
}
