"use client";

import { motion } from "framer-motion";
import {
  ArrowRight,
  CandlestickChart,
  ShieldCheck,
  Workflow,
} from "lucide-react";
import { useTranslations } from "next-intl";
import { ButtonLink, Eyebrow, SectionHeading } from "@/components/ui/primitives";

const transition = { duration: 0.65, ease: [0.22, 1, 0.36, 1] } as const;

function WorkflowColumn({
  index,
  title,
  copy,
}: {
  index: string;
  title: string;
  copy: string;
}) {
  return (
    <motion.div
      initial={{ opacity: 0, y: 22 }}
      whileInView={{ opacity: 1, y: 0 }}
      viewport={{ once: true, amount: 0.35 }}
      transition={{ ...transition, delay: 0.06 }}
      className="premium-hover border-t border-white/8 py-6"
    >
      <div className="mono text-[11px] uppercase tracking-[0.28em] text-[var(--color-accent)]">
        {index}
      </div>
      <h3 className="mt-4 text-xl font-semibold text-white">{title}</h3>
      <p className="mt-3 max-w-sm text-sm leading-7 text-[var(--color-text-soft)]">
        {copy}
      </p>
    </motion.div>
  );
}

function DeskPreview({ variant = "full" }: { variant?: "hero" | "full" }) {
  const compact = variant === "hero";

  return (
    <div
      className={`landing-poster surface-strong premium-hover relative overflow-hidden border border-[var(--color-border-strong)] ${
        compact ? "rounded-[1.8rem]" : "rounded-[2rem]"
      }`}
    >
      <div className="absolute inset-0 bg-[radial-gradient(circle_at_top_right,rgba(216,155,73,0.18),transparent_28%)]" />
      <div className="absolute inset-x-0 top-0 h-px bg-gradient-to-r from-transparent via-[var(--color-accent)]/70 to-transparent" />
      <div
        className={`grid ${
          compact
            ? "gap-4 p-4 lg:grid-cols-[1.08fr_0.92fr]"
            : "gap-6 p-6 lg:grid-cols-[1.1fr_0.9fr]"
        }`}
      >
        <div className={compact ? "space-y-4" : "space-y-6"}>
          <div className="flex items-center justify-between border-b border-white/8 pb-4">
            <div>
              <div className="mono text-[11px] uppercase tracking-[0.3em] text-[var(--color-text-muted)]">
                Desk Overview
              </div>
              <div
                className={`mt-2 font-semibold text-white ${
                  compact ? "text-lg" : "text-xl"
                }`}
              >
                FX Macro Desk
              </div>
            </div>
            <Eyebrow tone="accent">Operator Grade</Eyebrow>
          </div>

          <div
            className={`grid ${
              compact ? "gap-3 md:grid-cols-3" : "gap-5 md:grid-cols-3"
            }`}
          >
            {[
              ["VaR 99%", "2.48M", "amber"],
              ["ES 99%", "3.31M", "neutral"],
              ["Headroom", "38%", "success"],
            ].map(([label, value, tone]) => (
              <div
                key={label}
                className={`rounded-[1.4rem] border border-white/8 bg-black/18 ${
                  compact ? "p-3.5" : "p-4"
                }`}
              >
                <div className="mono text-[11px] uppercase tracking-[0.26em] text-[var(--color-text-muted)]">
                  {label}
                </div>
                <div
                  className={`mt-4 font-semibold text-white ${
                    compact ? "text-[1.8rem]" : "text-3xl"
                  }`}
                >
                  {value}
                </div>
                <div
                  className="mt-4 h-1 rounded-full"
                  style={{
                    background:
                      tone === "success"
                        ? "linear-gradient(90deg, rgba(95,212,166,0.35), rgba(95,212,166,0.95))"
                        : tone === "amber"
                          ? "linear-gradient(90deg, rgba(242,180,93,0.35), rgba(242,180,93,0.95))"
                          : "linear-gradient(90deg, rgba(216,155,73,0.18), rgba(216,155,73,0.65))",
                  }}
                />
              </div>
            ))}
          </div>

          <div className="overflow-hidden rounded-[1.4rem] border border-white/8 bg-black/24">
            <div className="flex items-center justify-between border-b border-white/8 px-5 py-4">
              <div>
                <div className="mono text-[11px] uppercase tracking-[0.26em] text-[var(--color-text-muted)]">
                  Champion / Challenger
                </div>
                <div className="mt-2 text-lg font-semibold text-white">
                  Model posture
                </div>
              </div>
              <div className="text-sm text-[var(--color-text-soft)]">live ranking</div>
            </div>
            <div
              className={`grid ${
                compact ? "gap-3 px-4 py-4" : "gap-4 px-5 py-5"
              }`}
            >
              {[
                ["GARCH", "Champion", "96.4", "var(--color-green)"],
                ["FHS", "Challenger", "92.1", "var(--color-accent)"],
                ["Hist", "Fallback", "88.0", "var(--color-text-soft)"],
              ].map(([model, role, score, color]) => (
                <div
                  key={model}
                  className="grid grid-cols-[1fr_auto_auto] items-center gap-4 border-b border-white/6 pb-3 last:border-b-0 last:pb-0"
                >
                  <div>
                    <div className="text-base font-semibold text-white">{model}</div>
                    <div className="text-xs uppercase tracking-[0.24em] text-[var(--color-text-muted)]">
                      {role}
                    </div>
                  </div>
                  <div className="mono text-sm text-[var(--color-text-soft)]">score</div>
                  <div
                    className={`mono font-semibold ${
                      compact ? "text-lg" : "text-xl"
                    }`}
                    style={{ color }}
                  >
                    {score}
                  </div>
                </div>
              ))}
            </div>
          </div>
        </div>

        <div className={compact ? "space-y-4" : "space-y-5"}>
          <div
            className={`rounded-[1.4rem] border border-white/8 bg-black/28 ${
              compact ? "p-4" : "p-5"
            }`}
          >
            <div className="mono text-[11px] uppercase tracking-[0.26em] text-[var(--color-text-muted)]">
              Capital Pressure
            </div>
            <div className="mt-4 grid gap-3">
              {[
                ["EURUSD", 0.82],
                ["GBPUSD", 0.58],
                ["USDJPY", 0.46],
                ["AUDUSD", 0.31],
              ].map(([symbol, load]) => (
                <div key={symbol} className="space-y-2">
                  <div className="flex items-center justify-between text-sm text-[var(--color-text-soft)]">
                    <span className="mono tracking-[0.18em] text-white">{symbol}</span>
                    <span>{Math.round(Number(load) * 100)}%</span>
                  </div>
                  <div className="h-2 rounded-full bg-white/6">
                    <div
                      className="h-full rounded-full bg-gradient-to-r from-[var(--color-accent)] via-[#f2c37b] to-[#fff0d6]"
                      style={{ width: `${Number(load) * 100}%` }}
                    />
                  </div>
                </div>
              ))}
            </div>
          </div>

          <div
            className={`rounded-[1.4rem] border border-white/8 bg-black/28 ${
              compact ? "p-4" : "p-5"
            }`}
          >
            <div className="mono text-[11px] uppercase tracking-[0.26em] text-[var(--color-text-muted)]">
              Risk Decision
            </div>
            <div className="mt-4 space-y-4">
              <div className="flex items-center justify-between">
                <div>
                  <div className="text-base font-semibold text-white">
                    EURUSD +4.0M
                  </div>
                  <div className="text-sm text-[var(--color-text-soft)]">
                    proposed buy notional
                  </div>
                </div>
                <span className="rounded-full border border-amber-300/20 bg-amber-300/10 px-3 py-1 text-xs font-semibold uppercase tracking-[0.24em] text-[var(--color-amber)]">
                  Reduce
                </span>
              </div>
              <div className="grid grid-cols-2 gap-4">
                {[
                  ["Requested", "4.0M"],
                  ["Approved", "2.7M"],
                  ["VaR After", "2.69M"],
                  ["Headroom", "14%"],
                ].map(([label, value]) => (
                  <div key={label} className="border-t border-white/8 pt-3">
                    <div className="mono text-[11px] uppercase tracking-[0.26em] text-[var(--color-text-muted)]">
                      {label}
                    </div>
                    <div className="mt-2 text-lg font-semibold text-white">
                      {value}
                    </div>
                  </div>
                ))}
              </div>
            </div>
          </div>

          {!compact ? (
            <div className="rounded-[1.4rem] border border-white/8 bg-gradient-to-br from-[var(--color-accent-soft)] via-transparent to-transparent p-5">
              <div className="mono text-[11px] uppercase tracking-[0.26em] text-[var(--color-text-muted)]">
                Audit
              </div>
              <div className="mt-4 space-y-3 text-sm text-[var(--color-text-soft)]">
                <div className="flex items-center justify-between">
                  <span>backtest.run</span>
                  <span className="mono">13:42 UTC</span>
                </div>
                <div className="flex items-center justify-between">
                  <span>capital.rebalance</span>
                  <span className="mono">13:39 UTC</span>
                </div>
                <div className="flex items-center justify-between">
                  <span>decision.evaluate</span>
                  <span className="mono">13:31 UTC</span>
                </div>
              </div>
            </div>
          ) : null}
        </div>
      </div>
    </div>
  );
}

export function LandingPage({ locale }: { locale: "en" | "fr" }) {
  const t = useTranslations("landing");
  const nav = useTranslations("nav");
  const alternateLocale = locale === "en" ? "fr" : "en";

  return (
    <div className="min-h-screen">
      <header className="absolute inset-x-0 top-0 z-30">
        <div className="mx-auto flex max-w-[1600px] items-center justify-between px-[var(--page-gutter)] py-7">
          <div className="flex items-center gap-3">
            <div className="flex size-11 items-center justify-center rounded-2xl border border-[var(--color-border-strong)] bg-[var(--color-accent-soft)] text-sm font-semibold tracking-[0.24em] text-[var(--color-accent)]">
              VR
            </div>
            <div>
              <div className="text-sm font-semibold uppercase tracking-[0.3em] text-white">
                VaR Risk Desk
              </div>
              <div className="text-xs text-[var(--color-text-muted)]">FX platform</div>
            </div>
          </div>
          <nav className="hidden items-center gap-8 text-sm text-[var(--color-text-soft)] md:flex">
            <a href="#platform">{nav("platform")}</a>
            <a href="#workflow">{nav("workflow")}</a>
            <a href="#validation">{nav("credibility")}</a>
            <a href="#preview">{nav("preview")}</a>
          </nav>
          <div className="flex items-center gap-3">
            <ButtonLink variant="ghost" href={`/${alternateLocale}`}>
              {nav("switchToFr")}
            </ButtonLink>
            <ButtonLink href="/desk">
              {nav("openDesk")}
              <ArrowRight className="ml-2 size-4" />
            </ButtonLink>
          </div>
        </div>
      </header>

      <main>
        <section className="hero-grid relative min-h-screen overflow-hidden px-[var(--page-gutter)] pb-12 pt-28 md:pt-32">
          <div className="absolute inset-0 bg-[radial-gradient(circle_at_18%_14%,rgba(216,155,73,0.18),transparent_22%),radial-gradient(circle_at_82%_12%,rgba(114,177,255,0.14),transparent_24%),linear-gradient(180deg,rgba(255,255,255,0.02),transparent_30%)]" />
          <div className="absolute inset-x-0 bottom-0 h-32 bg-gradient-to-t from-[rgba(8,9,11,0.92)] to-transparent" />

          <div className="relative mx-auto grid min-h-[calc(100svh-7rem)] max-w-[1600px] items-center gap-10 lg:grid-cols-[0.86fr_1.14fr] lg:gap-12">
            <motion.div
              initial={false}
              animate={{ opacity: 1, y: 0 }}
              transition={transition}
              className="flex max-w-[620px] flex-col justify-center py-6"
            >
              <Eyebrow tone="accent">{t("eyebrow")}</Eyebrow>
              <h1 className="mt-7 text-balance text-[clamp(3.1rem,6.8vw,6rem)] font-semibold leading-[0.92] tracking-[-0.075em] text-white">
                {t("title")}
              </h1>
              <p className="mt-6 max-w-[34rem] text-base leading-8 text-[var(--color-text-soft)] md:text-lg">
                {t("subtitle")}
              </p>
              <div className="mt-9 flex flex-wrap gap-4">
                <ButtonLink href="/desk">{t("primaryCta")}</ButtonLink>
                <ButtonLink href="#workflow" variant="secondary">
                  {t("secondaryCta")}
                </ButtonLink>
              </div>
              <div className="mt-10 grid gap-4 sm:grid-cols-3">
                {[
                  [t("heroPrimaryMetric"), t("heroPrimaryValue")],
                  [t("heroSecondaryMetric"), t("heroSecondaryValue")],
                  [t("heroTertiaryMetric"), t("heroTertiaryValue")],
                ].map(([label, value]) => (
                  <div
                    key={label}
                    className="hero-stat rounded-[1.3rem] border border-white/8 bg-black/18 px-4 py-4 backdrop-blur-md"
                  >
                    <div className="mono text-[11px] uppercase tracking-[0.24em] text-[var(--color-text-muted)]">
                      {label}
                    </div>
                    <div className="mt-3 text-[1.55rem] font-semibold tracking-[-0.05em] text-white">
                      {value}
                    </div>
                  </div>
                ))}
              </div>
            </motion.div>

            <motion.div
              initial={false}
              animate={{ opacity: 1, y: 0, scale: 1 }}
              transition={{ ...transition, delay: 0.1 }}
              className="lg:pl-4"
            >
              <DeskPreview variant="hero" />
            </motion.div>
          </div>
        </section>

        <section id="platform" className="px-[var(--page-gutter)] py-24">
          <div className="mx-auto grid max-w-[1600px] gap-10 lg:grid-cols-[0.8fr_1.2fr]">
            <SectionHeading title={t("supportTitle")} copy={t("supportBody")} />
            <div className="grid gap-6 md:grid-cols-3">
              {[
                {
                  icon: CandlestickChart,
                  title: "Risk Surface",
                  body: "Dense, operator-grade overview with the current posture visible at a glance.",
                },
                {
                  icon: ShieldCheck,
                  title: "Governance Ready",
                  body: "Decision trail, capital history and report continuity built into the workflow.",
                },
                {
                  icon: Workflow,
                  title: "Desk Flow",
                  body: "One visual system across models, attribution, capital, decisions and simulation.",
                },
              ].map(({ icon: Icon, title, body }, index) => (
                <motion.div
                  key={title}
                  initial={{ opacity: 0, y: 24 }}
                  whileInView={{ opacity: 1, y: 0 }}
                  viewport={{ once: true, amount: 0.35 }}
                  transition={{ ...transition, delay: index * 0.08 }}
                  className="surface premium-hover rounded-[1.6rem] p-6"
                >
                  <div className="flex size-12 items-center justify-center rounded-2xl border border-[var(--color-border-strong)] bg-[var(--color-accent-soft)]">
                    <Icon className="size-5 text-[var(--color-accent)]" />
                  </div>
                  <h3 className="mt-6 text-xl font-semibold text-white">{title}</h3>
                  <p className="mt-3 text-sm leading-7 text-[var(--color-text-soft)]">
                    {body}
                  </p>
                </motion.div>
              ))}
            </div>
          </div>
        </section>

        <section id="workflow" className="px-[var(--page-gutter)] py-24">
          <div className="mx-auto max-w-[1600px]">
            <SectionHeading title={t("workflowTitle")} copy={t("workflowLead")} />
            <div className="mt-14 grid gap-8 md:grid-cols-2 xl:grid-cols-4">
              <WorkflowColumn
                index="01"
                title={t("workflowMeasureTitle")}
                copy={t("workflowMeasureBody")}
              />
              <WorkflowColumn
                index="02"
                title={t("workflowValidateTitle")}
                copy={t("workflowValidateBody")}
              />
              <WorkflowColumn
                index="03"
                title={t("workflowAllocateTitle")}
                copy={t("workflowAllocateBody")}
              />
              <WorkflowColumn
                index="04"
                title={t("workflowDecideTitle")}
                copy={t("workflowDecideBody")}
              />
            </div>
          </div>
        </section>

        <section id="validation" className="px-[var(--page-gutter)] py-24">
          <div className="mx-auto grid max-w-[1600px] gap-10 lg:grid-cols-[1fr_1fr]">
            <motion.div
              initial={{ opacity: 0, y: 22 }}
              whileInView={{ opacity: 1, y: 0 }}
              viewport={{ once: true, amount: 0.28 }}
              transition={transition}
              className="surface-strong premium-hover rounded-[2rem] p-8 md:p-10"
            >
              <SectionHeading title={t("credibilityTitle")} copy={t("credibilityBody")} />
              <div className="mt-10 space-y-5">
                {[t("credibilityListOne"), t("credibilityListTwo"), t("credibilityListThree")].map(
                  (item, index) => (
                    <div
                      key={item}
                      className="flex items-start gap-4 border-t border-white/8 pt-5"
                    >
                      <div className="mono text-[12px] uppercase tracking-[0.3em] text-[var(--color-accent)]">
                        0{index + 1}
                      </div>
                      <p className="max-w-xl text-sm leading-7 text-[var(--color-text-soft)]">
                        {item}
                      </p>
                    </div>
                  ),
                )}
              </div>
            </motion.div>
            <div className="grid gap-6">
              <motion.div
                initial={{ opacity: 0, y: 22 }}
                whileInView={{ opacity: 1, y: 0 }}
                viewport={{ once: true, amount: 0.28 }}
                transition={{ ...transition, delay: 0.08 }}
                className="surface premium-hover rounded-[1.7rem] p-7"
              >
                <div className="mono text-[11px] uppercase tracking-[0.3em] text-[var(--color-text-muted)]">
                  Validation pulse
                </div>
                <div className="mt-4 flex items-end justify-between gap-4">
                  <div className="text-5xl font-semibold tracking-[-0.06em] text-white">
                    4.8%
                  </div>
                  <div className="text-right text-sm text-[var(--color-text-soft)]">
                    exception rate vs 5.0% expected
                  </div>
                </div>
                <div className="mt-8 h-48 rounded-[1.3rem] bg-[linear-gradient(180deg,rgba(216,155,73,0.08),transparent),linear-gradient(90deg,rgba(255,255,255,0.02)_1px,transparent_1px),linear-gradient(rgba(255,255,255,0.04)_1px,transparent_1px)] bg-[length:auto,48px_48px,48px_48px] bg-center" />
              </motion.div>
              <motion.div
                initial={{ opacity: 0, y: 22 }}
                whileInView={{ opacity: 1, y: 0 }}
                viewport={{ once: true, amount: 0.28 }}
                transition={{ ...transition, delay: 0.14 }}
                className="surface premium-hover rounded-[1.7rem] p-7"
              >
                <div className="mono text-[11px] uppercase tracking-[0.3em] text-[var(--color-text-muted)]">
                  Desk logic
                </div>
                <div className="mt-6 grid gap-4 sm:grid-cols-2">
                  {[
                    ["Model regime", "Champion/challenger"],
                    ["Budgeting", "Capital + headroom"],
                    ["Decisioning", "Accept / reduce / reject"],
                    ["Simulation", "Paper-trade follow-through"],
                  ].map(([label, value]) => (
                    <div key={label} className="border-t border-white/8 pt-4">
                      <div className="mono text-[11px] uppercase tracking-[0.24em] text-[var(--color-text-muted)]">
                        {label}
                      </div>
                      <div className="mt-2 text-base font-semibold text-white">
                        {value}
                      </div>
                    </div>
                  ))}
                </div>
              </motion.div>
            </div>
          </div>
        </section>

        <section id="preview" className="px-[var(--page-gutter)] py-24">
          <div className="mx-auto max-w-[1600px]">
            <SectionHeading
              title={t("previewTitle")}
              copy={t("previewBody")}
              align="center"
            />
            <div className="mt-14">
              <DeskPreview variant="full" />
            </div>
          </div>
        </section>

        <section className="px-[var(--page-gutter)] py-24">
          <div className="mx-auto max-w-[1600px]">
            <motion.div
              initial={{ opacity: 0, y: 22 }}
              whileInView={{ opacity: 1, y: 0 }}
              viewport={{ once: true, amount: 0.35 }}
              transition={transition}
              className="surface-strong premium-hover flex flex-col items-start justify-between gap-8 rounded-[2.2rem] px-8 py-10 md:flex-row md:items-end md:px-12 md:py-12"
            >
              <div className="max-w-2xl">
                <Eyebrow tone="accent">Desk Ready</Eyebrow>
                <h2 className="mt-6 text-balance text-4xl font-semibold tracking-[-0.05em] text-white md:text-5xl">
                  {t("finalTitle")}
                </h2>
                <p className="mt-4 text-sm leading-7 text-[var(--color-text-soft)] md:text-base">
                  {t("finalBody")}
                </p>
              </div>
              <ButtonLink href="/desk">
                {t("finalPrimaryCta")}
                <ArrowRight className="ml-2 size-4" />
              </ButtonLink>
            </motion.div>
          </div>
        </section>
      </main>
    </div>
  );
}
