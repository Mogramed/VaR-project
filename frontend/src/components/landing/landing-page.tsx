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

const transition = { duration: 0.5, ease: [0.22, 1, 0.36, 1] } as const;

function WorkflowColumn({ index, title, copy }: { index: string; title: string; copy: string }) {
  return (
    <motion.div
      initial={{ opacity: 0, y: 16 }}
      whileInView={{ opacity: 1, y: 0 }}
      viewport={{ once: true, amount: 0.35 }}
      transition={transition}
      className="border-t border-[var(--color-border)] py-5"
    >
      <div className="text-[10px] font-semibold uppercase tracking-wider text-[var(--color-accent)]">
        {index}
      </div>
      <h3 className="mt-3 text-base font-semibold text-[var(--color-text)]">{title}</h3>
      <p className="mt-2 max-w-sm text-[13px] leading-relaxed text-[var(--color-text-soft)]">{copy}</p>
    </motion.div>
  );
}

function DeskPreview({ variant = "full" }: { variant?: "hero" | "full" }) {
  const compact = variant === "hero";

  return (
    <div className="overflow-hidden rounded-[var(--radius-xl)] border border-[var(--color-border-strong)] bg-[var(--color-surface)]">
      <div className={`grid ${compact ? "gap-3 p-4 lg:grid-cols-[1.08fr_0.92fr]" : "gap-4 p-5 lg:grid-cols-[1.1fr_0.9fr]"}`}>
        {/* Left column */}
        <div className="space-y-3">
          <div className="flex items-center justify-between border-b border-[var(--color-border)] pb-3">
            <div>
              <div className="text-[10px] font-semibold uppercase tracking-wider text-[var(--color-text-muted)]">Desk Overview</div>
              <div className="mt-1 text-base font-semibold text-[var(--color-text)]">FX Macro Desk</div>
            </div>
            <Eyebrow tone="accent">Operator</Eyebrow>
          </div>

          <div className="grid gap-2 md:grid-cols-3">
            {([
              ["VaR 99%", "2.48M", "var(--color-amber)"],
              ["ES 99%", "3.31M", "var(--color-text-soft)"],
              ["Headroom", "38%", "var(--color-green)"],
            ] as const).map(([label, value, color]) => (
              <div key={label} className="rounded-[var(--radius-md)] border border-[var(--color-border)] bg-[var(--color-bg)] p-3">
                <div className="text-[9px] font-semibold uppercase tracking-wider text-[var(--color-text-muted)]">{label}</div>
                <div className="mono mt-2 text-2xl font-semibold" style={{ color }}>{value}</div>
              </div>
            ))}
          </div>

          <div className="rounded-[var(--radius-md)] border border-[var(--color-border)] bg-[var(--color-bg)]">
            <div className="flex items-center justify-between border-b border-[var(--color-border)] px-3.5 py-2.5">
              <div>
                <div className="text-[9px] font-semibold uppercase tracking-wider text-[var(--color-text-muted)]">Champion / Challenger</div>
                <div className="mt-0.5 text-sm font-semibold text-[var(--color-text)]">Model posture</div>
              </div>
              <span className="text-[10px] text-[var(--color-text-muted)]">live</span>
            </div>
            <div className="space-y-0 divide-y divide-[var(--color-border)]">
              {([
                ["GARCH", "Champion", "96.4", "var(--color-green)"],
                ["FHS", "Challenger", "92.1", "var(--color-accent)"],
                ["Hist", "Fallback", "88.0", "var(--color-text-muted)"],
              ] as const).map(([model, role, score, color]) => (
                <div key={model} className="flex items-center justify-between px-3.5 py-2.5">
                  <div>
                    <div className="text-[13px] font-semibold text-[var(--color-text)]">{model}</div>
                    <div className="text-[9px] uppercase tracking-wider text-[var(--color-text-muted)]">{role}</div>
                  </div>
                  <div className="mono text-lg font-semibold" style={{ color }}>{score}</div>
                </div>
              ))}
            </div>
          </div>
        </div>

        {/* Right column */}
        <div className="space-y-3">
          <div className="rounded-[var(--radius-md)] border border-[var(--color-border)] bg-[var(--color-bg)] p-3.5">
            <div className="text-[9px] font-semibold uppercase tracking-wider text-[var(--color-text-muted)]">Capital Pressure</div>
            <div className="mt-3 space-y-2.5">
              {([["EURUSD", 0.82], ["GBPUSD", 0.58], ["USDJPY", 0.46], ["AUDUSD", 0.31]] as const).map(([symbol, load]) => (
                <div key={symbol} className="space-y-1">
                  <div className="flex items-center justify-between text-[12px]">
                    <span className="mono font-medium text-[var(--color-text)]">{symbol}</span>
                    <span className="text-[var(--color-text-muted)]">{Math.round(load * 100)}%</span>
                  </div>
                  <div className="h-1 rounded-full bg-white/5">
                    <div className="h-full rounded-full bg-[var(--color-accent)]" style={{ width: `${load * 100}%` }} />
                  </div>
                </div>
              ))}
            </div>
          </div>

          <div className="rounded-[var(--radius-md)] border border-[var(--color-border)] bg-[var(--color-bg)] p-3.5">
            <div className="text-[9px] font-semibold uppercase tracking-wider text-[var(--color-text-muted)]">Risk Decision</div>
            <div className="mt-3 space-y-3">
              <div className="flex items-center justify-between">
                <div>
                  <div className="text-[13px] font-semibold text-[var(--color-text)]">EURUSD +4.0M</div>
                  <div className="text-[11px] text-[var(--color-text-muted)]">proposed buy</div>
                </div>
                <span className="rounded-[3px] bg-[var(--color-amber-soft)] px-1.5 py-0.5 text-[10px] font-semibold uppercase text-[var(--color-amber)]">Reduce</span>
              </div>
              <div className="grid grid-cols-2 gap-3">
                {([["Requested", "4.0M"], ["Approved", "2.7M"], ["VaR After", "2.69M"], ["Headroom", "14%"]] as const).map(([label, value]) => (
                  <div key={label} className="border-t border-[var(--color-border)] pt-2">
                    <div className="text-[9px] uppercase tracking-wider text-[var(--color-text-muted)]">{label}</div>
                    <div className="mt-0.5 text-sm font-semibold text-[var(--color-text)]">{value}</div>
                  </div>
                ))}
              </div>
            </div>
          </div>

          {!compact ? (
            <div className="rounded-[var(--radius-md)] border border-[var(--color-border)] bg-[var(--color-bg)] p-3.5">
              <div className="text-[9px] font-semibold uppercase tracking-wider text-[var(--color-text-muted)]">Audit</div>
              <div className="mt-2 space-y-1.5 text-[12px] text-[var(--color-text-soft)]">
                {([["backtest.run", "13:42"], ["capital.rebalance", "13:39"], ["decision.evaluate", "13:31"]] as const).map(([action, time]) => (
                  <div key={action} className="flex items-center justify-between">
                    <span>{action}</span>
                    <span className="mono text-[var(--color-text-muted)]">{time} UTC</span>
                  </div>
                ))}
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
    <div className="min-h-screen bg-[var(--color-bg)]">
      {/* Header */}
      <header className="absolute inset-x-0 top-0 z-30">
        <div className="mx-auto flex max-w-[1440px] items-center justify-between px-6 py-5">
          <div className="flex items-center gap-2.5">
            <div className="flex size-8 items-center justify-center rounded-[var(--radius-md)] bg-[var(--color-accent)] text-[11px] font-bold text-[#1a1206]">VR</div>
            <div>
              <div className="text-[13px] font-semibold text-[var(--color-text)]">VaR Risk Desk</div>
              <div className="text-[10px] text-[var(--color-text-muted)]">FX Platform</div>
            </div>
          </div>
          <nav className="hidden items-center gap-6 text-[13px] text-[var(--color-text-soft)] md:flex">
            <a href="#platform" className="hover:text-[var(--color-text)]">{nav("platform")}</a>
            <a href="#workflow" className="hover:text-[var(--color-text)]">{nav("workflow")}</a>
            <a href="#validation" className="hover:text-[var(--color-text)]">{nav("credibility")}</a>
            <a href="#preview" className="hover:text-[var(--color-text)]">{nav("preview")}</a>
          </nav>
          <div className="flex items-center gap-2">
            <ButtonLink variant="ghost" href={`/${alternateLocale}`}>{nav("switchToFr")}</ButtonLink>
            <ButtonLink href="/desk">{nav("openDesk")} <ArrowRight className="ml-1.5 size-3.5" /></ButtonLink>
          </div>
        </div>
      </header>

      <main>
        {/* Hero */}
        <section className="hero-grid relative min-h-screen overflow-hidden px-6 pb-12 pt-24 md:pt-28">
          <div className="relative mx-auto grid min-h-[calc(100svh-7rem)] max-w-[1440px] items-center gap-8 lg:grid-cols-[0.85fr_1.15fr] lg:gap-10">
            <motion.div initial={false} animate={{ opacity: 1, y: 0 }} transition={transition} className="flex max-w-[560px] flex-col justify-center py-6">
              <Eyebrow tone="accent">{t("eyebrow")}</Eyebrow>
              <h1 className="mt-5 text-balance text-[clamp(2.5rem,5.5vw,4.5rem)] font-semibold leading-[0.95] tracking-tight text-[var(--color-text)]">
                {t("title")}
              </h1>
              <p className="mt-4 max-w-[30rem] text-[15px] leading-relaxed text-[var(--color-text-soft)]">{t("subtitle")}</p>
              <div className="mt-7 flex flex-wrap gap-3">
                <ButtonLink href="/desk">{t("primaryCta")}</ButtonLink>
                <ButtonLink href="#workflow" variant="secondary">{t("secondaryCta")}</ButtonLink>
              </div>
              <div className="mt-8 grid gap-3 sm:grid-cols-3">
                {([
                  [t("heroPrimaryMetric"), t("heroPrimaryValue")],
                  [t("heroSecondaryMetric"), t("heroSecondaryValue")],
                  [t("heroTertiaryMetric"), t("heroTertiaryValue")],
                ] as const).map(([label, value]) => (
                  <div key={label} className="hero-stat rounded-[var(--radius-lg)] border border-[var(--color-border)] bg-[var(--color-surface)] px-3.5 py-3">
                    <div className="text-[9px] font-semibold uppercase tracking-wider text-[var(--color-text-muted)]">{label}</div>
                    <div className="mono mt-2 text-xl font-semibold text-[var(--color-text)]">{value}</div>
                  </div>
                ))}
              </div>
            </motion.div>

            <motion.div initial={false} animate={{ opacity: 1, y: 0, scale: 1 }} transition={{ ...transition, delay: 0.08 }}>
              <DeskPreview variant="hero" />
            </motion.div>
          </div>
        </section>

        {/* Platform */}
        <section id="platform" className="px-6 py-20">
          <div className="mx-auto grid max-w-[1440px] gap-8 lg:grid-cols-[0.8fr_1.2fr]">
            <SectionHeading title={t("supportTitle")} copy={t("supportBody")} />
            <div className="grid gap-4 md:grid-cols-3">
              {([
                { icon: CandlestickChart, title: "Risk Surface", body: "Dense operator-grade overview with current posture visible at a glance." },
                { icon: ShieldCheck, title: "Governance", body: "Decision trail, capital history and report continuity built into the workflow." },
                { icon: Workflow, title: "Desk Flow", body: "One system across models, attribution, capital, decisions and execution." },
              ] as const).map(({ icon: Icon, title, body }, i) => (
                <motion.div key={title} initial={{ opacity: 0, y: 16 }} whileInView={{ opacity: 1, y: 0 }} viewport={{ once: true, amount: 0.35 }} transition={{ ...transition, delay: i * 0.06 }}
                  className="rounded-[var(--radius-lg)] border border-[var(--color-border)] bg-[var(--color-surface)] p-5">
                  <div className="flex size-9 items-center justify-center rounded-[var(--radius-md)] bg-[var(--color-accent-soft)]">
                    <Icon className="size-4 text-[var(--color-accent)]" />
                  </div>
                  <h3 className="mt-4 text-[15px] font-semibold text-[var(--color-text)]">{title}</h3>
                  <p className="mt-2 text-[13px] leading-relaxed text-[var(--color-text-soft)]">{body}</p>
                </motion.div>
              ))}
            </div>
          </div>
        </section>

        {/* Workflow */}
        <section id="workflow" className="px-6 py-20">
          <div className="mx-auto max-w-[1440px]">
            <SectionHeading title={t("workflowTitle")} copy={t("workflowLead")} />
            <div className="mt-10 grid gap-6 md:grid-cols-2 xl:grid-cols-4">
              <WorkflowColumn index="01" title={t("workflowMeasureTitle")} copy={t("workflowMeasureBody")} />
              <WorkflowColumn index="02" title={t("workflowValidateTitle")} copy={t("workflowValidateBody")} />
              <WorkflowColumn index="03" title={t("workflowAllocateTitle")} copy={t("workflowAllocateBody")} />
              <WorkflowColumn index="04" title={t("workflowDecideTitle")} copy={t("workflowDecideBody")} />
            </div>
          </div>
        </section>

        {/* Validation */}
        <section id="validation" className="px-6 py-20">
          <div className="mx-auto grid max-w-[1440px] gap-6 lg:grid-cols-2">
            <motion.div initial={{ opacity: 0, y: 16 }} whileInView={{ opacity: 1, y: 0 }} viewport={{ once: true, amount: 0.28 }} transition={transition}
              className="rounded-[var(--radius-xl)] border border-[var(--color-border-strong)] bg-[var(--color-surface-strong)] p-7">
              <SectionHeading title={t("credibilityTitle")} copy={t("credibilityBody")} />
              <div className="mt-8 space-y-4">
                {[t("credibilityListOne"), t("credibilityListTwo"), t("credibilityListThree")].map((item, i) => (
                  <div key={item} className="flex items-start gap-3 border-t border-[var(--color-border)] pt-4">
                    <span className="text-[11px] font-semibold text-[var(--color-accent)]">0{i + 1}</span>
                    <p className="text-[13px] leading-relaxed text-[var(--color-text-soft)]">{item}</p>
                  </div>
                ))}
              </div>
            </motion.div>

            <div className="space-y-4">
              <motion.div initial={{ opacity: 0, y: 16 }} whileInView={{ opacity: 1, y: 0 }} viewport={{ once: true, amount: 0.28 }} transition={{ ...transition, delay: 0.06 }}
                className="rounded-[var(--radius-lg)] border border-[var(--color-border)] bg-[var(--color-surface)] p-5">
                <div className="text-[10px] font-semibold uppercase tracking-wider text-[var(--color-text-muted)]">Validation pulse</div>
                <div className="mt-3 flex items-end justify-between gap-3">
                  <div className="mono text-4xl font-semibold text-[var(--color-text)]">4.8%</div>
                  <div className="text-right text-[12px] text-[var(--color-text-soft)]">exception rate vs 5.0% expected</div>
                </div>
                <div className="mt-6 h-32 rounded-[var(--radius-md)] bg-[var(--color-bg)]" />
              </motion.div>

              <motion.div initial={{ opacity: 0, y: 16 }} whileInView={{ opacity: 1, y: 0 }} viewport={{ once: true, amount: 0.28 }} transition={{ ...transition, delay: 0.1 }}
                className="rounded-[var(--radius-lg)] border border-[var(--color-border)] bg-[var(--color-surface)] p-5">
                <div className="text-[10px] font-semibold uppercase tracking-wider text-[var(--color-text-muted)]">Desk logic</div>
                <div className="mt-4 grid gap-3 sm:grid-cols-2">
                  {([["Model regime", "Champion/challenger"], ["Budgeting", "Capital + headroom"], ["Decisioning", "Accept / reduce / reject"], ["Simulation", "Paper-trade follow-through"]] as const).map(([label, value]) => (
                    <div key={label} className="border-t border-[var(--color-border)] pt-3">
                      <div className="text-[9px] uppercase tracking-wider text-[var(--color-text-muted)]">{label}</div>
                      <div className="mt-1 text-[13px] font-semibold text-[var(--color-text)]">{value}</div>
                    </div>
                  ))}
                </div>
              </motion.div>
            </div>
          </div>
        </section>

        {/* Preview */}
        <section id="preview" className="px-6 py-20">
          <div className="mx-auto max-w-[1440px]">
            <SectionHeading title={t("previewTitle")} copy={t("previewBody")} align="center" />
            <div className="mt-10">
              <DeskPreview variant="full" />
            </div>
          </div>
        </section>

        {/* CTA */}
        <section className="px-6 py-20">
          <div className="mx-auto max-w-[1440px]">
            <motion.div initial={{ opacity: 0, y: 16 }} whileInView={{ opacity: 1, y: 0 }} viewport={{ once: true, amount: 0.35 }} transition={transition}
              className="flex flex-col items-start justify-between gap-6 rounded-[var(--radius-xl)] border border-[var(--color-border-strong)] bg-[var(--color-surface-strong)] px-8 py-8 md:flex-row md:items-end">
              <div className="max-w-2xl">
                <Eyebrow tone="accent">Desk Ready</Eyebrow>
                <h2 className="mt-4 text-balance text-3xl font-semibold tracking-tight text-[var(--color-text)] md:text-4xl">{t("finalTitle")}</h2>
                <p className="mt-3 text-[14px] text-[var(--color-text-soft)]">{t("finalBody")}</p>
              </div>
              <ButtonLink href="/desk">{t("finalPrimaryCta")} <ArrowRight className="ml-1.5 size-3.5" /></ButtonLink>
            </motion.div>
          </div>
        </section>
      </main>
    </div>
  );
}
