"use client";

import { type LucideIcon, Loader2 } from "lucide-react";
import { AnimatePresence, motion } from "framer-motion";
import { cn } from "@/lib/utils";

/* ─── Labels & Hints ─── */

export function FieldLabel({
  children,
  htmlFor,
}: {
  children: React.ReactNode;
  htmlFor?: string;
}) {
  return (
    <label
      htmlFor={htmlFor}
      className="block text-[10px] font-semibold uppercase tracking-wider text-[var(--color-text-muted)]"
    >
      {children}
    </label>
  );
}

export function FieldHint({ children }: { children: React.ReactNode }) {
  return (
    <p className="mt-1 text-[11px] text-[var(--color-text-muted)]">
      {children}
    </p>
  );
}

/* ─── Inputs ─── */

const inputBase =
  "h-9 w-full rounded-[var(--radius-md)] border border-[var(--color-border)] bg-[var(--color-bg)] text-[13px] text-[var(--color-text)] outline-none transition-colors placeholder:text-[var(--color-text-muted)] hover:border-[var(--color-border-strong)] focus:border-[var(--color-accent)]/40 focus:ring-1 focus:ring-[var(--color-accent)]/20";

export function FieldInput(props: React.InputHTMLAttributes<HTMLInputElement>) {
  return (
    <input
      {...props}
      className={cn("mt-1 px-3", inputBase, props.className)}
    />
  );
}

export function FieldInputWithIcon({
  icon: Icon,
  ...props
}: React.InputHTMLAttributes<HTMLInputElement> & { icon: LucideIcon }) {
  return (
    <div className="relative mt-1">
      <Icon className="pointer-events-none absolute left-2.5 top-1/2 size-3.5 -translate-y-1/2 text-[var(--color-text-muted)]" />
      <input
        {...props}
        className={cn("pl-8 pr-3", inputBase, props.className)}
      />
    </div>
  );
}

export function FieldSelect(props: React.SelectHTMLAttributes<HTMLSelectElement>) {
  return (
    <select
      {...props}
      className={cn(
        "mt-1 h-9 w-full rounded-[var(--radius-md)] border border-[var(--color-border)] bg-[var(--color-bg)] px-3 text-[13px] text-[var(--color-text)] outline-none transition-colors hover:border-[var(--color-border-strong)] focus:border-[var(--color-accent)]/40 focus:ring-1 focus:ring-[var(--color-accent)]/20",
        props.className,
      )}
    />
  );
}

export function FieldTextarea(props: React.TextareaHTMLAttributes<HTMLTextAreaElement>) {
  return (
    <textarea
      {...props}
      className={cn(
        "mt-1 min-h-20 w-full rounded-[var(--radius-md)] border border-[var(--color-border)] bg-[var(--color-bg)] px-3 py-2 text-[13px] text-[var(--color-text)] outline-none transition-colors placeholder:text-[var(--color-text-muted)] hover:border-[var(--color-border-strong)] focus:border-[var(--color-accent)]/40 focus:ring-1 focus:ring-[var(--color-accent)]/20",
        props.className,
      )}
    />
  );
}

/* ─── Metric Tile ─── */

const tileGradients: Record<string, string> = {
  accent: "bg-gradient-to-br from-[var(--color-accent-soft)] to-transparent",
  success: "bg-gradient-to-br from-[var(--color-green-soft)] to-transparent",
  warning: "bg-gradient-to-br from-[var(--color-amber-soft)] to-transparent",
  danger: "bg-gradient-to-br from-[var(--color-red-soft)] to-transparent",
};

export function FormMetaTile({
  label,
  value,
  hint,
  tone = "neutral",
}: {
  label: string;
  value: string;
  hint?: string;
  tone?: "neutral" | "accent" | "success" | "warning" | "danger";
}) {
  const valueColor =
    tone === "success"
      ? "text-[var(--color-green)]"
      : tone === "warning"
        ? "text-[var(--color-amber)]"
        : tone === "danger"
          ? "text-[var(--color-red)]"
          : tone === "accent"
            ? "text-[var(--color-accent)]"
            : "text-[var(--color-text)]";

  const bg = tileGradients[tone] ?? "bg-[var(--color-bg)]";

  return (
    <div className={cn("rounded-[var(--radius-md)] border border-[var(--color-border)] px-3 py-2.5", bg)}>
      <div className="text-[10px] font-medium uppercase tracking-wider text-[var(--color-text-muted)]">
        {label}
      </div>
      <div className={cn("mono mt-1 text-base font-semibold tracking-tight", valueColor)}>
        {value}
      </div>
      {hint ? (
        <div className="mt-0.5 text-[10px] text-[var(--color-text-muted)]">{hint}</div>
      ) : null}
    </div>
  );
}

/* ─── Preset Pill ─── */

export function PresetPill({
  active,
  children,
  onClick,
}: {
  active?: boolean;
  children: React.ReactNode;
  onClick?: () => void;
}) {
  return (
    <button
      type="button"
      onClick={onClick}
      className={cn(
        "inline-flex h-7 items-center justify-center rounded-[var(--radius-sm)] border px-2.5 text-[11px] font-medium transition-all duration-150",
        active
          ? "border-[var(--color-accent)]/30 bg-[var(--color-accent-soft)] text-[var(--color-accent)] shadow-[0_0_8px_rgba(216,155,73,0.15)]"
          : "border-[var(--color-border)] text-[var(--color-text-muted)] hover:border-[var(--color-border-strong)] hover:text-[var(--color-text-soft)] hover:scale-[1.02]",
      )}
    >
      {children}
    </button>
  );
}

/* ─── Submit Button ─── */

export function SubmitButton({
  isPending,
  label,
  pendingLabel,
  variant = "primary",
  ...props
}: React.ButtonHTMLAttributes<HTMLButtonElement> & {
  isPending: boolean;
  label: string;
  pendingLabel: string;
  variant?: "primary" | "secondary";
}) {
  return (
    <button
      type="submit"
      disabled={isPending}
      {...props}
      className={cn(
        "inline-flex h-8 items-center gap-1.5 rounded-[var(--radius-md)] px-4 text-[12px] font-semibold transition disabled:opacity-50",
        variant === "primary"
          ? "bg-[var(--color-accent)] text-[#1a1206] hover:brightness-110"
          : "border border-[var(--color-border-strong)] bg-[var(--color-surface)] text-[var(--color-text)] hover:bg-[var(--color-surface-hover)]",
        props.className,
      )}
    >
      {isPending ? <Loader2 className="size-3.5 animate-spin" /> : null}
      {isPending ? pendingLabel : label}
    </button>
  );
}

/* ─── Form Error ─── */

export function FormError({ message }: { message: string | null | undefined }) {
  return (
    <AnimatePresence>
      {message ? (
        <motion.span
          initial={{ opacity: 0, x: -6 }}
          animate={{ opacity: 1, x: 0 }}
          exit={{ opacity: 0 }}
          className="text-[11px] text-[var(--color-red)]"
        >
          {message}
        </motion.span>
      ) : null}
    </AnimatePresence>
  );
}

/* ─── Form Section ─── */

export function FormSection({ title, children }: { title?: string; children: React.ReactNode }) {
  return (
    <div className="mt-4 first:mt-0">
      {title ? (
        <div className="mb-2 flex items-center gap-2">
          <span className="text-[10px] font-semibold uppercase tracking-wider text-[var(--color-text-muted)]">{title}</span>
          <div className="h-px flex-1 bg-[var(--color-border)]" />
        </div>
      ) : null}
      {children}
    </div>
  );
}
