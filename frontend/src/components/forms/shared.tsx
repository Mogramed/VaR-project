import { cn } from "@/lib/utils";

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
      className="mono block text-[11px] uppercase tracking-[0.24em] text-[var(--color-text-muted)]"
    >
      {children}
    </label>
  );
}

export function FieldHint({ children }: { children: React.ReactNode }) {
  return (
    <p className="mt-2 text-xs leading-6 text-[var(--color-text-muted)]">
      {children}
    </p>
  );
}

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
  const accentClass =
    tone === "success"
      ? "text-[var(--color-green)]"
      : tone === "warning"
        ? "text-[var(--color-amber)]"
        : tone === "danger"
          ? "text-[var(--color-red)]"
        : tone === "accent"
          ? "text-[var(--color-accent)]"
          : "text-white";

  return (
    <div className="rounded-[1.35rem] border border-white/8 bg-black/18 px-4 py-4">
      <div className="mono text-[11px] uppercase tracking-[0.24em] text-[var(--color-text-muted)]">
        {label}
      </div>
      <div className={cn("mt-3 text-xl font-semibold tracking-[-0.04em]", accentClass)}>
        {value}
      </div>
      {hint ? (
        <div className="mt-2 text-xs leading-6 text-[var(--color-text-muted)]">
          {hint}
        </div>
      ) : null}
    </div>
  );
}

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
        "inline-flex h-9 items-center justify-center rounded-full border px-3 text-xs font-medium uppercase tracking-[0.18em] transition duration-300",
        active
          ? "border-[var(--color-border-strong)] bg-[var(--color-accent-soft)] text-[var(--color-accent)]"
          : "border-white/8 bg-white/[0.03] text-[var(--color-text-soft)] hover:border-white/14 hover:bg-white/[0.05] hover:text-white",
      )}
    >
      {children}
    </button>
  );
}

export function FieldInput(props: React.InputHTMLAttributes<HTMLInputElement>) {
  return (
    <input
      {...props}
      className={cn(
        "mt-2 h-12 w-full rounded-2xl border border-white/8 bg-white/[0.04] px-4 text-sm text-white shadow-[inset_0_1px_0_rgba(255,255,255,0.02)] outline-none transition duration-200 placeholder:text-[var(--color-text-muted)] hover:border-white/14 hover:bg-white/[0.05] focus:border-[var(--color-border-strong)] focus:bg-white/[0.06] focus:ring-2 focus:ring-[rgba(216,155,73,0.16)]",
        props.className,
      )}
    />
  );
}

export function FieldSelect(props: React.SelectHTMLAttributes<HTMLSelectElement>) {
  return (
    <select
      {...props}
      className={cn(
        "mt-2 h-12 w-full rounded-2xl border border-white/8 bg-white/[0.04] px-4 text-sm text-white shadow-[inset_0_1px_0_rgba(255,255,255,0.02)] outline-none transition duration-200 hover:border-white/14 hover:bg-white/[0.05] focus:border-[var(--color-border-strong)] focus:bg-white/[0.06] focus:ring-2 focus:ring-[rgba(216,155,73,0.16)]",
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
        "mt-2 min-h-28 w-full rounded-2xl border border-white/8 bg-white/[0.04] px-4 py-3 text-sm text-white shadow-[inset_0_1px_0_rgba(255,255,255,0.02)] outline-none transition duration-200 placeholder:text-[var(--color-text-muted)] hover:border-white/14 hover:bg-white/[0.05] focus:border-[var(--color-border-strong)] focus:bg-white/[0.06] focus:ring-2 focus:ring-[rgba(216,155,73,0.16)]",
        props.className,
      )}
    />
  );
}
