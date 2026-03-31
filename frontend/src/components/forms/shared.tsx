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

  return (
    <div className="rounded-[var(--radius-md)] border border-[var(--color-border)] bg-[var(--color-bg)] px-3 py-2.5">
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
        "inline-flex h-7 items-center justify-center rounded-[var(--radius-sm)] border px-2.5 text-[11px] font-medium transition-colors",
        active
          ? "border-[var(--color-accent)]/30 bg-[var(--color-accent-soft)] text-[var(--color-accent)]"
          : "border-[var(--color-border)] text-[var(--color-text-muted)] hover:border-[var(--color-border-strong)] hover:text-[var(--color-text-soft)]",
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
        "mt-1 h-9 w-full rounded-[var(--radius-md)] border border-[var(--color-border)] bg-[var(--color-bg)] px-3 text-[13px] text-[var(--color-text)] outline-none transition-colors placeholder:text-[var(--color-text-muted)] hover:border-[var(--color-border-strong)] focus:border-[var(--color-accent)]/40 focus:ring-1 focus:ring-[var(--color-accent)]/20",
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
