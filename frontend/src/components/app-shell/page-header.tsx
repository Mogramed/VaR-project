import { cn } from "@/lib/utils";

export function PageHeader({
  eyebrow,
  title,
  description,
  aside,
  className,
}: {
  eyebrow: string;
  title: string;
  description?: string;
  aside?: React.ReactNode;
  className?: string;
}) {
  return (
    <div className={cn("flex items-center justify-between gap-4", className)}>
      <div className="min-w-0">
        <div className="flex items-center gap-2">
          <span className="text-[10px] font-semibold uppercase tracking-wider text-[var(--color-accent)]">
            {eyebrow}
          </span>
        </div>
        <h1 className="mt-0.5 truncate text-lg font-semibold tracking-tight text-[var(--color-text)]">
          {title}
        </h1>
      </div>
      {aside ? <div className="flex shrink-0 items-center gap-2">{aside}</div> : null}
    </div>
  );
}
