import { cva, type VariantProps } from "class-variance-authority";
import Link, { type LinkProps } from "next/link";
import { cn } from "@/lib/utils";

const badgeStyles = cva(
  "inline-flex items-center rounded-[3px] px-1.5 py-0.5 text-[10px] font-semibold uppercase tracking-wide",
  {
    variants: {
      tone: {
        neutral: "bg-white/5 text-[var(--color-text-muted)]",
        accent: "bg-[var(--color-accent-soft)] text-[var(--color-accent)]",
        success: "bg-[var(--color-green-soft)] text-[var(--color-green)]",
        warning: "bg-[var(--color-amber-soft)] text-[var(--color-amber)]",
        danger: "bg-[var(--color-red-soft)] text-[var(--color-red)]",
      },
    },
    defaultVariants: {
      tone: "neutral",
    },
  },
);

export function Eyebrow({
  children,
  tone,
  className,
}: React.HTMLAttributes<HTMLDivElement> & VariantProps<typeof badgeStyles>) {
  return <div className={cn(badgeStyles({ tone }), className)}>{children}</div>;
}

const buttonStyles = cva(
  "inline-flex items-center justify-center rounded-[var(--radius-md)] font-medium transition-colors duration-150",
  {
    variants: {
      variant: {
        primary:
          "bg-[var(--color-accent)] px-4 py-2 text-[13px] text-[#1a1206] hover:brightness-110",
        secondary:
          "border border-[var(--color-border-strong)] bg-[var(--color-surface)] px-4 py-2 text-[13px] text-[var(--color-text)] hover:bg-[var(--color-surface-hover)]",
        ghost:
          "px-2 py-1 text-[13px] text-[var(--color-text-soft)] hover:text-[var(--color-text)]",
      },
    },
    defaultVariants: {
      variant: "primary",
    },
  },
);

type ButtonLinkProps = LinkProps &
  React.AnchorHTMLAttributes<HTMLAnchorElement> &
  VariantProps<typeof buttonStyles>;

export function ButtonLink({
  children,
  className,
  variant,
  ...props
}: ButtonLinkProps) {
  return (
    <Link className={cn(buttonStyles({ variant }), className)} {...props}>
      {children}
    </Link>
  );
}

export function SectionHeading({
  title,
  copy,
  align = "left",
}: {
  title: string;
  copy?: string;
  align?: "left" | "center";
}) {
  return (
    <div className={cn("max-w-3xl space-y-2", align === "center" && "mx-auto text-center")}>
      <h2 className="text-balance text-2xl font-semibold tracking-tight text-[var(--color-text)] md:text-4xl">
        {title}
      </h2>
      {copy ? (
        <p className="max-w-2xl text-sm text-[var(--color-text-soft)]">
          {copy}
        </p>
      ) : null}
    </div>
  );
}

export function StatusBadge({
  label,
  tone = "neutral",
}: {
  label: string;
  tone?: VariantProps<typeof badgeStyles>["tone"];
}) {
  return <span className={badgeStyles({ tone })}>{label}</span>;
}
