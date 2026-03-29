import { cva, type VariantProps } from "class-variance-authority";
import Link, { type LinkProps } from "next/link";
import { cn } from "@/lib/utils";

const badgeStyles = cva(
  "inline-flex items-center gap-2 rounded-full border px-3 py-1 text-[10px] font-semibold uppercase tracking-[0.24em] transition duration-300",
  {
    variants: {
      tone: {
        neutral:
          "border-white/10 bg-white/5 text-[var(--color-text-soft)]",
        accent:
          "border-[var(--color-border-strong)] bg-[var(--color-accent-soft)] text-[var(--color-accent)]",
        success:
          "border-emerald-400/20 bg-emerald-400/10 text-[var(--color-green)]",
        warning:
          "border-amber-400/20 bg-amber-400/10 text-[var(--color-amber)]",
        danger:
          "border-red-400/20 bg-red-400/10 text-[var(--color-red)]",
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
  "inline-flex items-center justify-center rounded-full transition duration-300 will-change-transform",
  {
    variants: {
      variant: {
        primary:
          "bg-[var(--color-accent)] px-5 py-3 text-sm font-semibold text-[#1c1408] motion-safe:hover:-translate-y-[1px] hover:shadow-[0_18px_44px_rgba(216,155,73,0.22)]",
        secondary:
          "border border-white/12 bg-white/5 px-5 py-3 text-sm font-semibold text-[var(--color-text)] motion-safe:hover:-translate-y-[1px] hover:border-[var(--color-border-strong)] hover:bg-white/8",
        ghost:
          "px-0 py-0 text-sm font-medium text-[var(--color-text-soft)] hover:text-[var(--color-text)]",
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
    <div className={cn("max-w-3xl space-y-4", align === "center" && "mx-auto text-center")}>
      <h2 className="text-balance text-3xl font-semibold tracking-[-0.04em] text-white md:text-5xl">
        {title}
      </h2>
      {copy ? (
        <p className="max-w-2xl text-sm leading-7 text-[var(--color-text-soft)] md:text-base">
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
