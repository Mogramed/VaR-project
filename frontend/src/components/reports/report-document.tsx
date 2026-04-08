import type { ReactNode } from "react";
import Markdown from "react-markdown";
import type { Components } from "react-markdown";
import remarkGfm from "remark-gfm";
import { slugifyHeading } from "@/lib/utils";

export interface ReportHeading {
  level: number;
  text: string;
  id: string;
}

export function extractMarkdownHeadings(markdown: string): ReportHeading[] {
  return markdown
    .split(/\r?\n/)
    .map((line) => line.match(/^(#{1,4})\s+(.+)$/))
    .filter((v): v is RegExpMatchArray => v !== null)
    .map((m) => ({ level: m[1].length, text: m[2].trim(), id: slugifyHeading(m[2]) }));
}

function chartNameFromSrc(src: string): string | null {
  const clean = src.split(/[?#]/)[0] ?? "";
  const parts = clean.split(/[\\/]/).filter(Boolean);
  const name = parts.length ? parts[parts.length - 1] : "";
  return name ? name : null;
}

function isAbsoluteImageSource(src: string): boolean {
  if (src.startsWith("/desk/")) {
    return false;
  }
  return /^https?:\/\//i.test(src) || src.startsWith("data:") || src.startsWith("/");
}

function resolveReportImageSource({
  src,
  portfolioSlug,
  reportId,
  allowedChartNames,
}: {
  src: string;
  portfolioSlug?: string;
  reportId?: number | null;
  allowedChartNames: Set<string> | null;
}): string | null {
  if (isAbsoluteImageSource(src)) {
    return src;
  }
  const chartName = chartNameFromSrc(src);
  if (!chartName) {
    return null;
  }
  if (allowedChartNames && !allowedChartNames.has(chartName.toLowerCase())) {
    return null;
  }
  const params = new URLSearchParams();
  if (portfolioSlug) {
    params.set("portfolio_slug", portfolioSlug);
  }
  if (reportId != null && Number.isFinite(reportId)) {
    params.set("report_id", String(reportId));
  }
  const query = params.toString();
  return `/api/proxy/reports/charts/${encodeURIComponent(chartName)}${query ? `?${query}` : ""}`;
}

export function ReportDocument({
  content,
  reportPath,
  showSource = true,
  portfolioSlug,
  reportId,
  chartPaths = [],
}: {
  content: string;
  reportPath: string;
  showSource?: boolean;
  portfolioSlug?: string;
  reportId?: number | null;
  chartPaths?: string[];
}) {
  function textFromNode(node: ReactNode): string {
    if (typeof node === "string" || typeof node === "number") return String(node);
    if (Array.isArray(node)) return node.map(textFromNode).join(" ");
    if (node && typeof node === "object" && "props" in node) return textFromNode((node as { props?: { children?: ReactNode } }).props?.children);
    return "";
  }

  const allowedChartNames = chartPaths.length
    ? new Set(
      chartPaths
        .map((value) => chartNameFromSrc(String(value)))
        .filter((value): value is string => Boolean(value))
        .map((value) => value.toLowerCase()),
    )
    : null;

  const components: Components = {
    h1: ({ children }) => <h1 id={slugifyHeading(textFromNode(children))}>{children}</h1>,
    h2: ({ children }) => <h2 id={slugifyHeading(textFromNode(children))}>{children}</h2>,
    h3: ({ children }) => <h3 id={slugifyHeading(textFromNode(children))}>{children}</h3>,
    h4: ({ children }) => <h4 id={slugifyHeading(textFromNode(children))}>{children}</h4>,
    img: (props) => {
      const src = (props as { src?: unknown }).src;
      const alt = (props as { alt?: unknown }).alt;
      if (typeof src !== "string" || !src.trim()) {
        return null;
      }
      const resolvedSrc = resolveReportImageSource({
        src,
        portfolioSlug,
        reportId,
        allowedChartNames,
      });
      if (!resolvedSrc) {
        return null;
      }
      /* eslint-disable-next-line @next/next/no-img-element */
      return <img
        src={resolvedSrc}
        alt={typeof alt === "string" ? alt : ""}
        loading="lazy"
        className="my-3 rounded-[var(--radius-md)] border border-[var(--color-border)]"
      />;
    },
  };

  return (
    <article className="rounded-[var(--radius-lg)] border border-[var(--color-border-strong)] bg-[var(--color-surface-strong)] p-5 report-document">
      <div className="report-prose max-w-none">
        <Markdown remarkPlugins={[remarkGfm]} components={components}>
          {content}
        </Markdown>
      </div>
      {showSource ? (
        <div className="mt-6 border-t border-[var(--color-border)] pt-3 text-[11px] text-[var(--color-text-muted)]">
          Source: <span className="mono">{reportPath}</span>
        </div>
      ) : null}
    </article>
  );
}

export function ReportTableOfContents({ headings, reportPath, chartCount }: { headings: ReportHeading[]; reportPath: string; chartCount: number }) {
  return (
    <aside className="space-y-3 print:hidden">
      <div className="rounded-[var(--radius-lg)] border border-[var(--color-border)] bg-[var(--color-surface)] p-3.5">
        <div className="mb-2 text-[10px] font-semibold uppercase tracking-wider text-[var(--color-text-muted)]">Contents</div>
        {headings.length === 0 ? (
          <p className="text-[11px] text-[var(--color-text-muted)]">No headings found.</p>
        ) : (
          <nav className="space-y-0.5">
            {headings.filter((h) => h.level <= 3).map((h) => (
              <a key={h.id} href={`#${h.id}`}
                className="block rounded-[var(--radius-sm)] px-2 py-1 text-[11px] text-[var(--color-text-soft)] transition-colors hover:bg-[var(--color-surface-hover)] hover:text-[var(--color-text)]"
                style={{ paddingLeft: `${0.5 + (h.level - 1) * 0.6}rem` }}>
                {h.text}
              </a>
            ))}
          </nav>
        )}
      </div>
      <div className="rounded-[var(--radius-lg)] border border-[var(--color-border)] bg-[var(--color-surface)] p-3.5">
        <div className="mb-2 text-[10px] font-semibold uppercase tracking-wider text-[var(--color-text-muted)]">Details</div>
        <div className="space-y-1.5 text-[11px]">
          <div className="flex justify-between"><span className="text-[var(--color-text-muted)]">Charts</span><span className="mono text-[var(--color-text)]">{chartCount}</span></div>
          <div className="space-y-1"><span className="text-[var(--color-text-muted)]">Source</span><div className="mono break-all text-[10px] text-[var(--color-text-muted)]">{reportPath}</div></div>
        </div>
      </div>
    </aside>
  );
}
