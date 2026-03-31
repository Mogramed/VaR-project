import type { ReactNode } from "react";
import Markdown from "react-markdown";
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

export function ReportDocument({ content, reportPath, showSource = true }: { content: string; reportPath: string; showSource?: boolean }) {
  function textFromNode(node: ReactNode): string {
    if (typeof node === "string" || typeof node === "number") return String(node);
    if (Array.isArray(node)) return node.map(textFromNode).join(" ");
    if (node && typeof node === "object" && "props" in node) return textFromNode((node as { props?: { children?: ReactNode } }).props?.children);
    return "";
  }

  return (
    <article className="rounded-[var(--radius-lg)] border border-[var(--color-border-strong)] bg-[var(--color-surface-strong)] p-5 report-document">
      <div className="report-prose max-w-none">
        <Markdown remarkPlugins={[remarkGfm]} components={{
          h1: ({ children }) => <h1 id={slugifyHeading(textFromNode(children))}>{children}</h1>,
          h2: ({ children }) => <h2 id={slugifyHeading(textFromNode(children))}>{children}</h2>,
          h3: ({ children }) => <h3 id={slugifyHeading(textFromNode(children))}>{children}</h3>,
          h4: ({ children }) => <h4 id={slugifyHeading(textFromNode(children))}>{children}</h4>,
        }}>{content}</Markdown>
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
