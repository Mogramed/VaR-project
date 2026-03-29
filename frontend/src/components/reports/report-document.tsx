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
    .filter((value): value is RegExpMatchArray => value !== null)
    .map((match) => ({
      level: match[1].length,
      text: match[2].trim(),
      id: slugifyHeading(match[2]),
    }));
}

export function ReportDocument({
  content,
  reportPath,
  showSource = true,
}: {
  content: string;
  reportPath: string;
  showSource?: boolean;
}) {
  function textFromNode(node: ReactNode): string {
    if (typeof node === "string" || typeof node === "number") {
      return String(node);
    }
    if (Array.isArray(node)) {
      return node.map((item) => textFromNode(item)).join(" ");
    }
    if (node && typeof node === "object" && "props" in node) {
      return textFromNode((node as { props?: { children?: ReactNode } }).props?.children);
    }
    return "";
  }

  return (
    <article className="surface-strong rounded-[2rem] border border-white/10 px-6 py-8 md:px-8 report-document">
      <div className="report-prose max-w-none">
        <Markdown
          remarkPlugins={[remarkGfm]}
          components={{
            h1: ({ children }) => {
              const text = textFromNode(children);
              return <h1 id={slugifyHeading(text)}>{children}</h1>;
            },
            h2: ({ children }) => {
              const text = textFromNode(children);
              return <h2 id={slugifyHeading(text)}>{children}</h2>;
            },
            h3: ({ children }) => {
              const text = textFromNode(children);
              return <h3 id={slugifyHeading(text)}>{children}</h3>;
            },
            h4: ({ children }) => {
              const text = textFromNode(children);
              return <h4 id={slugifyHeading(text)}>{children}</h4>;
            },
          }}
        >
          {content}
        </Markdown>
      </div>
      {showSource ? (
        <div className="mt-10 border-t border-white/8 pt-5 text-sm text-[var(--color-text-muted)] print:text-neutral-500">
          Source markdown: <span className="mono break-all">{reportPath}</span>
        </div>
      ) : null}
    </article>
  );
}

export function ReportTableOfContents({
  headings,
  reportPath,
  chartCount,
}: {
  headings: ReportHeading[];
  reportPath: string;
  chartCount: number;
}) {
  return (
    <aside className="space-y-5 print:hidden">
      <section className="surface rounded-[1.7rem] p-5">
        <div className="mono text-[11px] uppercase tracking-[0.28em] text-[var(--color-text-muted)]">
          Document map
        </div>
        <div className="mt-4 space-y-2">
          {headings.length === 0 ? (
            <div className="text-sm text-[var(--color-text-muted)]">
              No markdown headings available in the current report.
            </div>
          ) : (
            headings
              .filter((item) => item.level <= 3)
              .map((heading) => (
                <a
                  key={heading.id}
                  href={`#${heading.id}`}
                  className="block rounded-xl px-3 py-2 text-sm text-[var(--color-text-soft)] transition hover:bg-white/[0.04] hover:text-white"
                  style={{ paddingLeft: `${0.75 + (heading.level - 1) * 0.7}rem` }}
                >
                  {heading.text}
                </a>
              ))
          )}
        </div>
      </section>

      <section className="surface rounded-[1.7rem] p-5">
        <div className="mono text-[11px] uppercase tracking-[0.28em] text-[var(--color-text-muted)]">
          Report details
        </div>
        <div className="mt-4 space-y-4 text-sm text-[var(--color-text-soft)]">
          <div className="flex items-center justify-between gap-3">
            <span>Charts embedded</span>
            <span className="mono text-white">{chartCount}</span>
          </div>
          <div className="space-y-2">
            <div>Markdown source</div>
            <div className="mono break-all text-[11px] text-[var(--color-text-muted)]">
              {reportPath}
            </div>
          </div>
        </div>
      </section>
    </aside>
  );
}
