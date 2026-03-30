import type { Metadata } from "next";
import type { CSSProperties } from "react";
import { QueryProvider } from "@/components/providers/query-provider";
import "./globals.css";

const rootFontStyle = {
  "--font-ui": '"Aptos", "Segoe UI Variable", "Segoe UI", system-ui, sans-serif',
  "--font-ui-mono": '"Cascadia Code", "IBM Plex Mono", "Consolas", monospace',
} as CSSProperties;

export const metadata: Metadata = {
  title: {
    default: "VaR Risk Desk Platform",
    template: "%s | VaR Risk Desk Platform",
  },
  description:
    "A premium MT5-first risk desk for portfolio VaR, execution guardrails, reconciliation and audit.",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html
      lang="en"
      className="h-full scroll-smooth"
      style={rootFontStyle}
      suppressHydrationWarning
    >
      <body className="min-h-full bg-[var(--color-bg)] text-[var(--color-text)] antialiased">
        <QueryProvider>{children}</QueryProvider>
      </body>
    </html>
  );
}
