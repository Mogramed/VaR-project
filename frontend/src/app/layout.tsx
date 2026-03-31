import type { Metadata } from "next";
import type { CSSProperties } from "react";
import { QueryProvider } from "@/components/providers/query-provider";
import "./globals.css";

const rootFontStyle = {
  "--font-ui": '"Aptos", "Segoe UI Variable", "Segoe UI", -apple-system, BlinkMacSystemFont, sans-serif',
  "--font-ui-mono": '"Cascadia Code", "IBM Plex Mono", "Consolas", "SF Mono", monospace',
} as CSSProperties;

export const metadata: Metadata = {
  title: {
    default: "VaR Risk Desk",
    template: "%s | VaR Risk Desk",
  },
  description: "MT5-first risk desk for portfolio VaR, execution guardrails, reconciliation and audit.",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html
      lang="en"
      className="h-full"
      style={rootFontStyle}
      suppressHydrationWarning
    >
      <body className="h-full bg-[var(--color-bg)] text-[var(--color-text)] antialiased">
        <QueryProvider>{children}</QueryProvider>
      </body>
    </html>
  );
}
