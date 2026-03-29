import type { Metadata } from "next";
import { IBM_Plex_Mono, Manrope } from "next/font/google";
import { QueryProvider } from "@/components/providers/query-provider";
import "./globals.css";

const manrope = Manrope({
  variable: "--font-ui",
  subsets: ["latin"],
});

const plexMono = IBM_Plex_Mono({
  variable: "--font-ui-mono",
  subsets: ["latin"],
  weight: ["400", "500", "600"],
});

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
      className={`${manrope.variable} ${plexMono.variable} h-full scroll-smooth`}
      suppressHydrationWarning
    >
      <body className="min-h-full bg-[var(--color-bg)] text-[var(--color-text)] antialiased">
        <QueryProvider>{children}</QueryProvider>
      </body>
    </html>
  );
}
