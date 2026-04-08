import { Suspense } from "react";
import { DeskChrome } from "@/components/app-shell/desk-chrome";
import { api } from "@/lib/api/client";

export default async function DeskLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  const [health, jobsStatus, portfolios, activeAlerts, recentAlerts, audit] = await Promise.all([
    api.safeHealth(),
    api.jobsStatus().catch(() => null),
    api.portfolios().catch(() => []),
    api.activeAlerts(8).catch(() => []),
    api.recentAlerts(8).catch(() => []),
    api.recentAudit(undefined, 8).catch(() => []),
  ]);

  return (
    <Suspense>
      <DeskChrome
        health={health}
        jobsStatus={jobsStatus}
        portfolios={portfolios}
        activeAlerts={activeAlerts}
        recentAlerts={recentAlerts}
        audit={audit}
      >
        {children}
      </DeskChrome>
    </Suspense>
  );
}
