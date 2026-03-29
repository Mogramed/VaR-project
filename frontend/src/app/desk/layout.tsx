import { DeskChrome } from "@/components/app-shell/desk-chrome";
import { api } from "@/lib/api/client";

export default async function DeskLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  const [health, jobsStatus, portfolios, alerts, audit] = await Promise.all([
    api.health(),
    api.jobsStatus().catch(() => null),
    api.portfolios(),
    api.recentAlerts(8).catch(() => []),
    api.recentAudit(undefined, 8).catch(() => []),
  ]);

  return (
    <DeskChrome health={health} jobsStatus={jobsStatus} portfolios={portfolios} alerts={alerts} audit={audit}>
      {children}
    </DeskChrome>
  );
}
