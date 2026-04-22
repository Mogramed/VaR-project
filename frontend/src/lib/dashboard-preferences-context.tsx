"use client";

import { createContext, useContext } from "react";
import type { DashboardPreferencesAPI } from "@/lib/dashboard-preferences";

const DashboardPreferencesContext = createContext<DashboardPreferencesAPI | null>(null);

export function DashboardPreferencesProvider({
  value,
  children,
}: {
  value: DashboardPreferencesAPI;
  children: React.ReactNode;
}) {
  return (
    <DashboardPreferencesContext.Provider value={value}>
      {children}
    </DashboardPreferencesContext.Provider>
  );
}

export function useDashboardPrefs(): DashboardPreferencesAPI {
  const ctx = useContext(DashboardPreferencesContext);
  if (!ctx) {
    throw new Error("useDashboardPrefs must be used within DashboardPreferencesProvider");
  }
  return ctx;
}
