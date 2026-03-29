export const routing = {
  locales: ["en", "fr"] as const,
  defaultLocale: "en" as const,
};

export type AppLocale = (typeof routing.locales)[number];
