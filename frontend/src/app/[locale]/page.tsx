import { hasLocale } from "next-intl";
import { notFound } from "next/navigation";
import { LandingPage } from "@/components/landing/landing-page";
import { routing } from "@/i18n/routing";

export default async function LocalizedLandingPage({
  params,
}: {
  params: Promise<{ locale: string }>;
}) {
  const { locale } = await params;

  if (!hasLocale(routing.locales, locale)) {
    notFound();
  }

  return <LandingPage locale={locale as (typeof routing.locales)[number]} />;
}
