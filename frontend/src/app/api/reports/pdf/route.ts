import { NextRequest, NextResponse } from "next/server";
import { launchPdfBrowser } from "@/lib/pdf-browser";

export const runtime = "nodejs";
export const dynamic = "force-dynamic";

function getInternalBaseUrl() {
  return (process.env.NEXT_INTERNAL_BASE_URL ?? `http://127.0.0.1:${process.env.PORT ?? 3000}`).replace(
    /\/+$/,
    "",
  );
}

function buildFilename(portfolioSlug?: string | null) {
  const slug = (portfolioSlug ?? "desk").replace(/[^\w-]+/g, "_");
  const stamp = new Date().toISOString().slice(0, 10);
  return `fx-risk-report-${slug}-${stamp}.pdf`;
}

export async function GET(request: NextRequest) {
  const portfolioSlug = request.nextUrl.searchParams.get("portfolio");
  const target = new URL("/report-export", getInternalBaseUrl());
  if (portfolioSlug) {
    target.searchParams.set("portfolio", portfolioSlug);
  }

  const browser = await launchPdfBrowser();
  try {
    const page = await browser.newPage({
      colorScheme: "dark",
      viewport: { width: 1440, height: 2000 },
    });

    await page.goto(target.toString(), { waitUntil: "networkidle", timeout: 120000 });
    await page.emulateMedia({ media: "screen" });
    await page.waitForSelector("[data-report-export-root='true']", { timeout: 30000 });
    await page.waitForFunction(
      () => {
        const charts = document.querySelectorAll("[data-chart-surface]");
        if (charts.length === 0) {
          return true;
        }
        return Array.from(charts).every((node) => node.querySelector("svg"));
      },
      { timeout: 30000 },
    );
    await page.waitForTimeout(400);

    const pdf = await page.pdf({
      format: "A4",
      printBackground: true,
      margin: { top: "18mm", right: "14mm", bottom: "18mm", left: "14mm" },
      displayHeaderFooter: true,
      headerTemplate: `<div style="width:100%;font-size:8px;padding:4px 14mm 6px;color:#7e848f;display:flex;justify-content:space-between;align-items:center;border-bottom:1px solid rgba(255,255,255,0.06)">
          <span style="text-transform:uppercase;letter-spacing:0.22em;font-family:monospace">FX Risk Desk Platform</span>
          <span style="color:#d89b49;font-size:6px">\u25CF</span>
        </div>`,
      footerTemplate: `<div style="width:100%;font-size:8px;padding:6px 14mm 4px;color:#7e848f;display:flex;justify-content:space-between;align-items:center;border-top:1px solid rgba(255,255,255,0.06)">
          <span style="letter-spacing:0.08em">Confidential \u2014 Internal Use Only</span>
          <span style="font-family:monospace">Page <span class="pageNumber"></span> of <span class="totalPages"></span></span>
        </div>`,
    });

    return new NextResponse(Buffer.from(pdf), {
      status: 200,
      headers: {
        "Content-Type": "application/pdf",
        "Content-Disposition": `attachment; filename="${buildFilename(portfolioSlug)}"`,
        "Cache-Control": "no-store",
      },
    });
  } catch (error) {
    const detail =
      error instanceof Error ? error.message : "Failed to generate PDF report.";
    return NextResponse.json({ detail }, { status: 500 });
  } finally {
    await browser.close();
  }
}
