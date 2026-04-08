import createNextIntlPlugin from "next-intl/plugin";
import path from "node:path";
import type { NextConfig } from "next";

const withNextIntl = createNextIntlPlugin("./src/i18n/request.ts");

const nextConfig: NextConfig = {
  reactStrictMode: true,
  output: "standalone",
  turbopack: {
    root: path.resolve(__dirname),
  },
  async redirects() {
    return [
      {
        source: "/desk/simulation",
        destination: "/desk/blotter",
        permanent: true,
      },
    ];
  },
};

export default withNextIntl(nextConfig);
