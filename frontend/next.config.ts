import createNextIntlPlugin from "next-intl/plugin";
import type { NextConfig } from "next";

const withNextIntl = createNextIntlPlugin("./src/i18n/request.ts");

const nextConfig: NextConfig = {
  reactStrictMode: true,
  output: "standalone",
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
