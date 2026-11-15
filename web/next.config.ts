import type { NextConfig } from "next";

const config: NextConfig = {
  output: "standalone",
  poweredByHeader: false,
  experimental: { proxyClientMaxBodySize: "64mb" },
  async rewrites() {
    return [
      { source: "/api/:path*", destination: "http://server:8000/api/:path*" },
    ];
  },
};
export default config;
