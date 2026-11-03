import type { NextConfig } from "next";

const config: NextConfig = {
  // The Docker image runs the standalone server, which carries only the files it needs.
  output: "standalone",
  reactStrictMode: true,
  poweredByHeader: false,
};

export default config;
