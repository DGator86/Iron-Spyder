/** @type {import('next').NextConfig} */
const nextConfig = {
  reactStrictMode: true,
  // ECharts ships an ESM build that benefits from tree-shaking at the app layer.
  transpilePackages: ["echarts"],
};

export default nextConfig;
