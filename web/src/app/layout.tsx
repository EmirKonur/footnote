import "@fontsource-variable/newsreader/opsz.css";
import "@fontsource-variable/newsreader/opsz-italic.css";
import "@fontsource/ibm-plex-mono/400.css";
import "@fontsource/ibm-plex-mono/500.css";
import "./globals.css";

import type { Metadata, Viewport } from "next";
import { connection } from "next/server";
import type { ReactNode } from "react";

import { AppShell } from "@/components/app-shell";

export const metadata: Metadata = {
  title: { default: "Footnote", template: "%s · Footnote" },
  description: "Ask your documents. Every claim carries a footnote you can check.",
};

export const viewport: Viewport = {
  themeColor: [
    { media: "(prefers-color-scheme: dark)", color: "#0b1428" },
    { media: "(prefers-color-scheme: light)", color: "#f5efe2" },
  ],
};

// Runs before first paint so the stored theme applies without a flash of the other one.
const themeScript = `(() => {
  try {
    const stored = localStorage.getItem("footnote-theme");
    const theme = stored === "night" || stored === "reading-room"
      ? stored
      : matchMedia("(prefers-color-scheme: light)").matches ? "reading-room" : "night";
    document.documentElement.dataset.theme = theme;
  } catch {
    document.documentElement.dataset.theme = "night";
  }
})();`;

export default async function RootLayout({ children }: { children: ReactNode }) {
  // Read the API location per request rather than baking it in at build time, so one
  // image works behind any address (and the e2e suite can point two servers at two APIs).
  await connection();
  const apiUrl = process.env.FOOTNOTE_API_URL ?? "http://localhost:4410";
  return (
    <html lang="en" suppressHydrationWarning>
      <head>
        <script dangerouslySetInnerHTML={{ __html: themeScript }} />
      </head>
      <body>
        <a className="skip-link" href="#main">
          Skip to content
        </a>
        <AppShell apiUrl={apiUrl}>{children}</AppShell>
      </body>
    </html>
  );
}
