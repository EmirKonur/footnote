"use client";

import { Keyboard, Lamp, Moon } from "lucide-react";
import Link from "next/link";
import { useCallback, useSyncExternalStore } from "react";

import { useShell } from "@/components/app-shell";
import { useHotkeys } from "@/components/shortcuts";

type Theme = "night" | "reading-room";

function readTheme(): Theme {
  return document.documentElement.dataset.theme === "reading-room" ? "reading-room" : "night";
}

function subscribe(onChange: () => void): () => void {
  const observer = new MutationObserver(onChange);
  observer.observe(document.documentElement, { attributes: true, attributeFilter: ["data-theme"] });
  return () => observer.disconnect();
}

export function useTheme(): [Theme, () => void] {
  const theme = useSyncExternalStore(subscribe, readTheme, () => "night" as const);
  const toggle = useCallback(() => {
    const next: Theme = readTheme() === "night" ? "reading-room" : "night";
    document.documentElement.dataset.theme = next;
    try {
      localStorage.setItem("footnote-theme", next);
    } catch {
      // Private browsing can refuse storage; the theme still applies for this visit.
    }
  }, []);
  return [theme, toggle];
}

export function Logo() {
  return (
    <svg viewBox="0 0 64 64" aria-hidden="true">
      <rect width="64" height="64" rx="12" fill="var(--bg-sunken)" />
      <rect x="2.5" y="2.5" width="59" height="59" rx="10" fill="none" stroke="var(--accent)" strokeOpacity="0.4" />
      <circle cx="32" cy="12.5" r="3.4" fill="var(--accent)" />
      <path d="M29.2 16.5h5.6l-1.1 31.8L32 55l-1.7-6.7z" fill="var(--accent)" />
      <path
        d="M19.5 22.2c0-1.3 1-2.2 2.3-2.2h20.4c1.3 0 2.3.9 2.3 2.2s-1 2.3-2.3 2.3H21.8c-1.3 0-2.3-1-2.3-2.3z"
        fill="var(--ink)"
      />
    </svg>
  );
}

function ModeBadge() {
  const { health } = useShell();
  if (!health) return null;
  if (health.generation === "disabled") {
    return (
      <span
        className="tag"
        title="No Anthropic credentials found. Questions return ranked passages; set ANTHROPIC_API_KEY for cited answers."
      >
        Retrieval only
      </span>
    );
  }
  if (health.generation === "fixtures") {
    return (
      <span
        className="tag tag--accent"
        title="FOOTNOTE_FAKE_LLM=fixtures: answers are assembled from retrieved sentences by word overlap. No model is called."
      >
        Fixture answers
      </span>
    );
  }
  return (
    <span className="tag tag--accent" title="Answers are written by Claude with API citations">
      {health.model}
    </span>
  );
}

export function AppHeader() {
  const { crumbs, openShortcuts } = useShell();
  const [theme, toggleTheme] = useTheme();
  useHotkeys({ t: toggleTheme, "?": openShortcuts });
  const nextName = theme === "night" ? "reading room" : "night";

  return (
    <header className="app-header">
      <Link href="/" className="brand" aria-label="Footnote library">
        <Logo />
        <span>
          Footnote<sup aria-hidden>†</sup>
        </span>
      </Link>
      {crumbs && <nav className="app-header__crumbs" aria-label="Breadcrumb">{crumbs}</nav>}
      <div className="app-header__end">
        <ModeBadge />
        <button
          type="button"
          className="button button--quiet"
          onClick={toggleTheme}
          aria-label={`Switch to the ${nextName} theme`}
          title={`Switch to the ${nextName} theme (t)`}
        >
          {theme === "night" ? <Lamp className="icon" aria-hidden /> : <Moon className="icon" aria-hidden />}
        </button>
        <button
          type="button"
          className="button button--quiet"
          onClick={openShortcuts}
          aria-label="Keyboard shortcuts"
          title="Keyboard shortcuts (?)"
        >
          <Keyboard className="icon" aria-hidden />
        </button>
      </div>
    </header>
  );
}
