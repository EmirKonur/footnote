"use client";

import { RefreshCw } from "lucide-react";
import { createContext, use, useCallback, useEffect, useMemo, useState, type ReactNode } from "react";

import { AppHeader } from "@/components/app-header";
import { ShortcutsDialog } from "@/components/shortcuts";
import { createApi, type FootnoteApi } from "@/lib/api";
import type { Health } from "@/lib/types";

interface ShellContext {
  api: FootnoteApi;
  health: Health | null;
  offline: boolean;
  retryHealth: () => void;
  crumbs: ReactNode;
  setCrumbs: (crumbs: ReactNode) => void;
  openShortcuts: () => void;
}

const Context = createContext<ShellContext | null>(null);

export function useShell(): ShellContext {
  const value = use(Context);
  if (!value) throw new Error("useShell must be used inside <AppShell>");
  return value;
}

export function AppShell({ apiUrl, children }: { apiUrl: string; children: ReactNode }) {
  const api = useMemo(() => createApi(apiUrl), [apiUrl]);
  const [health, setHealth] = useState<Health | null>(null);
  const [offline, setOffline] = useState(false);
  const [crumbs, setCrumbs] = useState<ReactNode>(null);
  const [shortcutsOpen, setShortcutsOpen] = useState(false);
  const [attempt, setAttempt] = useState(0);

  useEffect(() => {
    let cancelled = false;
    api
      .health()
      .then((value) => {
        if (cancelled) return;
        setHealth(value);
        setOffline(false);
      })
      .catch(() => {
        if (!cancelled) setOffline(true);
      });
    return () => {
      cancelled = true;
    };
  }, [api, attempt]);

  const retryHealth = useCallback(() => setAttempt((n) => n + 1), []);
  const openShortcuts = useCallback(() => setShortcutsOpen(true), []);
  const value = useMemo(
    () => ({ api, health, offline, retryHealth, crumbs, setCrumbs, openShortcuts }),
    [api, health, offline, retryHealth, crumbs, openShortcuts],
  );

  return (
    <Context value={value}>
      <AppHeader />
      {offline && (
        <div className="offline-banner" role="alert">
          <span>
            Can&apos;t reach the Footnote API at {api.baseUrl}. Start it with <code>make api</code>, then retry.
          </span>
          <button type="button" className="button" onClick={retryHealth}>
            <RefreshCw className="icon" aria-hidden />
            Retry
          </button>
        </div>
      )}
      <main id="main">{children}</main>
      <ShortcutsDialog open={shortcutsOpen} onClose={() => setShortcutsOpen(false)} />
    </Context>
  );
}
