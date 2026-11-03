"use client";

import { useEffect, useRef } from "react";

/** Keys handled while focus is outside text fields. Escape is always delivered. */
export function useHotkeys(bindings: Record<string, () => void>): void {
  const latest = useRef(bindings);
  useEffect(() => {
    latest.current = bindings;
  });
  useEffect(() => {
    function onKey(event: KeyboardEvent) {
      if (event.metaKey || event.ctrlKey || event.altKey) return;
      const target = event.target as HTMLElement | null;
      const typing = target?.closest("input, textarea, select, [contenteditable='true']");
      if (typing && event.key !== "Escape") return;
      const action = latest.current[event.key];
      if (!action) return;
      event.preventDefault();
      action();
    }
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, []);
}

const SHORTCUTS: [string, string][] = [
  ["/", "Focus the question box"],
  ["n", "Start a new conversation"],
  ["u", "Upload documents"],
  ["Esc", "Stop the answer that is streaming"],
  ["[ ]", "Previous or next highlight in the viewer"],
  ["b", "Back (viewer to conversation, collection to library)"],
  ["t", "Switch between Night and Reading room"],
  ["?", "Show this list"],
];

export function ShortcutsDialog({ open, onClose }: { open: boolean; onClose: () => void }) {
  const ref = useRef<HTMLDialogElement>(null);
  useEffect(() => {
    const dialog = ref.current;
    if (!dialog) return;
    if (open && !dialog.open) dialog.showModal();
    if (!open && dialog.open) dialog.close();
  }, [open]);

  return (
    <dialog ref={ref} className="sheet" aria-labelledby="shortcuts-title" onClose={onClose}>
      <div className="side__head">
        <h2 id="shortcuts-title">Keyboard shortcuts</h2>
        <form method="dialog">
          <button className="button button--quiet" aria-label="Close">
            Esc
          </button>
        </form>
      </div>
      <dl className="shortcut-list">
        {SHORTCUTS.map(([keys, what]) => (
          <div key={keys}>
            <dt>
              {keys.split(" ").map((key) => (
                <kbd key={key}>{key}</kbd>
              ))}
            </dt>
            <dd>{what}</dd>
          </div>
        ))}
      </dl>
    </dialog>
  );
}
