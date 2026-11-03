"use client";

import { ArrowUp, Square } from "lucide-react";
import { useLayoutEffect, useRef, useState, type FormEvent, type KeyboardEvent, type RefObject } from "react";

import type { GenerationMode } from "@/lib/types";

const HINTS: Record<GenerationMode, string> = {
  anthropic: "Answers cite their sources. Shift+Enter for a new line.",
  fixtures: "Fixture mode: answers are stitched from retrieved sentences, not written by a model.",
  disabled: "Retrieval only: you will get ranked passages. Add ANTHROPIC_API_KEY for written answers.",
};

export function Composer({ onAsk, onStop, streaming, disabled, mode, inputRef }: {
  onAsk: (question: string) => void;
  onStop: () => void;
  streaming: boolean;
  disabled: boolean;
  mode: GenerationMode | null;
  inputRef: RefObject<HTMLTextAreaElement | null>;
}) {
  const [value, setValue] = useState("");
  const formRef = useRef<HTMLFormElement>(null);

  useLayoutEffect(() => {
    const field = inputRef.current;
    if (!field) return;
    field.style.height = "auto";
    field.style.height = `${field.scrollHeight}px`;
  }, [value, inputRef]);

  function submit(event: FormEvent) {
    event.preventDefault();
    const question = value.trim();
    if (!question || streaming || disabled) return;
    onAsk(question);
    setValue("");
  }

  function onKeyDown(event: KeyboardEvent<HTMLTextAreaElement>) {
    if (event.key === "Enter" && !event.shiftKey && !event.nativeEvent.isComposing) {
      event.preventDefault();
      formRef.current?.requestSubmit();
    }
  }

  return (
    <div className="composer">
      <form ref={formRef} onSubmit={submit}>
        <label className="visually-hidden" htmlFor="question">
          Ask a question about this collection
        </label>
        <div className="composer__box">
          <textarea
            id="question"
            ref={inputRef}
            className="field"
            rows={1}
            value={value}
            placeholder="Ask about these documents…"
            onChange={(event) => setValue(event.target.value)}
            onKeyDown={onKeyDown}
            disabled={disabled}
          />
          {streaming ? (
            <button type="button" className="button" onClick={onStop} title="Stop (Esc)">
              <Square className="icon" aria-hidden />
              Stop
            </button>
          ) : (
            <button className="button button--primary" disabled={disabled || !value.trim()} title="Ask (Enter)">
              <ArrowUp className="icon" aria-hidden />
              Ask
            </button>
          )}
        </div>
        <div className="composer__hint">
          <span className="meta">{mode ? HINTS[mode] : "Connecting…"}</span>
          <span className="meta" aria-hidden>
            <kbd>/</kbd> to focus
          </span>
        </div>
      </form>
    </div>
  );
}
