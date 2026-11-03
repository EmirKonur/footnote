import { describe, expect, it } from "vitest";

import { SSEParser, readEventStream } from "@/lib/sse";
import { pageStarts, parseRange, pieces } from "@/lib/viewer";

describe("SSEParser", () => {
  it("buffers messages split across network chunks", () => {
    const parser = new SSEParser();
    expect(parser.push('event: text\ndata: {"segment":0,')).toEqual([]);
    expect(parser.push('"text":"hi"}\n\nevent: done\ndata: {}\n')).toEqual([
      { event: "text", data: '{"segment":0,"text":"hi"}' },
    ]);
    expect(parser.push("\n")).toEqual([{ event: "done", data: "{}" }]);
  });

  it("joins multi-line data, ignores comments, and normalises CRLF", () => {
    const parser = new SSEParser();
    expect(parser.push(": keep-alive\r\n\r\ndata: line one\r\ndata: line two\r\n\r\n")).toEqual([
      { event: "message", data: "line one\nline two" },
    ]);
  });

  it("reads a whole stream, including a final message without a trailing blank line", async () => {
    const encoder = new TextEncoder();
    const body = new ReadableStream<Uint8Array>({
      start(controller) {
        controller.enqueue(encoder.encode("event: a\ndata: 1\n\nevent: b\nda"));
        controller.enqueue(encoder.encode("ta: 2"));
        controller.close();
      },
    });
    const seen: string[] = [];
    for await (const message of readEventStream(body)) seen.push(`${message.event}=${message.data}`);
    expect(seen).toEqual(["a=1", "b=2"]);
  });
});

describe("viewer ranges", () => {
  const text = "Heading\n\nLimit all lines to a maximum of 79 characters. More text.";
  const block = { kind: "paragraph" as const, start: 9, end: text.length, level: 0, page: null };

  it("splits a block around a highlight that starts inside it", () => {
    expect(pieces(text, block, { start: 9, end: 55 })).toEqual([
      { text: "Limit all lines to a maximum of 79 characters.", marked: true },
      { text: " More text.", marked: false },
    ]);
  });

  it("clips a highlight that spans several blocks", () => {
    const heading = { kind: "heading" as const, start: 0, end: 7, level: 1, page: null };
    expect(pieces(text, heading, { start: 3, end: 30 })).toEqual([
      { text: "Hea", marked: false },
      { text: "ding", marked: true },
    ]);
  });

  it("rejects ranges outside the document", () => {
    expect(parseRange("9", "55", text.length)).toEqual({ start: 9, end: 55 });
    expect(parseRange("55", "9", text.length)).toBeNull();
    expect(parseRange("0", "9999", text.length)).toBeNull();
    expect(parseRange("abc", "4", text.length)).toBeNull();
    expect(parseRange(null, "4", text.length)).toBeNull();
  });

  it("finds where each page begins", () => {
    const blocks = [1, 1, 2, 2, 3].map((page, i) => ({ kind: "paragraph" as const, start: i, end: i + 1, level: 0, page }));
    expect([...pageStarts(blocks)]).toEqual([
      [0, 1],
      [2, 2],
      [4, 3],
    ]);
  });
});
