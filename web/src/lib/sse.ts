/**
 * Incremental parser for `text/event-stream` bodies read through fetch().
 *
 * EventSource only supports GET, and asking a question is a POST, so the stream is read
 * from a ReadableStream instead. Network chunks split messages at arbitrary byte
 * offsets; the parser buffers until it sees the blank line that ends a message.
 */

export interface ServerSentEvent {
  event: string;
  data: string;
}

export class SSEParser {
  private buffer = "";

  /** Feed decoded text; returns every message completed by this chunk. */
  push(chunk: string): ServerSentEvent[] {
    this.buffer += chunk.replace(/\r\n?/g, "\n");
    const messages: ServerSentEvent[] = [];
    let boundary = this.buffer.indexOf("\n\n");
    while (boundary !== -1) {
      const raw = this.buffer.slice(0, boundary);
      this.buffer = this.buffer.slice(boundary + 2);
      const message = parseMessage(raw);
      if (message) messages.push(message);
      boundary = this.buffer.indexOf("\n\n");
    }
    return messages;
  }
}

function parseMessage(raw: string): ServerSentEvent | null {
  let event = "message";
  const data: string[] = [];
  for (const line of raw.split("\n")) {
    if (line === "" || line.startsWith(":")) continue; // comments are keep-alives
    const colon = line.indexOf(":");
    const field = colon === -1 ? line : line.slice(0, colon);
    let value = colon === -1 ? "" : line.slice(colon + 1);
    if (value.startsWith(" ")) value = value.slice(1);
    if (field === "event") event = value;
    else if (field === "data") data.push(value);
  }
  return data.length ? { event, data: data.join("\n") } : null;
}

/** Read an SSE response body to the end, yielding parsed messages as they complete. */
export async function* readEventStream(body: ReadableStream<Uint8Array>): AsyncGenerator<ServerSentEvent> {
  const reader = body.getReader();
  const decoder = new TextDecoder();
  const parser = new SSEParser();
  try {
    for (;;) {
      const { value, done } = await reader.read();
      if (done) break;
      yield* parser.push(decoder.decode(value, { stream: true }));
    }
    yield* parser.push(decoder.decode() + "\n\n");
  } finally {
    reader.releaseLock();
  }
}
