import type { ViewerBlock } from "@/lib/types";

export interface Range {
  start: number;
  end: number;
}

export interface BlockPiece {
  text: string;
  marked: boolean;
}

/**
 * Split one block's text around a highlight. Offsets are into the document's canonical
 * text, the same coordinates the API uses for chunks and citations, so a highlight is
 * the intersection of two integer ranges.
 */
export function pieces(text: string, block: ViewerBlock, range: Range | null): BlockPiece[] {
  const own = text.slice(block.start, block.end);
  if (!range || range.end <= block.start || range.start >= block.end) return [{ text: own, marked: false }];
  const from = Math.max(range.start, block.start) - block.start;
  const to = Math.min(range.end, block.end) - block.start;
  return [
    { text: own.slice(0, from), marked: false },
    { text: own.slice(from, to), marked: true },
    { text: own.slice(to), marked: false },
  ].filter((piece) => piece.text !== "");
}

/** Parse ?start=&end= into a range, rejecting anything outside the document. */
export function parseRange(start: string | null, end: string | null, length: number): Range | null {
  if (start === null || end === null) return null;
  const s = Number(start);
  const e = Number(end);
  if (!Number.isInteger(s) || !Number.isInteger(e) || s < 0 || e <= s || e > length) return null;
  return { start: s, end: e };
}

/** Indices of blocks that begin a new page, for drawing page rules in PDFs. */
export function pageStarts(blocks: ViewerBlock[]): Map<number, number> {
  const starts = new Map<number, number>();
  let page: number | null = null;
  blocks.forEach((block, index) => {
    if (block.page !== null && block.page !== page) {
      starts.set(index, block.page);
      page = block.page;
    }
  });
  return starts;
}
