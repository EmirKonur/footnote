import { act, cleanup, fireEvent, render, screen } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";

import { AnswerBody } from "@/components/answer-view";
import type { Footnote, Segment } from "@/lib/types";

const footnote = (n: number, overrides: Partial<Footnote> = {}): Footnote => ({
  n,
  document_id: "doc_8",
  document_title: "PEP 8",
  document_path: "pep-0008.html",
  chunk_id: 1,
  heading_path: ["PEP 8", "Maximum Line Length"],
  page: null,
  start: 10 * n,
  end: 10 * n + 5,
  cited_text: `quoted passage ${n}`,
  verified: true,
  ...overrides,
});

function renderAnswer(segments: Segment[], footnotes: Footnote[]) {
  const onFocusNote = vi.fn();
  const view = render(
    <AnswerBody
      segments={segments}
      streaming={false}
      context={{ collectionId: "col_1", conversationId: "conv_1", turn: "t1", footnotes, active: null, onFocusNote }}
    />,
  );
  return { ...view, onFocusNote };
}

afterEach(cleanup);

describe("AnswerBody", () => {
  it("renders footnotes as superscript links to the cited span", () => {
    const { container } = renderAnswer(
      [
        { text: "Lines stop at 79", footnotes: [1] },
        { text: " and comments at 72.", footnotes: [2] },
      ],
      [footnote(1), footnote(2)],
    );
    const refs = [...container.querySelectorAll("sup.ref a")];
    expect(refs.map((a) => a.textContent)).toEqual(["1", "2"]);
    expect(refs[0]?.getAttribute("href")).toBe("/c/col_1/doc/doc_8?start=10&end=15&n=1&from=conv_1");
    expect(refs[0]?.getAttribute("aria-label")).toBe("Footnote 1: PEP 8, Maximum Line Length");
    // Text between two references means no separator.
    expect(container.querySelector(".ref__sep")).toBeNull();
    expect(container.textContent).toBe("Lines stop at 791 and comments at 72.2");
  });

  it("separates adjacent references with a comma", () => {
    const { container } = renderAnswer([{ text: "Both rules apply.", footnotes: [1, 2] }], [footnote(1), footnote(2)]);
    expect(container.querySelectorAll(".ref__sep")).toHaveLength(1);
    expect(container.querySelector("p")?.textContent).toBe("Both rules apply.1,2");
  });

  it("shows the cited passage on hover and reports which note is in focus", () => {
    const { onFocusNote } = renderAnswer([{ text: "Claim.", footnotes: [1] }], [footnote(1)]);
    const link = screen.getByRole("link", { name: /Footnote 1/ });

    act(() => {
      fireEvent.mouseEnter(link);
    });
    const card = screen.getByRole("tooltip");
    expect(card.textContent).toContain("quoted passage 1");
    expect(link.getAttribute("aria-describedby")).toBe(card.id);
    expect(onFocusNote).toHaveBeenLastCalledWith({ turn: "t1", n: 1 });

    act(() => {
      fireEvent.mouseLeave(link);
    });
    expect(screen.queryByRole("tooltip")).toBeNull();
    expect(onFocusNote).toHaveBeenLastCalledWith(null);
  });

  it("points unverified citations at the footnote rail instead of the document", () => {
    renderAnswer([{ text: "Claim.", footnotes: [3] }], [footnote(3, { verified: false, start: null, end: null })]);
    const link = screen.getByRole("link", { name: /Footnote 3/ });
    expect(link.getAttribute("href")).toBe("#note-t1-3");
    act(() => {
      fireEvent.focus(link);
    });
    expect(screen.getByRole("tooltip").textContent).toContain("unverified quote");
  });
});
