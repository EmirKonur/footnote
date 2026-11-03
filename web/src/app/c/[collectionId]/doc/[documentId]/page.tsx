import { DocumentViewer } from "@/components/document-viewer";

export default async function DocumentPage({
  params,
  searchParams,
}: {
  params: Promise<{ collectionId: string; documentId: string }>;
  searchParams: Promise<{ start?: string; end?: string; n?: string; from?: string }>;
}) {
  const { collectionId, documentId } = await params;
  const query = await searchParams;
  return (
    <DocumentViewer
      collectionId={collectionId}
      documentId={documentId}
      start={query.start ?? null}
      end={query.end ?? null}
      footnote={query.n ?? null}
      fromConversation={query.from ?? null}
    />
  );
}
