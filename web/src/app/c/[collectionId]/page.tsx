import { Workspace } from "@/components/workspace";

export default async function CollectionPage({
  params,
  searchParams,
}: {
  params: Promise<{ collectionId: string }>;
  searchParams: Promise<{ conversation?: string }>;
}) {
  const { collectionId } = await params;
  const { conversation } = await searchParams;
  return <Workspace collectionId={collectionId} conversationId={conversation ?? null} />;
}
