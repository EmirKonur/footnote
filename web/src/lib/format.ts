export function plural(count: number, noun: string): string {
  return `${count.toLocaleString("en")} ${noun}${count === 1 ? "" : "s"}`;
}

export function formatDate(iso: string): string {
  return new Date(iso).toLocaleDateString("en", { day: "numeric", month: "short", year: "numeric" });
}

export function formatBytes(bytes: number): string {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(0)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
}
