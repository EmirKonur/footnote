import Link from "next/link";

export const metadata = { title: "Not found" };

export default function NotFound() {
  return (
    <div className="lost">
      <p className="label">Error 404</p>
      <h1>
        This page has no source<sup>†</sup>
      </h1>
      <p>
        The address doesn&apos;t match a collection, document, or conversation. It may have been deleted, or the
        link was copied from another Footnote instance.
      </p>
      <p>
        <Link href="/" className="button">
          Back to the library
        </Link>
      </p>
      <p className="lost__note">
        † Footnote only links to passages it can find in its own index. This one isn&apos;t there.
      </p>
    </div>
  );
}
