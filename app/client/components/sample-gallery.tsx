"use client"
import { useEffect, useState } from "react"
import Link from "next/link"
import type { CatalogueItem } from "@/components/catalogue-explorer"
import { Button, buttonVariants } from "@/components/ui/button"
import { Card, CardContent } from "@/components/ui/card"

export function SampleGallery() {
    const [items, setItems] = useState<CatalogueItem[] | null>(null),
        [error, setError] = useState(""),
        [retry, setRetry] = useState(0)
    useEffect(() => {
        const abort = new AbortController()
        fetch("/api/catalogue/samples", { signal: abort.signal })
            .then(async (r) => {
                if (!r.ok)
                    throw new Error(
                        "Start the API and make sure the catalogue dataset is available to load samples.",
                    )
                const body = await r.json()
                setItems(body.items)
            })
            .catch((e) => {
                if (!abort.signal.aborted) setError(e.message)
            })
        return () => abort.abort()
    }, [retry])
    return (
        <section
            aria-labelledby="sample-heading"
            className="mx-auto max-w-7xl px-4 py-16 md:px-6"
        >
            <div className="mb-6 flex flex-wrap items-end justify-between gap-4">
                <div>
                    <h2
                        id="sample-heading"
                        className="mt-3 text-3xl font-semibold tracking-tight"
                    >
                        Start with a sample.
                    </h2>
                    <p className="mt-3 max-w-2xl text-sm leading-6 text-muted-foreground">
                        Choose a product to run a live analysis. These examples
                        come from the training catalogue and demonstrate the
                        workflow, not performance on unseen images.
                    </p>
                </div>
                <Link
                    href="/catalogue"
                    className={buttonVariants({ variant: "outline" })}
                >
                    Explore the catalogue
                </Link>
            </div>
            {error ? (
                <div
                    role="status"
                    className="rounded-xl border p-6 text-sm text-muted-foreground"
                >
                    {error}
                    <Button
                        variant="link"
                        onClick={() => {
                            setError("")
                            setRetry((n) => n + 1)
                        }}
                    >
                        Retry samples
                    </Button>
                </div>
            ) : items === null ? (
                <p role="status" className="py-8 text-sm text-muted-foreground">
                    Loading sample images...
                </p>
            ) : !items.length ? (
                <p className="py-8 text-sm text-muted-foreground">
                    No sample images are available in the configured dataset.
                </p>
            ) : (
                <div className="grid grid-cols-2 gap-4 md:grid-cols-3 lg:grid-cols-6">
                    {items.map((item) => (
                        <Link
                            key={item.id}
                            href={`/analyse?sample=${item.id}`}
                            aria-label={`Analyse sample: ${item.articleType}`}
                            className="rounded-xl outline-none focus-visible:ring-2 focus-visible:ring-ring"
                        >
                            <Card className="h-full gap-0 overflow-hidden py-0 transition-colors hover:bg-muted/40">
                                {/* eslint-disable-next-line @next/next/no-img-element */}
                                <img
                                    loading="lazy"
                                    src={`/api/gallery/${item.id}/image`}
                                    alt={
                                        item.productDisplayName ||
                                        item.articleType
                                    }
                                    className="aspect-3/4 w-full bg-muted/30 object-contain"
                                />
                                <CardContent className="p-3">
                                    <p className="text-sm font-semibold">
                                        {item.articleType}
                                    </p>
                                    <p className="mt-1 text-xs text-muted-foreground">
                                        Analyse sample
                                    </p>
                                </CardContent>
                            </Card>
                        </Link>
                    ))}
                </div>
            )}
        </section>
    )
}
