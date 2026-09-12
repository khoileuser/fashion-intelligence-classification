"use client"
import { useEffect, useState } from "react"
import Link from "next/link"
import { Search, ArrowRight, Loader2, X } from "lucide-react"
import { Button, buttonVariants } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { DropdownSelect, DropdownOption } from "@/components/dropdown-select"
import { Card, CardContent } from "@/components/ui/card"
import { Badge } from "@/components/ui/badge"
import { Alert, AlertTitle, AlertDescription } from "@/components/ui/alert"
export type CatalogueItem = {
    id: string
    articleType: string
    season: string
    usage: string
    gender: string
    baseColour: string
    productDisplayName: string
    score?: number
}
type CatalogueResponse = {
    items: CatalogueItem[]
    total: number
    page: number
    pages: number
    facets: Record<string, string[]>
    reference: CatalogueItem | null
}
const filters = [
    ["articleType", "Article type"],
    ["season", "Season"],
    ["usage", "Occasion"],
    ["gender", "Catalogue gender"],
    ["baseColour", "Colour"],
]
export function CatalogueExplorer() {
    const [query, setQuery] = useState(""),
        [search, setSearch] = useState(""),
        [selected, setSelected] = useState<Record<string, string>>({}),
        [page, setPage] = useState(1),
        [similar, setSimilar] = useState("")
    const [data, setData] = useState<CatalogueResponse | null>(null),
        [busy, setBusy] = useState(true),
        [error, setError] = useState(""),
        [retry, setRetry] = useState(0)
    useEffect(() => {
        const abort = new AbortController()
        const params = new URLSearchParams({
            q: search,
            page: String(page),
            ...selected,
        })
        if (similar) params.set("similar_to", similar)
        fetch(`/api/catalogue?${params}`, { signal: abort.signal })
            .then(async (r) => {
                const body = await r.json()
                if (!r.ok)
                    throw new Error(
                        body.detail || "Could not load the catalogue.",
                    )
                if (!abort.signal.aborted) setData(body)
            })
            .catch((e) => {
                if (!abort.signal.aborted) setError(e.message)
            })
            .finally(() => {
                if (!abort.signal.aborted) setBusy(false)
            })
        return () => abort.abort()
    }, [search, selected, page, similar, retry])
    function refresh() {
        setBusy(true)
        setError("")
    }
    function reset() {
        refresh()
        setQuery("")
        setSearch("")
        setSelected({})
        setSimilar("")
        setPage(1)
    }
    return (
        <div className="space-y-6">
            <Card>
                <CardContent className="space-y-4">
                    <form
                        className="flex gap-2"
                        onSubmit={(e) => {
                            e.preventDefault()
                            refresh()
                            setSearch(query)
                            setPage(1)
                            setRetry((n) => n + 1)
                        }}
                    >
                        <Input
                            aria-label="Search catalogue"
                            placeholder="Search products, categories, or item IDs"
                            value={query}
                            onChange={(e) => setQuery(e.target.value)}
                        />
                        <Button type="submit">
                            <Search />
                            Search
                        </Button>
                    </form>
                    <div className="flex flex-wrap gap-3">
                        {filters.map(([key, title]) => (
                            <DropdownSelect
                                key={key}
                                aria-label={title}
                                value={selected[key] || ""}
                                onValueChange={(value) => {
                                    refresh()
                                    setSelected({
                                        ...selected,
                                        [key]: value,
                                    })
                                    setPage(1)
                                }}
                            >
                                <DropdownOption value="">
                                    All {title.toLowerCase()}
                                </DropdownOption>
                                {data?.facets[key]?.map((value) => (
                                    <DropdownOption key={value} value={value}>
                                        {value}
                                    </DropdownOption>
                                ))}
                            </DropdownSelect>
                        ))}
                        <Button variant="ghost" onClick={reset}>
                            Reset filters
                        </Button>
                    </div>
                </CardContent>
            </Card>
            {similar && (
                <div className="flex flex-wrap items-center justify-between gap-3 rounded-xl border bg-muted/40 p-4">
                    <div>
                        <p className="font-medium">
                            Visually similar to{" "}
                            {data?.reference?.productDisplayName ||
                                `item ${similar}`}
                        </p>
                        <p className="text-xs text-muted-foreground">
                            Cosine similarity ranks visual features, not match
                            probability. The reference item is excluded.
                        </p>
                    </div>
                    <Button
                        variant="outline"
                        onClick={() => {
                            refresh()
                            setSimilar("")
                            setPage(1)
                        }}
                    >
                        <X />
                        Exit similarity search
                    </Button>
                </div>
            )}
            {error ? (
                <Alert>
                    <AlertTitle>Catalogue unavailable</AlertTitle>
                    <AlertDescription>{error}</AlertDescription>
                    <Button
                        className="mt-3"
                        variant="outline"
                        onClick={() => {
                            refresh()
                            setRetry((n) => n + 1)
                        }}
                    >
                        Try again
                    </Button>
                </Alert>
            ) : busy ? (
                <div
                    role="status"
                    className="flex items-center justify-center gap-3 py-20 text-muted-foreground"
                >
                    <Loader2 className="size-5 animate-spin" />
                    Loading catalogue...
                </div>
            ) : (
                <>
                    <p role="status" className="text-sm text-muted-foreground">
                        {data?.total.toLocaleString()} products /{" "}
                        {similar
                            ? "Sorted by visual similarity"
                            : "Catalogue order"}
                    </p>
                    <div className="grid gap-5 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4">
                        {data?.items.map((item) => (
                            <Card
                                key={item.id}
                                className="gap-0 overflow-hidden py-0"
                            >
                                <div className="aspect-3/4 bg-muted/40">
                                    {/* eslint-disable-next-line @next/next/no-img-element */}
                                    <img
                                        loading="lazy"
                                        src={`/api/gallery/${item.id}/image`}
                                        alt={
                                            item.productDisplayName ||
                                            item.articleType
                                        }
                                        className="h-full w-full object-contain"
                                    />
                                </div>
                                <CardContent className="space-y-3 p-4">
                                    <p className="text-xs text-muted-foreground">
                                        Item {item.id}
                                        {item.score !== undefined
                                            ? ` / Similarity ${item.score.toFixed(3)}`
                                            : ""}
                                    </p>
                                    <h2 className="line-clamp-2 min-h-10 text-sm font-semibold">
                                        {item.productDisplayName ||
                                            `${item.baseColour} ${item.articleType}`}
                                    </h2>
                                    <div className="flex flex-wrap gap-1">
                                        {[
                                            item.articleType,
                                            item.season,
                                            item.usage,
                                            item.gender,
                                        ]
                                            .filter(Boolean)
                                            .map((value, i) => (
                                                <Badge
                                                    key={`${value}-${i}`}
                                                    variant="outline"
                                                >
                                                    {value}
                                                </Badge>
                                            ))}
                                    </div>
                                    <div className="flex flex-wrap gap-2 pt-2">
                                        <Link
                                            href={`/analyse?sample=${item.id}`}
                                            className={buttonVariants({
                                                variant: "outline",
                                                size: "sm",
                                            })}
                                        >
                                            Analyse{" "}
                                            <ArrowRight className="size-3" />
                                        </Link>
                                        <Button
                                            size="sm"
                                            onClick={() => {
                                                refresh()
                                                setSimilar(item.id)
                                                setPage(1)
                                            }}
                                        >
                                            Find similar
                                        </Button>
                                    </div>
                                </CardContent>
                            </Card>
                        ))}
                    </div>
                    {!data?.items.length && (
                        <div className="rounded-xl border border-dashed p-16 text-center">
                            <p>No products match these filters.</p>
                            <Button variant="link" onClick={reset}>
                                Clear filters and browse again
                            </Button>
                        </div>
                    )}
                    <div className="flex items-center justify-center gap-4">
                        <Button
                            variant="outline"
                            disabled={!data || data.page <= 1}
                            onClick={() => {
                                refresh()
                                setPage((data?.page || 1) - 1)
                            }}
                        >
                            Previous
                        </Button>
                        <span className="text-sm text-muted-foreground">
                            Page {data?.page || 1} of {data?.pages || 1}
                        </span>
                        <Button
                            variant="outline"
                            disabled={!data || data.page >= data.pages}
                            onClick={() => {
                                refresh()
                                setPage((data?.page || 1) + 1)
                            }}
                        >
                            Next
                        </Button>
                    </div>
                </>
            )}
        </div>
    )
}
