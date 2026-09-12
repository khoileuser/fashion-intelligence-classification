"use client"
import { useEffect, useRef, useState } from "react"
import { Download, Upload, Play, Square, RotateCcw, Trash2 } from "lucide-react"
import data from "@/lib/insights-data.json"
import { batchCsv, targets, type BatchRow } from "@/lib/batch"
import { createClientId } from "@/lib/client-id"
import { Button } from "@/components/ui/button"
import { Badge } from "@/components/ui/badge"
import {
    Card,
    CardHeader,
    CardTitle,
    CardDescription,
    CardContent,
} from "@/components/ui/card"
import { Alert, AlertTitle, AlertDescription } from "@/components/ui/alert"
import { Progress } from "@/components/ui/progress"
import { DropdownSelect, DropdownOption } from "@/components/dropdown-select"
import {
    Table,
    TableHeader,
    TableBody,
    TableHead,
    TableRow,
    TableCell,
} from "@/components/ui/table"
import type { Prediction } from "@/lib/types"
export function BatchAnalyser() {
    const [rows, setRows] = useState<BatchRow[]>([]),
        [busy, setBusy] = useState(false),
        [error, setError] = useState(""),
        [filter, setFilter] = useState("all")
    const input = useRef<HTMLInputElement>(null),
        controller = useRef<AbortController | null>(null),
        urls = useRef(new Set<string>()),
        running = useRef(false)
    useEffect(() => {
        const previews = urls.current
        return () => {
            controller.current?.abort()
            previews.forEach((url) => URL.revokeObjectURL(url))
        }
    }, [])
    useEffect(() => {
        const warn = (e: BeforeUnloadEvent) => {
            e.preventDefault()
            e.returnValue = ""
        }
        if (rows.length) window.addEventListener("beforeunload", warn)
        return () => window.removeEventListener("beforeunload", warn)
    }, [rows.length])
    function update(id: string, patch: Partial<BatchRow>) {
        setRows((current) =>
            current.map((r) => (r.id === id ? { ...r, ...patch } : r)),
        )
    }
    function add(files: FileList | null) {
        if (!files || busy) return
        setError("")
        const selected = Array.from(files)
        if (rows.length + selected.length > 20) {
            setError(
                "A batch can contain up to 20 images. Remove an item or clear the batch first.",
            )
            return
        }
        const valid: BatchRow[] = []
        const rejected: string[] = []
        for (const file of selected) {
            if (
                !["image/jpeg", "image/png", "image/webp"].includes(
                    file.type,
                ) ||
                file.size > 10 * 1024 * 1024
            ) {
                rejected.push(file.name)
                continue
            }
            const preview = URL.createObjectURL(file)
            urls.current.add(preview)
            valid.push({
                id: createClientId(),
                file,
                preview,
                status: "queued",
                corrections: {},
            })
        }
        setRows((current) => [...current, ...valid])
        if (rejected.length)
            setError(
                `Skipped ${rejected.join(", ")}. Use JPG, PNG, or WebP files up to 10 MB each.`,
            )
    }
    async function run() {
        if (running.current) return
        running.current = true
        setBusy(true)
        setError("")
        const abort = new AbortController()
        controller.current = abort
        try {
            for (const row of rows.filter(
                (r) => r.status === "queued" || r.status === "error",
            )) {
                if (abort.signal.aborted) break
                update(row.id, { status: "processing", error: undefined })
                const form = new FormData()
                form.append("file", row.file)
                try {
                    const response = await fetch("/api/predict?top_k=3", {
                        method: "POST",
                        body: form,
                        signal: abort.signal,
                    })
                    const body = await response.json()
                    if (!response.ok)
                        throw new Error(
                            typeof body.detail === "string"
                                ? body.detail
                                : "Image analysis failed.",
                        )
                    const predictions = body.predictions as Record<
                        string,
                        Prediction
                    >
                    if (
                        !predictions ||
                        targets.some(
                            ([key]) =>
                                !predictions[key] ||
                                typeof predictions[key].label !== "string" ||
                                !Number.isFinite(predictions[key].confidence),
                        )
                    )
                        throw new Error(
                            "The service returned incomplete predictions.",
                        )
                    update(row.id, {
                        status: "done",
                        predictions,
                        corrections: {},
                    })
                } catch (e) {
                    if (abort.signal.aborted) {
                        update(row.id, { status: "queued" })
                        break
                    }
                    update(row.id, {
                        status: "error",
                        error:
                            e instanceof Error ? e.message : "Analysis failed.",
                    })
                }
            }
        } finally {
            running.current = false
            setBusy(false)
            controller.current = null
        }
    }
    function remove(row: BatchRow) {
        URL.revokeObjectURL(row.preview)
        urls.current.delete(row.preview)
        setRows((current) => current.filter((r) => r.id !== row.id))
    }
    function clear() {
        urls.current.forEach((url) => URL.revokeObjectURL(url))
        urls.current.clear()
        setRows([])
        setError("")
    }
    function download() {
        const content = batchCsv(rows)
        const url = URL.createObjectURL(
            new Blob([content], { type: "text/csv;charset=utf-8;" }),
        )
        const a = document.createElement("a")
        a.href = url
        a.download = "threadline-all-results.csv"
        a.click()
        setTimeout(() => URL.revokeObjectURL(url), 1000)
    }
    const done = rows.filter((r) => r.status === "done").length,
        failed = rows.filter((r) => r.status === "error").length
    const visible = rows.filter(
        (r) =>
            filter === "all" ||
            (filter === "done" && r.status === "done") ||
            (filter === "error" && r.status === "error"),
    )
    return (
        <section className="space-y-6 px-4 pb-12 md:px-6">
            <Card>
                <CardHeader>
                    <CardTitle>Build a reviewed catalogue</CardTitle>
                    <CardDescription>
                        Up to 20 images, 10 MB each. Results stay in this page
                        session; export before navigating away or refreshing.
                    </CardDescription>
                </CardHeader>
                <CardContent className="space-y-5">
                    <input
                        ref={input}
                        type="file"
                        multiple
                        accept="image/jpeg,image/png,image/webp"
                        className="sr-only"
                        aria-label="Upload batch images"
                        disabled={busy}
                        onChange={(e) => {
                            add(e.target.files)
                            e.target.value = ""
                        }}
                    />
                    <div
                        onDragOver={(e) => e.preventDefault()}
                        onDrop={(e) => {
                            e.preventDefault()
                            add(e.dataTransfer.files)
                        }}
                        className="rounded-xl border border-dashed bg-muted/30 p-8 text-center"
                    >
                        <Upload className="mx-auto mb-3 size-7 text-muted-foreground" />
                        <p className="mb-4 text-sm text-muted-foreground">
                            Drop product images here to start your collection.
                        </p>
                        <Button
                            variant="outline"
                            onClick={() => input.current?.click()}
                            disabled={busy}
                        >
                            Choose images
                        </Button>
                    </div>
                    {error && (
                        <Alert className="border-destructive text-destructive">
                            <AlertTitle>Check your batch</AlertTitle>
                            <AlertDescription>{error}</AlertDescription>
                        </Alert>
                    )}
                    <div className="flex flex-wrap gap-2">
                        <Button
                            onClick={run}
                            disabled={
                                busy ||
                                !rows.some(
                                    (r) =>
                                        r.status === "queued" ||
                                        r.status === "error",
                                )
                            }
                        >
                            <Play />
                            Analyse queued images
                        </Button>
                        {busy && (
                            <Button
                                variant="outline"
                                onClick={() => controller.current?.abort()}
                            >
                                <Square />
                                Stop
                            </Button>
                        )}
                        <Button
                            variant="outline"
                            disabled={busy || !failed}
                            onClick={run}
                        >
                            <RotateCcw />
                            Retry failed ({failed})
                        </Button>
                        <Button
                            variant="ghost"
                            disabled={busy || !rows.length}
                            onClick={clear}
                        >
                            <Trash2 />
                            Clear batch
                        </Button>
                    </div>
                    <div
                        role="status"
                        aria-live="polite"
                        className="text-sm text-muted-foreground"
                    >
                        {done} of {rows.length} analysed / {failed} failed
                        {busy ? " / Processing..." : ""}
                    </div>
                    <Progress
                        value={
                            rows.length
                                ? ((done + failed) / rows.length) * 100
                                : 0
                        }
                        aria-label="Batch progress"
                    />
                </CardContent>
            </Card>
            <div className="flex flex-wrap items-center justify-between gap-3">
                <DropdownSelect
                    aria-label="Filter batch results"
                    value={filter}
                    onValueChange={(value) => setFilter(value)}
                >
                    <DropdownOption value="all">
                        All items ({rows.length})
                    </DropdownOption>
                    <DropdownOption value="done">
                        Analysed ({done})
                    </DropdownOption>
                    <DropdownOption value="error">
                        Failed ({failed})
                    </DropdownOption>
                </DropdownSelect>
                <div className="flex flex-wrap gap-2">
                    <Button disabled={!done} onClick={download}>
                        <Download />
                        Export
                    </Button>
                </div>
            </div>
            <Card>
                <CardContent>
                    <Table>
                        <TableHeader>
                            <TableRow>
                                <TableHead>Product</TableHead>
                                {targets.map(([key, label]) => (
                                    <TableHead key={key}>{label}</TableHead>
                                ))}
                                <TableHead>
                                    <span className="sr-only">Remove</span>
                                </TableHead>
                            </TableRow>
                        </TableHeader>
                        <TableBody>
                            {visible.map((row) => (
                                <TableRow key={row.id}>
                                    <TableCell>
                                        <div className="flex w-48 items-center gap-3">
                                            {/* eslint-disable-next-line @next/next/no-img-element */}
                                            <img
                                                src={row.preview}
                                                alt={row.file.name}
                                                className="size-14 rounded-md border bg-muted object-contain"
                                            />
                                            <div className="min-w-0">
                                                <p
                                                    className="truncate text-sm font-medium"
                                                    title={row.file.name}
                                                >
                                                    {row.file.name}
                                                </p>
                                                <Badge
                                                    variant="outline"
                                                    className="mt-1"
                                                >
                                                    {row.status}
                                                </Badge>
                                            </div>
                                        </div>
                                        {row.error && (
                                            <p
                                                role="alert"
                                                className="mt-2 max-w-56 whitespace-normal text-xs text-destructive"
                                            >
                                                {row.error}
                                            </p>
                                        )}
                                    </TableCell>
                                    {targets.map(([key, label]) => {
                                        const prediction =
                                            row.predictions?.[key]
                                        return (
                                            <TableCell key={key}>
                                                {prediction ? (
                                                    <div className="min-w-36 space-y-2">
                                                        <DropdownSelect
                                                            className="max-w-48"
                                                            aria-label={`${label} for ${row.file.name}`}
                                                            value={
                                                                row.corrections[
                                                                    key
                                                                ] ??
                                                                prediction.label
                                                            }
                                                            onValueChange={(
                                                                value,
                                                            ) => {
                                                                const corrections =
                                                                    {
                                                                        ...row.corrections,
                                                                    }
                                                                if (
                                                                    value ===
                                                                    prediction.label
                                                                )
                                                                    delete corrections[
                                                                        key
                                                                    ]
                                                                else
                                                                    corrections[
                                                                        key
                                                                    ] = value
                                                                update(row.id, {
                                                                    corrections,
                                                                })
                                                            }}
                                                        >
                                                            {Array.from(
                                                                new Set([
                                                                    prediction.label,
                                                                    ...(data.find(
                                                                        (t) =>
                                                                            t.target ===
                                                                            key,
                                                                    )?.labels ??
                                                                        []),
                                                                ]),
                                                            )
                                                                .sort()
                                                                .map(
                                                                    (label) => (
                                                                        <DropdownOption
                                                                            key={
                                                                                label
                                                                            }
                                                                            value={
                                                                                label
                                                                            }
                                                                        >
                                                                            {
                                                                                label
                                                                            }
                                                                        </DropdownOption>
                                                                    ),
                                                                )}
                                                        </DropdownSelect>
                                                        <p className="max-w-48 whitespace-normal text-xs text-muted-foreground">
                                                            {prediction.label} /{" "}
                                                            {(
                                                                prediction.confidence *
                                                                100
                                                            ).toFixed(1)}
                                                            %
                                                        </p>
                                                        {prediction.needs_review && (
                                                            <Badge
                                                                variant="outline"
                                                                title={
                                                                    prediction.review_reason
                                                                }
                                                            >
                                                                Needs review
                                                            </Badge>
                                                        )}
                                                        {row.corrections[
                                                            key
                                                        ] && (
                                                            <Badge variant="secondary">
                                                                Edited
                                                            </Badge>
                                                        )}
                                                    </div>
                                                ) : (
                                                    <span className="text-muted-foreground">
                                                        Pending
                                                    </span>
                                                )}
                                            </TableCell>
                                        )
                                    })}

                                    <TableCell>
                                        <Button
                                            variant="ghost"
                                            size="icon"
                                            disabled={busy}
                                            aria-label={`Remove ${row.file.name}`}
                                            onClick={() => remove(row)}
                                        >
                                            <Trash2 />
                                        </Button>
                                    </TableCell>
                                </TableRow>
                            ))}
                            {!visible.length && (
                                <TableRow>
                                    <TableCell
                                        colSpan={6}
                                        className="h-32 text-center text-muted-foreground"
                                    >
                                        {rows.length
                                            ? "No items match this filter."
                                            : "Your collection will appear here. Add images to begin."}
                                    </TableCell>
                                </TableRow>
                            )}
                        </TableBody>
                    </Table>
                </CardContent>
            </Card>
        </section>
    )
}
