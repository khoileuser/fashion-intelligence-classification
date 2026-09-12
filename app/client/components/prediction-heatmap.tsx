"use client"
import { useEffect, useRef, useState } from "react"
import type { Prediction } from "@/lib/types"
import { targets } from "@/lib/batch"
import {
    Card,
    CardHeader,
    CardTitle,
    CardDescription,
    CardContent,
} from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { DropdownSelect, DropdownOption } from "@/components/dropdown-select"
import { Input } from "@/components/ui/input"
import { Badge } from "@/components/ui/badge"
type Explanation = {
    target: string
    label: string
    confidence: number
    layer: string
    heatmap: string
    has_signal: boolean
    method: string
}
export function PredictionHeatmap({
    file,
    preview,
    predictions,
}: {
    file: File
    preview: string
    predictions: Record<string, Prediction>
}) {
    const [target, setTarget] = useState("articleType"),
        [label, setLabel] = useState(predictions.articleType.label),
        [opacity, setOpacity] = useState(65),
        [shown, setShown] = useState(true),
        [result, setResult] = useState<Explanation | null>(null),
        [busy, setBusy] = useState(false),
        [error, setError] = useState("")
    const request = useRef<AbortController | null>(null)
    useEffect(() => () => request.current?.abort(), [])
    function reset() {
        request.current?.abort()
        setResult(null)
        setError("")
        setBusy(false)
    }
    async function explain() {
        request.current?.abort()
        const abort = new AbortController()
        request.current = abort
        setBusy(true)
        setError("")
        setResult(null)
        const form = new FormData()
        form.append("file", file)
        try {
            const r = await fetch(
                `/api/explain?${new URLSearchParams({ target, label })}`,
                { method: "POST", body: form, signal: abort.signal },
            )
            const body = await r.json()
            if (!r.ok)
                throw new Error(body.detail || "Could not generate a heatmap.")
            if (!abort.signal.aborted) {
                setResult(body)
                setShown(true)
            }
        } catch (e) {
            if (!abort.signal.aborted)
                setError(
                    e instanceof Error
                        ? e.message
                        : "Could not generate a heatmap.",
                )
        } finally {
            if (!abort.signal.aborted) setBusy(false)
        }
    }
    return (
        <Card className="mt-8">
            <CardHeader>
                <CardTitle>What influenced this prediction?</CardTitle>
                <CardDescription>
                    Generate a Grad-CAM heatmap for an attribute and one of its
                    predicted classes.
                </CardDescription>
            </CardHeader>
            <CardContent className="space-y-5">
                <div className="flex flex-wrap items-center gap-3">
                    <DropdownSelect
                        aria-label="Heatmap attribute"
                        value={target}
                        onValueChange={(value) => {
                            reset()
                            setTarget(value)
                            setLabel(predictions[value].label)
                        }}
                    >
                        {targets.map(([key, title]) => (
                            <DropdownOption key={key} value={key}>
                                {title}
                            </DropdownOption>
                        ))}
                    </DropdownSelect>
                    <DropdownSelect
                        aria-label="Heatmap class"
                        value={label}
                        onValueChange={(value) => {
                            reset()
                            setLabel(value)
                        }}
                    >
                        {predictions[target].top_k.map((p) => (
                            <DropdownOption key={p.label} value={p.label}>
                                {p.label}
                            </DropdownOption>
                        ))}
                    </DropdownSelect>
                    <Button onClick={explain} disabled={busy}>
                        {busy ? "Generating heatmap..." : "Generate heatmap"}
                    </Button>
                </div>
                {error && (
                    <p role="alert" className="text-sm text-destructive">
                        {error}
                    </p>
                )}
                {busy && (
                    <p role="status" className="text-sm text-muted-foreground">
                        Computing class gradients...
                    </p>
                )}
                {result && (
                    <div className="grid gap-6 md:grid-cols-2">
                        <div className="rounded-xl border bg-muted/30 p-4">
                            <div className="relative h-80 w-full">
                                {/* eslint-disable-next-line @next/next/no-img-element */}
                                <img
                                    src={preview}
                                    alt="Original image for heatmap comparison"
                                    className="h-full w-full object-contain"
                                />
                                {shown && result.has_signal && (
                                    // eslint-disable-next-line @next/next/no-img-element
                                    <img
                                        src={result.heatmap}
                                        alt={`Grad-CAM overlay for ${result.label}`}
                                        className="absolute inset-0 h-full w-full object-contain"
                                        style={{ opacity: opacity / 100 }}
                                    />
                                )}
                            </div>
                        </div>
                        <div className="space-y-4">
                            <Badge variant="outline">
                                {result.method} / {result.label}
                            </Badge>
                            <p className="text-sm leading-7 text-muted-foreground">
                                Red regions show stronger positive contributions
                                to the selected class score. This is a coarse
                                diagnostic visualization, not an object boundary
                                or proof that a prediction is correct.
                            </p>
                            {!result.has_signal ? (
                                <p role="status" className="text-sm">
                                    No positive activation signal was found for
                                    this class. The original image is shown.
                                </p>
                            ) : (
                                <>
                                    <Button
                                        variant="outline"
                                        aria-pressed={shown}
                                        onClick={() => setShown((v) => !v)}
                                    >
                                        {shown
                                            ? "Show original"
                                            : "Show heatmap"}
                                    </Button>
                                    <label
                                        className="block text-sm"
                                        htmlFor="heatmap-opacity"
                                    >
                                        Overlay opacity: {opacity}%
                                    </label>
                                    <Input
                                        id="heatmap-opacity"
                                        type="range"
                                        min={0}
                                        max={100}
                                        value={opacity}
                                        onChange={(e) =>
                                            setOpacity(Number(e.target.value))
                                        }
                                        className="max-w-xs"
                                    />
                                </>
                            )}
                            <p className="text-xs text-muted-foreground">
                                Class confidence:{" "}
                                {(result.confidence * 100).toFixed(1)}%. Maps
                                are normalized independently; intensity is not
                                comparable across classes.
                            </p>
                        </div>
                    </div>
                )}
            </CardContent>
        </Card>
    )
}
