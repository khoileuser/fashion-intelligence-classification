"use client"

import { ChangeEvent, DragEvent, useEffect, useRef, useState } from "react"
import { AlertCircle, CheckCircle2, ImageIcon, Loader2, Search, Upload, X } from "lucide-react"

import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert"
import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"
import { Card, CardContent, CardDescription, CardFooter, CardHeader, CardTitle } from "@/components/ui/card"
import { Progress } from "@/components/ui/progress"
import { addHistoryEntry, createThumbnail } from "@/lib/history"
import { cn } from "@/lib/utils"
import type { AnalysisResponse, HealthResponse, Prediction, SimilarItem } from "@/lib/types"

const MAX_FILE_BYTES = 10 * 1024 * 1024
const TARGET_ORDER = ["articleType", "season", "gender", "usage"]

function targetLabel(target: string) {
  const labels: Record<string, string> = {
    articleType: "Article type",
    season: "Season",
    gender: "Gender",
    usage: "Occasion",
  }
  return labels[target] ?? target
}

function percent(value: number) {
  return `${Math.round(value * 100)}%`
}

function PredictionCard({ target, prediction }: { target: string; prediction: Prediction }) {
  return (
    <Card className="gap-5">
      <CardHeader>
        <CardDescription>{targetLabel(target)}</CardDescription>
        {prediction.needs_review && (
          <Badge variant="outline">Needs review</Badge>
        )}
        {prediction.review_reason && (
          <p className="text-xs text-muted-foreground">{prediction.review_reason}</p>
        )}
        <div className="flex items-end justify-between gap-4">
          <CardTitle className="text-xl leading-tight">{prediction.label}</CardTitle>
          <span className="text-sm font-semibold">{percent(prediction.confidence)}</span>
        </div>
      </CardHeader>
      <CardContent className="space-y-4">
        {prediction.top_k.map((item, index) => (
          <div className="space-y-1.5" key={`${item.label}-${index}`}>
            <div className="flex justify-between gap-4 text-xs">
              <span className="truncate text-muted-foreground">{item.label}</span>
              <span>{percent(item.confidence)}</span>
            </div>
            <Progress value={item.confidence * 100} className="h-1.5" />
          </div>
        ))}
      </CardContent>
    </Card>
  )
}

function SimilarCard({ item, rank }: { item: SimilarItem; rank: number }) {
  const id = String(item.id)
  const title = item.productDisplayName || item.articleType || `Catalogue item ${id}`
  const score = Math.max(0, Math.min(1, Number(item.score)))

  return (
    <Card className="gap-0 overflow-hidden py-0">
      <div className="relative aspect-[3/4] overflow-hidden bg-muted">
        {/* The API proxy serves these dynamic catalogue images. */}
        {/* eslint-disable-next-line @next/next/no-img-element */}
        <img
          src={`/api/backend/gallery/${encodeURIComponent(id)}/image`}
          alt={String(title)}
          className="h-full w-full object-contain transition-transform duration-300 hover:scale-[1.02]"
        />
        <Badge className="absolute left-3 top-3 bg-background text-foreground shadow-sm">#{rank}</Badge>
      </div>
      <CardContent className="space-y-3 p-4">
        <div>
          <p className="text-xs text-muted-foreground">Item {id}</p>
          <h3 className="mt-1 line-clamp-2 min-h-10 text-sm font-semibold leading-5">{String(title)}</h3>
        </div>
        <div className="flex min-h-6 flex-wrap gap-1.5">
          {[item.gender, item.season, item.usage].filter(Boolean).map((tag) => (
            <Badge variant="secondary" key={String(tag)}>{String(tag)}</Badge>
          ))}
        </div>
        <div className="flex items-center justify-between border-t pt-3 text-xs">
          <span className="text-muted-foreground" title="Cosine similarity ranks visual features; it is not a probability of a correct match.">Similarity</span>
          <span className="font-semibold">{score.toFixed(3)}</span>
        </div>
      </CardContent>
    </Card>
  )
}

export function FashionAnalyser() {
  const inputRef = useRef<HTMLInputElement>(null)
  const [file, setFile] = useState<File | null>(null)
  const [preview, setPreview] = useState<string | null>(null)
  const [result, setResult] = useState<AnalysisResponse | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [busy, setBusy] = useState(false)
  const [dragging, setDragging] = useState(false)
  const [health, setHealth] = useState<"checking" | "ready" | "partial" | "offline">("checking")

  useEffect(() => {
    const controller = new AbortController()
    fetch("/api/backend/health", { cache: "no-store", signal: controller.signal })
      .then(async (response) => {
        if (!response.ok) throw new Error("API unavailable")
        const body = (await response.json()) as HealthResponse
        setHealth(Object.values(body.models).every(Boolean) ? "ready" : "partial")
      })
      .catch((requestError: unknown) => {
        if ((requestError as Error).name !== "AbortError") setHealth("offline")
      })
    return () => controller.abort()
  }, [])

  useEffect(() => {
    return () => {
      if (preview) URL.revokeObjectURL(preview)
    }
  }, [preview])

  function chooseFile(nextFile: File | undefined) {
    setError(null)
    setResult(null)
    if (!nextFile) return
    if (!nextFile.type.startsWith("image/")) {
      setError("Choose a JPG, PNG, or WebP image.")
      return
    }
    if (nextFile.size > MAX_FILE_BYTES) {
      setError("The image must be smaller than 10 MB.")
      return
    }
    if (preview) URL.revokeObjectURL(preview)
    setFile(nextFile)
    setPreview(URL.createObjectURL(nextFile))
  }

  function onInput(event: ChangeEvent<HTMLInputElement>) {
    chooseFile(event.target.files?.[0])
  }

  function onDrop(event: DragEvent<HTMLDivElement>) {
    event.preventDefault()
    setDragging(false)
    chooseFile(event.dataTransfer.files?.[0])
  }

  async function analyse() {
    if (!file) return
    setBusy(true)
    setError(null)
    setResult(null)
    const formData = new FormData()
    formData.append("file", file)

    try {
      const response = await fetch("/api/backend/analyse?prediction_top_k=3&search_top_k=5", {
        method: "POST",
        body: formData,
      })
      const body = await response.json()
      if (!response.ok) {
        throw new Error(typeof body.detail === "string" ? body.detail : "Analysis failed.")
      }
      const analysisResult = body as AnalysisResponse
      setResult(analysisResult)

      try {
        const thumbnail = await createThumbnail(file)
        addHistoryEntry({
          id: crypto.randomUUID(),
          createdAt: new Date().toISOString(),
          fileName: file.name,
          thumbnail,
          predictions: analysisResult.predictions,
        })
      } catch (historyError) {
        console.warn("Analysis completed, but history could not be saved.", historyError)
      }
    } catch (requestError) {
      setError(requestError instanceof Error ? requestError.message : "Analysis failed.")
    } finally {
      setBusy(false)
    }
  }

  function reset() {
    if (preview) URL.revokeObjectURL(preview)
    setFile(null)
    setPreview(null)
    setResult(null)
    setError(null)
    if (inputRef.current) inputRef.current.value = ""
  }

  const status = {
    checking: { label: "Checking service", className: "text-muted-foreground" },
    ready: { label: "Ready", className: "border-emerald-200 bg-emerald-50 text-emerald-700 dark:border-emerald-900 dark:bg-emerald-950 dark:text-emerald-300" },
    partial: { label: "Models unavailable", className: "border-amber-200 bg-amber-50 text-amber-700 dark:border-amber-900 dark:bg-amber-950 dark:text-amber-300" },
    offline: { label: "Service offline", className: "border-red-200 bg-red-50 text-red-700 dark:border-red-900 dark:bg-red-950 dark:text-red-300" },
  }[health]

  return (
    <section id="analyser" className="scroll-mt-20 px-4 pb-20 md:px-6">
      <div className="mx-auto max-w-5xl">
        <div className="mb-6 flex items-end justify-between gap-4">
          <div>
            <h2 className="text-2xl font-bold tracking-tight">Analyse a garment</h2>
            <p className="mt-1 text-sm text-muted-foreground">Choose a clear product photo to get started.</p>
          </div>
          <Badge variant="outline" className={status.className}>
            <span className={cn("mr-1.5 size-1.5 rounded-full bg-current", health === "checking" && "animate-pulse")} />
            {status.label}
          </Badge>
        </div>

        <div className="grid gap-6 md:grid-cols-2">
          <Card>
            <CardHeader>
              <CardTitle>Upload image</CardTitle>
              <CardDescription>JPG, PNG, or WebP up to 10 MB</CardDescription>
            </CardHeader>
            <CardContent>
              <div
                className={cn(
                  "relative flex h-80 items-center justify-center overflow-hidden rounded-lg border border-dashed bg-muted/30 transition-colors",
                  dragging && "border-primary bg-muted",
                )}
                onDragOver={(event) => { event.preventDefault(); setDragging(true) }}
                onDragLeave={() => setDragging(false)}
                onDrop={onDrop}
              >
                {preview ? (
                  <>
                    {/* Blob URLs cannot be optimized by next/image. */}
                    {/* eslint-disable-next-line @next/next/no-img-element */}
                    <img src={preview} alt="Selected fashion item" className="h-full w-full object-contain" />
                    <Button
                      type="button"
                      variant="secondary"
                      size="sm"
                      className="absolute bottom-3 right-3 shadow-sm"
                      onClick={() => inputRef.current?.click()}
                    >
                      Replace image
                    </Button>
                  </>
                ) : (
                  <button type="button" onClick={() => inputRef.current?.click()} className="flex h-full w-full flex-col items-center justify-center p-8 text-center">
                    <span className="mb-4 flex size-12 items-center justify-center rounded-full border bg-background">
                      <Upload className="size-5" aria-hidden="true" />
                    </span>
                    <span className="font-medium">Drop an image here</span>
                    <span className="mt-1 text-sm text-muted-foreground">or click to choose a file</span>
                  </button>
                )}
                <input ref={inputRef} type="file" accept="image/jpeg,image/png,image/webp" onChange={onInput} hidden />
              </div>
              {file && <p className="mt-3 truncate text-xs text-muted-foreground">Selected: {file.name}</p>}
              {error && (
                <Alert className="mt-4 border-red-200 bg-red-50 text-red-800 dark:border-red-900 dark:bg-red-950 dark:text-red-200">
                  <AlertCircle className="size-4" />
                  <AlertTitle>Couldn&apos;t complete the analysis</AlertTitle>
                  <AlertDescription className="text-red-700 dark:text-red-300">{error}</AlertDescription>
                </Alert>
              )}
            </CardContent>
            <CardFooter className="gap-2">
              <Button type="button" size="lg" className="flex-1" disabled={!file || busy} onClick={analyse}>
                {busy ? <Loader2 className="animate-spin" /> : <Search />}
                {busy ? "Analysing..." : "Analyse image"}
              </Button>
              {file && (
                <Button type="button" variant="outline" size="icon-lg" aria-label="Clear selection" onClick={reset}>
                  <X />
                </Button>
              )}
            </CardFooter>
          </Card>

          <Card>
            <CardHeader>
              <CardTitle>Prediction result</CardTitle>
              <CardDescription>{result ? "The strongest prediction for each attribute" : "Your results will appear here after analysis"}</CardDescription>
            </CardHeader>
            <CardContent className="flex flex-1">
              {result ? (
                <div className="w-full divide-y">
                  {TARGET_ORDER.map((target) => {
                    const prediction = result.predictions[target]
                    if (!prediction) return null
                    return (
                      <div className="flex items-center justify-between gap-4 py-4 first:pt-0 last:pb-0" key={target}>
                        <div className="flex items-center gap-3">
                          {prediction.needs_review ? (
                            <AlertCircle className="size-4 text-amber-600" aria-hidden="true" />
                          ) : (
                            <CheckCircle2 className="size-4 text-emerald-600" aria-hidden="true" />
                          )}
                          <div>
                            <p className="text-xs text-muted-foreground">{targetLabel(target)}</p>
                            <p className="font-medium">{prediction.label}</p>
                            {prediction.needs_review && (
                              <p className="text-xs text-amber-700 dark:text-amber-400">Needs review</p>
                            )}
                          </div>
                        </div>
                        <Badge variant="secondary">{percent(prediction.confidence)}</Badge>
                      </div>
                    )
                  })}
                </div>
              ) : (
                <div className="flex min-h-80 w-full flex-col items-center justify-center rounded-lg border border-dashed bg-muted/20 p-8 text-center">
                  <ImageIcon className="mb-4 size-9 text-muted-foreground/60" strokeWidth={1.5} aria-hidden="true" />
                  <p className="font-medium">No result yet</p>
                  <p className="mt-1 max-w-64 text-sm leading-6 text-muted-foreground">Upload a garment image and select Analyse image.</p>
                </div>
              )}
            </CardContent>
          </Card>
        </div>

        {result && (
          <div className="mt-16" aria-live="polite">
            <div className="mb-6">
              <h2 className="text-2xl font-bold tracking-tight">Prediction details</h2>
              <p className="mt-1 text-sm text-muted-foreground">Top model outputs and confidence scores.</p>
            </div>
            <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
              {TARGET_ORDER.map((target) => result.predictions[target] && (
                <PredictionCard key={target} target={target} prediction={result.predictions[target]} />
              ))}
            </div>

            <div className="mb-6 mt-16 flex items-end justify-between gap-4">
              <div>
                <h2 className="text-2xl font-bold tracking-tight">Similar catalogue items</h2>
                <p className="mt-1 text-sm text-muted-foreground">Products with the closest visual features.</p>
              </div>
              <p className="text-sm text-muted-foreground">{result.similar_items.length} matches</p>
            </div>
            <div className="grid gap-5 sm:grid-cols-2 lg:grid-cols-5">
              {result.similar_items.map((item, index) => (
                <SimilarCard item={item} rank={index + 1} key={`${item.id}-${index}`} />
              ))}
            </div>
          </div>
        )}
      </div>
    </section>
  )
}
