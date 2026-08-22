"use client"

import { useMemo, useState, useSyncExternalStore } from "react"
import { Clock3, Trash2 } from "lucide-react"

import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"
import { Card, CardContent } from "@/components/ui/card"
import { clearHistory, HISTORY_STORAGE_KEY, HISTORY_UPDATED_EVENT, type HistoryEntry } from "@/lib/history"

const ITEMS_PER_PAGE = 6
const TARGETS = [
  ["articleType", "Type"],
  ["season", "Season"],
  ["gender", "Gender"],
  ["usage", "Occasion"],
] as const

function subscribeToHistory(onStoreChange: () => void) {
  window.addEventListener(HISTORY_UPDATED_EVENT, onStoreChange)
  window.addEventListener("storage", onStoreChange)
  return () => {
    window.removeEventListener(HISTORY_UPDATED_EVENT, onStoreChange)
    window.removeEventListener("storage", onStoreChange)
  }
}

function getHistorySnapshot() {
  return window.localStorage.getItem(HISTORY_STORAGE_KEY) ?? "[]"
}

export function AnalysisHistory() {
  const [page, setPage] = useState(1)
  const historySnapshot = useSyncExternalStore(subscribeToHistory, getHistorySnapshot, () => "[]")
  const entries = useMemo(() => {
    try {
      const parsed: unknown = JSON.parse(historySnapshot)
      return Array.isArray(parsed) ? (parsed as HistoryEntry[]) : []
    } catch {
      return []
    }
  }, [historySnapshot])

  const pageCount = Math.max(1, Math.ceil(entries.length / ITEMS_PER_PAGE))
  const currentPage = Math.min(page, pageCount)
  const visibleEntries = entries.slice((currentPage - 1) * ITEMS_PER_PAGE, currentPage * ITEMS_PER_PAGE)

  return (
    <section id="history" className="scroll-mt-20 border-t px-4 py-16 md:px-6 md:py-20">
      <div className="mx-auto max-w-6xl">
        <div className="mb-8 flex items-end justify-between gap-4">
          <div>
            <h2 className="text-3xl font-bold tracking-tight">Analysis history</h2>
            <p className="mt-2 text-muted-foreground">Your latest analyses are saved on this device.</p>
          </div>
          {entries.length > 0 && (
            <Button type="button" variant="outline" onClick={() => { setPage(1); clearHistory() }}>
              <Trash2 />
              Clear history
            </Button>
          )}
        </div>

        {visibleEntries.length === 0 ? (
          <div className="flex min-h-56 flex-col items-center justify-center rounded-xl border border-dashed bg-muted/20 p-8 text-center">
            <Clock3 className="mb-4 size-9 text-muted-foreground/60" strokeWidth={1.5} />
            <p className="font-medium">No analyses yet</p>
            <p className="mt-1 text-sm text-muted-foreground">Completed analyses will appear here automatically.</p>
          </div>
        ) : (
          <>
            <div className="grid gap-5 sm:grid-cols-2 lg:grid-cols-3">
              {visibleEntries.map((entry) => (
                <Card className="gap-0 overflow-hidden py-0" key={entry.id}>
                  {/* History thumbnails are generated data URLs. */}
                  {/* eslint-disable-next-line @next/next/no-img-element */}
                  <img src={entry.thumbnail} alt={entry.fileName} className="aspect-[4/3] w-full bg-muted object-contain" />
                  <CardContent className="space-y-4 p-5">
                    <div>
                      <p className="truncate font-medium">{entry.fileName}</p>
                      <p className="mt-1 text-xs text-muted-foreground">
                        {new Intl.DateTimeFormat(undefined, { dateStyle: "medium", timeStyle: "short" }).format(new Date(entry.createdAt))}
                      </p>
                    </div>
                    <div className="grid grid-cols-2 gap-3 border-t pt-4">
                      {TARGETS.map(([target, label]) => {
                        const prediction = entry.predictions[target]
                        return prediction ? (
                          <div key={target} className="min-w-0">
                            <p className="text-xs text-muted-foreground">{label}</p>
                            <p className="truncate text-sm font-medium">{prediction.label}</p>
                          </div>
                        ) : null
                      })}
                    </div>
                  </CardContent>
                </Card>
              ))}
            </div>
            {pageCount > 1 && (
              <div className="mt-8 flex items-center justify-center gap-3">
                <Button type="button" variant="outline" disabled={currentPage === 1} onClick={() => setPage((value) => value - 1)}>Previous</Button>
                <Badge variant="secondary">Page {currentPage} of {pageCount}</Badge>
                <Button type="button" variant="outline" disabled={currentPage === pageCount} onClick={() => setPage((value) => value + 1)}>Next</Button>
              </div>
            )}
          </>
        )}
      </div>
    </section>
  )
}
