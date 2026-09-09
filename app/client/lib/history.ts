import type { AnalysisResponse } from "@/lib/types"

export const HISTORY_STORAGE_KEY = "threadline.analysis-history.v1"
export const HISTORY_UPDATED_EVENT = "threadline:history-updated"
export const MAX_HISTORY_ITEMS = 12

export type HistoryEntry = {
  id: string
  createdAt: string
  fileName: string
  thumbnail: string
  predictions: AnalysisResponse["predictions"]
}

export function readHistory(): HistoryEntry[] {
  try {
    const value = window.localStorage.getItem(HISTORY_STORAGE_KEY)
    if (!value) return []
    const parsed: unknown = JSON.parse(value)
    return Array.isArray(parsed) ? (parsed as HistoryEntry[]) : []
  } catch {
    return []
  }
}

export function addHistoryEntry(entry: HistoryEntry) {
  const next = [entry, ...readHistory()].slice(0, MAX_HISTORY_ITEMS)
  window.localStorage.setItem(HISTORY_STORAGE_KEY, JSON.stringify(next))
  window.dispatchEvent(new Event(HISTORY_UPDATED_EVENT))
}

export function clearHistory() {
  window.localStorage.removeItem(HISTORY_STORAGE_KEY)
  window.dispatchEvent(new Event(HISTORY_UPDATED_EVENT))
}

export async function createThumbnail(file: File) {
  const bitmap = await createImageBitmap(file)
  const maxSide = 480
  const scale = Math.min(1, maxSide / Math.max(bitmap.width, bitmap.height))
  const canvas = document.createElement("canvas")
  canvas.width = Math.max(1, Math.round(bitmap.width * scale))
  canvas.height = Math.max(1, Math.round(bitmap.height * scale))
  const context = canvas.getContext("2d")
  if (!context) throw new Error("Image preview could not be created")
  context.drawImage(bitmap, 0, 0, canvas.width, canvas.height)
  bitmap.close()
  return canvas.toDataURL("image/jpeg", 0.76)
}
