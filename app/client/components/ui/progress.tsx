import { cn } from "@/lib/utils"

function Progress({ value = 0, className }: { value?: number; className?: string }) {
  const normalizedValue = Math.max(0, Math.min(100, value))

  return (
    <div
      data-slot="progress"
      className={cn("h-2 w-full overflow-hidden rounded-full bg-secondary", className)}
      role="progressbar"
      aria-valuemin={0}
      aria-valuemax={100}
      aria-valuenow={Math.round(normalizedValue)}
    >
      <div className="h-full rounded-full bg-primary transition-[width]" style={{ width: `${normalizedValue}%` }} />
    </div>
  )
}

export { Progress }
