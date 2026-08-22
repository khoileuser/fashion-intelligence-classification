import * as React from "react"

import { cn } from "@/lib/utils"

function Alert({ className, ...props }: React.ComponentProps<"div">) {
  return <div data-slot="alert" role="alert" className={cn("grid w-full grid-cols-[0_1fr] gap-y-1 rounded-lg border px-4 py-3 text-sm has-[>svg]:grid-cols-[calc(var(--spacing)*4)_1fr] has-[>svg]:gap-x-3", className)} {...props} />
}

function AlertTitle({ className, ...props }: React.ComponentProps<"div">) {
  return <div data-slot="alert-title" className={cn("col-start-2 font-medium", className)} {...props} />
}

function AlertDescription({ className, ...props }: React.ComponentProps<"div">) {
  return <div data-slot="alert-description" className={cn("col-start-2 text-muted-foreground", className)} {...props} />
}

export { Alert, AlertDescription, AlertTitle }
