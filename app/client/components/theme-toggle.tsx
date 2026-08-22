"use client"

import { useSyncExternalStore } from "react"
import { Moon, Sun } from "lucide-react"
import { useTheme } from "next-themes"

import { Button } from "@/components/ui/button"

export function ThemeToggle() {
  const { resolvedTheme, setTheme } = useTheme()
  const mounted = useSyncExternalStore(
    () => () => undefined,
    () => true,
    () => false,
  )

  return (
    <Button
      type="button"
      variant="outline"
      size="icon"
      aria-label={mounted && resolvedTheme === "dark" ? "Use light theme" : "Use dark theme"}
      title={mounted && resolvedTheme === "dark" ? "Use light theme" : "Use dark theme"}
      onClick={() => setTheme(resolvedTheme === "dark" ? "light" : "dark")}
    >
      {mounted && resolvedTheme === "dark" ? <Sun /> : <Moon />}
    </Button>
  )
}
