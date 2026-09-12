"use client"
import Link from "next/link"
import { usePathname } from "next/navigation"
import { Shirt } from "lucide-react"
import { ThemeToggle } from "@/components/theme-toggle"
import { cn } from "@/lib/utils"
export function SiteHeader() {
    const pathname = usePathname()
    return (
        <header className="sticky top-0 z-50 border-b bg-background/95 backdrop-blur">
            <nav
                aria-label="Main navigation"
                className="mx-auto flex max-w-7xl flex-wrap items-center justify-between gap-4 px-4 py-4 md:px-6"
            >
                <Link
                    href="/"
                    className="flex items-center gap-2 text-lg font-semibold tracking-tight"
                >
                    <span className="rounded-lg bg-primary p-2 text-primary-foreground">
                        <Shirt className="size-5" />
                    </span>
                    Threadline
                    <span className="hidden text-xs font-normal text-muted-foreground lg:inline">
                        {" "}
                        / Fashion intelligence
                    </span>
                </Link>
                <div className="flex flex-wrap items-center gap-3 text-sm md:gap-6">
                    {[
                        ["/", "Home"],
                        ["/analyse", "Analyse"],
                        ["/catalogue", "Catalogue"],
                        ["/insights", "Model insights"],
                    ].map(([href, label]) => (
                        <Link
                            key={href}
                            href={href}
                            aria-current={
                                pathname === href ? "page" : undefined
                            }
                            className={cn(
                                "transition-colors hover:text-foreground",
                                pathname === href
                                    ? "font-semibold text-foreground"
                                    : "text-muted-foreground",
                            )}
                        >
                            {label}
                        </Link>
                    ))}
                    <ThemeToggle />
                </div>
            </nav>
        </header>
    )
}
