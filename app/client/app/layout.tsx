import type { Metadata } from "next"
import { Geist } from "next/font/google"

import { ThemeProvider } from "@/components/theme-provider"

import { SiteHeader } from "@/components/site-header"
import "./globals.css"

const geist = Geist({ subsets: ["latin"], variable: "--font-geist-sans" })

export const metadata: Metadata = {
    title: "Threadline | Fashion Intelligence",
    description:
        "Classify fashion attributes and discover visually similar products.",
}

export default function RootLayout({
    children,
}: Readonly<{ children: React.ReactNode }>) {
    return (
        <html
            lang="en"
            data-scroll-behavior="smooth"
            className={geist.variable}
            suppressHydrationWarning
        >
            <body>
                <ThemeProvider>
                    <a
                        href="#main-content"
                        className="sr-only focus:not-sr-only focus:absolute focus:z-100 focus:bg-background focus:p-4"
                    >
                        Skip to content
                    </a>
                    <SiteHeader />
                    {children}
                </ThemeProvider>
            </body>
        </html>
    )
}
