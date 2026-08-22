import type { Metadata } from "next"
import { Geist } from "next/font/google"

import { ThemeProvider } from "@/components/theme-provider"

import "./globals.css"

const geist = Geist({ subsets: ["latin"], variable: "--font-geist-sans" })

export const metadata: Metadata = {
  title: "Threadline | Fashion Intelligence",
  description: "Classify fashion attributes and discover visually similar products.",
}

export default function RootLayout({ children }: Readonly<{ children: React.ReactNode }>) {
  return (
    <html lang="en" className={geist.variable} suppressHydrationWarning>
      <body>
        <ThemeProvider>{children}</ThemeProvider>
      </body>
    </html>
  )
}
