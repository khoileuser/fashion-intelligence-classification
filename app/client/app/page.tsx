import { Shirt } from "lucide-react"

import { FashionAnalyser } from "@/components/fashion-analyser"

export default function Home() {
  return (
    <main className="min-h-screen">
      <nav className="sticky top-0 z-50 border-b bg-background/95 backdrop-blur">
        <div className="mx-auto flex h-16 max-w-7xl items-center justify-between px-4 md:px-6">
          <a href="#top" className="flex items-center gap-2 font-semibold">
            <span className="flex size-9 items-center justify-center rounded-lg bg-primary text-primary-foreground">
              <Shirt className="size-5" aria-hidden="true" />
            </span>
            <span>Threadline</span>
          </a>
          <div className="flex items-center gap-5 text-sm">
            <a className="text-muted-foreground transition-colors hover:text-foreground" href="#analyser">Analyser</a>
            <a className="hidden text-muted-foreground transition-colors hover:text-foreground sm:block" href="#how-it-works">How it works</a>
          </div>
        </div>
      </nav>

      <section id="top" className="mx-auto flex max-w-4xl flex-col items-center px-4 pb-16 pt-20 text-center md:pb-20 md:pt-28">
        <div className="mb-6 flex size-20 items-center justify-center rounded-2xl border bg-muted/50">
          <Shirt className="size-9" strokeWidth={1.5} aria-hidden="true" />
        </div>
        <p className="mb-3 text-sm font-medium text-muted-foreground">Fashion image analysis</p>
        <h1 className="max-w-3xl text-balance text-4xl font-bold tracking-tight sm:text-5xl md:text-6xl">
          Understand what you&apos;re looking at.
        </h1>
        <p className="mt-5 max-w-2xl text-pretty text-base leading-7 text-muted-foreground md:text-lg">
          Upload a product photo to identify its type, season, gender, and occasion, then find similar pieces in the catalogue.
        </p>
      </section>

      <FashionAnalyser />

      <section id="how-it-works" className="border-t bg-muted/30">
        <div className="mx-auto max-w-6xl px-4 py-16 md:px-6 md:py-20">
          <div className="mx-auto mb-10 max-w-2xl text-center">
            <h2 className="text-3xl font-bold tracking-tight">One photo, four useful details</h2>
            <p className="mt-3 text-muted-foreground">The analyser turns a fashion image into catalogue-ready attributes.</p>
          </div>
          <div className="grid gap-px overflow-hidden rounded-xl border bg-border sm:grid-cols-2 lg:grid-cols-4">
            {[
              ["Article type", "The garment category and silhouette."],
              ["Season", "The most likely seasonal collection."],
              ["Gender", "The catalogue gender classification."],
              ["Occasion", "The expected setting or use case."],
            ].map(([title, description]) => (
              <div className="bg-background p-6" key={title}>
                <h3 className="font-semibold">{title}</h3>
                <p className="mt-2 text-sm leading-6 text-muted-foreground">{description}</p>
              </div>
            ))}
          </div>
        </div>
      </section>

      <footer className="border-t">
        <div className="mx-auto flex max-w-7xl flex-col gap-2 px-4 py-8 text-sm text-muted-foreground sm:flex-row sm:items-center sm:justify-between md:px-6">
          <p>Threadline fashion intelligence</p>
          <p>Classification and visual catalogue search</p>
        </div>
      </footer>
    </main>
  )
}
