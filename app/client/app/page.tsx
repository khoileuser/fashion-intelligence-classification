import { SampleGallery } from "@/components/sample-gallery"
import Link from "next/link"
import {
    ArrowRight,
    Layers,
    ScanLine,
    ChartNoAxesCombined,
    Shirt,
} from "lucide-react"
import { Badge } from "@/components/ui/badge"
import { buttonVariants } from "@/components/ui/button"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
export default function Home() {
    return (
        <main id="main-content">
            <section className="mx-auto grid max-w-7xl items-center gap-12 px-4 py-16 md:px-6 md:py-24 lg:grid-cols-2">
                <div>
                    <Badge variant="outline" className="mb-6">
                        FROM IMAGE TO INFORMATION
                    </Badge>
                    <h1 className="max-w-xl text-5xl font-semibold leading-[1.08] tracking-tight md:text-7xl">
                        A clearer view
                        <br />
                        of every garment.
                    </h1>
                    <p className="mt-6 max-w-lg text-lg leading-8 text-muted-foreground">
                        Turn fashion photos into searchable product information.
                        Identify attributes, discover similar pieces, and review
                        a whole collection in one place.
                    </p>
                    <div className="mt-8 flex flex-wrap gap-3">
                        <Link
                            href="/analyse"
                            className={buttonVariants({ size: "lg" })}
                        >
                            Analyse <ArrowRight />
                        </Link>
                        <Link
                            href="/insights"
                            className={buttonVariants({
                                variant: "outline",
                                size: "lg",
                            })}
                        >
                            Model insights
                        </Link>
                    </div>
                </div>
                <div className="relative rounded-3xl border bg-muted/60 p-6 md:p-10">
                    <div className="mb-5 flex justify-between text-xs text-muted-foreground">
                        <span>THE ANALYSIS WORKSPACE</span>
                        <span>Illustrative preview</span>
                    </div>
                    <div className="grid gap-4 sm:grid-cols-2">
                        <div className="flex min-h-64 flex-col items-center justify-center rounded-2xl bg-background p-6">
                            <Shirt
                                className="size-32 text-foreground/70"
                                strokeWidth={0.8}
                            />
                            <span className="mt-6 text-xs text-muted-foreground">
                                Your next catalogue entry
                            </span>
                        </div>
                        <div className="flex flex-col justify-center gap-3">
                            {[
                                ["Article type", "Shirts"],
                                ["Season", "Summer"],
                                ["Catalogue gender", "Men"],
                                ["Occasion", "Casual"],
                            ].map(([title, value]) => (
                                <div
                                    key={title}
                                    className="rounded-xl border bg-card p-3"
                                >
                                    <p className="text-xs text-muted-foreground">
                                        {title}
                                    </p>
                                    <p className="mt-1 font-medium">{value}</p>
                                </div>
                            ))}
                        </div>
                    </div>
                </div>
            </section>
            <section className="border-y bg-muted/30">
                <div className="mx-auto max-w-7xl px-4 py-16 md:px-6">
                    <div className="mb-8 flex flex-wrap items-end justify-between gap-4">
                        <h2 className="text-3xl font-semibold tracking-tight">
                            A photo is just the beginning.
                        </h2>
                    </div>
                    <div className="grid gap-5 md:grid-cols-3">
                        {[
                            {
                                icon: ScanLine,
                                title: "Understand each item",
                                text: "Predict article type, season, catalogue gender, and occasion, with ranked alternatives and visual matches.",
                            },
                            {
                                icon: Layers,
                                title: "Work through a collection",
                                text: "Queue multiple images, correct attributes, and download a catalogue-ready CSV.",
                            },
                            {
                                icon: ChartNoAxesCombined,
                                title: "Look behind the prediction",
                                text: "Explore measured performance, compare model families, and inspect where classifications are confused.",
                            },
                        ].map(({ icon: Icon, title, text }) => (
                            <Card key={title}>
                                <CardHeader>
                                    <Icon className="mb-4 size-6" />
                                    <CardTitle>{title}</CardTitle>
                                </CardHeader>
                                <CardContent className="leading-7 text-muted-foreground">
                                    {text}
                                </CardContent>
                            </Card>
                        ))}
                    </div>
                </div>
            </section>
            <SampleGallery />
        </main>
    )
}
