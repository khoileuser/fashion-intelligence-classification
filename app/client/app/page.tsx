import { SampleGallery } from "@/components/sample-gallery"
import Link from "next/link"
import {
    ArrowRight,
    Layers,
    ScanLine,
    ChartNoAxesCombined,
    CheckCheck,
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
                            Start analysing <ArrowRight />
                        </Link>
                        <Link
                            href="/insights"
                            className={buttonVariants({
                                variant: "outline",
                                size: "lg",
                            })}
                        >
                            Explore model insights
                        </Link>
                    </div>
                    <p className="mt-6 text-xs text-muted-foreground">
                        Four classification tasks / Visual search / CSV export
                    </p>
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
                    <div className="mt-4 flex items-center gap-3 rounded-xl border bg-card p-4 text-sm">
                        <CheckCheck className="size-5" />
                        <span>Analyse. Review. Export.</span>
                        <Badge variant="secondary" className="ml-auto">
                            Your workflow
                        </Badge>
                    </div>
                </div>
            </section>
            <section className="border-y bg-muted/30">
                <div className="mx-auto max-w-7xl px-4 py-16 md:px-6">
                    <div className="mb-8 flex flex-wrap items-end justify-between gap-4">
                        <h2 className="text-3xl font-semibold tracking-tight">
                            A photo is just the beginning.
                        </h2>
                        <p className="max-w-md text-sm leading-6 text-muted-foreground">
                            A practical workspace backed by measurable machine
                            learning experiments.
                        </p>
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
            <section className="mx-auto max-w-7xl px-4 py-16 md:px-6">
                <h2 className="text-3xl font-semibold tracking-tight">
                    From upload to a reviewed catalogue.
                </h2>
                <div className="mt-8 grid gap-8 md:grid-cols-3">
                    {[
                        [
                            "01",
                            "Bring your images",
                            "Use a clear product photo for a single analysis, or upload up to 20 images for batch processing.",
                        ],
                        [
                            "02",
                            "Make the final call",
                            "Inspect model confidence and review flags. Adjust the attributes before exporting.",
                        ],
                        [
                            "03",
                            "Take your results with you",
                            "Export analysed entries with original predictions and corrections preserved separately.",
                        ],
                    ].map(([n, title, text]) => (
                        <div key={n}>
                            <span className="font-mono text-sm text-muted-foreground">
                                {n} /
                            </span>
                            <h3 className="mt-3 font-semibold">{title}</h3>
                            <p className="mt-2 text-sm leading-7 text-muted-foreground">
                                {text}
                            </p>
                        </div>
                    ))}
                </div>
            </section>
            <SampleGallery />
        </main>
    )
}
