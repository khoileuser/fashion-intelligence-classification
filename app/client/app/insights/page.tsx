import { ModelInsights } from "@/components/model-insights"
export default function InsightsPage() {
    return (
        <main
            id="main-content"
            className="mx-auto max-w-7xl px-4 py-10 md:px-6"
        >
            <p className="text-xs tracking-widest text-muted-foreground">
                BEHIND THE PREDICTIONS
            </p>
            <h1 className="mt-3 text-4xl font-semibold tracking-tight">
                Model insights.
            </h1>
            <p className="mt-3 mb-8 text-muted-foreground">
                Measured results, model comparisons, and the limits of the
                current experiments.
            </p>
            <ModelInsights />
        </main>
    )
}
