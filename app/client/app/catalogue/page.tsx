import { CatalogueExplorer } from "@/components/catalogue-explorer"
export default function CataloguePage() {
    return (
        <main
            id="main-content"
            className="mx-auto max-w-7xl px-4 py-10 md:px-6"
        >
            <p className="text-xs tracking-widest text-muted-foreground">
                THE COLLECTION
            </p>
            <h1 className="mt-3 text-4xl font-semibold tracking-tight">
                Find your next reference.
            </h1>
            <p className="mb-8 mt-3 text-muted-foreground">
                Browse product attributes, explore visual neighbours, or send
                any item to the analyser.
            </p>
            <CatalogueExplorer />
        </main>
    )
}
