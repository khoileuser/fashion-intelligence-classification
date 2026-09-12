import { FashionAnalyser } from "@/components/fashion-analyser"
import { AnalysisHistory } from "@/components/analysis-history"
import { BatchAnalyser } from "@/components/batch-analyser"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
export default async function AnalysePage({
    searchParams,
}: {
    searchParams: Promise<{ sample?: string }>
}) {
    const { sample } = await searchParams
    const initialSample =
        sample && /^\d{1,20}$/.test(sample) ? sample : undefined
    return (
        <main id="main-content" className="mx-auto max-w-7xl py-10">
            <div className="px-4 md:px-6">
                <p className="text-xs tracking-widest text-muted-foreground">
                    YOUR WORKSPACE
                </p>
                <h1 className="mt-3 text-4xl font-semibold tracking-tight">
                    Meet your next collection.
                </h1>
                <p className="mt-3 text-muted-foreground">
                    Explore one item or turn a batch of images into reviewed
                    product attributes.
                </p>
            </div>
            <Tabs defaultValue="single" className="mt-8">
                <TabsList className="mx-4 mb-8 md:mx-6">
                    <TabsTrigger value="single">Single image</TabsTrigger>
                    <TabsTrigger value="batch">
                        Batch review & export
                    </TabsTrigger>
                </TabsList>
                <TabsContent value="single" keepMounted>
                    <FashionAnalyser
                        key={initialSample || "upload"}
                        initialSample={initialSample}
                    />
                    <AnalysisHistory />
                </TabsContent>
                <TabsContent value="batch" keepMounted>
                    <BatchAnalyser />
                </TabsContent>
            </Tabs>
        </main>
    )
}
