"use client"
import { useState } from "react"
import data from "@/lib/insights-data.json"
import {
    Card,
    CardHeader,
    CardTitle,
    CardContent,
    CardDescription,
} from "@/components/ui/card"
import { DropdownSelect, DropdownOption } from "@/components/dropdown-select"
import { Input } from "@/components/ui/input"
import { Badge } from "@/components/ui/badge"
import {
    Table,
    TableHeader,
    TableBody,
    TableHead,
    TableRow,
    TableCell,
} from "@/components/ui/table"
const pct = (n: number | string) => `${(Number(n) * 100).toFixed(1)}%`
export function ModelInsights() {
    const [target, setTarget] = useState(data[0].target),
        [query, setQuery] = useState("")
    const task = data.find((t) => t.target === target)!
    const classes = task.per_class.filter((r) =>
        r[""].toLowerCase().includes(query.toLowerCase()),
    )
    return (
        <div className="space-y-6">
            <div className="flex flex-wrap items-center justify-between gap-4">
                <DropdownSelect
                    aria-label="Classification task"
                    value={target}
                    onValueChange={(value) => {
                        setTarget(value)
                        setQuery("")
                    }}
                >
                    {data.map((t) => (
                        <DropdownOption key={t.target} value={t.target}>
                            {t.title}
                        </DropdownOption>
                    ))}
                </DropdownSelect>
                <Badge variant="outline">
                    Selected: {task.selected.replaceAll("_", " ")}
                </Badge>
            </div>
            <div className="grid gap-4 sm:grid-cols-3">
                {[
                    [
                        "Test accuracy",
                        pct(task.test_metrics.accuracy),
                        "Overall share of correct predictions.",
                    ],
                    [
                        "Test macro-F1",
                        pct(task.test_metrics.macro_f1),
                        "Weights each represented test class equally, including rare classes.",
                    ],
                    [
                        "Calibration error (ECE)",
                        pct(task.test_metrics.ece),
                        "Confidence versus correctness; lower is better.",
                    ],
                ].map(([title, value, desc]) => (
                    <Card key={title}>
                        <CardHeader>
                            <CardDescription>{title}</CardDescription>
                            <CardTitle className="text-4xl">{value}</CardTitle>
                        </CardHeader>
                        <CardContent className="text-sm text-muted-foreground">
                            {desc}
                        </CardContent>
                    </Card>
                ))}
            </div>
            <Card>
                <CardHeader>
                    <CardTitle>Model family comparison</CardTitle>
                    <CardDescription>
                        Validation results used for selection, separate from the
                        test metrics above.
                    </CardDescription>
                </CardHeader>
                <CardContent>
                    <Table>
                        <TableHeader>
                            <TableRow>
                                {[
                                    "Model",
                                    "Accuracy",
                                    "Macro-F1",
                                    "Parameters",
                                    "Epochs",
                                    "Selection",
                                ].map((h) => (
                                    <TableHead key={h}>{h}</TableHead>
                                ))}
                            </TableRow>
                        </TableHeader>
                        <TableBody>
                            {task.comparison.map((r) => (
                                <TableRow key={r[""]}>
                                    <TableCell className="font-medium">
                                        {r[""].replaceAll("_", " ")}
                                    </TableCell>
                                    <TableCell>
                                        {pct(r.validation_accuracy)}
                                    </TableCell>
                                    <TableCell>
                                        {pct(r.validation_macro_f1)}
                                    </TableCell>
                                    <TableCell>
                                        {Number(
                                            r.complexity_parameters,
                                        ).toLocaleString()}
                                    </TableCell>
                                    <TableCell>{r.epochs_run}</TableCell>
                                    <TableCell>
                                        {r.selected === "True" ? (
                                            <Badge>Selected</Badge>
                                        ) : (
                                            "Not selected"
                                        )}
                                    </TableCell>
                                </TableRow>
                            ))}
                        </TableBody>
                    </Table>
                </CardContent>
            </Card>
            <div className="grid items-start gap-6 lg:grid-cols-2">
                {[
                    [
                        "confusion",
                        "Confusion matrix",
                        "Open the full-size figure to inspect individual class labels.",
                    ],
                    [
                        "learning_curves",
                        "Training history",
                        "Saved learning curves from the classification experiment.",
                    ],
                ].map(([suffix, title, desc]) => (
                    <Card key={suffix}>
                        <CardHeader>
                            <CardTitle>{title}</CardTitle>
                            <CardDescription>{desc}</CardDescription>
                        </CardHeader>
                        <CardContent>
                            <a
                                href={`/insights/${task.stem}_${suffix}.png`}
                                target="_blank"
                                rel="noreferrer"
                                aria-label={`Open full-size ${task.title} ${title.toLowerCase()}`}
                            >
                                {/* eslint-disable-next-line @next/next/no-img-element */}
                                <img
                                    src={`/insights/${task.stem}_${suffix}.png`}
                                    alt={`${task.title} ${title.toLowerCase()}`}
                                    className="w-full rounded-lg border bg-white"
                                />
                            </a>
                        </CardContent>
                    </Card>
                ))}
            </div>
            <Card>
                <CardHeader>
                    <CardTitle>Per-class test performance</CardTitle>
                    <CardDescription>
                        Support is the number of evaluated examples. Classes
                        with no support cannot be assessed.
                    </CardDescription>
                    <Input
                        aria-label="Search class labels"
                        placeholder="Search class labels..."
                        value={query}
                        onChange={(e) => setQuery(e.target.value)}
                        className="mt-3 max-w-sm"
                    />
                </CardHeader>
                <CardContent>
                    <div className="max-h-96 overflow-auto">
                        <Table>
                            <TableHeader>
                                <TableRow>
                                    {[
                                        "Class",
                                        "Precision",
                                        "Recall",
                                        "F1",
                                        "Support",
                                    ].map((h) => (
                                        <TableHead key={h}>{h}</TableHead>
                                    ))}
                                </TableRow>
                            </TableHeader>
                            <TableBody>
                                {classes.map((r) => (
                                    <TableRow key={r[""]}>
                                        <TableCell>{r[""]}</TableCell>
                                        <TableCell>
                                            {pct(r.precision)}
                                        </TableCell>
                                        <TableCell>{pct(r.recall)}</TableCell>
                                        <TableCell>
                                            {pct(r["f1-score"])}
                                        </TableCell>
                                        <TableCell>
                                            {Number(r.support).toLocaleString()}
                                        </TableCell>
                                    </TableRow>
                                ))}
                                {!classes.length && (
                                    <TableRow>
                                        <TableCell colSpan={5}>
                                            No matching classes.
                                        </TableCell>
                                    </TableRow>
                                )}
                            </TableBody>
                        </Table>
                    </div>
                </CardContent>
            </Card>
            <Card>
                <CardHeader>
                    <CardTitle>Dataset context & limitations</CardTitle>
                </CardHeader>
                <CardContent className="space-y-4 text-sm leading-7 text-muted-foreground">
                    <div className="flex flex-wrap gap-3">
                        {Object.entries(task.split_sizes).map(
                            ([key, value]) => (
                                <Badge variant="secondary" key={key}>
                                    {key}: {value.toLocaleString()}
                                </Badge>
                            ),
                        )}
                    </div>
                    <p>
                        Season, occasion, and gender are catalogue annotations
                        inferred from product appearance. They are not
                        statements about the wearer. Ambiguous items,
                        backgrounds, lighting changes, and underrepresented
                        classes can affect results.
                    </p>
                    <p>
                        Source: saved experiment summaries, comparison tables,
                        per-class reports, and figures. This dashboard is a
                        snapshot; regenerate it after retraining.
                    </p>
                </CardContent>
            </Card>
        </div>
    )
}
