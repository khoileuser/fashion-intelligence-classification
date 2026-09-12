import type { Prediction } from "@/lib/types";
export type BatchRow = {
  id: string;
  file: File;
  preview: string;
  status: "queued" | "processing" | "done" | "error";
  predictions?: Record<string, Prediction>;
  corrections: Record<string, string>;
  error?: string;
};
export const targets = [
  ["articleType", "Article type"],
  ["season", "Season"],
  ["gender", "Gender"],
  ["usage", "Occasion"],
] as const;
export function csvCell(value: unknown) {
  let text = String(value ?? "");
  // Prevent spreadsheet applications from treating user-controlled filenames as formulas.
  if (/^[\s]*[=+@-]/.test(text) || /^[\t\r\n]/.test(text)) text = "'" + text;
  return `"${text.replaceAll('"', '""')}"`;
}
export function batchCsv(rows: BatchRow[]) {
  const header = [
    "filename",
    ...targets.flatMap(([target]) => [
      `${target}_predicted`,
      `${target}_confidence`,
      `${target}_needs_review`,
      `${target}_corrected`,
      `${target}_final`,
    ]),
  ];
  const lines = rows
    .filter((r) => r.status === "done")
    .map((r) => [
      r.file.name,
      ...targets.flatMap(([target]) => {
        const p = r.predictions![target];
        return [
          p.label,
          p.confidence,
          p.needs_review ?? false,
          r.corrections[target] ?? "",
          r.corrections[target] ?? p.label,
        ];
      }),
    ]);
  return (
    "\uFEFF" +
    [header, ...lines].map((line) => line.map(csvCell).join(",")).join("\r\n")
  );
}
