import fs from "node:fs/promises";
import { FileBlob, SpreadsheetFile, Workbook } from "@oai/artifact-tool";

const base = "C:/Users/Administrator/Documents/ChatGPT/New project/table1_clean_only_20260922";
const csvText = await fs.readFile(`${base}/formal_results_snapshot_20260922.csv`, "utf8");
const records = csvText.trimEnd().split(/\r?\n/).map((line) => {
  const fields = [];
  let value = "";
  let quoted = false;
  for (let i = 0; i < line.length; i++) {
    const ch = line[i];
    if (ch === '"' && quoted && line[i + 1] === '"') { value += '"'; i++; }
    else if (ch === '"') quoted = !quoted;
    else if (ch === "," && !quoted) { fields.push(value); value = ""; }
    else value += ch;
  }
  fields.push(value);
  return fields;
});
const headers = records[0].map((h, i) => i === 0 ? h.replace(/^\uFEFF/, "") : h);
const numCols = new Set(["paired_seed", "ratio", "selection_seed", "training_seed", "image_size", "epochs", "dynamics_epochs", "ACC", "BA", "worst_class_recall"]);
const boolCols = new Set(["deterministic"]);
const rows = records.slice(1).map((r) => Object.fromEntries(headers.map((h, i) => {
  const v = r[i] ?? "";
  if (v === "") return [h, null];
  if (boolCols.has(h)) return [h, v.toLowerCase() === "true"];
  if (numCols.has(h)) return [h, Number(v)];
  return [h, v];
})));
const datasets = ["organsmnist", "organamnist", "pathmnist", "tissuemnist", "bloodmnist"];
const seeds = [0, 1, 2, 3, 4, 5, 6, 2026];
const methods = ["random", "el2n_top", "forgetting", "eva", "facility", "fps", "herding", "graph_a2"];
const completed = rows.filter((r) => r.status === "COMPLETE");
if (rows.length !== 320 || completed.length === 0) throw new Error(`Unexpected snapshot totals: rows=${rows.length}, complete=${completed.length}`);
if (rows.some((r) => r.status === "INCOMPLETE" && (r.ACC !== null || r.BA !== null || r.worst_class_recall !== null))) throw new Error("Incomplete records must not contain result metrics");

const wb = Workbook.create();
const summary = wb.worksheets.add("Summary");
summary.showGridLines = false;
summary.getRange("A1").values = [["Table 1 clean-only experiment status"]];
summary.getRange("A1").format = { font: { name: "Arial", size: 14, bold: true, color: "#17324D" } };
summary.getRange("A2:B4").values = [
  ["Snapshot date", "2026-09-22 (HPC live snapshot; may continue changing)"],
  ["Completed model units", completed.length],
  ["Not completed", rows.length - completed.length],
];
summary.getRange("A6:J6").values = [["Dataset", ...seeds.map(String), "Completed units / 64"]];
const completionRows = datasets.map((ds) => {
  const counts = seeds.map((seed) => rows.filter((r) => r.dataset === ds && r.paired_seed === seed && r.status === "COMPLETE").length);
  return [ds, ...counts, counts.reduce((a, b) => a + b, 0)];
});
summary.getRange("A7:J11").values = completionRows;
summary.getRange("A13:C13").values = [["Algorithm", "Completed", "Expected"]];
summary.getRange("A14:C21").values = methods.map((m) => [m, completed.filter((r) => r.algorithm === m).length, datasets.length * seeds.length]);
summary.getRange("A23:B23").values = [["Protocol", "2% clean selection → clean selected training → clean test; paired selection_seed = training_seed; no MedMNIST-C corruptions."]];
summary.getRange("A24:B29").values = [
  ["Seeds", "0, 1, 2, 3, 4, 5, 6, 2026"],
  ["Methods", methods.join(", ")],
  ["Training", "1000 epochs; dynamics 200 epochs; deterministic=true; image size 224; augmentation=0"],
  ["Metric fields", "ACC, BA, worst-class recall are fractions (Excel displays percentages)."],
  ["Blank metrics", "Incomplete cells intentionally have blank result fields; status is INCOMPLETE."],
  ["Config hash", "Per-run config_hash is included on Runs for audit/provenance."],
];
summary.getRange("A6:J6").format = { fill: "#244A64", font: { name: "Arial", size: 10, bold: true, color: "#FFFFFF" }, horizontalAlignment: "center", verticalAlignment: "center", wrapText: true };
summary.getRange("A13:C13").format = { fill: "#244A64", font: { name: "Arial", size: 10, bold: true, color: "#FFFFFF" }, horizontalAlignment: "center", verticalAlignment: "center" };
summary.getRange("A1:J29").format.font = { name: "Arial", size: 10, color: "#202A33" };
summary.getRange("A1").format.font = { name: "Arial", size: 14, bold: true, color: "#17324D" };
summary.getRange("A6:J6").format.font = { name: "Arial", size: 10, bold: true, color: "#FFFFFF" };
summary.getRange("A13:C13").format.font = { name: "Arial", size: 10, bold: true, color: "#FFFFFF" };
summary.getRange("A7:J11").format.borders = { preset: "insideHorizontal", style: "thin", color: "#D9E1E8" };
summary.getRange("A14:C21").format.borders = { preset: "insideHorizontal", style: "thin", color: "#D9E1E8" };
summary.getRange("B3:B4").format.numberFormat = "#,##0";
summary.getRange("A1:J29").format.verticalAlignment = "center";
summary.getRange("A1:J29").format.wrapText = true;
summary.getRange("A1:A29").format.columnWidth = 22;
summary.getRange("B1:B29").format.columnWidth = 58;
summary.getRange("C1:I29").format.columnWidth = 11;
summary.getRange("J1:J29").format.columnWidth = 20;

const detail = wb.worksheets.add("Runs");
detail.showGridLines = false;
detail.getRangeByIndexes(0, 0, rows.length + 1, headers.length).values = [headers, ...rows.map((r) => headers.map((h) => r[h] ?? null))];
const table = detail.tables.add(`A1:T${rows.length + 1}`, true, "FormalRuns");
table.style = "TableStyleMedium2";
detail.freezePanes.freezeRows(1);
detail.getRange(`A1:T${rows.length + 1}`).format.font = { name: "Arial", size: 10, color: "#202A33" };
detail.getRange("A1:T1").format.font = { name: "Arial", size: 10, bold: true, color: "#FFFFFF" };
detail.getRange(`E2:E${rows.length + 1}`).format.numberFormat = "0.00%";
detail.getRange(`F2:J${rows.length + 1}`).format.numberFormat = "0";
detail.getRange(`Q2:S${rows.length + 1}`).format.numberFormat = "0.00%";
detail.getRange(`A2:T${rows.length + 1}`).format.verticalAlignment = "center";
const widths = { A: 17, B: 12, C: 19, D: 15, E: 11, F: 14, G: 14, H: 12, I: 12, J: 17, K: 16, L: 15, M: 17, N: 14, O: 11, P: 68, Q: 12, R: 12, S: 20, T: 100 };
for (const [col, width] of Object.entries(widths)) detail.getRange(`${col}1:${col}${rows.length + 1}`).format.columnWidth = width;

wb.recalculate();
const inspection = await wb.inspect({ kind: "sheet,table", maxChars: 2500, tableMaxRows: 5, tableMaxCols: 8 });
console.log(inspection.ndjson);
const preview = await wb.render({ sheetName: "Summary", autoCrop: "all", scale: 1, format: "png" });
await fs.writeFile(`${base}/formal_results_summary_preview.png`, new Uint8Array(await preview.arrayBuffer()));
const out = await SpreadsheetFile.exportXlsx(wb);
const xlsxPath = `${base}/formal_results_snapshot_20260922.xlsx`;
await out.save(xlsxPath);
const reopened = await SpreadsheetFile.importXlsx(await FileBlob.load(xlsxPath));
const runCheck = reopened.worksheets.getItem("Runs").getRange("A1:T5").values;
const exportedRows = reopened.worksheets.getItem("Runs").getUsedRange().values;
if (exportedRows.length !== 321 || exportedRows[0].length !== 20 || exportedRows[1][3] !== "COMPLETE") throw new Error("XLSX re-import verification failed");
console.log(JSON.stringify({ exportedRows: exportedRows.length - 1, columns: exportedRows[0].length, sample: runCheck[1].slice(0, 4) }));
