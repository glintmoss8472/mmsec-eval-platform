import { formatRiskLevel } from "./uiLabels";

export interface RunPresentationLike {
  run_id?: string;
  model_adapter?: string;
  victim_model_adapters?: string[];
  benchmark_tag?: string;
  dataset_name?: string;
  risk_level?: string;
}

export function isDemoRun(run: RunPresentationLike): boolean {
  const dataset = `${String(run.dataset_name || "")} ${String(run.benchmark_tag || "")}`.toLowerCase();
  const modelText = `${String(run.model_adapter || "")} ${(run.victim_model_adapters || []).join(" ")}`.toLowerCase();
  return (
    String(run.run_id || "").includes("demo")
    || modelText.includes("fixture_vlm")
    || modelText.includes("fixture")
    || modelText.split(/\s+/).includes("dummy")
    || dataset.includes("seed_bootstrap")
    || dataset.includes("toy")
    || dataset.includes("demo")
    || dataset.includes("演示")
  );
}

export function riskText(level: string | undefined, emptyText = "暂无风险结论"): string {
  if (!level) return emptyText;
  const raw = String(level).trim().toLowerCase();
  const formatted =
    raw === "critical" || raw === "极高" ? "极高" :
    raw === "high" || raw === "高" ? "高" :
    raw === "medium" || raw === "moderate" || raw === "中" ? "中" :
    raw === "low" || raw === "低" ? "低" :
    raw === "minimal" || raw === "极低" ? "极低" :
    formatRiskLevel(String(level));
  if (formatted === "极高") return "极高风险";
  if (formatted === "高") return "高风险";
  if (formatted === "中") return "中风险";
  if (formatted === "低") return "低风险";
  if (formatted === "极低") return "极低风险";
  return formatted;
}

export function riskTone(level: string | undefined): "red" | "orange" | "green" {
  const formatted = formatRiskLevel(String(level || ""));
  if (formatted.includes("高")) return "red";
  if (formatted.includes("中")) return "orange";
  return "green";
}

export function riskBucket(level: string | undefined): "low" | "medium" | "high" {
  const tone = riskTone(level);
  if (tone === "red") return "high";
  if (tone === "orange") return "medium";
  return "low";
}

export function isHighRisk(level: string | undefined): boolean {
  return riskBucket(level) === "high";
}
