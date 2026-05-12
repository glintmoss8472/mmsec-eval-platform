// 文件说明：该文件属于前端业务工具，集中实现 riskPolicy 相关逻辑。
export interface RiskThresholdRow {
  level: string;
  range: string;
  meaning: string;
}

export interface RiskComponentRow {
  key: string;
  label: string;
  defaultWeight: string;
  meaning: string;
}

export const RISK_THRESHOLDS: RiskThresholdRow[] = [
  { level: "极低", range: "0.00 <= 风险分数 < 0.20", meaning: "当前攻击影响很弱，作为运行记录保留。" },
  { level: "低", range: "0.20 <= 风险分数 < 0.40", meaning: "存在可观测扰动影响，但暂未形成主要风险。" },
  { level: "中", range: "0.40 <= 风险分数 < 0.60", meaning: "任务破坏、输出失稳或低扰动可达性已有明显信号，需要复核证据。" },
  { level: "高", range: "0.60 <= 风险分数 < 0.80", meaning: "攻击已明显破坏图文匹配、问答或描述结果，应优先复核样本。" },
  { level: "极高", range: "0.80 <= 风险分数 <= 1.00", meaning: "攻击强且证据链集中，应作为答辩重点风险案例展示。" },
];

export const RETRIEVAL_RISK_COMPONENTS: RiskComponentRow[] = [
  { key: "task_damage", label: "任务破坏风险", defaultWeight: "0.30", meaning: "攻击越能破坏图文匹配、问答或描述目标，风险越高。" },
  { key: "output_instability", label: "输出失稳风险", defaultWeight: "0.25", meaning: "攻击后召回、排名、答案或描述越不稳定，风险越高。" },
  { key: "semantic_disguise", label: "语义伪装风险", defaultWeight: "0.15", meaning: "输入仍保留语义或外观而攻击仍有效时，风险越高。" },
  { key: "low_perturbation", label: "低扰动可达风险", defaultWeight: "0.15", meaning: "较低扰动即可造成任务破坏时，风险越高。" },
  { key: "tail_case", label: "尾部案例风险", defaultWeight: "0.15", meaning: "最坏样本或历史稳定性信号越突出，风险越高。" },
];

/** 中文注释：实现 riskLevelFromScore 的核心流程，支撑前端业务工具中的业务语义和异常边界。 */
export function riskLevelFromScore(score: number): string {
  if (!Number.isFinite(score)) return "待判定";
  if (score >= 0.8) return "极高";
  if (score >= 0.6) return "高";
  if (score >= 0.4) return "中";
  if (score >= 0.2) return "低";
  return "极低";
}

/** 中文注释：实现 riskTone 的核心流程，支撑前端业务工具中的业务语义和异常边界。 */
export function riskTone(level: string): "red" | "orange" | "green" {
  const text = String(level || "");
  if (text.includes("高")) return "red";
  if (text.includes("中")) return "orange";
  return "green";
}
