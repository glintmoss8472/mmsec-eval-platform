// 文件说明：该文件属于前端业务工具，集中实现 caseBundleText 相关逻辑。
export type CaseStage = "clean" | "adv";

/** 中文注释：实现 asRecord 的核心流程，支撑前端业务工具中的业务语义和异常边界。 */
function asRecord(value: unknown): Record<string, unknown> {
  return typeof value === "object" && value !== null ? (value as Record<string, unknown>) : {};
}

/** 中文注释：实现 stageRecord 的核心流程，支撑前端业务工具中的业务语义和异常边界。 */
function stageRecord(root: unknown, stage: CaseStage): Record<string, unknown> {
  return asRecord(asRecord(root)[stage]);
}

/** 中文注释：实现 textValue 的核心流程，支撑前端业务工具中的业务语义和异常边界。 */
function textValue(value: unknown): string {
  if (value === null || value === undefined) {
    return "";
  }
  return String(value).trim();
}

/** 中文注释：实现 extractAttackText 的核心流程，支撑前端业务工具中的业务语义和异常边界。 */
function extractAttackText(outputText: string): string {
  const marker = "攻击文本：";
  return outputText.includes(marker) ? outputText.split(marker, 2)[1].trim() : "";
}

/** 中文注释：实现 stripEmbeddedInputText 的核心流程，支撑前端业务工具中的业务语义和异常边界。 */
function stripEmbeddedInputText(outputText: string): string {
  const marker = "，攻击文本：";
  return outputText.includes(marker) ? outputText.split(marker, 2)[0].trim() : outputText;
}

/** 中文注释：实现 formatCaseOutputText 的核心流程，支撑前端业务工具中的业务语义和异常边界。 */
export function formatCaseOutputText(value: unknown): string {
  const raw = value === null || value === undefined || value === "" ? "未记录输出说明" : value;
  return stripEmbeddedInputText(textValue(raw)
    .replace(/clip similarity=/gi, "CLIP 相似度=")
    .replace(/score=/gi, "分数="));
}

/** 中文注释：实现 caseOutputText 的核心流程，支撑前端业务工具中的业务语义和异常边界。 */
export function caseOutputText(bundle: Record<string, unknown>, stage: CaseStage): string {
  const node = stageRecord(bundle.outputs, stage);
  const text = textValue(node.text);
  const reason = textValue(node.reason);
  const score = node.score === null || node.score === undefined ? "" : node.score;
  return formatCaseOutputText(text || reason || score || "未记录输出说明");
}

/** 中文注释：实现 caseInputText 的核心流程，支撑前端业务工具中的业务语义和异常边界。 */
export function caseInputText(bundle: Record<string, unknown>, stage: CaseStage): string {
  const inputs = asRecord(bundle.inputs);
  const direct = textValue(stageRecord(inputs, stage).text);
  if (direct) {
    return direct;
  }
  if (stage === "clean") {
    return textValue(asRecord(bundle.sample).text);
  }
  const outputNode = stageRecord(bundle.outputs, "adv");
  const outputText = textValue(outputNode.text) || textValue(outputNode.reason);
  return textValue(asRecord(bundle.adversarial).text) || extractAttackText(outputText);
}


/** 中文注释：实现 isLikelyEnglishText 的核心流程，支撑前端业务工具中的业务语义和异常边界。 */
export function isLikelyEnglishText(value: unknown): boolean {
  const text = textValue(value);
  if (!text || /[\u4e00-\u9fff]/.test(text)) return false;
  const letters = (text.match(/[A-Za-z]/g) || []).length;
  return letters >= 3;
}

/** 中文注释：实现 caseInputLabel 的核心流程，支撑前端业务工具中的业务语义和异常边界。 */
export function caseInputLabel(stage: CaseStage, taskKind: string, value: unknown): string {
  const kind = String(taskKind || "").toLowerCase();
  const english = isLikelyEnglishText(value);
  if (english && kind === "vqa") return stage === "clean" ? "原始英文问题" : "攻击后英文问题";
  if (english && kind === "caption") return stage === "clean" ? "原始英文描述" : "攻击后英文描述";
  if (english) return stage === "clean" ? "原始英文标注" : "攻击后英文文本";
  if (kind === "vqa") return "问题文本";
  if (kind === "caption") return "描述指令";
  return stage === "clean" ? "原始输入文本" : "攻击后输入文本";
}

/** 中文注释：实现 caseOutputLabel 的核心流程，支撑前端业务工具中的业务语义和异常边界。 */
export function caseOutputLabel(stage: CaseStage, taskKind: string, value: unknown): string {
  const kind = String(taskKind || "").toLowerCase();
  if (kind !== "vqa" && kind !== "caption") return "检索分数";
  if (isLikelyEnglishText(value)) return stage === "clean" ? "原始英文输出" : "攻击后英文输出";
  return stage === "clean" ? "原始输出" : "攻击后输出";
}
