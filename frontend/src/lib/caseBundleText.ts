// 文件说明：该文件属于前端业务工具，集中实现 caseBundleText 相关逻辑。
export type CaseStage = "clean" | "adv";

/** 整理 `as record` 前端辅助逻辑，保持数据转换和展示口径一致。 */
function asRecord(value: unknown): Record<string, unknown> {
  return typeof value === "object" && value !== null ? (value as Record<string, unknown>) : {};
}

/** 整理 `stage record` 前端辅助逻辑，保持数据转换和展示口径一致。 */
function stageRecord(root: unknown, stage: CaseStage): Record<string, unknown> {
  return asRecord(asRecord(root)[stage]);
}

/** 生成 `文本 value` 展示值，统一页面标签、颜色和缺省文案。 */
function textValue(value: unknown): string {
  if (value === null || value === undefined) {
    return "";
  }
  return String(value).trim();
}

/** 生成 `extract 攻击 文本` 展示值，统一页面标签、颜色和缺省文案。 */
function extractAttackText(outputText: string): string {
  const marker = "攻击文本：";
  return outputText.includes(marker) ? outputText.split(marker, 2)[1].trim() : "";
}

/** 生成 `strip embedded input 文本` 展示值，统一页面标签、颜色和缺省文案。 */
function stripEmbeddedInputText(outputText: string): string {
  const marker = "，攻击文本：";
  return outputText.includes(marker) ? outputText.split(marker, 2)[0].trim() : outputText;
}

/** 格式化 `format 案例 output 文本`，统一页面展示文本和缺省值。 */
export function formatCaseOutputText(value: unknown): string {
  const raw = value === null || value === undefined || value === "" ? "未记录输出说明" : value;
  return stripEmbeddedInputText(textValue(raw)
    .replace(/clip similarity=/gi, "CLIP 相似度=")
    .replace(/score=/gi, "分数="));
}

/** 生成 `案例 output 文本` 展示值，统一页面标签、颜色和缺省文案。 */
export function caseOutputText(bundle: Record<string, unknown>, stage: CaseStage): string {
  const node = stageRecord(bundle.outputs, stage);
  const text = textValue(node.text);
  const reason = textValue(node.reason);
  const score = node.score === null || node.score === undefined ? "" : node.score;
  return formatCaseOutputText(text || reason || score || "未记录输出说明");
}

/** 生成 `案例 input 文本` 展示值，统一页面标签、颜色和缺省文案。 */
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


/** 判断 `是否 likely english 文本` 状态，支撑页面分支渲染或按钮可用性。 */
export function isLikelyEnglishText(value: unknown): boolean {
  const text = textValue(value);
  if (!text || /[\u4e00-\u9fff]/.test(text)) return false;
  const letters = (text.match(/[A-Za-z]/g) || []).length;
  return letters >= 3;
}

/** 生成 `案例 input label` 展示值，统一页面标签、颜色和缺省文案。 */
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

/** 生成 `案例 output label` 展示值，统一页面标签、颜色和缺省文案。 */
export function caseOutputLabel(stage: CaseStage, taskKind: string, value: unknown): string {
  const kind = String(taskKind || "").toLowerCase();
  if (kind !== "vqa" && kind !== "caption") return "检索分数";
  if (isLikelyEnglishText(value)) return stage === "clean" ? "原始英文输出" : "攻击后英文输出";
  return stage === "clean" ? "原始输出" : "攻击后输出";
}
