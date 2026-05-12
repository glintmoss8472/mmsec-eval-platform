// 文件说明：该文件属于前端页面，集中实现 ExperimentBundle.test 相关逻辑。
import { describe, expect, it } from "vitest";

import { surrogateSupportedForAttack, victimSelectableForLaunch } from "./ExperimentStudioPage";

describe("ExperimentStudioPage launch guards", () => {
  it("rejects OpenAI-compatible victims as FGSM surrogates", () => {
    expect(surrogateSupportedForAttack("fgsm", "clip_hf")).toBe(true);
    expect(surrogateSupportedForAttack("fgsm", "blip_itm")).toBe(true);
    expect(surrogateSupportedForAttack("fgsm", "vilt_itm")).toBe(true);
    expect(surrogateSupportedForAttack("fgsm", "openai_qwen25_vl")).toBe(false);
  });

  it("uses the attack catalog for surrogate policies", () => {
    expect(surrogateSupportedForAttack("advclip", "openai_qwen25_vl")).toBe(true);
    expect(surrogateSupportedForAttack("tmm", "vilt_itm")).toBe(true);
    expect(surrogateSupportedForAttack("tmm", "openai_qwen25_vl")).toBe(false);
    expect(surrogateSupportedForAttack("advedm_plus", "clip_hf")).toBe(true);
    expect(surrogateSupportedForAttack("advedm_plus", "blip_itm")).toBe(false);
  });

  it("allows launchable victims to enter the live run queue", () => {
    expect(victimSelectableForLaunch("ready")).toBe(true);
    expect(victimSelectableForLaunch("launchable")).toBe(true);
    expect(victimSelectableForLaunch("launch_blocked")).toBe(false);
    expect(victimSelectableForLaunch("missing_assets")).toBe(false);
  });
});
