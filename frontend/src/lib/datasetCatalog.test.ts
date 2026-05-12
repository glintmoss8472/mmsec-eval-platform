// 文件说明：该文件属于前端业务工具，集中实现 datasetCatalog.test 相关逻辑。
import { describe, expect, it } from "vitest";

import { datasetCatalogMap } from "./datasetCatalog";

describe("datasetCatalog", () => {
  it("uses the prepared Flickr30k directory for the Flickr1k test slice", () => {
    const flickr1k = datasetCatalogMap.get("flickr1k");

    expect(flickr1k?.override.root).toBe("data/flickr30k");
    expect(flickr1k?.override.captions_file).toBe("captions_index_single.jsonl");
  });

  it("documents the real generation datasets separately from VLR retrieval datasets", () => {
    const vqa = datasetCatalogMap.get("vqa_v2_coco_val");
    const objectProbe = datasetCatalogMap.get("coco_object_probe_val");
    const caption = datasetCatalogMap.get("coco_caption_object_val");

    expect(vqa?.tier).toBe("generation");
    expect(vqa?.override.kind).toBe("generation_jsonl");
    expect(vqa?.override.task_kind).toBe("vqa");
    expect(objectProbe?.override.task_kind).toBe("vqa");
    expect(caption?.tier).toBe("generation");
    expect(caption?.override.task_kind).toBe("caption");
  });

});
