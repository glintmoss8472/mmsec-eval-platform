# 文件说明：该文件属于自动化测试，集中实现 test datasets toy 相关逻辑。
from mmsec_eval.datasets.toy_shapes import ToyShapesDataset


# 中文注释：验证 test_toy_dataset_generate 覆盖的业务场景，防止自动化测试后续改动破坏既有行为。
def test_toy_dataset_generate():
    ds = ToyShapesDataset(num_samples=8, image_size=64, seed=1)
    items = ds.generate()
    assert len(items) == 8
    assert items[0].image.shape == (64, 64, 3)

