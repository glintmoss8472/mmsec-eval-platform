from mmsec_eval.datasets.toy_shapes import ToyShapesDataset


def test_toy_dataset_generate():
    ds = ToyShapesDataset(num_samples=8, image_size=64, seed=1)
    items = ds.generate()
    assert len(items) == 8
    assert items[0].image.shape == (64, 64, 3)

