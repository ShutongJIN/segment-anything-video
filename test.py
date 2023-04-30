from metaseg import SegAutoMaskPredictor, SegManualMaskPredictor


results = SegAutoMaskPredictor().video_predict(
    source="./video/test.mp4",
    model_type="vit_b", # vit_l, vit_h, vit_b
    points_per_side=16, 
    points_per_batch=64,
    min_area=1000,
    output_path="./out/output.mp4",
)
