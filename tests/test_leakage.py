"""Leakage audit (static): nothing on the inference path may read tracking data. GT poses appear only in evaluation scoring and the training target."""
import inspect, re
from usmotion4d.models import encoder, correspondence, pose_support, pose_estimator
from usmotion4d.evaluation import pose_eval

BAD = re.compile(r"tforms|\.poses|plane_poses|calib\.delta|\bTg\b|ground.?truth|\bgt\b", re.I)


def test_models_never_touch_tracking():
    for mod in (encoder, correspondence, pose_support):
        assert not BAD.search(inspect.getsource(mod)), mod.__name__
    src = inspect.getsource(pose_estimator)
    assert not re.search(r"tforms|\.poses|plane_poses|tusrec", src)
    assert "calib.delta" not in src  # token_geometry uses only the (static) calibration


def test_prediction_functions_read_frames_only():
    for fn in (pose_eval.encode_scan, pose_eval.predict_offset):
        assert not BAD.search(inspect.getsource(fn)), fn.__name__
    src = inspect.getsource(pose_eval.evaluate_scan)
    pred_part = src.split("Tg = ")[0].split("feats = ")[0]
    assert not BAD.search(pred_part)
    # the predicted trajectory Tp is built from D1 only (predictions), before any GT is used to score it
    assert re.search(r"Tp = calib\.sequential\(D1\)", src) and src.index("calib.sequential(D1)") > src.index("predict_offset(model, feats, 1")
