"""Temporary preview harness: deterministic forecasts for visual UI checks only."""
from pathlib import Path
import runpy
import sys
from unittest.mock import patch
import numpy as np

root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root))

def test_forecast(pipeline, history, prediction_length):
    return np.resize(np.asarray(history[-4:], dtype=float), prediction_length) * 0.97

with patch('core.load_chronos_pipeline', return_value=(object(), 'cpu')), patch('core.chronos_point_forecast', side_effect=test_forecast):
    runpy.run_path(str(root / 'app.py'), run_name='__main__')
