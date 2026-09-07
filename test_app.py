"""UI behavior checks. Chronos inference is substituted; GPR and evaluation are real."""
from io import BytesIO
from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
import streamlit as st
from streamlit.testing.v1 import AppTest

APP_DIR = Path(__file__).resolve().parent


def seasonal_test_forecast(pipeline, history, prediction_length):
    return np.resize(np.asarray(history[-4:], dtype=float), prediction_length) * 0.97


class InterfaceTests(unittest.TestCase):
    def setUp(self):
        st.cache_resource.clear()
        self.loader = patch("core.load_chronos_pipeline", return_value=(object(), "cpu"))
        self.forecaster = patch("core.chronos_point_forecast", side_effect=seasonal_test_forecast)
        self.load_mock = self.loader.start()
        self.forecast_mock = self.forecaster.start()
        self.addCleanup(self.loader.stop)
        self.addCleanup(self.forecaster.stop)
        self.addCleanup(st.cache_resource.clear)

    def app(self):
        app = AppTest.from_file(str(APP_DIR / "app.py"), default_timeout=60).run()
        self.assertEqual(len(app.exception), 0)
        return app

    def test_results_persist_and_changed_settings_require_a_new_run(self):
        app = self.app()
        self.assertFalse(app.button[0].disabled)
        self.load_mock.assert_not_called()
        app.button[0].click().run()
        self.assertEqual(len(app.exception), 0)
        self.assertEqual(len(app.error), 0)
        result = app.session_state["comparison_result"]
        self.assertEqual(len(result["future"]), 4)
        self.assertEqual([tab.label for tab in app.tabs], ["Next quarters", "Historical accuracy"])
        self.assertFalse(result["metrics"].empty)
        np.testing.assert_allclose(result["future"]["corrected_forecast"],
                                   result["future"]["raw_forecast"] + result["future"]["gpr_correction"])
        calls = self.forecast_mock.call_count
        app.run()
        self.assertEqual(self.forecast_mock.call_count, calls)
        self.assertEqual(len(app.tabs), 2)
        app.selectbox(key="horizon").set_value(1).run()
        self.assertTrue(any("previous results" in item.value for item in app.warning))
        self.assertEqual(len(app.session_state["comparison_result"]["future"]), 4)
        app.button[0].click().run()
        self.assertEqual(len(app.exception), 0)
        self.assertEqual(len(app.session_state["comparison_result"]["future"]), 1)
        self.assertEqual(len(app.warning), 0)

    def test_upload_without_a_file_disables_run_and_keeps_guidance(self):
        app = self.app()
        app.radio(key="data_source").set_value("Upload CSV").run()
        self.assertEqual(len(app.exception), 0)
        self.assertTrue(app.button[0].disabled)
        self.assertTrue(any("Add quarterly production data" in item.value for item in app.markdown))

    def test_invalid_and_short_uploads_are_actionable(self):
        demo = pd.read_csv(APP_DIR / "sample_demo_palaysynthetic.csv")
        payloads = [b"year,quarter\n2024,1", b"year,quarter,production_mt\n",
                    b"year,quarter,production_mt\n2024,1,inf",
                    demo.iloc[:20].to_csv(index=False).encode()]
        for payload in payloads:
            with self.subTest(payload=payload[:40]):
                app = self.app()
                upload = BytesIO(payload)
                upload.name = "production.csv"
                with patch("streamlit.file_uploader", return_value=upload):
                    app.radio(key="data_source").set_value("Upload CSV").run()
                self.assertEqual(len(app.exception), 0)
                self.assertTrue(app.button[0].disabled)
                self.assertTrue(len(app.error) or len(app.warning))

    def test_minimum_length_upload_can_forecast_without_enough_evaluation_rows(self):
        app = self.app()
        data = pd.read_csv(APP_DIR / "sample_demo_palaysynthetic.csv").iloc[:37]
        upload = BytesIO(data.to_csv(index=False).encode())
        upload.name = "short-series.csv"
        with patch("streamlit.file_uploader", return_value=upload):
            app.radio(key="data_source").set_value("Upload CSV").run()
            self.assertFalse(app.button[0].disabled)
            self.assertEqual(app.slider(key="backtest_origins").value, 5)
            app.slider(key="min_gpr_rows").set_value(24).run()
            app.button[0].click().run()
        self.assertEqual(len(app.exception), 0)
        self.assertEqual(len(app.error), 0)
        result = app.session_state["comparison_result"]
        self.assertEqual(len(result["future"]), 4)
        self.assertTrue(result["metrics"].empty)
        self.assertTrue(any("not enough past errors" in item.value for item in app.info))

    def test_failed_run_shows_an_error_and_retains_previous_result(self):
        app = self.app()
        app.button[0].click().run()
        signature = app.session_state["comparison_result"]["signature"]
        with patch("core.build_residual_dataset", side_effect=RuntimeError("Test inference failure")):
            app.button[0].click().run()
        self.assertEqual(len(app.exception), 0)
        self.assertEqual(len(app.error), 1)
        self.assertTrue(any(item.value == "Test inference failure" for item in app.code))
        self.assertEqual(app.session_state["comparison_result"]["signature"], signature)
        self.assertEqual(len(app.tabs), 2)


if __name__ == "__main__":
    unittest.main()
