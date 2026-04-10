import os

import pytest

from controller.task_run_logger import TaskRunLogger, _OPENPYXL_AVAILABLE


@pytest.mark.skipif(not _OPENPYXL_AVAILABLE, reason="openpyxl not installed")
def test_end_run_recovers_if_workbook_deleted(tmp_path):
    excel_path = tmp_path / "logs" / "task_runs.xlsx"
    logger = TaskRunLogger(excel_path=str(excel_path))

    logger.start_run(
        task_id="task-1",
        task_text="go to C1",
        scenario_name="scene-1",
        initial_snapshot={},
    )

    os.remove(excel_path)

    # Should not raise even if the workbook disappeared mid-run.
    logger.end_run(run_status="completed")

    assert excel_path.exists()
