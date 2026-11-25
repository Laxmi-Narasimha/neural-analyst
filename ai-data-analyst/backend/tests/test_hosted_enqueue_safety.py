from uuid import uuid4

import pytest
from fastapi import BackgroundTasks

from app.core.config import DeploymentMode, JobExecutor, settings
from app.workers.dispatcher import enqueue_dataset_processing


def test_hosted_queue_failure_does_not_run_compute_in_api_process(monkeypatch):
    from app.workers.celery_app import process_dataset_task

    old_executor = settings.job_executor
    old_mode = settings.deployment_mode

    def fail_enqueue(*_args, **_kwargs):
        raise ConnectionError("broker unavailable")

    monkeypatch.setattr(process_dataset_task, "delay", fail_enqueue)
    settings.job_executor = JobExecutor.CELERY
    settings.deployment_mode = DeploymentMode.SAAS
    background = BackgroundTasks()
    try:
        with pytest.raises(RuntimeError, match="queue unavailable"):
            enqueue_dataset_processing(
                background_tasks=background,
                dataset_id=uuid4(),
                job_id=uuid4(),
            )
        assert not background.tasks
    finally:
        settings.job_executor = old_executor
        settings.deployment_mode = old_mode
