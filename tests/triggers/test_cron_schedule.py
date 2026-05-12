from koa.triggers.cron.models import CronJob, CronScheduleSpec
from koa.triggers.cron.schedule import compute_job_next_run_at_ms


def test_cron_schedule_computes_next_run_with_timezone():
    job = CronJob(
        id="daily-briefing",
        schedule=CronScheduleSpec(expr="0 7 * * *", tz="America/Los_Angeles"),
    )

    next_run = compute_job_next_run_at_ms(job, 1_778_566_800_000)

    assert next_run is not None
    assert next_run > 1_778_566_800_000
