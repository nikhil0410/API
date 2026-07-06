"""DataOps deployment: schedule the pipeline to run every 2 minutes.

This registers Prefect deployments with a 120-second interval schedule and
serves them. While this process runs, Prefect triggers the flows automatically
every 2 minutes and streams the logs to the dashboard.

Usage:
    1. Start the Prefect server/dashboard in one terminal:
           prefect server start
       (dashboard at http://127.0.0.1:4200)
    2. Point the API at the local server (once per shell):
           prefect config set PREFECT_API_URL=http://127.0.0.1:4200/api
    3. Run this script in another terminal:
           python deploy.py
"""
from __future__ import annotations

from prefect import serve

import config
from flows import data_pipeline_flow, full_pipeline_flow

if __name__ == "__main__":
    interval = config.SCHEDULE_INTERVAL_SECONDS

    data_deployment = data_pipeline_flow.to_deployment(
        name="data-pipeline-every-2-min",
        interval=interval,
        tags=["dataops", "student-success"],
        description="Ingest, preprocess and run EDA every 2 minutes.",
    )
    full_deployment = full_pipeline_flow.to_deployment(
        name="full-pipeline-every-2-min",
        interval=interval,
        tags=["dataops", "mlops", "student-success"],
        description="Run the full data + ML pipeline every 2 minutes.",
    )

    print(f"Serving deployments on a {interval}s ({interval // 60} min) schedule. "
          f"Open the dashboard at http://127.0.0.1:4200 -> Deployments.")
    serve(data_deployment, full_deployment)
