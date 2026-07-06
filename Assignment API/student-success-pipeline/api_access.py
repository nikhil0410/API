"""Sub-Objective 3: API Access.

Uses Prefect's built-in REST API (via the Python client) to retrieve and display
key application details such as flows, deployments, flow runs and their states.

Prerequisites: the Prefect server must be running and PREFECT_API_URL set, e.g.
    prefect server start
    prefect config set PREFECT_API_URL=http://127.0.0.1:4200/api

Usage:
    python api_access.py
"""
from __future__ import annotations

import asyncio

from prefect.client.orchestration import get_client


async def fetch_application_details() -> dict:
    """Call the built-in Prefect API and gather >= 4 application details."""
    details: dict = {}
    async with get_client() as client:
        # Detail 1: API health / connectivity
        try:
            await client.api_healthcheck()
            details["api_health"] = "healthy"
        except Exception as exc:  # pragma: no cover - depends on server state
            details["api_health"] = f"unavailable ({exc})"

        # Detail 2: server / API version
        try:
            details["api_version"] = await client.api_version()
        except Exception:
            details["api_version"] = "unknown"

        # Detail 3: registered flows
        flows = await client.read_flows()
        details["flows"] = [{"name": f.name, "id": str(f.id)} for f in flows]

        # Detail 4: deployments (incl. schedules)
        deployments = await client.read_deployments()
        details["deployments"] = [
            {
                "name": d.name,
                "id": str(d.id),
                "paused": getattr(d, "paused", None),
                "tags": list(d.tags or []),
            }
            for d in deployments
        ]

        # Detail 5: recent flow runs and their states
        flow_runs = await client.read_flow_runs(limit=5)
        details["recent_flow_runs"] = [
            {
                "name": fr.name,
                "state": fr.state_name,
                "start_time": str(fr.start_time) if fr.start_time else None,
            }
            for fr in flow_runs
        ]

        # Detail 6: work pools (cloud-native execution infrastructure)
        try:
            work_pools = await client.read_work_pools()
            details["work_pools"] = [
                {"name": wp.name, "type": wp.type} for wp in work_pools
            ]
        except Exception:
            details["work_pools"] = []

    return details


def display_details() -> None:
    details = asyncio.run(fetch_application_details())

    print("=" * 60)
    print(" PREFECT APPLICATION DETAILS (via built-in REST API)")
    print("=" * 60)
    print(f"1. API health        : {details['api_health']}")
    print(f"2. API version       : {details['api_version']}")
    print(f"3. Registered flows  : {len(details['flows'])}")
    for f in details["flows"]:
        print(f"     - {f['name']}")
    print(f"4. Deployments       : {len(details['deployments'])}")
    for d in details["deployments"]:
        state = "paused" if d["paused"] else "active"
        print(f"     - {d['name']} [{state}] tags={d['tags']}")
    print(f"5. Recent flow runs  : {len(details['recent_flow_runs'])}")
    for fr in details["recent_flow_runs"]:
        print(f"     - {fr['name']}: {fr['state']} @ {fr['start_time']}")
    print(f"6. Work pools        : {len(details['work_pools'])}")
    for wp in details["work_pools"]:
        print(f"     - {wp['name']} ({wp['type']})")
    print("=" * 60)


if __name__ == "__main__":
    display_details()
