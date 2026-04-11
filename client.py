"""Social Media Auditor Environment — Client for connecting to the environment server."""
from __future__ import annotations

from typing import Any

from openenv_core.env_client import EnvClient

from models import AuditAction, AuditObservation


class SocialMediaAuditorEnv(EnvClient):
    """Client for the Social Media Auditor environment.

    Usage (async):
        async with SocialMediaAuditorEnv(base_url="https://your-space.hf.space") as client:
            result = await client.reset()
            result = await client.step(AuditAction(...))

    Usage (sync):
        with SocialMediaAuditorEnv(base_url="...").sync() as client:
            result = client.reset()
            result = client.step(AuditAction(...))
    """

    action_type = AuditAction
    observation_type = AuditObservation
