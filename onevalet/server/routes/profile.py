"""Profile extraction routes (internal, service-to-service)."""

import logging

from fastapi import APIRouter, HTTPException, Request

from ..app import require_app, verify_service_key
from ...services.profile_extraction import ProfileExtractionService
from ...providers.email.resolver import AccountResolver
from ...providers.email.factory import EmailProviderFactory

logger = logging.getLogger(__name__)
router = APIRouter()

# Singleton service instance (in-memory job tracking)
_service = ProfileExtractionService()


@router.post("/api/internal/profile/extract")
async def start_profile_extraction(request: Request, tenant_id: str):
    """
    Start profile extraction for a tenant by scanning all linked email accounts.

    Requires X-Service-Key header.
    Returns a job_id for polling status.
    """
    verify_service_key(request)
    app = require_app()

    # Ensure app is fully initialized (sets up credential store + LLM client)
    await app._ensure_initialized()

    # Resolve all email accounts for this tenant
    try:
        accounts = await AccountResolver.resolve_accounts(tenant_id, ["all"])
    except Exception as e:
        logger.error(f"Failed to resolve email accounts for {tenant_id}: {e}")
        raise HTTPException(500, f"Failed to resolve email accounts: {e}")

    if not accounts:
        raise HTTPException(404, "No email accounts found for this tenant")

    # Create email providers
    providers = []
    for acc in accounts:
        provider = EmailProviderFactory.create_provider(acc)
        if provider:
            providers.append(provider)

    if not providers:
        raise HTTPException(500, "Failed to create email providers")

    # Start background extraction
    job_id = _service.start_extraction(
        tenant_id=tenant_id,
        providers=providers,
        llm_client=app._llm_client,
    )

    logger.info(f"Profile extraction started: job={job_id}, tenant={tenant_id}, providers={len(providers)}")
    return {"job_id": job_id, "status": "started"}


@router.get("/api/internal/profile/extract/{job_id}/status")
async def get_extraction_status(request: Request, job_id: str):
    """
    Get the status of a profile extraction job.

    Requires X-Service-Key header.
    When status is "completed", the response includes the extracted profile.
    """
    verify_service_key(request)

    job = _service.get_job_status(job_id)
    if not job:
        raise HTTPException(404, "Job not found")

    result = {
        "job_id": job["job_id"],
        "status": job["status"],
        "progress": job.get("progress", {}),
    }

    if job["status"] == "completed" and job.get("profile"):
        result["profile"] = job["profile"]

    if job["status"] == "failed":
        result["error"] = job.get("error")

    return result
