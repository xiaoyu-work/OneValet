# Reminder → Push Notification Pipeline: Required Fixes

## PRIORITY 1: CRITICAL (Pipeline Broken Without These)

### Fix 1: Initialize PushNotification in OneValet

**File**: `onevalet/app.py` (in `_initialize()` method)

**Problem**: PushNotification is never instantiated, so push delivery has nowhere to go.

**Current Code** (~line 250):
```python
# CallbackNotification — if callbacks.notify_url configured
callback_url = cfg.get("callbacks", {}).get("notify_url") if isinstance(cfg.get("callbacks"), dict) else None
callback_notification = None
if callback_url:
    callback_notification = CallbackNotification(callback_url=callback_url)
    self._trigger_engine._notifications.append(callback_notification)
    logger.info(f"CallbackNotification configured: {callback_url}")
```

**Fixed Code**:
```python
# CallbackNotification — if callbacks.notify_url configured
callback_url = cfg.get("callbacks", {}).get("notify_url") if isinstance(cfg.get("callbacks"), dict) else None
callback_notification = None
if callback_url:
    callback_notification = CallbackNotification(callback_url=callback_url)
    self._trigger_engine._notifications.append(callback_notification)
    logger.info(f"CallbackNotification configured: {callback_url}")

# PushNotification — if push config provided
push_config = cfg.get("push", {})
if push_config.get("enabled", False):
    try:
        # Option 1: Use koi-backend NotificationService
        if push_config.get("backend_url"):
            from koiai.services.notification import NotificationService
            push_service = NotificationService()
            push_notification = PushNotification(push_sender=push_service.push_sender_callback)
            self._trigger_engine._notifications.append(push_notification)
            logger.info(f"PushNotification configured via koi-backend: {push_config.get('backend_url')}")
        # Option 2: Use direct push sender callable
        elif push_config.get("sender_callable"):
            push_notification = PushNotification(push_sender=push_config["sender_callable"])
            self._trigger_engine._notifications.append(push_notification)
            logger.info("PushNotification configured via custom sender")
    except Exception as e:
        logger.warning(f"Failed to configure PushNotification: {e}")
```

**Config to add to `config.yaml`**:
```yaml
push:
  enabled: true
  backend_url: "http://koi-backend:8000"  # or internal service URL
  # OR use a custom callable if needed
```

---

### Fix 2: Implement Push Endpoint in koi-backend

**File**: `koiai/routes/push.py` (add new endpoint)

**Problem**: OneValet needs somewhere to POST reminders when they fire.

**Add this new endpoint**:
```python
"""
POST /api/internal/push
Receives push notification delivery requests from OneValet.
Dispatches to NotificationService.
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional, Dict, Any

class PushPayload(BaseModel):
    tenant_id: str
    title: str
    body: str
    data: Optional[Dict[str, Any]] = None

@router.post("/api/internal/push")
async def send_push_from_onevalet(payload: PushPayload):
    """
    Receives push notification from OneValet cron system.
    This is the callback endpoint that OneValet posts to.
    """
    try:
        from koiai.services.notification import NotificationService
        service = NotificationService()
        
        success = await service.notify(
            user_id=payload.tenant_id,
            title=payload.title,
            body=payload.body,
            data=payload.data
        )
        
        if not success:
            logger.warning(f"Push notification not sent for user {payload.tenant_id}")
            # Don't fail - might be no tokens registered
        
        return {
            "status": "ok",
            "delivered": success,
            "user_id": payload.tenant_id
        }
    except Exception as e:
        logger.error(f"Failed to process push from OneValet: {e}")
        raise HTTPException(500, "Push delivery failed")
```

**Update OneValet config**:
```yaml
callbacks:
  notify_url: "http://koi-backend:8000/api/internal/push"
```

---

## PRIORITY 2: HIGH (Error Handling & Reliability)

### Fix 3: Add Retry Logic to Expo Push

**File**: `koi-backend/koiai/services/notification.py`

**Problem**: If Expo is down, push notifications are lost forever. No retry logic.

**Replace `_send_push()` method**:
```python
async def _send_push(
    self,
    tokens: List[str],
    title: str,
    body: str,
    data: Optional[Dict[str, Any]] = None
) -> bool:
    """Send push notification via Expo Push API with retry logic."""
    messages = [
        {
            "to": token,
            "title": title,
            "body": body,
            "data": data or {},
            "sound": "default",
        }
        for token in tokens
    ]

    max_retries = 3
    for attempt in range(max_retries):
        try:
            async with httpx.AsyncClient() as client:
                resp = await client.post(
                    EXPO_PUSH_URL,
                    json=messages,
                    timeout=10.0
                )
                
                if resp.status_code == 200:
                    logger.info(f"Push sent to {len(tokens)} device(s) on attempt {attempt + 1}")
                    return True
                elif resp.status_code == 429:  # Rate limit
                    if attempt < max_retries - 1:
                        wait_time = 2 ** attempt  # Exponential backoff
                        logger.warning(f"Expo rate limited, retrying in {wait_time}s")
                        await asyncio.sleep(wait_time)
                        continue
                    else:
                        logger.error(f"Expo rate limited, max retries exceeded")
                        return False
                else:
                    logger.error(f"Expo push failed: {resp.status_code} {resp.text}")
                    if attempt < max_retries - 1:
                        wait_time = 2 ** attempt
                        await asyncio.sleep(wait_time)
                        continue
                    return False
                    
        except httpx.TimeoutError:
            if attempt < max_retries - 1:
                wait_time = 2 ** attempt
                logger.warning(f"Expo timeout, retrying in {wait_time}s")
                await asyncio.sleep(wait_time)
                continue
            else:
                logger.error(f"Expo timeout after {max_retries} attempts")
                return False
        except Exception as e:
            logger.error(f"Push notification error (attempt {attempt + 1}): {e}")
            if attempt < max_retries - 1:
                wait_time = 2 ** attempt
                await asyncio.sleep(wait_time)
                continue
            return False
    
    return False
```

**Add to imports**:
```python
import asyncio
```

---

### Fix 4: Add Delivery Tracking to koi-backend

**File**: `koi-backend/koiai/services/notification.py`

**Problem**: No way to know if push was actually delivered or failed.

**Add new table to database**:
```sql
CREATE TABLE IF NOT EXISTS push_delivery_log (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id VARCHAR(255) NOT NULL,
    token VARCHAR(255) NOT NULL,
    message_title VARCHAR(255),
    status VARCHAR(50),  -- 'sent' | 'failed' | 'rate_limited'
    error_message TEXT,
    sent_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    INDEX idx_user_sent (user_id, sent_at)
);
```

**Add logging to notification service**:
```python
async def _send_push(self, tokens: List[str], title: str, body: str, data: Optional[Dict[str, Any]] = None) -> bool:
    """..."""
    # ... existing code ...
    
    # Log each delivery attempt
    for token in tokens:
        try:
            await db_client.insert_push_delivery_log(
                user_id=user_id,  # Need to pass this in
                token=token,
                message_title=title,
                status="sent" if success else "failed",
                error_message=error_msg if not success else None
            )
        except Exception as e:
            logger.warning(f"Failed to log push delivery: {e}")
    
    return success
```

---

### Fix 5: Validate Push Token Registration in koi-app

**File**: `koi-app/services/push.ts`

**Problem**: No error handling if registration fails.

**Current Code**:
```typescript
const tokenData = await N.getExpoPushTokenAsync();
const token = tokenData.data;
const platform = Platform.OS as 'ios' | 'android';

await registerPushToken(token, platform);  // ← No error handling
_currentPushToken = token;
```

**Fixed Code**:
```typescript
const tokenData = await N.getExpoPushTokenAsync();
const token = tokenData.data;
const platform = Platform.OS as 'ios' | 'android';

try {
    const response = await registerPushToken(token, platform);
    if (!response.ok) {
        const error = await response.json();
        console.error('Push token registration failed:', error);
        return null;  // ← Return null on failure
    }
    _currentPushToken = token;
    console.log('✅ Push token registered successfully');
    return token;
} catch (error) {
    console.error('Push token registration error:', error);
    return null;
}
```

---

## PRIORITY 3: MEDIUM (Observability & User Experience)

### Fix 6: Add Message Size Validation

**File**: `koi-backend/koiai/services/notification.py`

**Problem**: Expo has message size limits; oversized messages fail silently.

**Add validation**:
```python
async def notify(
    self,
    user_id: str,
    title: str,
    body: str,
    data: Optional[Dict[str, Any]] = None
) -> bool:
    """Send notification via user's preferred channel."""
    
    # Validate message size (Expo limit is ~4KB per message)
    max_size = 4000
    if len(body) > max_size:
        logger.warning(f"Message body too large ({len(body)} bytes), truncating")
        body = body[:max_size - 100] + "..."
    
    if title and len(title) > 255:
        title = title[:252] + "..."
    
    # ... rest of method ...
```

---

### Fix 7: Add Delivery Status Endpoint

**File**: `koi-backend/koiai/routes/push.py`

**Problem**: No way to check if reminders were delivered.

**Add endpoint**:
```python
@router.get("/api/push/delivery-status/{job_id}")
async def get_delivery_status(job_id: str, current_user: dict = Depends(get_current_user)):
    """Get delivery status for a reminder."""
    try:
        logs = db_client.query(
            "SELECT * FROM push_delivery_log WHERE job_id = ? ORDER BY sent_at DESC LIMIT 10",
            (job_id,)
        )
        return {
            "job_id": job_id,
            "deliveries": [
                {
                    "token": log["token"][:20] + "...",
                    "status": log["status"],
                    "error": log["error_message"],
                    "sent_at": log["sent_at"].isoformat()
                }
                for log in logs
            ]
        }
    except Exception as e:
        logger.error(f"Failed to get delivery status: {e}")
        raise HTTPException(500, "Failed to get delivery status")
```

---

### Fix 8: Log Reminder Fire Events

**File**: `onevalet/triggers/cron/executor.py`

**Problem**: Difficult to debug when reminders don't fire.

**Add structured logging**:
```python
async def execute(self, job: CronJob) -> CronRunEntry:
    """Execute a cron job, handling concurrency, backoff, and delivery."""
    now = _now_ms()
    
    logger.info(
        f"[CRON] Executing job: {job.id} ({job.name})",
        extra={
            "job_id": job.id,
            "job_name": job.name,
            "user_id": job.user_id,
            "schedule_type": job.schedule.kind,
            "delivery_mode": job.delivery.mode if job.delivery else "none",
        }
    )
    
    # ... rest of method ...
    
    # Log delivery
    logger.info(
        f"[CRON] Delivery complete for {job.id}: {delivery_result.status}",
        extra={
            "job_id": job.id,
            "delivery_status": delivery_result.status,
            "delivered": delivery_result.delivered,
            "error": delivery_result.error,
        }
    )
```

---

## PRIORITY 4: NICE-TO-HAVE (Polish)

### Fix 9: Handle Duplicate Tokens

**File**: `koi-backend/koiai/routes/push.py`

**Problem**: User registers same token multiple times (device reinstall, browser refresh).

**Add deduplication**:
```python
@router.post("/api/push/register")
async def register_push_token(
    request: PushTokenRequest,
    current_user: dict = Depends(get_current_user)
):
    """Register an Expo push token for the current user."""
    user_id = current_user["user_id"]

    if request.platform not in ("ios", "android"):
        raise HTTPException(400, "platform must be 'ios' or 'android'")

    # Check if token already exists
    existing = db_client.get_push_token(user_id, request.token)
    if existing:
        logger.info(f"Push token already registered for user {user_id}")
        return {"status": "ok", "action": "already_registered"}

    result = db_client.save_push_token(user_id, request.token, request.platform)
    if not result:
        raise HTTPException(500, "Failed to register push token")

    logger.info(f"Push token registered for user {user_id} ({request.platform})")
    return {"status": "ok", "action": "registered"}
```

---

### Fix 10: Add Push Token Pruning

**File**: `koi-backend/koiai/services/notification.py`

**Problem**: Old tokens accumulate; Expo returns "InvalidCredentials" for expired tokens.

**Add cleanup routine**:
```python
async def cleanup_invalid_tokens(self, user_id: str, invalid_tokens: List[str]) -> int:
    """Remove tokens that Expo reports as invalid."""
    removed = 0
    for token in invalid_tokens:
        try:
            success = db_client.delete_push_token(user_id, token)
            if success:
                removed += 1
                logger.info(f"Removed invalid token for user {user_id}")
        except Exception as e:
            logger.warning(f"Failed to remove token: {e}")
    return removed
```

**Call after Expo failure**:
```python
if "errors" in resp.json():
    errors = resp.json()["errors"]
    invalid_tokens = [e["details"]["expoPushToken"] for e in errors 
                      if e.get("code") == "INVALID_CREDENTIALS"]
    if invalid_tokens:
        await self.cleanup_invalid_tokens(user_id, invalid_tokens)
```

---

## IMPLEMENTATION ORDER

1. **Fix 1 + Fix 2** (Critical) - Gets push delivery working
2. **Fix 3** (Critical) - Adds reliability
3. **Fix 4** (Critical) - Adds observability
4. **Fix 5** (High) - Prevents silent failures in app
5. **Fix 6 + Fix 7** (Medium) - Better debugging
6. **Fix 8** (Medium) - Better logging
7. **Fix 9 + Fix 10** (Nice) - Polish and edge cases

---

## TESTING CHECKLIST

After implementing fixes:

- [ ] Create a reminder via CronAgent
- [ ] Verify job stored in PostgreSQL: `SELECT * FROM cron_jobs WHERE name LIKE '%reminder%'`
- [ ] Wait for job to fire
- [ ] Check OneValet logs for "[CRON] Executing job" message
- [ ] Check koi-backend logs for push delivery attempt
- [ ] Check push_delivery_log table for delivery record
- [ ] Verify device received notification
- [ ] Test retry logic: temporarily disable Expo endpoint, create reminder, re-enable
- [ ] Test invalid token cleanup: manually insert invalid token, create reminder
- [ ] Check token pruning: verify invalid tokens were removed

