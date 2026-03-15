# Reminder → Push Notification Pipeline: Executive Summary

## THE PIPELINE (Simplified)

```
User: "Remind me tomorrow 9am"
  ↓
CronAgent calls cron_add() with delivery_mode="announce"
  ↓
CronService stores job in PostgreSQL
  ↓
Timer loop fires when next_run_at_ms is due
  ↓
CronExecutor runs orchestrator → CronDeliveryHandler
  ↓
CronDeliveryHandler calls push_sender callback
  ↓
koi-backend NotificationService.notify()
  ↓
Get user's push tokens from database
  ↓
POST to Expo Push API: https://exp.host/--/api/v2/push/send
  ↓
Device receives push → notification shown to user
```

---

## KEY FILES BY REPO

### OneValet
- **Reminder Creation**: `builtin_agents/cron/agent.py` (CronAgent), `builtin_agents/todo/tools.py` (set_reminder)
- **Scheduling**: `triggers/cron/service.py` (CronService timer loop)
- **Execution**: `triggers/cron/executor.py` (runs orchestrator + delivery)
- **Delivery**: `triggers/cron/delivery.py` (routes to notification channels)
- **Storage**: `triggers/cron/pg_store.py` (PostgreSQL persistence)
- **Models**: `triggers/cron/models.py` (CronJob, Schedule, DeliveryConfig)
- **Initialization**: `app.py` (sets up CronService, notifications)

### koi-backend
- **Push Registration**: `routes/push.py` (POST/DELETE `/api/push/register`)
- **Notification Service**: `services/notification.py` (NotificationService class)
- **Storage**: Database client methods: `get_user_push_tokens()`, `save_push_token()`
- **Expo Integration**: Calls `https://exp.host/--/api/v2/push/send` with list of tokens

### koi-app
- **Push Setup**: `services/push.ts` (setupPushNotifications, registerPushToken)
- **Push Reception**: `hooks/useNotifications.ts` (foreground + tap handlers)
- **Expo Config**: `app.json` (expo-notifications plugin configured)

---

## HOW REMINDERS ARE CREATED

### Via CronAgent (User: "Remind me daily at 8am")
```
cron_add(
  name="Daily reminder",
  instruction="Check your tasks",
  schedule_type="cron",
  schedule_value="0 8 * * *",
  delivery_mode="announce"
)
```

### Via set_reminder() Tool (User: "Remind me tomorrow")
```
set_reminder(
  schedule_datetime="2025-03-15T14:00:00",
  reminder_message="Take medicine"
  # Automatically sets delivery_mode="announce"
)
```

---

## HOW REMINDERS FIRE

1. **CronService starts**: loads jobs from PostgreSQL into memory
2. **Timer loop** (every 0.1-60s): 
   - Checks `job.state.next_run_at_ms <= current_time`
   - Marks job as running
   - Calls `CronExecutor.execute(job)`
3. **CronExecutor**:
   - Runs orchestrator with `AgentTurnPayload(message=reminder_message)`
   - Captures result/error
   - Updates job state in PostgreSQL
   - Calls `CronDeliveryHandler.deliver()`
4. **CronDeliveryHandler**:
   - If `job.delivery.mode == "ANNOUNCE"`:
     - Routes to push notification handler
     - Calls: `await push_sender(user_id, title, message, metadata)`
5. **One-shot reminders** (schedule_type="at"):
   - Auto-deleted after successful execution
6. **Recurring reminders**:
   - `next_run_at_ms` recomputed after execution

---

## HOW PUSH NOTIFICATIONS ARE SENT

### OneValet → koi-backend
- CronDeliveryHandler has list of `self._notifications`
- If `callbacks.notify_url` configured in OneValet:
  - Creates `CallbackNotification(callback_url=notify_url)`
  - Posts JSON to that URL with: `{ tenant_id, title, body, metadata }`
- **Problem**: No PushNotification is created by default!

### koi-backend Delivery
```python
NotificationService.notify(user_id, title, body, data):
  1. Get user's notification channel: "push" | "sms" | "both"
  2. If "push":
     a. Query: tokens = db_client.get_user_push_tokens(user_id)
     b. For each token, build message: { to: token, title, body, sound: "default" }
     c. POST to Expo: POST https://exp.host/--/api/v2/push/send
  3. If "sms" (Pro tier only):
     a. Get phone number from user profile
     b. Send via SMS provider
```

### koi-app Reception
```typescript
setupPushNotifications():
  - Get Expo token: await N.getExpoPushTokenAsync()
  - POST to /api/push/register { token, platform }
  - Configure handler: shouldShowAlert, shouldPlaySound, etc.

useNotifications():
  - Foreground: notification arrives → inject into chat UI
  - Tap: user taps notification → navigate to chat screen
  - Background: notification shows in OS notification center
```

---

## CRITICAL GAPS & BROKEN LINKS

### 🔴 MAJOR ISSUES:

1. **PushNotification Not Initialized**
   - OneValet's `PushNotification` class expects a `push_sender` callback
   - **Never created in app.py!**
   - Only `CallbackNotification` is instantiated (requires `callbacks.notify_url` config)
   - **Result**: If no callback URL, push notifications have nowhere to go

2. **Missing Backhaul from koi-backend to OneValet**
   - OneValet calls `callbacks.notify_url` for delivery
   - koi-backend's `/api/push/register` only handles token registration
   - **No endpoint for koi-backend to trigger push sends!**
   - Circular dependency not implemented

3. **No Error Handling or Retries**
   - If Expo push fails, logged but not retried
   - No retry logic, exponential backoff, or dead letter queue
   - If Expo is down, reminders silently lost

4. **Push Token Registration Not Verified**
   - koi-app calls `registerPushToken()` but doesn't validate success
   - If registration fails, app continues with stale token

5. **Background Notification Handling Missing**
   - Only handles foreground (app open) and tap events
   - Background delivery depends on Expo/OS, not app code

### 🟡 MODERATE ISSUES:

6. **Message Payload Not Validated**
   - Expo has message size limits
   - No truncation or validation before sending

7. **Delivery Status Not Tracked**
   - OneValet tracks `last_delivery_status` in CronJob
   - koi-backend has no delivery log table
   - Difficult to debug: "Did the push send?"

8. **User ID Consistency**
   - Must be same across OneValet, koi-backend, koi-app
   - No validation that IDs match

---

## WHAT ACTUALLY WORKS TODAY

✅ Reminders can be created via CronAgent or TodoAgent
✅ Reminders are persisted in OneValet's PostgreSQL
✅ Timer loop successfully fires jobs
✅ Orchestrator runs with reminder message
✅ Push tokens can be registered via koi-app
✅ Expo-notifications configured in koi-app
✅ Foreground notification reception works (if tokens exist)
✅ One-shot reminders auto-delete after firing
✅ Recurring reminders recompute next run time

❌ **But push delivery is broken unless OneValet is explicitly configured with a callback URL**

---

## CONFIGURATION REQUIRED TO MAKE IT WORK

### Option 1: Via Callback URL (Current Design)
```yaml
# OneValet config.yaml
callbacks:
  notify_url: "https://koi-backend.example.com/api/internal/push"
```

Then koi-backend must implement:
```python
@router.post("/api/internal/push")
async def receive_push_from_onevalet(request):
    data = await request.json()
    # data = { tenant_id, title, body, metadata }
    return await notification_service.notify(data.tenant_id, data.title, data.body)
```

### Option 2: Direct Push Sender (Better)
Initialize in OneValet app.py:
```python
from koi_backend.notification import NotificationService

notification_service = NotificationService()
push_notification = PushNotification(push_sender=notification_service.push_sender_callback)
self._trigger_engine._notifications.append(push_notification)
```

---

## SEQUENCE DIAGRAM: Full Flow

```
User              koi-app              koi-backend         OneValet            Expo
 │                   │                      │                 │               │
 │─ "Remind me"─→   │                      │                 │               │
 │                   │─ registerPushToken()─→                 │               │
 │                   │←─ save_push_token()──│                 │               │
 │                   │                      │                 │               │
 │                   │─ setupPushNotifications()              │               │
 │                   │   (get Expo token)   │                 │               │
 │                   │                      │                 │               │
 │                                          │ ← cron_add()   │               │
 │                                          │ (CronAgent)    │               │
 │                                          │                 │               │
 │                                          │ Timer loop fires at scheduled time
 │                                          │  ↓              │               │
 │                                          │ CronExecutor    │               │
 │                                          │  → delivers    │               │
 │                                          │                │─ _send_push() │
 │                                          │ get_tokens()   │  (Expo API)   │
 │                                          │ ←──────────────│               │
 │                                          │                │────────────→ │
 │                                          │                │ (message)    │
 │ ←─ push notification ─────────────────────────────────────────────────── │
 │ (displayed in Notification Center or   │                 │               │
 │  injected into chat UI if app open)    │                 │               │
 │                                        │                 │               │
 └────────────────────────────────────────────────────────────────────────────
```

---

## TROUBLESHOOTING CHECKLIST

If reminders don't send push notifications:

1. ✓ Check OneValet config has `callbacks.notify_url` (or PushNotification initialized)
2. ✓ Verify koi-backend can receive HTTP POST at that URL
3. ✓ Confirm user has registered push token: `GET /api/push/tokens/{user_id}`
4. ✓ Check user's notification channel preference: `GET /api/user/{user_id}/notification_channel`
5. ✓ Verify job exists in OneValet: `SELECT * FROM cron_jobs WHERE user_id = ?`
6. ✓ Check job's delivery config: `delivery.mode = 'ANNOUNCE'`
7. ✓ Verify job fired: `SELECT * FROM cron_run_log WHERE job_id = ? ORDER BY ts DESC LIMIT 1`
8. ✓ Look for delivery status: `job.state.last_delivery_status`
9. ✓ Check Expo API response logs
10. ✓ Verify Expo project ID in app.json matches Expo account

