# Reminder → Push Notification Pipeline Documentation

This folder contains a complete analysis of the reminder creation → push notification delivery pipeline across three repositories: OneValet, koi-backend, and koi-app.

## 📚 Documents

### 1. **REMINDER_PIPELINE_ANALYSIS.md** (11 KB)
**Purpose**: Complete technical breakdown and understanding

- **ARCHITECTURE OVERVIEW**: High-level pipeline diagram
- **1. OneValet Reminder Creation & Scheduling**: 
  - Entry points (CronAgent, TodoAgent)
  - Storage model (CronJob, Schedule, DeliveryConfig)
  - Storage backend (PostgreSQL)
- **2. OneValet Scheduling & Triggering**:
  - CronService timer loop mechanics
  - Job execution (CronExecutor)
  - Delivery system (CronDeliveryHandler)
  - PushNotification channel
- **3. OneValet → koi-backend Handoff**: How push tokens flow
- **4. koi-backend Notification Service**:
  - Push token registration and retrieval
  - Expo Push API integration
- **5. koi-app Push Setup**:
  - Token registration flow
  - Notification reception handlers
- **COMPLETE PIPELINE SUMMARY**: Full end-to-end sequence
- **IDENTIFIED GAPS & BROKEN LINKS**: 7 critical issues with evidence
- **DATA FLOW ISSUES**: User ID mapping, delivery tracking, message size
- **RECOMMENDATIONS**: How to fix each gap
- **CONFIGURATION CHECKLIST**: What needs to be set up

**Read this when**: You want to understand the full system architecture and how data flows through it.

---

### 2. **PIPELINE_QUICK_REFERENCE.txt** (25 KB)
**Purpose**: Fast lookup guide with visual organization

- **REMINDER CREATION ENTRY POINTS**: CronAgent vs TodoAgent
- **STORAGE**: Database schema for both OneValet (PostgreSQL) and koi-backend
- **SCHEDULING & FIRING**: How CronService and CronExecutor work
- **PUSH NOTIFICATION DELIVERY**: Three-step delivery process (OneValet → koi-backend → Expo → Device)
- **PUSH TOKEN LIFECYCLE**: 4-phase token management
- **KEY MODELS & TYPES**: Data structures (CronJob, Schedule, DeliveryConfig, etc.)
- **CONFIGURATION**: What needs to be in config.yaml, .env, app.json
- **🔴 CRITICAL GAPS**: 5 major issues preventing push delivery
- **FILES TO EXAMINE**: Organized by repo with line numbers
- **SEQUENCE: HOW TO TEST**: Step-by-step testing procedure
- **DEBUG CHECKLIST**: Troubleshooting steps (10 checks)

**Read this when**: You need to quickly find a file, understand a concept, or debug a specific issue.

---

### 3. **FIXES_NEEDED.md** (15 KB)
**Purpose**: Actionable code fixes organized by priority

- **PRIORITY 1: CRITICAL** (Pipeline broken without these)
  - Fix 1: Initialize PushNotification in OneValet
  - Fix 2: Implement push endpoint in koi-backend
  
- **PRIORITY 2: HIGH** (Error handling & reliability)
  - Fix 3: Add retry logic to Expo push
  - Fix 4: Add delivery tracking to koi-backend
  - Fix 5: Validate push token registration in koi-app
  
- **PRIORITY 3: MEDIUM** (Observability & debugging)
  - Fix 6: Add message size validation
  - Fix 7: Add delivery status endpoint
  - Fix 8: Log reminder fire events
  
- **PRIORITY 4: NICE-TO-HAVE** (Polish)
  - Fix 9: Handle duplicate tokens
  - Fix 10: Add push token pruning
  
- **IMPLEMENTATION ORDER**: Recommended sequence
- **TESTING CHECKLIST**: What to verify after implementing

Each fix includes:
- Problem statement
- File location
- Current code (if applicable)
- Fixed code (copy-paste ready)
- Configuration (if needed)

**Read this when**: You're ready to implement fixes. Use the code snippets directly.

---

## 🎯 Quick Start by Use Case

### "I want to understand how reminders work"
1. Read **REMINDER_PIPELINE_ANALYSIS.md** sections 1-2
2. Look at **PIPELINE_QUICK_REFERENCE.txt** "Entry Points" and "Scheduling"

### "I need to debug why reminders aren't being created"
1. Check **PIPELINE_QUICK_REFERENCE.txt** "Configuration Checklist"
2. Use **PIPELINE_QUICK_REFERENCE.txt** "Files to Examine" to find code
3. Follow **PIPELINE_QUICK_REFERENCE.txt** "Debug Checklist"

### "I need to debug why push notifications aren't arriving"
1. Follow **PIPELINE_QUICK_REFERENCE.txt** "Debug Checklist" (especially items 1-9)
2. Check **REMINDER_PIPELINE_ANALYSIS.md** "Identified Gaps" to understand what might be broken
3. Review **FIXES_NEEDED.md** to see what's missing

### "I need to implement end-to-end push notifications"
1. Read **REMINDER_PIPELINE_ANALYSIS.md** to understand the flow
2. Check **FIXES_NEEDED.md** "Priority 1" (Fix 1-2)
3. Implement those two fixes first
4. Then do Priority 2 and 3 fixes

### "I need to understand the data model"
1. Check **PIPELINE_QUICK_REFERENCE.txt** "Key Models & Types"
2. Read **REMINDER_PIPELINE_ANALYSIS.md** "Reminder Storage Model"

### "I found a bug - how do I fix it?"
1. Find the issue in **REMINDER_PIPELINE_ANALYSIS.md** "Identified Gaps"
2. Look for corresponding fix in **FIXES_NEEDED.md**
3. Copy code and implement
4. Run tests from **PIPELINE_QUICK_REFERENCE.txt** "Testing Checklist"

---

## 🔍 Key Insights

### What Works Well ✅
- OneValet's CronService timer-based scheduling is solid
- PostgreSQL persistence handles job storage well
- Expo Push API integration is straightforward
- koi-app's expo-notifications library is properly configured

### What's Broken 🔴
- **Critical**: PushNotification channel is never initialized in OneValet
- **Critical**: No koi-backend endpoint to receive push delivery requests
- **High**: No retry logic for Expo failures → reminders lost if Expo is down
- **High**: No delivery tracking → can't tell if push was sent

### The Single Point of Failure
The pipeline breaks at this step:
```
CronExecutor → CronDeliveryHandler → Push channel → ???

The PushNotification instance is never created, so delivery stops.
Even if it were created, there's no endpoint in koi-backend to receive it.
```

### To Get Working
1. Initialize PushNotification in OneValet (Fix 1)
2. Add push endpoint to koi-backend (Fix 2)
3. Add retry logic (Fix 3)
4. That's minimum for a working system

---

## 📊 Statistics

- **One-shot reminders**: Schedule type "at" - auto-deleted after firing
- **Recurring reminders**: Schedule types "every" (interval) or "cron" (expression)
- **Scheduling precision**: Millisecond-accurate next_run_at_ms
- **Timer loop cadence**: Checks jobs every 0.1-60 seconds
- **Job timeout**: Default 120 seconds per job
- **Backoff on error**: Exponential: 30s, 60s, 300s, 900s, 3600s
- **Stuck job cleanup**: Auto-cleared if running >2 hours
- **Push API timeout**: 10 seconds per POST to Expo
- **Message size limit**: ~4KB per message (Expo)

---

## 🔗 File Locations

**OneValet**:
- Cron agent: `builtin_agents/cron/agent.py`
- Todo reminder tool: `builtin_agents/todo/tools.py` 
- Scheduling: `triggers/cron/service.py`
- Execution: `triggers/cron/executor.py`
- Delivery: `triggers/cron/delivery.py`
- Models: `triggers/cron/models.py`
- Storage: `triggers/cron/pg_store.py`
- App init: `app.py` (lines ~235-260)

**koi-backend**:
- Push registration: `routes/push.py`
- Notifications: `services/notification.py`
- Database: `storage/database.py`

**koi-app**:
- Push service: `services/push.ts`
- Notifications hook: `hooks/useNotifications.ts`
- Config: `app.json` (lines ~58-68)

---

## 📋 Database Tables

**OneValet (PostgreSQL)**:
- `cron_jobs` - Job definitions
- `cron_run_log` - Run history

**koi-backend**:
- `push_tokens` - User's registered tokens
- `push_delivery_log` - (Missing, need to add - see Fix 4)

---

## 🧪 Testing

Quick test to verify the pipeline works:

```bash
# 1. Ensure device has push token registered
# (Check: db_client.get_user_push_tokens(user_id))

# 2. Create a reminder in 1 minute
# "Remind me in 1 minute to test push"

# 3. Wait 1 minute and check:
# - OneValet logs for "[CRON] Executing job"
# - koi-backend logs for push delivery attempt
# - Device notification appears

# 4. Check database:
SELECT * FROM cron_jobs WHERE created_at > now() - interval '5 min';
SELECT * FROM cron_run_log WHERE job_id = '<job_id>' ORDER BY ts DESC LIMIT 1;
```

---

## 💡 Tips

- **Turn on debug logging** in OneValet to see cron execution details
- **Check push_tokens table** - if empty, device never registered
- **Monitor Expo logs** - sign in at expo.dev to see push delivery status
- **Use structured logging** - parse JSON logs to filter by job_id
- **Test with short reminder times** - don't wait hours to verify
- **Check user's notification_channel** - must be "push" or "both" (not "sms")

---

## 📞 Questions?

- **"Where do reminders get created?"** → See PIPELINE_QUICK_REFERENCE.txt "Reminder Creation Entry Points"
- **"How does scheduling work?"** → See REMINDER_PIPELINE_ANALYSIS.md section 2
- **"Why aren't reminders firing?"** → See PIPELINE_QUICK_REFERENCE.txt "Debug Checklist"
- **"Where's the code that sends push?"** → See PIPELINE_QUICK_REFERENCE.txt "Files to Examine"
- **"What's broken?"** → See REMINDER_PIPELINE_ANALYSIS.md "Identified Gaps"
- **"How do I fix it?"** → See FIXES_NEEDED.md

---

**Last Updated**: March 2025
**Accuracy**: Verified against OneValet commit, koi-backend code, koi-app TypeScript
**Status**: All 7 critical gaps identified and documented with recommended fixes
