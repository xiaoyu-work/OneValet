# Proactive Personal Agent — Feature Design Spec

> **Vision:** Koi proactively reaches out to users at the right time, right place, with the right information — without being asked.

## 20 Proactive Scenarios

### Schedule-Driven (F1-F6)
1. **Smart Departure Reminder** — calculate real travel time, alert when time to leave
2. **Meeting Prep Summary** — 10 min before: attendee context + recent emails
3. **Evening Summary** — day recap + tomorrow preview
4. **Weekly Planning** — Sunday: next week overview
5. **Idle Time Suggestions** — 3h gap detected → suggest tasks
6. **Long Inactivity Check** — 24h no interaction → important items summary

### Location-Driven (F7-F10)
7. **Arrive Home/Office** — contextual greeting with relevant info
8. **Store/Shopping Reminder** — geofence at store → show shopping list
9. **Departure Weather Alert** — leaving home → weather warning
10. **Nearby Package Pickup** — near pickup point → show code

### Cross-Dimensional (F11-F20)
11. **Smart Travel Prep** — travel day: flight + weather + hotel + docs
12. **Commute Companion** — regular commute → offer audio email briefing
13. **Lunch Suggestion** — noon + free + location → nearby restaurants
14. **Rainy Schedule Adjust** — weather + outdoor event → suggest moving
15. **Travel Mode** — far from home → local info auto-switch
16. **Flight Delay Cascade** — delay → reschedule downstream events
17. **Subscription Decision** — renewal + usage stats → keep or cancel?
18. **Birthday/Anniversary Prep** — 3 days before → gift/restaurant ideas
19. **Bill Due Reminder** — email bill detected → auto-reminder
20. **Task Rollover** — evening incomplete tasks → move to tomorrow?

## Implementation: Phase 1 (Quick Wins)

All Phase 1 features are new cron job instructions in `_ensure_proactive_jobs()`.
No new infrastructure needed — just agent prompts.

See full spec at: docs/superpowers/specs/2026-04-06-proactive-agent-design.md
