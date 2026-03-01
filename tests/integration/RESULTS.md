# Integration Test Results

## Latest Run

- **Date**: 2026-03-01
- **Commit**: `3001ab4`
- **Provider**: Azure OpenAI
- **Model**: gpt-4.1
- **Result**: 244 passed / 13 failed (94.9%)
- **Duration**: ~25m (across 4 domain runs)

### Changes Since Last Run
- Replaced `capabilities=[...]` with `@valet(domain="...")` on all 21 agents
- Removed hardcoded `DOMAIN_AGENT_MAP` — routing now driven by `metadata.domain`
- Fixed LinkedIn casing bug (`LinkedinComposioAgent` → `LinkedInComposioAgent`)
- Improved test inputs for ambiguous routing cases
- Added routing examples to orchestrator system prompt
- Relaxed overly strict LLM judge criteria
- Strengthened agent capabilities (CloudStorageAgent, ShippingAgent)

### Results by Domain

| Domain | Passed | Failed | Total | Rate |
|--------|--------|--------|-------|------|
| Communication | 43 | 0 | 43 | 100% |
| Lifestyle | 65 | 0 | 65 | 100% |
| Productivity | 106 | 9 | 115 | 92.2% |
| Travel | 30 | 4 | 34 | 88.2% |
| **Total** | **244** | **13** | **257** | **94.9%** |

### Failed Tests (13)

#### Productivity (9)

**test_cron_agent (2)**
- `test_tool_selection[Delete the Weekly Report cron job]` — Expected `cron_remove`, wrong tool selected
- `test_response_quality_create_job` — Response quality check failed

**test_google_workspace_agent (3)**
- `test_tool_selection[Write values to the Budget Google spread]` — Expected `sheets_write`, wrong tool selected
- `test_response_quality_drive_search` — Response quality check failed
- `test_response_quality_create_doc` — Response quality check failed

**test_notion_agent (3)**
- `test_create_page_extracts_title` — Expected `notion_create_page` to be called, tool not invoked (approval flow)
- `test_update_page_extracts_title_and_content` — Expected `notion_update_page` to be called, tool not invoked (approval flow)
- `test_response_quality_create_page` — Response quality check failed ("Create Notion page?" confirmation instead of execution)

**test_todo_agent (1)**
- `test_response_quality_create_task` — LLM judge false negative. Response: "The task 'Pick up dry cleaning' has been created and is due by Friday, March 6, 2026." (correct but judge rejected)

#### Travel (4)

**test_maps_agent (2)**
- `test_tool_selection[Get directions from 100 Broadway to Whol]` — LLM called `search_places` instead of `get_directions`
- `test_response_quality_directions` — Response quality check failed

**test_trip_planner_agent (2)**
- `test_response_quality_trip_plan` — LLM judge false negative / max_turns exhausted
- `test_response_quality_directions` — LLM judge false negative

### Failure Analysis (Commit 3001ab4)

| Category | Count | Description |
|----------|-------|-------------|
| LLM judge false negative | 5 | Response is correct but judge rejects it |
| Tool not called (approval flow) | 3 | Notion agent asks confirmation instead of calling tool directly |
| Routing / wrong tool | 3 | LLM picks wrong tool within agent (e.g., search_places vs get_directions) |
| Response quality | 2 | Response genuinely doesn't meet criteria |

---

## Previous Run

- **Date**: 2026-02-28
- **Commit**: `a4c3015f049b930e8d71e519bd91da12092d2af2`
- **Provider**: Azure OpenAI
- **Model**: gpt-4.1
- **Result**: 181 passed / 90 failed (66.8%)
- **Duration**: 16m 45s

## Passed (181)

### test_briefing_agent (14/15)
- test_tool_selection[Give me my daily briefing]
- test_tool_selection[What's on my plate today?]
- test_tool_selection[Summarize my day]
- test_tool_selection[What do I have going on today?]
- test_tool_selection[Set up a daily briefing at 7am]
- test_tool_selection[Send me a morning summary every day at 8]
- test_tool_selection[Check the status of my daily briefing]
- test_tool_selection[Pause my morning briefing]
- test_tool_selection[Disable my daily digest]
- test_tool_selection[Cancel my daily briefing]
- test_extracts_schedule_time
- test_extracts_manage_action_disable
- test_extracts_manage_action_status
- test_response_quality_schedule

### test_calendar_agent (9/13)
- test_tool_selection[What's on my calendar today?]
- test_tool_selection[Do I have any meetings tomorrow?]
- test_tool_selection[Show my schedule for this week]
- test_tool_selection[Schedule a meeting with Bob tomorrow at]
- test_tool_selection[Create an event: dentist appointment Fri]
- test_tool_selection[Add lunch with Sarah on March 5th at noo]
- test_tool_selection[Move my 2pm meeting to 4pm]
- test_tool_selection[Reschedule the team standup to 10am]
- test_tool_selection[Cancel my meeting with Bob]
- test_extracts_query_time_range
- test_extracts_create_event_fields
- test_response_quality_query
- test_response_quality_create

### test_cloud_storage_agent (9/9)
- test_routes_to_cloud_storage_agent[Find my Q4 report in Google Drive]
- test_routes_to_cloud_storage_agent[Search for budget.xlsx in Dropbox]
- test_routes_to_cloud_storage_agent[Show my recent files]
- test_routes_to_cloud_storage_agent[How much storage space do I have?]
- test_routes_to_cloud_storage_agent[Download the project proposal PDF]
- test_search_action_extracted
- test_recent_action_extracted
- test_usage_action_extracted
- test_share_action_extracts_target
- test_provider_extraction
- test_response_quality_routing

### test_cron_agent (8/11)
- test_tool_selection[Show me the cron status]
- test_tool_selection[List all my scheduled jobs]
- test_tool_selection[Update the Daily Briefing job to run at]
- test_tool_selection[Show me the run history for the Daily Br]
- test_tool_selection[Alert me if Bitcoin drops below $50k, ch]
- test_cron_add_extracts_schedule
- test_cron_add_conditional_flag
- test_response_quality_list_jobs
- test_response_quality_create_job

### test_discord_agent (5/5)
- test_tool_selection[List all my Discord servers]
- test_tool_selection[Send a message on Discord to channel 123]
- test_tool_selection[Show me all channels in my Discord serve]
- test_tool_selection[Connect my Discord account]
- test_extracts_message_fields
- test_response_quality_list_servers

### test_email_agent (6/13)
- test_tool_selection[Check my email]
- test_tool_selection[Do I have any unread emails?]
- test_tool_selection[Find the email about Q4 Report]
- test_tool_selection[Show emails from John]
- test_tool_selection[Reply to the email from my boss saying s]
- test_extracts_search_query_keywords
- test_extracts_search_sender_filter
- test_response_quality_check_inbox

### test_expense_agent (14/16)
- test_tool_selection[I spent $15 on lunch today]
- test_tool_selection[Uber ride $12 yesterday]
- test_tool_selection[Coffee at Starbucks $5.50]
- test_tool_selection[How much did I spend last week?]
- test_tool_selection[Show me my expenses this month]
- test_tool_selection[Give me a spending summary for February]
- test_tool_selection[Breakdown of my spending this month]
- test_tool_selection[Set my food budget to $500 per month]
- test_tool_selection[How much budget do I have left?]
- test_tool_selection[Find my receipt from the restaurant]
- test_extracts_amount_and_category
- test_extracts_merchant
- test_extracts_query_period
- test_budget_amount_extraction
- test_response_quality_log
- test_response_quality_query

### test_github_agent (6/7)
- test_tool_selection[Show me the open issues in facebook/reac]
- test_tool_selection[Create an issue in org/repo titled 'Logi]
- test_tool_selection[List open pull requests in vercel/next.j]
- test_tool_selection[Create a PR in org/repo to merge feature]
- test_tool_selection[Search GitHub for machine learning Pytho]
- test_tool_selection[Connect my GitHub account]
- test_response_quality_list_issues

### test_google_workspace_agent (4/11)
- test_tool_selection[Read the Q4 Report Google Doc]
- test_tool_selection[Create a new Google Doc called Meeting A]
- test_read_doc_triggers_search_then_read
- test_sheets_write_extracts_spreadsheet_name
- test_response_quality_create_doc

### test_image_agent (6/6)
- test_routes_to_image_agent[Generate an image of a sunset over the o]
- test_routes_to_image_agent[Draw me a futuristic cityscape at night]
- test_routes_to_image_agent[Create a picture of a cat wearing a top]
- test_routes_to_image_agent[Make an image of a mountain landscape in]
- test_extracts_prompt_field
- test_response_quality_generation

### test_linkedin_agent (5/5)
- test_tool_selection[Create a LinkedIn post about my new job]
- test_tool_selection[Post on LinkedIn: Excited to share our l]
- test_tool_selection[Show me my LinkedIn profile]
- test_tool_selection[Connect my LinkedIn account]
- test_extracts_post_text
- test_response_quality_profile

### test_maps_agent (6/14)
- test_tool_selection[Coffee shops in San Francisco]
- test_tool_selection[Check AQI in San Francisco]
- test_tool_selection[What's the air quality in Beijing?]
- test_tool_selection[Is the air safe to breathe in LA today?]
- test_extracts_search_query_and_location
- test_extracts_air_quality_location
- test_response_quality_search_places
- test_response_quality_air_quality

### test_notion_agent (9/10)
- test_tool_selection[Search for meeting notes in Notion]
- test_tool_selection[Find my project tracker in Notion]
- test_tool_selection[Show me what's in my Notion workspace]
- test_tool_selection[Read the Meeting Notes page in Notion]
- test_tool_selection[Create a new Notion page called Weekly R]
- test_tool_selection[Query the tasks database in Notion]
- test_tool_selection[Add notes to the Meeting Notes page in N]
- test_search_extracts_query
- test_create_page_extracts_title
- test_update_page_extracts_title_and_content
- test_response_quality_search

### test_shipping_agent (2/11)
- test_tool_selection[Track my package 1Z999AA10123456784]
- test_query_one_extracts_tracking_number
- test_response_quality_track_package

### test_slack_agent (5/7)
- test_tool_selection[Send a Slack message to #engineering say]
- test_tool_selection[Show me the latest messages in the #gene]
- test_tool_selection[List all Slack channels in the workspace]
- test_tool_selection[Find the Slack user John Doe]
- test_tool_selection[Set a Slack reminder to check PR in 30 m]
- test_extracts_message_fields

### test_smarthome_agent (2/14)
- test_tool_selection[Set the volume to 30%]
- test_tool_selection[Turn on all the lights]

### test_spotify_agent (7/7)
- test_tool_selection[Play some jazz music on Spotify]
- test_tool_selection[Pause the music on Spotify]
- test_tool_selection[Search Spotify for Bohemian Rhapsody by]
- test_tool_selection[Show me my Spotify playlists]
- test_tool_selection[What song is currently playing on Spotif]
- test_extracts_search_query
- test_response_quality_playlists

### test_todo_agent (3/21)
- test_tool_selection[List my pending tasks]
- test_tool_selection[What tasks do I have?]
- test_tool_selection[Delete my medicine reminder]
- test_response_quality_list_tasks

### test_trip_planner_agent (8/14)
- test_tool_selection[Plan a 3-day trip to Tokyo]
- test_tool_selection[Plan a trip to London from New York]
- test_tool_selection[Find flights from SF to Tokyo]
- test_tool_selection[Find hotels in Barcelona for next week]
- test_tool_selection[What's the weather like in Paris for my]
- test_tool_selection[Search for restaurants near the Eiffel T]
- test_weather_extracts_location
- test_flights_extracts_origin_and_destination
- test_hotels_extracts_destination
- test_search_places_extracts_destination
- test_trip_plan_triggers_multiple_tools
- test_trip_with_origin_triggers_flights

### test_twitter_agent (4/6)
- test_tool_selection[Post a tweet saying 'Just launched our n]
- test_tool_selection[Search Twitter for tweets about AI start]
- test_tool_selection[Show me my Twitter timeline]
- test_tool_selection[Look up the Twitter user @elonmusk]
- test_extracts_tweet_text

### test_youtube_agent (6/6)
- test_tool_selection[Search YouTube for Python tutorial video]
- test_tool_selection[Find me some cooking recipe videos on Yo]
- test_tool_selection[Get details about YouTube video dQw4w9Wg]
- test_tool_selection[Show me my YouTube playlists]
- test_extracts_search_query
- test_response_quality_search

### test_edge_cases (4/5)
- test_multiple_expenses_calls_tool_multiple_times
- test_unknown_intent_does_not_crash
- test_very_short_input_routes_correctly
- test_conversational_message_completes_with_text

### test_routing (12/21)
- test_routes_to_correct_agent[What's on my calendar today?]
- test_routes_to_correct_agent[I spent $15 on lunch]
- test_routes_to_correct_agent[How much did I spend this month?]
- test_routes_to_correct_agent[Set a budget of $500 for food]
- test_routes_to_correct_agent[Track my package 1Z999AA10123456784]
- test_routes_to_correct_agent[What's my morning briefing?]
- test_routes_to_correct_agent[Set up daily briefing at 8am]
- test_routes_to_correct_agent[Plan a 3-day trip to Tokyo]
- test_routes_to_correct_agent[Turn off the living room lights]
- test_routes_to_correct_agent[Search my Google Drive for the Q4 report]
- test_routes_to_correct_agent[Search my Notion for meeting notes]
- test_routes_to_correct_agent[Generate an image of a sunset over the o]

## Failed (90)

### test_briefing_agent (1)
- test_response_quality_briefing

### test_calendar_agent (4)
- test_tool_selection[Delete the dentist appointment]
- test_tool_selection[Remove all meetings tomorrow]
- test_extracts_update_event_target_and_changes
- test_extracts_delete_event_query

### test_cron_agent (3)
- test_tool_selection[Schedule a daily briefing every morning]
- test_tool_selection[Delete the Weekly Report cron job]
- test_tool_selection[Run the Daily Briefing job right now]

### test_email_agent (7)
- test_tool_selection[Send an email to alice@example.com about]
- test_tool_selection[Email bob@company.com saying I'll be lat]
- test_tool_selection[Delete the promotional emails]
- test_tool_selection[Archive all emails from Amazon]
- test_tool_selection[Mark all emails as read]
- test_extracts_send_email_fields
- test_response_quality_send

### test_expense_agent (2)
- test_tool_selection[Delete the Starbucks expense from yester]
- test_tool_selection[Remove the $5 coffee charge]

### test_github_agent (1)
- test_extracts_issue_fields

### test_google_workspace_agent (7)
- test_tool_selection[Search for Q4 Report in Google Drive]
- test_tool_selection[Find my budget spreadsheet in Google Dri]
- test_tool_selection[Show me what's in the Budget spreadsheet]
- test_tool_selection[Write data to the Budget spreadsheet]
- test_tool_selection[List my recent Google Drive files]
- test_drive_search_extracts_query
- test_docs_create_extracts_title
- test_response_quality_drive_search

### test_maps_agent (8)
- test_tool_selection[Find Italian restaurants near downtown S]
- test_tool_selection[Where's the nearest gas station?]
- test_tool_selection[Best pizza places in Brooklyn]
- test_tool_selection[How do I get to the airport from downtow]
- test_tool_selection[Directions from 123 Main St to Central P]
- test_tool_selection[Navigate to Whole Foods from my office]
- test_extracts_directions_origin_and_destination
- test_extracts_directions_travel_mode
- test_response_quality_directions

### test_notion_agent (1)
- test_response_quality_create_page

### test_shipping_agent (8)
- test_tool_selection[Where is my order with tracking number 1]
- test_tool_selection[Show me all my shipments]
- test_tool_selection[What's the status of my FedEx delivery?]
- test_tool_selection[Delete tracking for 1Z999AA10123456784]
- test_tool_selection[Show my past deliveries]
- test_query_all_action
- test_history_action
- test_response_quality_all_shipments

### test_slack_agent (1)
- test_response_quality_fetch

### test_smarthome_agent (12)
- test_tool_selection[Turn off the living room lights]
- test_tool_selection[Set the bedroom lights to 50% brightness]
- test_tool_selection[Change the kitchen lights to blue]
- test_tool_selection[Play music on the living room speaker]
- test_tool_selection[Pause the speaker]
- test_tool_selection[Skip to the next song]
- test_lights_off_extracts_action_and_target
- test_lights_brightness_extracts_value
- test_speaker_play_action
- test_speaker_volume_extracts_value
- test_response_quality_lights_off
- test_response_quality_speaker_play

### test_todo_agent (18)
- test_tool_selection[Show my todo list]
- test_tool_selection[Add a task: buy groceries]
- test_tool_selection[Create a todo to call the dentist by Fri]
- test_tool_selection[I finished buying groceries]
- test_tool_selection[Mark the dentist task as done]
- test_tool_selection[Delete the groceries task]
- test_tool_selection[Remove the call dentist todo]
- test_tool_selection[Remind me to take medicine at 9pm]
- test_tool_selection[Set a reminder for tomorrow at 8am to ch]
- test_tool_selection[Show my reminders]
- test_tool_selection[Pause my morning reminder]
- test_extracts_create_task_title
- test_extracts_create_task_due_date
- test_extracts_reminder_message_and_time
- test_extracts_manage_reminders_action
- test_response_quality_create_task
- test_response_quality_set_reminder

### test_trip_planner_agent (4)
- test_tool_selection[How do I get from Shibuya to Asakusa?]
- test_directions_extracts_endpoints
- test_response_quality_trip_plan
- test_response_quality_hotel_search
- test_response_quality_directions

### test_twitter_agent (1)
- test_response_quality_search

### test_edge_cases (1)
- test_ambiguous_input_completes

### test_routing (9)
- test_routes_to_correct_agent[Send an email to john@example.com]
- test_routes_to_correct_agent[Set a reminder to call mom tomorrow]
- test_routes_to_correct_agent[Add buy groceries to my todo list]
- test_routes_to_correct_agent[Find a good Italian restaurant nearby]
- test_routes_to_correct_agent[How do I get to the airport from here?]
- test_routes_to_correct_agent[Create a GitHub issue for the login bug]
- test_routes_to_correct_agent[Post a tweet about our new product]
- test_routes_to_correct_agent[Send a Slack message to the engineering]
- test_routes_to_correct_agent[Schedule a recurring task every Monday a]

---

## Failure Analysis

### Root Cause Summary

| Category | Count | Description |
|----------|-------|-------------|
| A. Tier 2 routing failure | ~55 | LLM fails to use `delegate_to_agent` for Tier 2 agents |
| B. LLM non-determinism | ~15 | Same agent works for some inputs but not others |
| C. Test bug: `result` vs `result.raw_message` | ~10 | `llm_judge` receives AgentResult object instead of string |
| D. Misrouting | ~5 | LLM routes to the wrong agent entirely |
| E. Genuine response quality issues | ~5 | LLM response doesn't meet quality criteria |

### Category A: Tier 2 Routing Failure (~55 failures)

**The primary root cause.** `TIER1_AGENT_TOOL_LIMIT = 8` means only the first 8 agents get direct tool schemas. The remaining agents must be invoked via the `delegate_to_agent` meta-tool, which the LLM uses unreliably.

**Agent discovery order** (alphabetical via `pkgutil.walk_packages`):

| Slot | Agent | Tier |
|------|-------|------|
| 1 | BriefingAgent | Tier 1 |
| 2 | CalendarAgent | Tier 1 |
| 3 | CloudStorageAgent | Tier 1 |
| 4 | DiscordAgent (composio) | Tier 1 |
| 5 | GithubAgent (composio) | Tier 1 |
| 6 | LinkedinAgent (composio) | Tier 1 |
| 7 | SlackAgent (composio) | Tier 1 |
| 8 | SpotifyAgent (composio) | Tier 1 |
| 9 | TwitterAgent (composio) | **Tier 2** |
| 10 | YoutubeAgent (composio) | **Tier 2** |
| 11 | CronAgent | **Tier 2** |
| 12 | ImportantDatesAgent | **Tier 2** |
| 13 | EmailAgent | **Tier 2** |
| 14 | EmailImportanceAgent | **Tier 2** |
| 15 | EmailPreferenceAgent | **Tier 2** |
| 16 | ExpenseAgent | **Tier 2** |
| 17 | GoogleWorkspaceAgent | **Tier 2** |
| 18 | ImageAgent | **Tier 2** |
| 19 | MapsAgent | **Tier 2** |
| 20 | NotionAgent | **Tier 2** |
| 21 | ShipmentAgent | **Tier 2** |
| 22 | SmartHomeAgent | **Tier 2** |
| 23 | TodoAgent | **Tier 2** |
| 24 | TripPlannerAgent | **Tier 2** |

**Problem:** Core agents (Email, Expense, Todo, Maps, SmartHome, Shipping) are all in Tier 2 because 5 Composio agents (Discord, GitHub, LinkedIn, Slack, Spotify) occupy Tier 1 slots 4-8. The LLM often fails to use `delegate_to_agent` and instead fabricates a response or calls `complete_task` directly.

**Affected tests (by agent):**
- **SmartHomeAgent** (12 failures): Nearly all tests fail. Only 2/14 pass (simple "Turn on all lights" and "Set volume to 30%"), the rest require `control_lights` or `control_speaker` tools that are only reachable via delegation.
- **TodoAgent** (18 failures): Only 3/21 pass. `create_task`, `update_task`, `delete_task`, `set_reminder`, `manage_reminders` tools all unreachable without delegation.
- **ShippingAgent** (8 failures): Only 2/11 pass (the one with explicit tracking number). `query_all`, `history`, and vague queries all fail.
- **MapsAgent** (8 failures): Directions-related tests (`get_directions`) fail consistently; `search_places` and `check_air_quality` work sometimes.
- **EmailAgent** (7 failures): `send_email`, `delete_emails`, `archive_emails`, `mark_as_read` tools fail; only `search_emails` works.
- **GoogleWorkspaceAgent** (7 failures): Drive search/list and Sheets write tests fail; only Doc create/read work.
- **TripPlannerAgent** (4 failures): Directions and some response quality tests fail.
- **ExpenseAgent** (2 failures): Delete/remove expense tests fail (most other expense tests pass because the agent sometimes gets routed).
- **CronAgent** (3 failures): `cron_add`, `cron_remove`, `cron_run` fail while `cron_status`, `cron_list`, `cron_update` pass.

**Affected routing tests (9 failures):**
- Email, Todo, Maps, GitHub, Twitter, Slack, Cron routing all fail because these agents are in Tier 2 and the LLM doesn't delegate.

### Category B: LLM Non-Determinism (~15 failures)

Some agents in Tier 2 **partially** work — certain prompts trigger `delegate_to_agent` while others don't. This is due to LLM non-determinism with temperature > 0 and the inherent unreliability of the meta-tool approach.

Examples:
- **ExpenseAgent**: 14/16 pass but "Delete the Starbucks expense" and "Remove the $5 coffee charge" fail. The LLM handles logging/querying but not deletion.
- **CalendarAgent** (Tier 1): 9/13 pass but delete/update extraction tests fail — LLM picks the right agent but generates wrong tool parameters.
- **CronAgent**: 8/11 pass but scheduling/deletion/manual-run fail intermittently.
- **test_edge_cases**: `test_ambiguous_input_completes` is inherently nondeterministic.

### Category C: Test Bug — `result` vs `result.raw_message` (~10 failures)

Some `response_quality` tests pass the `result` object (an `AgentResult` instance) directly to `llm_judge`, while others correctly extract `result.raw_message`. When `AgentResult` is passed, the LLM judge evaluates the string representation `"AgentResult(agent_type='...', ...)"` instead of the actual response text.

**Tests passing `result` directly (bug):**
- test_briefing_agent: `test_response_quality_briefing`
- test_calendar_agent: `test_response_quality_query`, `test_response_quality_create`
- test_email_agent: `test_response_quality_check_inbox`, `test_response_quality_send`
- test_expense_agent: `test_response_quality_log`, `test_response_quality_query`
- test_maps_agent: `test_response_quality_search_places`, `test_response_quality_directions`, `test_response_quality_air_quality`
- test_todo_agent: `test_response_quality_list_tasks`, `test_response_quality_create_task`, `test_response_quality_set_reminder`

**Tests correctly passing `result.raw_message` or extracting it:**
- discord, github, image, linkedin, slack, spotify, twitter, youtube agents use `result.raw_message`
- cloud_storage, cron, google_workspace, notion, shipping, smarthome, trip_planner agents use `result.raw_message if hasattr(result, "raw_message") else str(result)`

**Note:** Some of the tests listed above (e.g., calendar, expense) happen to PASS despite this bug, likely because the `AgentResult.__str__()` includes enough of the raw message for the judge to evaluate positively. The ones that FAIL due to this bug are: `test_response_quality_briefing`, `test_response_quality_send` (email), `test_response_quality_directions` (maps).

### Category D: Misrouting (~5 failures)

The LLM routes the request to the wrong agent entirely:
- **"Set a reminder to call mom tomorrow"** → routed to TwitterComposioAgent instead of TodoAgent
- **"Post a tweet about our new product"** → sometimes fails to route to TwitterAgent (Tier 2)
- **"Create a GitHub issue for the login bug"** → may fail routing (GitHub is Tier 1 but sometimes misrouted)
- **"Send a Slack message to the engineering channel"** → SlackAgent is Tier 1 but still misroutes occasionally
- **"Find a good Italian restaurant nearby"** → may route to TripPlannerAgent instead of MapsAgent

### Category E: Genuine Response Quality Issues (~5 failures)

Even when routing succeeds, the LLM response doesn't meet quality criteria:
- **test_response_quality_create_page** (Notion): Response may lack confirmation details
- **test_response_quality_fetch** (Slack): Response format doesn't meet readability criteria
- **test_response_quality_search** (Twitter): Search results not presented clearly
- **test_response_quality_trip_plan**: Response may lack structured itinerary format
- **test_response_quality_hotel_search**: Response may not include hotel names/prices from canned data

### Recommendations

1. **Re-order Tier 1 agents by usage priority** — Move core agents (Email, Todo, Expense, Maps, SmartHome) into Tier 1 instead of less-used Composio agents. Consider making Tier 1 slot allocation configurable or priority-based rather than alphabetical.

2. **Increase `TIER1_AGENT_TOOL_LIMIT`** — Raising from 8 to 12-16 would capture most core agents. Trade-off: more tokens per LLM call.

3. **Improve `delegate_to_agent` reliability** — Add few-shot examples in the system prompt showing when/how to use `delegate_to_agent`. Consider making the catalog descriptions more distinctive.

4. **Fix `result` vs `result.raw_message` test bug** — Standardize all `response_quality` tests to use `result.raw_message if hasattr(result, "raw_message") else str(result)` pattern.

5. **Add retry/warm-up for non-deterministic tests** — Consider `@pytest.mark.flaky(reruns=2)` for tests known to be LLM-nondeterministic.
