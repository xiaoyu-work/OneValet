# Workflow Engine

Define multi-agent workflows with sequential, parallel, or conditional execution using YAML.

## Quick Start

### Define a Workflow (YAML)

```yaml
# workflows/expense.yaml
name: expense_reimbursement
description: Process expense reimbursements
type: interactive
triggers:
  - "expense"
  - "reimbursement"

stages:
  - name: collect
    run:
      - CollectExpenseAgent

  - name: validate
    run:
      - ValidateExpenseAgent

  - name: notify
    run:
      - NotifyAgent
```

## Workflow Types

| Type | Description |
|------|-------------|
| `interactive` | Triggered by user message |
| `scheduled` | Triggered by cron expression |
| `event_triggered` | Triggered by system events |

```yaml
# Scheduled workflow
name: daily_report
type: scheduled
schedule: "0 9 * * *"  # Every day at 9 AM

stages:
  - name: generate
    run:
      - ReportAgent
```

## Execution Patterns

### Sequential (run)

Agents execute one after another:

```yaml
stages:
  - name: process
    run:
      - AgentA
      - AgentB
      - AgentC
```

### Parallel

Agents execute simultaneously:

```yaml
stages:
  - name: search
    parallel:
      - FlightAgent
      - HotelAgent
      - CarAgent
```

### Then (Aggregator)

Collect results from parallel agents:

```yaml
stages:
  - name: search
    parallel:
      - FlightAgent
      - HotelAgent
    then: CombineResultsAgent
```

## Multi-Stage Workflows

```yaml
name: travel_booking
stages:
  - name: search
    parallel:
      - FlightAgent
      - HotelAgent
    then: CombineAgent

  - name: confirm
    run:
      - ConfirmationAgent

  - name: notify
    run:
      - NotifyAgent
```

## Parameters

### Template Parameters (Defaults)

```yaml
name: greeting_workflow
parameters:
  language: "en"
  formal: false

stages:
  - name: greet
    run:
      - GreetingAgent
```

### User Profile Overrides

Users can override parameters via their profile:

```yaml
# User profile
user_id: alice
workflow_preferences:
  greeting_workflow:
    language: "zh"
    formal: true
```

## Conditional Execution

```yaml
stages:
  - name: check
    run:
      - CheckAmountAgent

  - name: approve
    run:
      - AutoApproveAgent
    condition: "${check.amount} < 100"

  - name: manual_approve
    run:
      - ManagerApproveAgent
    condition: "${check.amount} >= 100"
```

## Variable Resolution

```yaml
stages:
  - name: step1
    run:
      - AgentA

  - name: step2
    run:
      - AgentB
    inputs:
      data: ${step1.output}           # From previous stage
      query: ${query}                  # From workflow inputs
      api_key: ${env:API_KEY}          # Environment variable
```

## Error Handling

```yaml
stages:
  - name: api_call
    run:
      - APIAgent
    retry:
      max_attempts: 3
      delay: 1.0
    on_error: continue  # or: stop
```

## Loading Workflows

Configure workflow directory in `flowagents.yaml`:

```yaml
workflows:
  directory: ./workflows
  auto_load: true
```

Workflows are automatically loaded and matched by triggers.

## Best Practices

1. **Keep stages focused** - One stage = one logical step
2. **Use parallel for independent operations** - Faster execution
3. **Add error handling** - Use retry for unreliable operations
4. **Use parameters** - Make workflows configurable
5. **Name stages clearly** - For readable variable references
