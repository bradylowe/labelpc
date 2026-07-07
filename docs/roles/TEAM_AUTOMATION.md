# Team Automation Model

## Purpose

This document describes how LabelPC agents should work together when development is coordinated by a recurring Project Manager run.

The goal is to make steady progress without requiring a human to manually prompt every next step. The Project Manager wakes on a schedule, reviews project state, delegates the next safe pieces of work, tracks blockers, and escalates when human judgment is needed.

## Core Idea

Run one scheduled Project Manager job at a predictable cadence, such as hourly during active development. The Project Manager reads the project context and role files, decides what needs attention, and coordinates specialist agents.

The PM loop should be conservative:
- Prefer one or two high-value next actions per run.
- Avoid spawning duplicate work.
- Avoid giving agents vague or unbounded assignments.
- Escalate decisions that affect release strategy, destructive changes, public communication, or large scope changes.

## Suggested Cadence

Start with one hourly PM run.

Other cadences can be added later:
- Every 10 minutes during an active sprint or focused build session.
- Daily for quiet monitoring.
- One-shot runs for release checks, dependency audits, or blocker triage.

Do not start many independent cron jobs until the single PM loop proves useful. The PM should be the coordinator, not another noisy worker.

## Project State Files

The automated team should use durable repository files for coordination.

Suggested future files:
- `docs/roles/PROJECT_MANAGER.md`: PM persona and responsibilities.
- `docs/roles/TEAM_AUTOMATION.md`: this operating model.
- `docs/MVP_ROADMAP.md`: milestone plan and current scope.
- `docs/runbooks/`: repeatable operational procedures.
- `docs/status/PRIORITIES.md`: current ordered priorities.
- `docs/status/RUN_LOG.md`: concise run history and decisions.
- `docs/status/BLOCKERS.md`: active blockers, owner, status, and escalation path.
- `docs/status/AGENT_HANDOFFS.md`: active assignments and handoff notes.

These status files should stay concise. They are coordination tools, not transcripts.

## Hourly PM Loop

Each scheduled PM run should:

1. Read `docs/roles/PROJECT_MANAGER.md`.
2. Read `docs/roles/TEAM_AUTOMATION.md`.
3. Read `docs/MVP_ROADMAP.md`.
4. Read current status files if they exist.
5. Check repository state, open branches, recent commits, tests, and pull requests when tooling is available.
6. Identify the highest-priority next action.
7. Decide whether to:
   - prompt a specialist agent to continue,
   - inspect a blocker,
   - update priorities,
   - create or refine a task,
   - summarize progress,
   - escalate to Brady.
8. Write a short run-log entry with what changed, what was assigned, and what remains blocked.
9. Go back to sleep.

## Specialist Agent Handoffs

Specialist prompts should include:
- Role to adopt.
- Repository path and branch.
- Exact task objective.
- Files or docs to read first.
- Expected output.
- Test or verification command.
- Stop condition.
- Handoff location for results.

Example roles:
- Senior Frontend Developer.
- Senior Backend Developer.
- Rendering / Engine Developer.
- Database Developer.
- Test Developer.
- DevOps / Container Developer.
- Product Owner.

## Blocker Handling

When a specialist reports a blocker, the PM should classify it:

- Missing decision: ask Brady or Product Owner.
- Technical uncertainty: run a spike or assign a senior dev.
- Test failure: assign Test Developer or owning dev.
- Dependency/security issue: assign DevOps/Backend and track severity.
- Scope conflict: update priorities or split the milestone.
- External permission/access issue: escalate to Brady.

The PM may resolve simple blockers itself by updating docs, clarifying scope, or assigning a smaller task. It should not silently make high-impact product or release decisions.

## Escalation Rules

Escalate to Brady in the Telegram topic when:
- A blocker needs human product judgment.
- A destructive repo action is proposed.
- A branch/tag/release action needs approval.
- A security issue affects release readiness.
- A specialist cannot make progress after a clear retry.
- Two plausible architecture paths have meaningful tradeoffs.
- Work is drifting away from the agreed MVP scope.

Escalations should be short and decision-oriented. Include the recommended option when possible.

## Guardrails

- Do not merge to `master` automatically.
- Do not create release tags automatically.
- Do not delete major code paths automatically.
- Do not overwrite human changes.
- Do not spawn unbounded parallel agents.
- Do not keep retrying the same failed task without changing strategy.
- Prefer small branches and reviewable pull requests.
- Keep run logs concise and useful.

## Release Cycle Shape

The PM should manage the rebuild toward a staged release cycle:

1. Freeze historical `master` with a branch or tag, likely `v4.2.6` unless the team chooses another name.
2. Create or use a development/integration branch for rebuild PRs.
3. Land roadmap, team, and architecture docs first.
4. Build MVP 0 as an architecture proof.
5. Build MVP 1 as the minimal annotation loop.
6. Stabilize tests, dependency posture, and documentation.
7. Decide when the development branch is ready to merge into `master`.
8. Begin the major-version release cycle.

## Success Criteria

The automated team model is working when:
- Each PM run leaves a clear, concise trace.
- Agents receive bounded tasks and produce reviewable output.
- Blockers are tracked rather than lost in chat.
- Tests and documentation improve alongside features.
- Brady only gets escalations that need human judgment.
- The repository moves toward MVP 0 without accumulating unmanaged branches or stale work.
