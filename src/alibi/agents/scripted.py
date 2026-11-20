"""Scripted fixture policy: a deterministic stand-in for a chat model.

THIS IS NOT A MODEL. It is hand-written control flow that drives the real
DeepAgents graph (delegation, tools, case files, the review interrupt) so the
integration can be exercised offline with zero inference. Its findings are produced
by the same templating code as the BR baseline. Every run that uses it is labeled
"Scripted fixture · no model inference", and its results are never reported as AI
performance or as an AI benchmark arm.
"""

from __future__ import annotations

import json
from typing import Any

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult

from alibi import baselines, checks_families
from alibi.agents.tools import FAMILY_CHECK_TOOLS, finding_id_for
from alibi.contracts import Finding

SCRIPTED_MODEL_ID = "scripted-fixture-policy-v1 (no model)"
FINDINGS_MARKER = "FINDINGS_JSON:"


def _calls(messages: list[BaseMessage]) -> list[str]:
    return [tc["name"] for m in messages if isinstance(m, AIMessage) for tc in m.tool_calls]


def _results(messages: list[BaseMessage], name: str) -> list[Any]:
    out = []
    for m in messages:
        if isinstance(m, ToolMessage) and m.name == name:
            try:
                out.append(json.loads(m.content if isinstance(m.content, str) else json.dumps(m.content)))
            except (ValueError, TypeError):
                out.append({"error": "unparseable tool result"})
    return out


def _ai(role: str, messages: list[BaseMessage], calls: list[tuple[str, dict[str, Any]]]) -> AIMessage:
    n = sum(isinstance(m, AIMessage) for m in messages)
    return AIMessage(content="", tool_calls=[{"name": name, "args": args, "id": f"{role}-{n}-{i}", "type": "tool_call"}
                                             for i, (name, args) in enumerate(calls)])


def _usable(res: Any, key: str) -> bool:
    return isinstance(res, dict) and key in res and "error" not in res


def _family_calls(role: str, messages: list[BaseMessage], bound: tuple[str, ...]) -> AIMessage | None:
    """On a bundle of the remaining case families, run its extra checks once, in one turn."""
    wanted = [t for t in FAMILY_CHECK_TOOLS if t in bound]
    if wanted and wanted[0] not in _calls(messages):
        return _ai(role, messages, [(t, {}) for t in wanted])
    return None


def _family_obs(messages: list[BaseMessage], bound: tuple[str, ...]) -> tuple[dict[str, Any], list[str]]:
    """The family check results as the baseline templating expects them, plus any that could not be read."""
    obs: dict[str, Any] = {}
    problems: list[str] = []
    for tool, key in FAMILY_CHECK_TOOLS.items():
        if tool not in bound:
            continue
        res = (_results(messages, tool) or [None])[0]
        if isinstance(res, dict) and "check" in res and "error" not in res:
            obs[key] = res
        else:
            problems.append(f"{tool} result unavailable")
    return (obs if len(obs) == len([t for t in FAMILY_CHECK_TOOLS if t in bound]) else {}), problems


# --------------------------------------------------------------------------- integrity-auditor


def auditor_policy(messages: list[BaseMessage], bound: tuple[str, ...] = ()) -> AIMessage:
    called = _calls(messages)
    if "check_local_vs_collector" not in called:
        return _ai("integrity-auditor", messages, [("check_local_vs_collector", {})])
    if "check_transfer_sessions" not in called:
        return _ai("integrity-auditor", messages, [("check_transfer_sessions", {})])
    rec = (_results(messages, "check_local_vs_collector") or [{}])[0]
    xfer = (_results(messages, "check_transfer_sessions") or [{}])[0]
    linked = [t for t in (xfer.get("transfers", []) if _usable(xfer, "transfers") else [])
              if t["policy_status"] == "not_approved" and t["session_link"] == "single_overlapping_session"]
    if linked and "check_account_vs_person" not in called:
        return _ai("integrity-auditor", messages,
                   [("check_account_vs_person", {"session_id": linked[0]["sessions"][0]["session_id"]})])
    more = _family_calls("integrity-auditor", messages, bound)
    if more is not None:
        return more
    presence = (_results(messages, "check_account_vs_person") or [None])[0]
    fam, fam_problems = _family_obs(messages, bound)
    candidates: list[Finding] = []
    problems = list(fam_problems)
    if _usable(rec, "files"):
        from alibi.runners import lab_removal_findings  # one finding per removed line, as on the fast path

        candidates += lab_removal_findings(rec, produced_by="integrity-auditor")
    else:
        problems.append("local/collector reconciliation result unavailable")
    if _usable(xfer, "transfers"):
        candidates += baselines.findings_from_transfers(xfer, produced_by="integrity-auditor")
    else:
        problems.append("transfer correlation result unavailable")
    candidates += checks_families.findings(fam, produced_by="integrity-auditor") if fam else []
    attribution = baselines.with_family_attribution(
        baselines.attribution_from(xfer if _usable(xfer, "transfers") else {"transfers": []}, presence, candidates), candidates)
    return AIMessage(content=json.dumps({
        "candidates": [c.model_dump(mode="json") for c in candidates],
        "attribution": attribution.model_dump(mode="json"),
        "problems": problems,
        "coverage_limits": _limits(rec, xfer, fam),
    }))


def _limits(rec: Any, xfer: Any, fam: dict[str, Any] | None = None) -> list[str]:
    """What the checks could not assess on this bundle. Empty when every needed source is present."""
    out = baselines.coverage_limits(rec if _usable(rec, "files") else {"files": []},
                                    xfer if isinstance(xfer, dict) else {})
    return out + [x for x in checks_families.coverage_limits(fam or {}) if x not in out]


_UNKNOWN = " Some checks could not be run on this bundle; what they would have shown is unknown."


# --------------------------------------------------------------------------- skeptic


def skeptic_policy(messages: list[BaseMessage], bound: tuple[str, ...] = ()) -> AIMessage:
    called = _calls(messages)
    if "check_local_vs_collector" not in called:
        return _ai("skeptic", messages, [("check_local_vs_collector", {})])
    if "check_transfer_sessions" not in called:
        return _ai("skeptic", messages, [("check_transfer_sessions", {})])
    more = _family_calls("skeptic", messages, bound)
    if more is not None:
        return more
    fam, _ = _family_obs(messages, bound)
    rec = (_results(messages, "check_local_vs_collector") or [{}])[0]
    xfer = (_results(messages, "check_transfer_sessions") or [{}])[0]
    task = next((m.content for m in messages if isinstance(m, HumanMessage)), "")
    task = task if isinstance(task, str) else json.dumps(task)
    try:
        findings = json.loads(task.split(FINDINGS_MARKER, 1)[1]) if FINDINGS_MARKER in task else []
    except ValueError:
        findings = []
    unexplained = {r["collector_event_id"]: f for f in rec.get("files", []) for r in f["unexplained_local_absence"]} \
        if _usable(rec, "files") else {}
    rows = xfer.get("transfers", []) if _usable(xfer, "transfers") else []
    bad_flows = {t["flow_event_id"]: t for t in rows if t["policy_status"] == "not_approved"}
    bad_staged = {i for t in bad_flows.values() if t.get("staged_object")
                  for i in [t["staged_object"]["fs_stat_event_id"], *t["staged_object"]["export_event_ids"],
                            *t["staged_object"]["retrieval_flow_event_ids"]]}
    verdicts = []
    for f in findings:
        cited = set(f.get("evidence_ids", []))
        claim, cat = f.get("claim_type"), f.get("category")
        if cat != "incident_allegation":
            verdicts.append({"finding_id": f["id"], "verdict": "keep",
                             "reason": "Not an allegation. The benign or open reading stands; nothing here needs a suspect."})
        elif claim == "local_log_removal":
            hit = next((unexplained[c] for c in cited if c in unexplained), None)
            if hit:
                verdicts.append({"finding_id": f["id"], "verdict": "keep", "reason": (
                    f"Tried log rotation: the line postdates the host's retention boundary ({hit['local_coverage_start']}) "
                    "and later local records bracket it, so rotation does not explain it. Tried a collection gap: the "
                    "collector did receive the line. A local agent or filesystem fault cannot be excluded.")})
            else:
                verdicts.append({"finding_id": f["id"], "verdict": "downgrade",
                                 "reason": "No cited record is a collector-retained line unexplainedly absent on the host."})
        elif claim == "policy_violating_transfer":
            hit = next((bad_flows[c] for c in cited if c in bad_flows), None)
            if hit:
                verdicts.append({"finding_id": f["id"], "verdict": "keep", "reason": (
                    "Tried the scheduled backup and lab submissions: this destination is not on the approved list and "
                    "the volume is above the bulk threshold. A flow record does not show content, and the policy list "
                    "could be incomplete.")})
            else:
                verdicts.append({"finding_id": f["id"], "verdict": "downgrade",
                                 "reason": "The cited transfer is not a bulk transfer to a non-approved destination."})
        elif claim == "data_staging":
            if cited & bad_staged:
                verdicts.append({"finding_id": f["id"], "verdict": "keep", "reason": (
                    "Tried routine exports: this object matches the size of the non-approved transfer and was written "
                    "minutes before it. Matching size is not proof of identical content.")})
            else:
                verdicts.append({"finding_id": f["id"], "verdict": "downgrade",
                                 "reason": "The cited object is not linked to a non-approved transfer."})
        elif claim in checks_families.INCIDENT_CLAIMS and fam:
            try:
                probe = Finding.model_validate({k: f.get(k) for k in ("claim_type", "category", "host", "object_or_session",
                                                                      "t_start", "t_end", "evidence_ids")} | {"id": "tmp"})
                verdict, note = checks_families.predicate(probe, cited, fam)
            except Exception:  # noqa: BLE001 - a malformed brief is simply not supported
                verdict, note = "unresolved", "the finding could not be re-checked"
            if verdict == "supported":
                verdicts.append({"finding_id": f["id"], "verdict": "keep", "reason": (
                    "Tried the approved change records, stated maintenance and policy exceptions in the brief: none "
                    f"covers it. The check still reports: {note}. That shows what the records say, not who did it.")[:400]})
            else:
                verdicts.append({"finding_id": f["id"], "verdict": "downgrade", "reason": str(note)[:300]})
        else:
            verdicts.append({"finding_id": f["id"], "verdict": "downgrade", "reason": "No benign test applies; unsupported."})
    return AIMessage(content=json.dumps({
        "verdicts": verdicts,
        "attribution_caution": "Account activity identifies a credential. Do not name or clear a person from it.",
    }))


# --------------------------------------------------------------------------- lead-investigator

_TODOS = ["Read the brief and source inventory", "Delegate contradiction checks to the integrity auditor",
          "Record findings with cited evidence", "Have the skeptic test each finding for a benign explanation",
          "Assess account use vs. human identity", "Submit the report for human review"]


def _todos(done: int, active: int | None, items: list[str] | None = None) -> dict[str, Any]:
    return {"todos": [{"content": t, "status": "completed" if i < done else ("in_progress" if i == active else "pending")}
                      for i, t in enumerate(items or _TODOS)]}


def _finding_args(c: dict[str, Any]) -> dict[str, Any]:
    keys = ("claim_type", "category", "host", "t_start", "t_end", "evidence_ids", "rationale", "object_or_session",
            "confidence", "account_used", "attack_id", "technique_family", "counterevidence_ids", "assumptions",
            "alternative_explanations", "check_refs")
    return {k: c[k] for k in keys if c.get(k) not in (None, [], "")}


def _candidate_id(c: dict[str, Any]) -> str:
    f = Finding.model_validate({**c, "id": "tmp"})
    return finding_id_for(f.claim_type, f.category, f.host, f.object_or_session, f.t_start, f.t_end)


def lead_policy(messages: list[BaseMessage], bound: tuple[str, ...] = ()) -> AIMessage:
    role = "lead-investigator"
    called = _calls(messages)
    if not called:
        return _ai(role, messages, [("write_todos", _todos(0, 0))])
    if "get_case_brief" not in called:
        return _ai(role, messages, [("get_case_brief", {})])
    if "task" not in called:
        return _ai(role, messages, [("task", {"subagent_type": "integrity-auditor", "description": (
            "Run the registered contradiction checks on this evidence bundle: reconcile host logs with collector "
            "copies, correlate bulk transfers with the destination policy and sessions, and compare the linked "
            "session with badge presence. Return candidate findings with event ids as JSON.")})])
    task_results = [m for m in messages if isinstance(m, ToolMessage) and m.name == "task"]
    try:
        audit = json.loads(task_results[0].content) if task_results else {}
    except (ValueError, TypeError):
        audit = {}
    candidates = audit.get("candidates", []) if isinstance(audit, dict) else []
    attribution = audit.get("attribution", {}) if isinstance(audit, dict) else {}
    n_inc = sum(c["category"] == "incident_allegation" for c in candidates)
    if "write_file" not in called:
        lines = ["# Case notes (scripted fixture)", "", f"Auditor returned {len(candidates)} candidate(s); "
                 f"{n_inc} incident allegation(s).", ""]
        lines += [f"- [{c['category']}] {c['claim_type']} on {c['host']} ({c.get('object_or_session')}): "
                  f"{len(c['evidence_ids'])} cited record(s)" for c in candidates]
        lines += ["", "Open questions:", "- Who had access to the credential used in the linked session?"] if n_inc else []
        return _ai(role, messages, [("write_file", {"file_path": "/case/notes.md", "content": "\n".join(lines) + "\n"})])
    if candidates and "submit_finding" not in called:
        return _ai(role, messages, [("submit_finding", _finding_args(c)) for c in candidates])
    if called.count("write_todos") == 1:
        return _ai(role, messages, [("write_todos", _todos(3, 3))])
    if candidates and called.count("task") == 1:
        brief = [{"id": _candidate_id(c), **{k: c.get(k) for k in ("claim_type", "category", "host", "object_or_session",
                                                                      "t_start", "t_end", "evidence_ids")}}
                 for c in candidates]
        return _ai(role, messages, [("task", {"subagent_type": "skeptic", "description": (
            "For each finding below, try to explain it benignly (scheduled log rotation, maintenance, approved "
            "transfers, collection gaps). Return a verdict per finding id: keep or downgrade, with the reason.\n"
            + FINDINGS_MARKER + json.dumps(brief))})])
    if "submit_attribution" not in called:
        batch: list[tuple[str, dict[str, Any]]] = []
        if len(task_results) >= 2:
            try:
                verdicts = json.loads(task_results[1].content).get("verdicts", [])
            except (ValueError, TypeError, AttributeError):
                verdicts = []
            batch += [("record_skeptic_verdict", {"finding_id": v["finding_id"], "verdict": v["verdict"],
                                                  "reason": v["reason"]}) for v in verdicts]
        att = {k: attribution.get(k) for k in ("statement", "account_used", "human_identity", "session_ref",
                                                "supporting_evidence_ids", "contrary_evidence_ids", "limits", "assumptions")
               if attribution.get(k) not in (None, [], "")}
        att.setdefault("statement", "No incident is alleged from this evidence, so there is no attribution to assess.")
        att.setdefault("human_identity", "not_assessed")
        batch.append(("submit_attribution", att))
        return _ai(role, messages, batch)
    if called.count("write_todos") == 2:
        return _ai(role, messages, [("write_todos", _todos(5, 5))])
    if "finalize_report" not in called:
        limits = [str(x) for x in (audit.get("coverage_limits") or [])] if isinstance(audit, dict) else []
        summary = (f"{n_inc} incident allegation(s), {len(candidates) - n_inc} benign explanation(s)/open note(s). "
                   + str(attribution.get("statement", "")) + (_UNKNOWN if limits else ""))
        questions = ["Who had access to the credential used in the linked session?",
                     "Was the destination known to anyone at the clinic?"] if n_inc else []
        questions += [f"Coverage limit: {x}"[:300] for x in limits]
        return _ai(role, messages, [("finalize_report", {"summary": summary, "open_questions": questions[:8]})])
    last = messages[-1]
    approved = isinstance(last, ToolMessage) and "finalized" in str(last.content)
    return AIMessage(content="The reviewer accepted the report; it is frozen for scoring." if approved else
                     "The reviewer did not accept the report. The original proposal is kept unchanged with the review history.")


# --------------------------------------------------------------------------- single agent (fast investigation)

_SINGLE_TODOS = ["Read the brief and source inventory", "Run the contradiction checks",
                 "Record findings with cited evidence", "Assess account use vs. human identity",
                 "Submit the report for human review"]


_LAB_TODOS = ["Read the brief and source inventory", "Compare the local log with the collector's copy",
              "Correlate bulk transfers with the destination policy", "Check staged exports and the linked session",
              "Compare activity with the scheduler's assignments", "Assess account use vs. worker identity",
              "Submit the report for human review"]


def _submitted_types(messages: list[BaseMessage]) -> set[str]:
    return {str((tc.get("args") or {}).get("claim_type")) for m in messages if isinstance(m, AIMessage)
            for tc in m.tool_calls if tc["name"] == "submit_finding"}


def lab_single_policy(messages: list[BaseMessage], bound: tuple[str, ...] = ()) -> AIMessage:
    """Fast investigation of a worker-lab bundle, with a bounded follow-up on the first unexplained export gap.

    Each turn below is one request against the run's allowance: the brief; the local-vs-collector comparison;
    recording what it shows; the transfer correlation; recording the transfers; the staged exports and the
    linked session; the scheduler's assignments; the attribution; the report. Findings are recorded as soon as
    the check that supports them has run, so a run stopped at its allowance keeps what it had established.
    Same templating as the BR baseline; no model is involved.
    """
    from alibi.lab import assignments as lab_assignments

    role = "single-agent"
    called = _calls(messages)
    done = _submitted_types(messages)
    if not called:
        return _ai(role, messages, [("write_todos", _todos(0, 0, _LAB_TODOS)), ("get_case_brief", {})])
    if "check_local_vs_collector" not in called:
        return _ai(role, messages, [("check_local_vs_collector", {})])
    rec = (_results(messages, "check_local_vs_collector") or [{}])[0]
    from alibi.runners import lab_removal_findings

    rec_found = lab_removal_findings(rec, produced_by=role) if _usable(rec, "files") else []
    # Pick a lead from the returned evidence. Routine activity and retention-explained gaps do not trigger it.
    lead = next((row["collector_event_id"] for file in rec.get("files", [])
                 for row in file["unexplained_local_absence"] if row["action"] == "export_bulk"), None)
    follow = lead is not None and "follow_export_output" in bound
    plan = list(_LAB_TODOS)
    if follow:
        plan.insert(2, f"Look up the output path recorded by export {lead}")

    def current_step(original_index: int) -> dict[str, Any]:
        index = plan.index(_LAB_TODOS[original_index])
        return _todos(index, index, plan)

    calls = []
    if rec_found and not done & {f.claim_type for f in rec_found}:
        calls.extend(("submit_finding", _finding_args(f.model_dump(mode="json"))) for f in rec_found)
    if follow and "follow_export_output" not in called:
        calls.extend([("write_todos", _todos(2, 2, plan)), ("follow_export_output", {"event_id": lead})])
    if calls:
        return _ai(role, messages, calls)
    if "check_transfer_sessions" not in called:
        return _ai(role, messages, [("write_todos", current_step(2)), ("check_transfer_sessions", {})])
    xfer = (_results(messages, "check_transfer_sessions") or [{}])[0]
    x_found = baselines.findings_from_transfers(xfer, produced_by=role) if _usable(xfer, "transfers") else []
    moved = [f for f in x_found if f.claim_type != "data_staging"]
    staged = [f for f in x_found if f.claim_type == "data_staging"]
    if moved and not done & {f.claim_type for f in moved}:
        return _ai(role, messages, [("submit_finding", _finding_args(f.model_dump(mode="json"))) for f in moved])
    linked = [t for t in (xfer.get("transfers", []) if _usable(xfer, "transfers") else [])
              if t["policy_status"] == "not_approved" and t["session_link"] == "single_overlapping_session"]
    step: list[tuple[str, dict[str, Any]]] = []
    if linked and "check_account_vs_person" not in called:
        step.append(("check_account_vs_person", {"session_id": linked[0]["sessions"][0]["session_id"]}))
    if staged and "data_staging" not in done:
        step += [("submit_finding", _finding_args(f.model_dump(mode="json"))) for f in staged]
    if step:
        return _ai(role, messages, step)
    if "check_job_assignments" not in called:
        return _ai(role, messages, [("write_todos", current_step(4)), ("check_job_assignments", {})])
    more = _family_calls(role, messages, bound)
    if more is not None:
        return more
    presence = (_results(messages, "check_account_vs_person") or [None])[0]
    assigned = (_results(messages, "check_job_assignments") or [None])[0]
    found = rec_found + x_found
    attribution = baselines.attribution_from(xfer if _usable(xfer, "transfers") else {"transfers": []}, presence, found)
    if isinstance(assigned, dict) and "error" not in assigned:
        try:
            attribution = lab_assignments.worker_link(attribution, assigned)
        except Exception:  # noqa: BLE001 - an unreadable result leaves the account-level statement
            pass
    candidates = [c.model_dump(mode="json") for c in found]
    n_inc = sum(c["category"] == "incident_allegation" for c in candidates)
    follow_results = _results(messages, "follow_export_output")
    follow_note = ""
    if follow_results:
        follow_note = follow_results[0].get("observation", f"The output lookup for export {lead} did not complete.")
    if "submit_attribution" not in called:
        lines = ["# Case notes (scripted fixture, single agent, worker lab)", "",
                 f"The checks produced {len(candidates)} candidate(s); {n_inc} incident allegation(s).", ""]
        lines += [f"- [{c['category']}] {c['claim_type']} on {c['host']} ({c.get('object_or_session')}): "
                  f"{len(c['evidence_ids'])} cited record(s)" for c in candidates]
        if follow_note:
            lines.append(follow_note)
        att = {k: v for k, v in attribution.model_dump(mode="json").items()
               if k in ("statement", "account_used", "human_identity", "session_ref", "supporting_evidence_ids",
                        "contrary_evidence_ids", "limits", "assumptions") and v not in (None, [], "")}
        return _ai(role, messages, [("write_file", {"file_path": "/case/notes.md", "content": "\n".join(lines) + "\n"}),
                                    ("write_todos", current_step(6)), ("submit_attribution", att)])
    if "finalize_report" not in called:
        limits = _limits(rec, xfer, {})
        summary = (f"{n_inc} incident allegation(s), {len(candidates) - n_inc} benign explanation(s)/open note(s). "
                   + attribution.statement + (_UNKNOWN if limits else ""))
        if follow_note:
            summary += " " + follow_note
        questions = ["Who had access to the credential used in the linked session?"] if n_inc else []
        questions += [f"Coverage limit: {x}"[:300] for x in limits]
        return _ai(role, messages, [("finalize_report", {"summary": summary, "open_questions": questions[:8]})])
    last = messages[-1]
    approved = isinstance(last, ToolMessage) and "finalized" in str(last.content)
    return AIMessage(content="The reviewer accepted the report; it is frozen for scoring." if approved else
                     "The reviewer did not accept the report. The original proposal is kept unchanged with the review history.")


def single_policy(messages: list[BaseMessage], bound: tuple[str, ...] = ()) -> AIMessage:
    """One agent, no delegation: runs the checks itself, then reports. Same templating as the BR baseline."""
    if "check_job_assignments" in bound:  # a worker-lab bundle: one step per request (see lab_single_policy)
        return lab_single_policy(messages, bound)
    role = "single-agent"
    called = _calls(messages)
    if not called:
        return _ai(role, messages, [("write_todos", _todos(0, 0, _SINGLE_TODOS))])
    if "get_case_brief" not in called:
        return _ai(role, messages, [("get_case_brief", {})])
    if "check_local_vs_collector" not in called:
        return _ai(role, messages, [("check_local_vs_collector", {}), ("check_transfer_sessions", {})])
    rec = (_results(messages, "check_local_vs_collector") or [{}])[0]
    xfer = (_results(messages, "check_transfer_sessions") or [{}])[0]
    linked = [t for t in (xfer.get("transfers", []) if _usable(xfer, "transfers") else [])
              if t["policy_status"] == "not_approved" and t["session_link"] == "single_overlapping_session"]
    if linked and "check_account_vs_person" not in called:
        return _ai(role, messages, [("check_account_vs_person", {"session_id": linked[0]["sessions"][0]["session_id"]})])
    more = _family_calls(role, messages, bound)
    if more is not None:
        return more
    presence = (_results(messages, "check_account_vs_person") or [None])[0]
    fam, _ = _family_obs(messages, bound)
    found: list[Finding] = []
    if _usable(rec, "files"):
        from alibi.runners import lab_removal_findings

        found += lab_removal_findings(rec, produced_by=role)
    if _usable(xfer, "transfers"):
        found += baselines.findings_from_transfers(xfer, produced_by=role)
    found += checks_families.findings(fam, produced_by=role) if fam else []
    attribution = baselines.with_family_attribution(
        baselines.attribution_from(xfer if _usable(xfer, "transfers") else {"transfers": []}, presence, found), found)
    candidates = [c.model_dump(mode="json") for c in found]
    n_inc = sum(c["category"] == "incident_allegation" for c in candidates)
    if "write_file" not in called:
        lines = ["# Case notes (scripted fixture, single agent)", "",
                 f"The checks produced {len(candidates)} candidate(s); {n_inc} incident allegation(s).", ""]
        lines += [f"- [{c['category']}] {c['claim_type']} on {c['host']} ({c.get('object_or_session')}): "
                  f"{len(c['evidence_ids'])} cited record(s)" for c in candidates]
        return _ai(role, messages, [("write_file", {"file_path": "/case/notes.md", "content": "\n".join(lines) + "\n"})])
    if candidates and "submit_finding" not in called:
        return _ai(role, messages, [("submit_finding", _finding_args(c)) for c in candidates])
    if "submit_attribution" not in called:
        att = {k: v for k, v in attribution.model_dump(mode="json").items()
               if k in ("statement", "account_used", "human_identity", "session_ref", "supporting_evidence_ids",
                        "contrary_evidence_ids", "limits", "assumptions") and v not in (None, [], "")}
        return _ai(role, messages, [("write_todos", _todos(3, 3, _SINGLE_TODOS)), ("submit_attribution", att)])
    if "finalize_report" not in called:
        limits = _limits(rec, xfer, fam)
        summary = (f"{n_inc} incident allegation(s), {len(candidates) - n_inc} benign explanation(s)/open note(s). "
                   + attribution.statement + (_UNKNOWN if limits else ""))
        questions = ["Who had access to the credential used in the linked session?",
                     "Was the destination known to anyone at the clinic?"] if n_inc else []
        questions += [f"Coverage limit: {x}"[:300] for x in limits]
        return _ai(role, messages, [("finalize_report", {"summary": summary, "open_questions": questions[:8]})])
    last = messages[-1]
    approved = isinstance(last, ToolMessage) and "finalized" in str(last.content)
    return AIMessage(content="The reviewer accepted the report; it is frozen for scoring." if approved else
                     "The reviewer did not accept the report. The original proposal is kept unchanged with the review history.")


POLICIES = {"lead-investigator": lead_policy, "integrity-auditor": auditor_policy, "skeptic": skeptic_policy,
            "single-agent": single_policy}


class ScriptedPolicyModel(BaseChatModel):
    """Deterministic policy over the message history. Makes no network call and no inference."""

    role: str
    bound_tools: tuple[str, ...] = ()

    @property
    def _llm_type(self) -> str:
        return "alibi-scripted-fixture"

    def bind_tools(self, tools: Any, **kwargs: Any) -> "ScriptedPolicyModel":
        names = tuple(getattr(t, "name", None) or (t.get("name") if isinstance(t, dict) else getattr(t, "__name__", "?"))
                      for t in tools)
        return self.model_copy(update={"bound_tools": names})

    def _generate(self, messages: list[BaseMessage], stop: list[str] | None = None, run_manager: Any = None,
                  **kwargs: Any) -> ChatResult:
        ai = POLICIES[self.role](messages, self.bound_tools)
        unbound = [tc["name"] for tc in ai.tool_calls if self.bound_tools and tc["name"] not in self.bound_tools]
        if unbound:  # a script bug must fail loudly, not silently skip a step
            raise RuntimeError(f"scripted {self.role} policy called tools that are not bound: {unbound}")
        return ChatResult(generations=[ChatGeneration(message=ai)])
