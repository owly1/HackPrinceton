"""An investigation's trace, derived from what was recorded while it ran.

Everything here is computed from the case's event log: real tool calls, real delegations, the plan as the
investigator wrote it, and its case-file writes and reads. Nothing is inferred about what a model was
thinking. Summaries are the summaries the tools returned.

`behaviors` states narrow, checkable facts from the order of recorded events: whether a plan was recorded,
whether anything was examined, whether the first plan is earlier in the log than the first examination, what
was delegated, which plan versions gained an item, and whether the case file was written and read back later.
"After" means "later in the recorded order"; it does not claim a cause. A plan and a first examination
recorded from the same model turn are ordered by the log, not by intent.

Where the log cannot say something, the trace does not fill it in silently:

- A delegation with no finish event is "not finished in the log". It is "running" only when the caller says
  the case still is, which the events alone cannot show.
- A finish event naming an agent that has no open span (the runner writes `subagent` once a resume has made it
  forget whose task was pending) closes the oldest delegation still open, and that span carries
  `closed_by: "unnamed finish event"` so the reading is visible. The lead's own finish event is never read
  this way.
- Whether a changed plan item is new or a rewording is decided by a fixed rule on the two texts: their word
  sets share at least half their words, or one normalised text begins with the other. Rewordings are listed
  as `reworded` pairs beside the raw `added` and `removed`, and a plan version counts as a revision only when
  it gained an item that is not a rewording.
"""

from __future__ import annotations

from typing import Any

TRACE_VERSION = "trace/2"
EXAMINATION_TOOLS = ("check_", "search_events", "get_events", "count_events", "follow_export_output")
UNNAMED_FINISH = "unnamed finish event"
NOT_FINISHED = "not finished in the log"
REWORDING_OVERLAP = 0.5


def _text(items: list[dict[str, Any]]) -> list[str]:
    return [str(i.get("text", "")).strip() for i in items if str(i.get("text", "")).strip()]


def _words(text: str) -> list[str]:
    """The text in lowercase with only its letters, digits and spaces kept, as words."""
    return "".join(c for c in text.lower() if c.isalnum() or c.isspace()).split()


def _rewording_overlap(old: str, new: str) -> float | None:
    """The share of words two plan items have in common (Jaccard), or None when the rule does not call one a
    rewording of the other: the share is under a half and neither normalised text begins with the other."""
    a, b = _words(old), _words(new)
    if not a or not b:
        return None  # nothing to compare, and an empty text would begin every other text
    overlap = len(set(a) & set(b)) / len(set(a) | set(b))
    ta, tb = " ".join(a), " ".join(b)
    return overlap if overlap >= REWORDING_OVERLAP or ta.startswith(tb) or tb.startswith(ta) else None


def _plan_changes(before: list[str], now: list[str]) -> dict[str, list[Any]]:
    """What one plan version changed against the version before it: the raw difference, and the same difference
    with rewordings told apart from new items.

    A dropped item is the source of at most one rewording, the closest pair first. So a version that lists more
    different items than the one before always reports at least one of them as new.
    """
    added = [t for t in now if t not in before]
    removed = [t for t in before if t not in now]
    scored = [(-overlap, a, r) for a, new in enumerate(added) for r, old in enumerate(removed)
              if (overlap := _rewording_overlap(old, new)) is not None]
    source: dict[int, int] = {}  # position in `added` -> position in `removed` of the item it rewords
    for _, a, r in sorted(scored):
        if a not in source and r not in source.values():
            source[a] = r
    return {"added": added, "removed": removed,
            "added_new": [t for a, t in enumerate(added) if a not in source],
            "reworded": [{"from": removed[source[a]], "to": t} for a, t in enumerate(added) if a in source],
            "removed_outright": [t for r, t in enumerate(removed) if r not in source.values()]}


def investigation_trace(events: list[dict[str, Any]], lead_roles: tuple[str, ...] = ("lead-investigator", "single-agent"),
                        still_running: bool = False) -> dict[str, Any]:
    """The trace of one case's recorded events. `still_running` is what the caller knows about the case and the
    log does not: only when it is true is a delegation without a finish event reported as "running"."""
    spans: list[dict[str, Any]] = []
    plans: list[dict[str, Any]] = []
    revisions: list[dict[str, Any]] = []
    files: list[dict[str, Any]] = []
    delegations: list[dict[str, Any]] = []
    open_by_agent: dict[str, list[dict[str, Any]]] = {}  # each agent's open spans, oldest first
    root: dict[str, Any] | None = None
    last_result: dict[str, Any] | None = None
    resumes: list[dict[str, Any]] = []
    usage: dict[str, Any] | None = None
    for e in events:
        kind, d, seq, ts = e.get("type"), e.get("data") or {}, e.get("seq"), e.get("ts")
        if kind == "agent":
            agent = str(d.get("agent"))
            lead = agent in lead_roles
            if d.get("status") == "started":
                span = {"span_id": f"a{seq}", "parent_id": None if lead else (root or {}).get("span_id"),
                        "kind": "run" if lead else "delegation", "actor": agent,
                        "task": d.get("task"), "start_seq": seq, "start": ts, "end_seq": None, "end": None,
                        "status": "running", "result_summary": None}
                spans.append(span)
                open_by_agent.setdefault(agent, []).append(span)
                if lead and root is None:
                    root = span
                elif not lead:
                    delegations.append(span)
                continue
            waiting = [s for s in delegations if s["end_seq"] is None]
            if open_by_agent.get(agent):
                span = open_by_agent[agent].pop()
                span.update(end_seq=seq, end=ts, status="finished", result_summary=d.get("summary"))
            elif not lead and waiting:
                # The event names nobody with an open span. Whose result this is cannot be read from the log,
                # so the oldest open delegation takes it and the span says how it was closed. The lead's own
                # finish event is never read this way: its summary is not a specialist's.
                span = waiting[0]
                open_by_agent[span["actor"]] = [s for s in open_by_agent[span["actor"]] if s is not span]
                span.update(end_seq=seq, end=ts, status="finished", result_summary=d.get("summary"),
                            closed_by=UNNAMED_FINISH)
            else:
                continue
            if not lead:
                last_result = {"seq": seq, "what": f"{agent} returned", "summary": d.get("summary")}
        elif kind == "tool":
            agent = str(d.get("agent"))
            mine = open_by_agent.get(agent)
            parent = mine[-1] if mine else root
            spans.append({"span_id": f"t{seq}", "parent_id": (parent or {}).get("span_id"), "kind": "tool", "actor": agent,
                          "tool": d.get("tool"), "arguments": d.get("args"), "result_summary": d.get("result_summary"),
                          "evidence_ids": list(d.get("evidence_ids") or []), "status": "ok" if d.get("ok", True) else "error",
                          "start_seq": seq, "start": ts, "ms": d.get("ms")})
            last_result = {"seq": seq, "what": f"{agent} · {d.get('tool')}", "summary": d.get("result_summary")}
        elif kind == "todo":
            items = list(d.get("items") or [])
            now = _text(items)
            change = _plan_changes(_text(plans[-1]["items"]) if plans else now, now)  # the first version changes nothing
            plans.append({"seq": seq, "t": ts, "items": items, "added": change["added"], "removed": change["removed"],
                          "added_new": change["added_new"], "reworded": change["reworded"], "after": last_result})
            if change["added_new"]:
                revisions.append({"seq": seq, "added": list(change["added_new"]), "added_new": list(change["added_new"]),
                                  "reworded": list(change["reworded"]), "removed": change["removed_outright"],
                                  "after": last_result})
        elif kind == "note":
            files.append({"seq": seq, "t": ts, "op": "edit" if d.get("op") == "edit_file" else "write",
                          "path": d.get("path"), "preview": d.get("preview")})
        elif kind == "note_read":
            files.append({"seq": seq, "t": ts, "op": "read", "path": d.get("path")})
        elif kind == "stage" and d.get("name") == "resume":
            resumes.append({"seq": seq, "t": ts, **(d.get("info") or {})})
        elif kind == "usage":
            usage = d.get("usage")
    for s in spans:  # a run interrupted before its finish event is as unfinished in the log as a delegation is
        if s["kind"] in ("run", "delegation") and s["end_seq"] is None:
            s["status"] = "running" if still_running else NOT_FINISHED
    examined = [s for s in spans if s["kind"] == "tool" and str(s.get("tool", "")).startswith(EXAMINATION_TOOLS)]
    first_look = min((s["start_seq"] for s in examined + delegations), default=None)
    plan_first: bool | None = None if not plans or first_look is None else plans[0]["seq"] < first_look
    writes = [f for f in files if f["op"] in ("write", "edit")]
    reads_back = [f for f in files if f["op"] == "read" and any(w["path"] == f["path"] and w["seq"] < f["seq"] for w in writes)]
    after_resume = [s for s in spans if s["kind"] == "tool" and resumes and s["start_seq"] > resumes[0]["seq"]]
    return {
        "version": TRACE_VERSION,
        "note": ("Built from the recorded event log. Summaries are what the tools returned, not a model's thoughts. "
                 "'After' is the recorded order and does not claim a cause. A plan and a first examination "
                 "recorded from the same model turn are ordered by the log, not by intent."),
        "spans": spans, "plan": plans, "case_file": files, "usage": usage,
        "behaviors": {
            "plan_recorded": bool(plans),
            "examined": first_look is not None,
            "plan_recorded_before_first_examination": plan_first,
            "planned_before_examining": plan_first is True,  # the older key, kept for existing callers
            "delegations": [{"span_id": s["span_id"], "to": s["actor"], "task": s["task"], "status": s["status"],
                             "result_summary": s["result_summary"], "closed_by": s.get("closed_by"),
                             "tool_calls": sum(1 for t in spans if t["kind"] == "tool" and t["parent_id"] == s["span_id"])}
                            for s in delegations],
            "plan_revisions": revisions,
            "case_file": {"writes": len(writes), "reads_after_a_write": len(reads_back),
                          "paths": sorted({str(f["path"]) for f in files})},
            "resumed": {"times": len(resumes), "resumes": resumes, "tool_calls_after_resume": len(after_resume)},
        },
    }


__all__ = ["TRACE_VERSION", "investigation_trace"]
