"""Repurposing agent loop and run driver.

Systems (``mode`` x ``verifier``):
  closed_book            LLM only, no retrieval, single call
  rag      + none        tools (KG + date-capped PubMed), output taken as-is
  rag      + filter      ... unverified proposals dropped / demoted
  rag      + revise      ... rejected proposals returned to the agent once for repair, then filter
Every system's output is verified post hoc, so fabricated-edge rates are comparable across systems
even when the verifier does not influence the ranking.
"""

from __future__ import annotations

import json
import logging
import re
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from ..data.kg import DRUG, KnowledgeGraph
from ..utils import append_jsonl, extract_json, read_jsonl
from ..verify.schema import Proposal
from ..verify.verifier import MechanismVerifier, Verification
from . import prompts
from .llm import AssistantTurn, LLMClient, ToolCall, Usage
from .tools import Toolbox

log = logging.getLogger(__name__)

_REACT = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.S)


@dataclass
class AgentSettings:
    mode: str = "rag"                 # closed_book | rag
    verifier: str = "revise"          # none | filter | revise
    unverified_policy: str = "demote"  # drop | demote
    top_k: int = 20
    max_steps: int = 16
    revise_rounds: int = 1
    tool_mode: str = "native"         # native | react
    use_kg: bool = True
    use_candidates: bool = True
    use_pubmed: bool = True

    @property
    def system_name(self) -> str:
        if self.mode == "closed_book":
            return "closed_book"
        parts = ["rag", f"verify-{self.verifier}"]
        if not self.use_kg:
            parts.append("no-kg")
        elif not self.use_candidates:
            parts.append("no-candidates")
        if not self.use_pubmed:
            parts.append("no-pubmed")
        if self.verifier != "none" and self.unverified_policy == "drop":
            parts.append("drop")
        return "_".join(parts)


class RepurposingAgent:
    def __init__(self, llm: LLMClient, kg: KnowledgeGraph, verifier: MechanismVerifier,
                 settings: AgentSettings, cutoff: str, pubmed=None,
                 known_by_disease: dict[int, set[int]] | None = None,
                 pubmed_max_results: int = 5):
        self.llm = llm
        self.kg = kg
        self.verifier = verifier
        self.s = settings
        self.cutoff = cutoff
        self.pubmed = pubmed
        self.known = known_by_disease or {}
        self.pubmed_max_results = pubmed_max_results
        if settings.tool_mode == "native" and not llm.supports_native_tools:
            self.s.tool_mode = "react"

    # ------------------------------------------------------------------ LLM plumbing
    def _system(self, tools: list[dict]) -> str:
        if self.s.mode == "closed_book":
            return prompts.SYSTEM_CLOSED_BOOK.format(cutoff=self.cutoff, k=self.s.top_k,
                                                     schema=prompts.CLOSED_BOOK_SCHEMA)
        text = prompts.SYSTEM_AGENT.format(cutoff=self.cutoff, k=self.s.top_k,
                                           max_steps=self.s.max_steps)
        if self.s.tool_mode == "react":
            text += prompts.REACT_SUFFIX.format(tools=json.dumps(tools, indent=1))
        return text

    def _chat(self, system: str, messages: list[dict], tools: list[dict] | None,
              usage: Usage) -> AssistantTurn:
        native = tools if (tools and self.s.tool_mode == "native") else None
        turn = self.llm.chat(system, messages, native)
        usage.add(turn.usage)
        if self.s.tool_mode == "react" and tools and not turn.tool_calls:
            for m in _REACT.finditer(turn.text):
                try:
                    obj = json.loads(m.group(1))
                    turn.tool_calls.append(ToolCall(uuid.uuid4().hex[:12], obj.get("name", ""),
                                                    obj.get("arguments", {}) or {}))
                except json.JSONDecodeError:
                    continue
            turn.tool_calls = turn.tool_calls[:1]
        messages.append({"role": "assistant", "text": turn.text, "tool_calls": turn.tool_calls,
                         "raw": turn.raw})
        return turn

    def _tool_message(self, results: list[dict]) -> dict:
        if self.s.tool_mode == "react":
            body = "\n".join(f'<tool_result name="{r["name"]}">{r["content"]}</tool_result>'
                             for r in results)
            return {"role": "user", "content": body}
        return {"role": "tool", "results": results}

    @staticmethod
    def _parse_text_proposals(text: str) -> list[dict] | None:
        try:
            obj = extract_json(text)
        except ValueError:
            return None
        if isinstance(obj, dict):
            obj = obj.get("proposals", obj.get("candidates"))
        return obj if isinstance(obj, list) else None

    # ------------------------------------------------------------------ main loop
    def _converse(self, system: str, messages: list[dict], toolbox: Toolbox | None,
                  usage: Usage, trace: dict, max_steps: int) -> list[dict] | None:
        tools = toolbox.specs() if toolbox else None
        nudged = False
        for step in range(max_steps + 1):
            last = step == max_steps
            if last and tools:
                messages.append({"role": "user", "content": prompts.BUDGET_EXHAUSTED})
            turn = self._chat(system, messages, tools, usage)
            trace["steps"] += 1
            trace["stop_reasons"].append(turn.stop_reason)
            if turn.stop_reason == "refusal":
                trace["errors"].append("refusal")
                return None
            if not turn.tool_calls:
                parsed = self._parse_text_proposals(turn.text)
                if parsed is not None or not tools or nudged or last:
                    return parsed
                nudged = True
                messages.append({"role": "user", "content": prompts.NUDGE_SUBMIT})
                continue
            results, submitted = [], None
            for call in turn.tool_calls:
                trace["tool_calls"][call.name] = trace["tool_calls"].get(call.name, 0) + 1
                if call.name == "submit_proposals":
                    props = call.arguments.get("proposals")
                    if isinstance(props, str):
                        props = self._parse_text_proposals(props)
                    if isinstance(props, list):
                        submitted = props
                        results.append({"id": call.id, "name": call.name,
                                        "content": json.dumps({"received": len(props)}),
                                        "is_error": False})
                    else:
                        results.append({"id": call.id, "name": call.name, "is_error": True,
                                        "content": "proposals must be a JSON array"})
                    continue
                content, is_err = toolbox.execute(call.name, call.arguments)
                results.append({"id": call.id, "name": call.name, "content": content,
                                "is_error": is_err})
            messages.append(self._tool_message(results))
            if submitted is not None:
                return submitted
        return None

    def run_disease(self, disease_idx: int) -> dict:
        t0 = time.time()
        usage = Usage()
        trace = {"steps": 0, "tool_calls": {}, "stop_reasons": [], "errors": [], "rounds": 0}
        name = self.kg.names[disease_idx]
        known_names = [self.kg.names[d] for d in sorted(self.known.get(disease_idx, set()))]
        toolbox = None
        if self.s.mode != "closed_book":
            toolbox = Toolbox(self.kg, self.pubmed, use_kg=self.s.use_kg,
                              use_candidates=self.s.use_candidates, use_pubmed=self.s.use_pubmed,
                              known_by_disease=self.known,
                              pubmed_max_results=self.pubmed_max_results)
        system = self._system(toolbox.specs() if toolbox else [])
        ref = f" (kg:{disease_idx})" if toolbox and self.s.use_kg else ""
        messages = [{"role": "user", "content": prompts.USER_TASK.format(
            disease_name=name, disease_ref=ref, cutoff=self.cutoff,
            known=prompts.format_known(known_names), k=self.s.top_k)}]
        raw_props = None
        try:
            steps = 1 if self.s.mode == "closed_book" else self.s.max_steps
            raw_props = self._converse(system, messages, toolbox, usage, trace, steps)
            proposals = self._to_proposals(raw_props)
            checks = [self.verifier.verify(p, disease_idx) for p in proposals]
            if self.s.mode != "closed_book" and self.s.verifier == "revise":
                for _ in range(self.s.revise_rounds):
                    failures = [(p, c) for p, c in zip(proposals, checks) if not c.verified]
                    if not failures:
                        break
                    trace["rounds"] += 1
                    messages.append({"role": "user", "content": prompts.USER_REVISE.format(
                        cutoff=self.cutoff, failures=self._format_failures(failures))})
                    revised = self._converse(system, messages, toolbox, usage, trace,
                                             max(4, self.s.max_steps // 2))
                    if revised is None:
                        break
                    proposals = self._to_proposals(revised)
                    checks = [self.verifier.verify(p, disease_idx) for p in proposals]
        except Exception as err:  # keep the run going; failures are reported per disease
            log.exception("Disease %s failed", disease_idx)
            trace["errors"].append(f"{type(err).__name__}: {err}")
            proposals, checks = self._to_proposals(raw_props), []
            checks = [self.verifier.verify(p, disease_idx) for p in proposals] if proposals else []
        ranked = self._rank(proposals, checks)
        return {
            "disease_idx": int(disease_idx), "disease_name": name,
            "system": self.s.system_name, "model": self.llm.model,
            "ranked": ranked,
            "proposals": [{**p.to_dict(), "verification": c.to_dict()}
                          for p, c in zip(proposals, checks)],
            "usage": usage.to_dict(), "trace": trace,
            "seconds": round(time.time() - t0, 2),
        }

    # ------------------------------------------------------------------ helpers
    def _to_proposals(self, raw: list | None) -> list[Proposal]:
        out: list[Proposal] = []
        for item in raw or []:
            try:
                p = Proposal.from_any(item)
            except (AttributeError, TypeError, ValueError):
                continue
            if p.drug:
                out.append(p)
        return out[: self.s.top_k]

    def _format_failures(self, failures: list[tuple[Proposal, Verification]]) -> str:
        lines = []
        for p, c in failures:
            lines.append(f"- {p.drug}: {c.summary()}")
        return "\n".join(lines)

    def _rank(self, proposals: list[Proposal], checks: list[Verification]) -> list[dict]:
        rows, seen = [], set()
        for p, c in zip(proposals, checks):
            idx = c.drug_idx if c.drug_idx is not None else self.kg.resolve(p.drug, DRUG)
            if idx is None or idx in seen:
                continue
            seen.add(idx)
            rows.append({"drug_idx": int(idx), "drug": self.kg.names[idx], "verified": c.verified})
        if self.s.mode != "closed_book" and self.s.verifier in ("filter", "revise"):
            good = [r for r in rows if r["verified"]]
            bad = [r for r in rows if not r["verified"]]
            rows = good if self.s.unverified_policy == "drop" else good + bad
        return rows


def run_agent(agent_factory: Callable[[], RepurposingAgent], disease_ids: list[int], out_path: Path,
              workers: int = 1, meta: dict | None = None) -> Path:
    """Run every disease (resumable: diseases already in ``out_path`` are skipped)."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if out_path.exists():
        done = {r["disease_idx"] for r in read_jsonl(out_path)}
    todo = [d for d in disease_ids if d not in done]
    if meta:
        (out_path.parent / "meta.json").write_text(json.dumps(meta, indent=2, default=str),
                                                   encoding="utf-8")
    log.info("%d diseases to run (%d already done)", len(todo), len(done))
    lock = threading.Lock()
    local = threading.local()

    def work(d: int) -> dict:
        if not hasattr(local, "agent"):
            local.agent = agent_factory()
        rec = local.agent.run_disease(d)
        append_jsonl(out_path, rec, lock)
        return rec

    from tqdm import tqdm

    if workers <= 1:
        for d in tqdm(todo, desc="diseases"):
            work(d)
    else:
        with ThreadPoolExecutor(workers) as pool:
            futures = [pool.submit(work, d) for d in todo]
            for f in tqdm(as_completed(futures), total=len(futures), desc="diseases"):
                f.result()
    return out_path
