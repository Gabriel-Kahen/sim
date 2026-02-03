import ast
import json
import os
from typing import Any, Dict, Optional

from .models import Patch, new_id


class LLMClient:
    """
    Gemini-backed Writer/Compiler. Raises if the LLM is unavailable or responses are invalid.
    """

    def __init__(self, api_key: Optional[str]):
        if not api_key:
            raise ValueError("Gemini API key is required")
        self.model_name = os.getenv("GEMINI_MODEL", "gemini-1.0-pro")
        try:
            import google.generativeai as genai

            genai.configure(api_key=api_key)
            self._model = genai.GenerativeModel(self.model_name)
        except Exception as e:
            raise RuntimeError(f"Failed to initialize Gemini model {self.model_name}: {e}") from e

    def _generate(self, prompt: str) -> str:
        try:
            resp = self._model.generate_content(prompt)
            return resp.text or ""
        except Exception as e:
            raise RuntimeError(f"Gemini generation failed: {e}") from e

    def _generate_json(self, prompt: str) -> str:
        """
        Ask Gemini to return JSON only.
        """
        try:
            resp = self._model.generate_content(
                prompt,
                generation_config={"response_mime_type": "application/json"},
            )
            return resp.text or ""
        except Exception as e:
            raise RuntimeError(f"Gemini JSON generation failed: {e}") from e

    def writer_pair(self, obs_u: Dict[str, Any], obs_v: Dict[str, Any]) -> Dict[str, str]:
        prompt = (
            "You are a creative socio-economic simulator. Given two nodes, write a concise outcome.\n"
            f"Node A: {obs_u}\nNode B: {obs_v}\n"
            "Return 2-3 sentences describing what happens, including any transfers, obligations, edge changes, reputational impacts. You may also change jobs or succession intents if warranted."
        )
        text = self._generate(prompt)
        return {"scene": text, "outcome_summary": text.split("\n")[0][:400]}

    def writer_production(self, obs: Dict[str, Any]) -> Dict[str, str]:
        prompt = (
            "Describe a short production event for this node in 1-2 sentences.\n"
            f"Node: {obs}\nInclude output value and brief rationale."
        )
        text = self._generate(prompt)
        return {"scene": text, "outcome_summary": text.split("\n")[0][:400]}

    def compiler(self, outcome_summary: str, context: Dict[str, Any]) -> Patch:
        prompt = (
            "You are a strict compiler. Given an outcome summary, emit JSON with patch ops.\n"
            "Allowed ops (prefer updates): update_node, update_edge. You may create_edge when a new relationship is formed (provide source/target ids).\n"
            "update_node changes must be one of: set, append, update_item, remove_item.\n"
            "Every op must include required fields: update_node/update_edge/delete_edge must include 'id'; create_node must include 'node'; create_edge must include 'edge'.\n"
            "Use only these node ids (copy exact, do NOT shorten or invent): {allowed_node_ids}\n"
            "If an id is not provided in allowed_node_ids, do not reference it; instead, skip that update.\n"
            "Example: {\"ops\":[{\"op\":\"update_node\",\"id\":\"agent-1\",\"changes\":[{\"type\":\"set\",\"path\":\"state.job\",\"value\":\"Farmer\"}]}],\"notes\":\"short\"}\n"
            f"Outcome: {outcome_summary}\n"
            "Context (ledger facts): {context}\n"
            "Respond with JSON ONLY, no prose, no code fences, no prefix/suffix text: {\"ops\": [...], \"notes\": \"...\"}"
        )
        raw = self._generate_json(prompt)
        try:
            data = self._coerce_json(raw)
            self._validate_patch_dict(data, context)
            if not data.get("ops"):
                return self._fallback_patch(outcome_summary, context, note="empty_ops")
            return Patch.from_dict(data)
        except Exception as e:
            return self._fallback_patch(outcome_summary, context, note=f"fallback_due_to_error: {e}")

    def summarize_tick(self, events: list[str], t: int) -> str:
        if not events:
            return f"No events at tick {t}."
        joined = "\n".join(f"- {e}" for e in events)
        prompt = (
            f"Summarize the socio-economic simulation events for tick {t} in 3-5 sentences.\n"
            "Be concrete about transfers, production, shocks, and reputational changes.\n"
            f"Events:\n{joined}"
        )
        text = self._generate(prompt)
        return text or f"Tick {t}: " + " | ".join(events)

    def propose_institution(self, participants: list[dict], outcome_summary: str, recent: list[str] | None = None) -> Dict[str, str]:
        """
        Ask the LLM for an open-ended institution spec (type/name/description).
        """
        recent_note = f" Avoid repeating these recent institution themes: {recent}." if recent else ""
        prompt = (
            "Two or more agents want to form a new institution that is NOT a family/kin/support network. "
            "Base it on their jobs/networks/interests (e.g., finance, trade, guild, research, security, education, healthcare, logistics, culture). "
            "Propose a JSON object with keys: \"name\",\"institution_type\",\"description\",\"policies\",\"governance\". "
            "Keep it short and grounded in the outcome. Avoid words: family, kin, mutual aid, support network, artisan collective, guild collective, co-op unless clearly implied. "
            f"{recent_note}\n"
            f"Participants (include jobs if known): {participants}\nOutcome: {outcome_summary}\n"
            "Respond with JSON only."
        )
        try:
            raw = self._generate_json(prompt)
            data = self._coerce_json(raw)
            if isinstance(data, dict):
                return data
        except Exception:
            pass
        # Fallback
        fallback_opts = [
            {
                "name": "Trade Exchange",
                "institution_type": "Trade consortium",
                "description": "A group coordinating commercial routes and pricing.",
                "policies": "Fair pricing and shared logistics",
                "governance": "Rotating council",
            },
            {
                "name": "Mutual Credit House",
                "institution_type": "Finance",
                "description": "A small lender/clearinghouse for members.",
                "policies": "Transparent ledgers and capped interest",
                "governance": "Elected treasurer board",
            },
            {
                "name": "Security Pact",
                "institution_type": "Security",
                "description": "Members coordinate protection and dispute response.",
                "policies": "Collective defense and mediation first",
                "governance": "Consensus among signatories",
            },
            {
                "name": "Workshop League",
                "institution_type": "Craft guild",
                "description": "Shared standards and training for makers.",
                "policies": "Quality standards and apprenticeships",
                "governance": "Elected master artisans",
            },
            {
                "name": "Scholars Circle",
                "institution_type": "Research/Education",
                "description": "Members share knowledge and fund study.",
                "policies": "Open exchange of findings",
                "governance": "Steward council",
            },
        ]
        return random.choice(fallback_opts)

    def propose_job(self, node: Dict[str, Any]) -> str:
        """
        Ask the LLM for a concise job/title given node traits/memories; fallback to a small generic set.
        """
        prompt = (
            "Suggest a concise job or role for this agent based on their traits and recent memories. "
            "Return a short title only (no prose). Examples: Merchant, Mediator, Bookkeeper, Harbor Inspector, Loan Officer.\n"
            f"Agent: {node}\n"
            "Respond with a short title only."
        )
        try:
            raw = self._generate(prompt)
            if raw:
                return raw.strip().splitlines()[0][:80]
        except Exception:
            pass
        return random.choice(["Merchant", "Mediator", "Bookkeeper", "Inspector", "Courier", "Broker"])

    def propose_shock_story(self, node: Dict[str, Any], env: Dict[str, Any]) -> str:
        prompt = (
            "Invent a brief shock description affecting this node. Keep it to one short clause.\n"
            f"Node: {node}\nEnvironment: {env}\n"
            "Avoid generic phrasing; be specific (e.g., 'Local crop blight cuts harvest')."
        )
        try:
            raw = self._generate(prompt)
            if raw:
                return raw.strip().splitlines()[0][:120]
        except Exception:
            pass
        return "A sudden disruption hits the local economy."

    def propose_inflow_story(self, env: Dict[str, Any]) -> str:
        prompt = (
            "Invent a brief inflow description for an exogenous resource injection. One short clause only.\n"
            f"Environment: {env}\n"
            "Example styles: 'Civic grant program releases funds', 'Trade surplus redistributed', 'Patron donates capital'."
        )
        try:
            raw = self._generate(prompt)
            if raw:
                return raw.strip().splitlines()[0][:120]
        except Exception:
            pass
        return "A grant program releases funds."

    def propose_edge_description(self, participants: list[dict], outcome_summary: str) -> str:
        prompt = (
            "Given two nodes and their recent interaction, provide a concise relationship label (e.g., 'child', 'business partner', "
            "'trusted ally', 'rival', 'loan provider', 'guild member with'). Avoid generic words like 'relationship' or 'tie'. "
            "Return one short phrase only.\n"
            f"Participants: {participants}\nOutcome: {outcome_summary}\n"
            "Respond with a short label only."
        )
        try:
            raw = self._generate(prompt)
            if raw:
                return raw.strip().splitlines()[0][:80]
        except Exception:
            pass
        return "associate"

    @staticmethod
    def _coerce_json(raw: str) -> Dict[str, Any]:
        cleaned = raw.strip()
        if cleaned.startswith("```"):
            # Strip Markdown fences like ```json ... ```
            cleaned = cleaned.strip("`")
            if cleaned.lower().startswith("json"):
                cleaned = cleaned[4:]
        # Replace smart quotes if present
        cleaned = cleaned.replace("“", '"').replace("”", '"').replace("’", "'")

        def try_parse(s: str) -> Any:
            try:
                return json.loads(s)
            except Exception:
                return ast.literal_eval(s)

        # Attempt direct parse
        if cleaned.startswith("{") or cleaned.startswith("["):
            try:
                return try_parse(cleaned)
            except Exception:
                pass

        # Extract balanced JSON object from first '{'
        start = cleaned.find("{")
        if start != -1:
            depth = 0
            for idx in range(start, len(cleaned)):
                ch = cleaned[idx]
                if ch == "{":
                    depth += 1
                elif ch == "}":
                    depth -= 1
                    if depth == 0:
                        candidate = cleaned[start : idx + 1]
                        try:
                            return try_parse(candidate)
                        except Exception:
                            break
        raise ValueError("No JSON object found in compiler response")

    @staticmethod
    def _validate_patch_dict(data: Dict[str, Any], context: Dict[str, Any]) -> None:
        if not isinstance(data, dict):
            raise ValueError("Compiler JSON is not an object")
        ops = data.get("ops")
        if not isinstance(ops, list):
            raise ValueError("Compiler JSON missing 'ops' list")
        allowed_nodes = set(context.get("allowed_node_ids", []) or [])
        cleaned_ops = []
        for idx, op in enumerate(ops):
            if not isinstance(op, dict):
                continue
            if "op" not in op:
                continue
            op_type = op["op"]
            if op_type == "create_node" and "node" not in op:
                continue
            if op_type == "create_edge" and "edge" not in op:
                continue
            if op_type in {"update_node", "update_edge", "delete_edge"} and "id" not in op:
                continue
            if op_type == "update_node" and allowed_nodes:
                raw_id = op["id"]
                if raw_id not in allowed_nodes:
                    suffix = raw_id.split("-")[-1]
                    matches = [aid for aid in allowed_nodes if aid.endswith(raw_id) or aid.endswith(suffix)]
                    if len(matches) == 1:
                        op["id"] = matches[0]
                    elif len(allowed_nodes) == 1:
                        op["id"] = next(iter(allowed_nodes))
                    else:
                        continue
            cleaned_ops.append(op)
        if not cleaned_ops:
            data["ops"] = []
            return
        data["ops"] = cleaned_ops

    def _fallback_patch(self, outcome_summary: str, context: Dict[str, Any], note: str) -> Patch:
        """
        Last-resort deterministic patch when compiler output is unusable.
        """
        ops = []
        allowed_nodes = context.get("allowed_node_ids") or [n.get("id") for n in context.get("nodes", []) if isinstance(n, dict)]
        for node_id in allowed_nodes[:2]:
            if not node_id:
                continue
            ops.append(
                {
                    "op": "update_node",
                    "id": node_id,
                    "changes": [
                        {
                            "type": "append",
                            "path": "state.memories",
                            "item": {"id": new_id("mem"), "description": outcome_summary},
                        }
                    ],
                }
            )
        return Patch(ops=ops, notes=f"[fallback] {note}: {outcome_summary}")
