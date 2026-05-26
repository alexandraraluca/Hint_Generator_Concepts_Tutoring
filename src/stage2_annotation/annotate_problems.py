"""Stage 2.2 - LLM-assisted annotation of problems with concepts.

For each problem packet (statement + representative solutions), call
`gpt-oss:20b` via Ollama and ask it to:

- pick a `primary_concept`,
- list secondary `concepts`,
- estimate `difficulty` and `expected_complexity`,
- list `common_pitfalls`.

The model is given the canonical concept vocabulary (ids only) from
`concepts_dag.json` and is told it MUST use only ids from that list. We
validate the output against `PROBLEMS_SCHEMA` and assert that all returned
concept ids exist in the DAG; invalid annotations are written to a separate
`problems_invalid.json` for manual review.

The script is *resumable*: it appends to `problems.json` and skips
problem_ids already present.
"""

from __future__ import annotations

import argparse
import textwrap
from pathlib import Path
from typing import Any

from tqdm import tqdm

from src.common.io_utils import read_json, write_json
from src.common.ollama_client import OllamaClient, OllamaConfig
from src.common.paths import (
    ANNOTATIONS_DIR,
    PROCESSED_DIR,
    ensure_dirs,
)
from src.common.schemas import validate

PACKETS_DIR = PROCESSED_DIR / "packets"
PROBLEMS_OUT = ANNOTATIONS_DIR / "problems.json"
PROBLEMS_INVALID = ANNOTATIONS_DIR / "problems_invalid.json"
CONCEPTS_DAG_PATH = ANNOTATIONS_DIR / "concepts_dag.json"


def _system_prompt(concept_ids: list[str]) -> str:
    return textwrap.dedent(
        f"""
        Ești un asistent expert în algoritmică care adnotează probleme de la
        cursul universitar Programarea Algoritmilor (PA), folosit la Politehnica
        București. Primești enunțul unei probleme și 1-2 soluții corecte
        (scor 100). Trebuie să întorci STRICT JSON cu următoarele câmpuri:

        - primary_concept (string): cel mai important concept algoritmic, ales
          DIN LISTA de id-uri permise.
        - concepts (array of string): toate conceptele relevante (incluzând
          primary_concept), id-uri DIN LISTA permisă; min 1, max 8.
        - difficulty (string): unul din "easy", "medium", "hard", "very_hard".
        - expected_complexity (string): exemplu "O(N log N)", "O(N + M)",
          "O(N^2)", etc.
        - common_pitfalls (array of string): 2-5 capcane tipice exprimate scurt
          în română (max 15 cuvinte fiecare). NU include cod.
        - title (string): un titlu scurt (max 8 cuvinte) inspirat din enunț.
        - summary (string): rezumat 1-2 propoziții, fără cod.
        - llm_confidence (number, 0..1): cât de sigur ești de etichetare.

        REGULI ABSOLUTE:
        - NU inventa concepte noi. Folosește doar id-uri din lista de mai jos.
        - JSON valid, fără text suplimentar înainte/după.
        - Scrie în limba română.

        Lista permisă de concept ids:
        {", ".join(concept_ids)}
        """
    ).strip()


def _user_prompt(packet: dict[str, Any]) -> str:
    rep = packet.get("representative_solutions") or []
    snippets = []
    for s in rep[:2]:
        snippets.append(
            f"--- Soluție {s['language']} (anon={s['anon_id']}, scor=100) ---\n"
            f"{s['code']}"
        )
    snippet_block = "\n\n".join(snippets) if snippets else "(fără soluții reprezentative)"
    statement = packet.get("statement_text") or "(enunț indisponibil)"
    return textwrap.dedent(
        f"""
        problem_id: {packet["problem_id"]}
        an: {packet["year"]}, temă: {packet["tema"]}, pid: {packet["pid"]}
        nr soluții totale pe disc: {packet["n_solutions"]}
        limbaje: {", ".join(packet.get("languages", []))}

        === Enunț (extras din PDF) ===
        {statement}

        === Soluții reprezentative ===
        {snippet_block}

        Întoarce JSON cu schema cerută în system prompt.
        """
    ).strip()


def _load_concept_ids() -> list[str]:
    dag = read_json(CONCEPTS_DAG_PATH)
    return [c["id"] for c in dag["concepts"]]


def _load_alias_map() -> dict[str, str]:
    """Build {alias_lowercased -> canonical_id} from the DAG."""
    dag = read_json(CONCEPTS_DAG_PATH)
    alias_map: dict[str, str] = {}
    for c in dag["concepts"]:
        cid = c["id"]
        alias_map[cid.lower()] = cid
        for a in c.get("aliases", []) or []:
            alias_map[a.lower()] = cid
    return alias_map


def _normalize_concept_id(raw: str, alias_map: dict[str, str]) -> str:
    """Return canonical id if `raw` matches an alias, else `raw` unchanged."""
    if not isinstance(raw, str):
        return raw
    return alias_map.get(raw.strip().lower(), raw)


def _load_existing(out_path: Path) -> dict[str, dict[str, Any]]:
    if not out_path.exists():
        return {}
    obj = read_json(out_path)
    return {p["problem_id"]: p for p in obj.get("problems", [])}


def _save_problems(out_path: Path, items: list[dict[str, Any]]) -> None:
    obj = {"version": "0.1", "problems": items}
    errs = validate("problems", obj)
    if errs:
        raise RuntimeError("schema validation failed: " + "; ".join(errs))
    write_json(out_path, obj)


def annotate_one(
    client: OllamaClient,
    packet: dict[str, Any],
    concept_ids: list[str],
    valid_id_set: set[str],
    alias_map: dict[str, str],
) -> tuple[dict[str, Any] | None, str | None]:
    """Return (annotation_dict, error_msg)."""
    try:
        result = client.chat_json(
            system=_system_prompt(concept_ids),
            user=_user_prompt(packet),
        )
    except Exception as e:  # noqa: BLE001
        return None, f"LLM call failed: {e!r}"

    annotation: dict[str, Any] = {
        "problem_id": packet["problem_id"],
        "year": packet["year"],
        "tema": packet["tema"],
        "pid": packet["pid"],
        "annotation_source": "llm",
    }
    for k in (
        "primary_concept",
        "concepts",
        "difficulty",
        "expected_complexity",
        "common_pitfalls",
        "title",
        "summary",
        "llm_confidence",
    ):
        if k in result:
            annotation[k] = result[k]

    if "concepts" in annotation:
        annotation["concepts"] = [
            _normalize_concept_id(c, alias_map) for c in annotation["concepts"]
        ]
        annotation["concepts"] = list(dict.fromkeys(annotation["concepts"]))
    if "primary_concept" in annotation:
        annotation["primary_concept"] = _normalize_concept_id(
            annotation["primary_concept"], alias_map
        )

    bad_concepts = [
        c for c in annotation.get("concepts", []) if c not in valid_id_set
    ]
    if bad_concepts:
        return annotation, f"unknown concept ids: {bad_concepts}"
    if (
        annotation.get("primary_concept")
        and annotation["primary_concept"] not in valid_id_set
    ):
        return annotation, (
            f"unknown primary_concept: {annotation['primary_concept']}"
        )
    if not annotation.get("concepts"):
        return annotation, "missing concepts"

    return annotation, None


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=0, help="0 = all packets")
    parser.add_argument("--model", type=str, default=None, help="override OLLAMA_MODEL")
    parser.add_argument(
        "--force",
        action="store_true",
        help="re-annotate problems already in problems.json",
    )
    args = parser.parse_args()

    ensure_dirs()
    concept_ids = _load_concept_ids()
    valid_id_set = set(concept_ids)
    alias_map = _load_alias_map()

    packets = sorted(PACKETS_DIR.glob("*.json"))
    if args.limit:
        packets = packets[: args.limit]

    existing = _load_existing(PROBLEMS_OUT)
    invalid_existing = (
        read_json(PROBLEMS_INVALID).get("problems", [])
        if PROBLEMS_INVALID.exists()
        else []
    )

    cfg = OllamaConfig()
    if args.model:
        cfg.model = args.model

    client = OllamaClient(cfg)
    if not client.health():
        print(
            f"WARN: Ollama not reachable at {cfg.base_url}; "
            "start `ollama serve` and `ollama pull {cfg.model}` first.",
        )
        return 2
    print(f"using model={cfg.model} ctx={cfg.num_ctx} temp={cfg.temperature}")

    new_annotations: list[dict[str, Any]] = list(existing.values())
    new_invalid: list[dict[str, Any]] = list(invalid_existing)

    n_done = 0
    for path in tqdm(packets, desc="annotating problems"):
        packet = read_json(path)
        pid = packet["problem_id"]
        if pid in existing and not args.force:
            continue
        ann, err = annotate_one(
            client, packet, concept_ids, valid_id_set, alias_map
        )
        if err:
            print(f"  ! {pid}: {err}")
            if ann is not None:
                ann["_error"] = err
                new_invalid = [x for x in new_invalid if x.get("problem_id") != pid]
                new_invalid.append(ann)
            continue
        # remove any prior version for this pid
        new_annotations = [x for x in new_annotations if x["problem_id"] != pid]
        new_annotations.append(ann)  # type: ignore[arg-type]
        n_done += 1
        # save after each success so we are resumable
        _save_problems(PROBLEMS_OUT, sorted(new_annotations, key=lambda x: x["problem_id"]))
        if new_invalid:
            write_json(
                PROBLEMS_INVALID,
                {"version": "0.1", "problems": new_invalid},
            )

    client.close()
    print(f"\nfinished: {n_done} new annotations, {len(new_invalid)} invalid")
    print(f"  -> {PROBLEMS_OUT}")
    if new_invalid:
        print(f"  -> {PROBLEMS_INVALID} (please review)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
