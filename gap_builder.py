"""
gap_builder.py
--------------
Builds a structured gap_object by comparing segments of a partial solution
against segments of the passing (100p) solution.

ALL string-level comparisons (key operations, data structures, algorithm names,
issue descriptions, concept inference) are powered by sentence-transformer
embeddings — NOT lexical overlap. Two sentences that mean the same thing but
share no words (e.g. "sortează după preț" vs "ordonează computerele după cost")
are treated as semantically equivalent.

The default model is multilingual (`paraphrase-multilingual-MiniLM-L12-v2`)
because the whole pipeline runs in Romanian.

Gap levels (priority order):
  intent          → student is solving the wrong problem entirely
  function        → right problem, wrong algorithm or wrong data structures
  implementation  → right algorithm, wrong execution (bugs, logic, output)

Usage:
  from gap_builder import build_gap_object
  gap = build_gap_object(partial_segments, passing_segments, codebert_sim)
"""

from __future__ import annotations

import json
import os
from typing import Any


# ── Embedding model (multilingual: works for Romanian and English) ─────────────
EMBED_MODEL_NAME = os.environ.get(
    "GAP_BUILDER_EMBED_MODEL",
    "paraphrase-multilingual-MiniLM-L12-v2",
)

# ── Similarity thresholds ──────────────────────────────────────────────────────
SIM_HIGH = 0.97   # codebert similarity bucket: diff-friendly
SIM_MEDIUM = 0.85
SIM_LOW = 0.0

# Per-comparison semantic thresholds.
OP_MATCH_THRESHOLD = 0.55       # two key-operation strings → semantically the same
DS_MATCH_THRESHOLD = 0.55       # two data-structure strings → same DS
ALGO_MATCH_THRESHOLD = 0.75     # two algorithm names → semantically the same
CONCEPT_MATCH_THRESHOLD = 0.45  # min sim to accept an inferred missing_concept
OUTPUT_ISSUE_THRESHOLD = 0.50   # min sim to flag potential issue as output-related


_EMBED_MODEL: Any = None
_EMBED_CACHE: dict[str, Any] = {}


def _get_embed_model():
    """Lazy-load sentence-transformers model (cached at module level)."""
    global _EMBED_MODEL
    if _EMBED_MODEL is None:
        from sentence_transformers import SentenceTransformer
        _EMBED_MODEL = SentenceTransformer(EMBED_MODEL_NAME)
    return _EMBED_MODEL


def _encode(texts: list[str]):
    """Batch-encode with per-string cache. Returns np.ndarray (N, D) or None.
    Falls back gracefully (returns None) if sentence-transformers / numpy missing.
    """
    if not texts:
        return None
    try:
        import numpy as np
        model = _get_embed_model()
    except Exception:
        return None

    out: list[Any] = [None] * len(texts)
    to_compute_idx: list[int] = []
    to_compute_text: list[str] = []
    for i, t in enumerate(texts):
        if t in _EMBED_CACHE:
            out[i] = _EMBED_CACHE[t]
        else:
            to_compute_idx.append(i)
            to_compute_text.append(t)
    if to_compute_text:
        new = model.encode(to_compute_text, normalize_embeddings=True)
        for i, t, emb in zip(to_compute_idx, to_compute_text, new):
            _EMBED_CACHE[t] = emb
            out[i] = emb
    return np.stack(out)


def _cosine(a, b) -> float:
    """Cosine between two normalized vectors → dot product."""
    return float((a * b).sum())


def _semantic_sim(text_a: str, text_b: str) -> float:
    """Cosine similarity in embedding space.

    Falls back to lexical Jaccard ONLY if embeddings are unavailable
    (sentence-transformers not installed). Returns 0.0 for empty inputs.
    """
    if not text_a or not text_b:
        return 0.0
    embs = _encode([text_a, text_b])
    if embs is not None:
        return _cosine(embs[0], embs[1])
    wa = set(text_a.lower().split())
    wb = set(text_b.lower().split())
    return len(wa & wb) / max(len(wa | wb), 1)


def _best_match(query: str, candidates: list[str]) -> tuple[int, float]:
    """Return (best_index, best_similarity) for `query` against `candidates`."""
    if not query or not candidates:
        return (-1, 0.0)
    embs = _encode([query] + candidates)
    if embs is None:
        sims = [_semantic_sim(query, c) for c in candidates]
    else:
        q = embs[0]
        sims = [_cosine(q, embs[i + 1]) for i in range(len(candidates))]
    best_i = max(range(len(sims)), key=lambda i: sims[i])
    return best_i, sims[best_i]


def _max_sim_to_anchors(text: str, anchors: list[str]) -> float:
    """Max cosine similarity between `text` and any anchor phrase."""
    if not text or not anchors:
        return 0.0
    embs = _encode([text] + anchors)
    if embs is None:
        return max((_semantic_sim(text, a) for a in anchors), default=0.0)
    q = embs[0]
    return max(_cosine(q, embs[i + 1]) for i in range(len(anchors)))


# ── Code-similarity buckets (from CodeBERT) ────────────────────────────────────

def similarity_bucket(sim: float) -> str:
    if sim >= SIM_HIGH:
        return "high"
    if sim >= SIM_MEDIUM:
        return "medium"
    return "low"


def hint_level_from_gap(gap_level: str, sim: float) -> str:
    if gap_level == "intent":
        return "macro"
    if gap_level == "function":
        return "structural"
    if sim >= SIM_HIGH:
        return "very_specific"
    if sim >= SIM_MEDIUM:
        return "specific"
    return "structural"


# ── Algorithm comparison: synonym sets + semantic fallback ─────────────────────

_ALGO_SYNONYMS = {
    "greedy": {"greedy", "greedy_sort", "greedy_selection", "greedy_with_sort"},
    "dp": {"dp", "dynamic_programming", "memoization", "programare_dinamica", "programare dinamica"},
    "binary_search": {"binary_search", "binary search", "bisection", "cautare binara", "căutare binară"},
    "bfs": {"bfs", "breadth_first", "breadth first search", "parcurgere bfs"},
    "dfs": {"dfs", "depth_first", "depth first search", "parcurgere dfs"},
    "two_pointers": {"two_pointers", "two pointers", "doi pointeri"},
    "sliding_window": {"sliding_window", "sliding window", "fereastra glisanta", "fereastră glisantă"},
    "dijkstra": {"dijkstra", "shortest_path", "drumuri minime"},
    "union_find": {"union_find", "union-find", "dsu", "disjoint_set"},
    "topological_sort": {"topological_sort", "sortare topologica", "sortare topologică"},
}


def _algo_canonical(a: str) -> str:
    for canon, variants in _ALGO_SYNONYMS.items():
        if a in variants:
            return canon
    return a


def _compare_algorithms(partial: dict, passing: dict) -> dict | None:
    """Different algorithms? Returns gap dict or None.
    Uses canonical synonym set first, then semantic similarity as fallback.
    """
    alg_p = partial.get("function", {}).get("algorithm", "").lower().strip()
    alg_r = passing.get("function", {}).get("algorithm", "").lower().strip()
    if not alg_p or not alg_r:
        return None
    if _algo_canonical(alg_p) == _algo_canonical(alg_r):
        return None
    if _semantic_sim(alg_p, alg_r) >= ALGO_MATCH_THRESHOLD:
        return None
    return {"algorithm_partial": alg_p, "algorithm_passing": alg_r}


# ── Key-operations comparison: best-match alignment, NOT positional zip ────────

def _compare_key_ops(partial: dict, passing: dict) -> dict:
    """Compare two sequences of key_operations using semantic alignment.

    Strategy:
      1. Encode all operations from both sides in a single batch.
      2. Build a full similarity matrix (n_partial × n_passing).
      3. For each partial step, find best passing match.
      4. `diverge_at_step` = first partial index whose best match falls below
         `OP_MATCH_THRESHOLD`. This correctly handles reorderings and extra
         steps because alignment is not positional.
      5. `missing_steps_in_partial` = passing steps that NO partial step matches
         well — i.e. logic that the student forgot to write.
    """
    ops_p = list(partial.get("implementation", {}).get("key_operations", []))
    ops_r = list(passing.get("implementation", {}).get("key_operations", []))
    n_p, n_r = len(ops_p), len(ops_r)

    base = {
        "ops_partial": ops_p,
        "ops_passing": ops_r,
        "n_steps_partial": n_p,
        "n_steps_passing": n_r,
        "length_difference": abs(n_p - n_r),
    }

    if not ops_p or not ops_r:
        return {
            **base,
            "diverge_at_step": -1,
            "missing_steps_in_partial": list(ops_r),
            "match_scores": [],
        }

    all_embs = _encode(ops_p + ops_r)
    if all_embs is None:
        sim_mat = [[_semantic_sim(ops_p[i], ops_r[j]) for j in range(n_r)] for i in range(n_p)]
    else:
        emb_p = all_embs[:n_p]
        emb_r = all_embs[n_p:]
        sim_mat = [[_cosine(emb_p[i], emb_r[j]) for j in range(n_r)] for i in range(n_p)]

    best_per_p: list[tuple[int, float]] = []
    for i in range(n_p):
        j_best = max(range(n_r), key=lambda j: sim_mat[i][j])
        best_per_p.append((j_best, sim_mat[i][j_best]))

    diverge_at = -1
    for i, (_, sim) in enumerate(best_per_p):
        if sim < OP_MATCH_THRESHOLD:
            diverge_at = i
            break

    matched_passing: set[int] = {j for (j, sim) in best_per_p if sim >= OP_MATCH_THRESHOLD}
    missing = [ops_r[j] for j in range(n_r) if j not in matched_passing]

    return {
        **base,
        "diverge_at_step": diverge_at,
        "missing_steps_in_partial": missing,
        "match_scores": [round(sim, 3) for (_, sim) in best_per_p],
    }


# ── Issue severity scoring (semantic, not keyword) ─────────────────────────────

_HIGH_SEVERITY_ANCHORS = [
    "rezultatul final este incorect",
    "depășire de tip / overflow",
    "indexare în afara limitelor vectorului",
    "valoare null accesată",
    "algoritmul nu garantează soluția optimă",
    "eșuează pentru anumite cazuri de test",
    "eroare de logică critică",
    "bucla devine infinită",
    "off by one la rezultat",
]

_MEDIUM_SEVERITY_ANCHORS = [
    "tratare incompletă a unui caz limită",
    "lipsește un pas",
    "potențial overflow în cazuri rare",
    "comparare incorectă într-un singur caz",
]

_LOW_SEVERITY_ANCHORS = [
    "cod redundant",
    "stil neoptim de scriere",
    "comentariu lipsă",
    "variabilă nefolosită",
    "sortarea este puțin ineficientă",
]


def _find_critical_issue(partial: dict) -> str:
    """Pick the most algorithmically critical issue from the partial's list.

    Each issue is scored as:
        max_sim_high_severity_anchors - 0.3 * max_sim_low_severity_anchors
    so that messages about correctness rank above messages about style/perf.
    """
    issues = partial.get("implementation", {}).get("potential_issues", [])
    if not issues:
        return ""

    def severity(issue: str) -> float:
        hi = _max_sim_to_anchors(issue, _HIGH_SEVERITY_ANCHORS)
        lo = _max_sim_to_anchors(issue, _LOW_SEVERITY_ANCHORS)
        return hi - 0.3 * lo

    return max(issues, key=severity)


def _check_intent_mismatch(partial: dict, passing: dict) -> bool:
    return not partial.get("intent", {}).get("matches_problem", True)


# ── Data-structure comparison: semantic matching ──────────────────────────────

_GENERIC_DS = {
    "arraylist", "list", "array", "vector", "string", "int[]",
    "vector de int", "listă", "tablou", "tablou de int",
}


def _check_data_structure_divergence(partial: dict, passing: dict) -> dict | None:
    """Detect data-structure divergence using semantic matching.

    `priority queue` ~ `heap`, `dictionar` ~ `hashmap` etc. should NOT count
    as missing/extra.
    """
    ds_p = [s.strip() for s in partial.get("function", {}).get("data_structures", []) if s and s.strip()]
    ds_r = [s.strip() for s in passing.get("function", {}).get("data_structures", []) if s and s.strip()]

    ds_p = [s for s in ds_p if s.lower() not in _GENERIC_DS]
    ds_r = [s for s in ds_r if s.lower() not in _GENERIC_DS]

    if not ds_p and not ds_r:
        return None

    def _present_in(item: str, pool: list[str]) -> bool:
        if not pool:
            return False
        _, sim = _best_match(item, pool)
        return sim >= DS_MATCH_THRESHOLD

    missing = [s for s in ds_r if not _present_in(s, ds_p)]
    extra = [s for s in ds_p if not _present_in(s, ds_r)]
    if missing or extra:
        return {"missing_in_partial": missing, "extra_in_partial": extra}
    return None


# ── Output-issue detection (semantic) ──────────────────────────────────────────

_OUTPUT_ISSUE_ANCHORS = [
    "rezultatul afișat este incorect",
    "ieșirea finală scade 1 din valoare, deși nu ar trebui",
    "scrierea rezultatului în fișier este greșită",
    "ordinea valorilor de ieșire este inversată",
    "valoarea finală tipărită este alta decât cea calculată",
    "the final output is wrong",
]


def _has_output_issue(partial: dict) -> bool:
    issues = partial.get("implementation", {}).get("potential_issues", [])
    if not issues:
        return False
    for issue in issues:
        if _max_sim_to_anchors(issue, _OUTPUT_ISSUE_ANCHORS) >= OUTPUT_ISSUE_THRESHOLD:
            return True
    return False


# ── Concept inference: semantic anchors per concept ────────────────────────────

_CONCEPT_ANCHORS: dict[str, list[str]] = {
    "integer_overflow": [
        "depășire de tip întreg (overflow)",
        "valoarea depășește limita unui int de 32 de biți",
        "trebuie folosit long long pentru produs sau sumă",
        "variabila este int dar suma poate depăși limita",
    ],
    "off_by_one_output": [
        "se scade 1 din rezultatul final, deși nu ar trebui",
        "ieșirea este cu 1 mai mică decât valoarea corectă",
        "eroare off-by-one la afișarea rezultatului",
    ],
    "index_boundary": [
        "indexare în afara limitelor vectorului",
        "accesare element la index invalid",
        "out of bounds la accesul listei",
        "i poate ajunge la sfârșitul listei și provoacă acces nevalid",
    ],
    "group_minimum_upgrade": [
        "trebuie actualizate toate calculatoarele cu valoarea minimă simultan",
        "selecția greedy nu tratează tot grupul cu același minim",
        "doar primul element este modificat când ar trebui actualizate toate cele cu valoarea minimă",
    ],
    "suboptimal_greedy_selection": [
        "alegerea greedy nu garantează soluția optimă",
        "criteriul de ordonare nu este corect pentru greedy",
        "alegere locală greșită care nu dă optimul global",
    ],
    "incorrect_output_extraction": [
        "rezultatul final nu este calculat sau extras corect",
        "ieșirea folosește o valoare greșită",
        "valoarea finală tipărită este alta decât cea calculată",
    ],
    "wrong_sort_criterion": [
        "sortarea este făcută după criteriul greșit",
        "ordinea elementelor nu permite greedy corect",
        "trebuie sortat după alt criteriu",
    ],
    "budget_tracking_error": [
        "bugetul rămas nu este actualizat corect",
        "costul cumulativ este calculat greșit",
        "se scade cost greșit din buget",
    ],
    "infinite_loop": [
        "bucla nu se oprește și intră în ciclu infinit",
        "condiția de terminare nu este îndeplinită niciodată",
    ],
    "wrong_data_structure": [
        "structura de date folosită nu permite operațiile necesare",
        "ar trebui folosit heap în loc de vector / map în loc de listă",
        "structura aleasă duce la complexitate prea mare",
    ],
    "missing_edge_case": [
        "nu se tratează un caz limită",
        "lipsește verificarea pentru cazul când lista e goală",
        "nu se gestionează cazul când N=0 sau N=1",
    ],
}


def _infer_missing_concept(
    primary_issue: str,
    key_op_info: dict,
    partial: dict,
    passing: dict,
) -> str:
    """Find concept whose anchor phrases are semantically closest to the issue.

    If no concept anchor reaches `CONCEPT_MATCH_THRESHOLD`, falls back to
    structural signals from key_op_info.
    """
    if primary_issue:
        best_concept: str | None = None
        best_sim = 0.0
        for concept, anchors in _CONCEPT_ANCHORS.items():
            sim = _max_sim_to_anchors(primary_issue, anchors)
            if sim > best_sim:
                best_sim = sim
                best_concept = concept
        if best_concept and best_sim >= CONCEPT_MATCH_THRESHOLD:
            return best_concept

    if key_op_info.get("diverge_at_step") == 0:
        return "wrong_initial_step"
    if key_op_info.get("length_difference", 0) > 2:
        return "missing_algorithmic_steps"
    return "implementation_logic_error"


# ── Main builder ───────────────────────────────────────────────────────────────

def build_gap_object(
    partial_segments: dict[str, Any],
    passing_segments: dict[str, Any],
    codebert_sim: float,
) -> dict[str, Any]:
    """Compare partial vs passing segments and return a structured gap_object."""
    sim_bucket = similarity_bucket(codebert_sim)

    if _check_intent_mismatch(partial_segments, passing_segments):
        return {
            "gap_level": "intent",
            "missing_concept": "problem_understanding",
            "evidence_partial": partial_segments.get("intent", {}).get("text", ""),
            "evidence_passing": passing_segments.get("intent", {}).get("text", ""),
            "primary_issue": "Studentul rezolvă o altă problemă",
            "similarity_bucket": sim_bucket,
            "hint_level": "macro",
            "ds_divergence": None,
            "key_op_divergence": None,
        }

    algo_gap = _compare_algorithms(partial_segments, passing_segments)
    if algo_gap:
        return {
            "gap_level": "function",
            "missing_concept": f"algoritm_gresit:{algo_gap['algorithm_partial']}_vs_{algo_gap['algorithm_passing']}",
            "evidence_partial": f"folosește {algo_gap['algorithm_partial']}",
            "evidence_passing": f"ar trebui să folosească {algo_gap['algorithm_passing']}",
            "primary_issue": _find_critical_issue(partial_segments),
            "similarity_bucket": sim_bucket,
            "hint_level": "structural",
            "ds_divergence": _check_data_structure_divergence(partial_segments, passing_segments),
            "key_op_divergence": None,
        }

    ds_gap = _check_data_structure_divergence(partial_segments, passing_segments)
    key_op_info = _compare_key_ops(partial_segments, passing_segments)
    primary_issue = _find_critical_issue(partial_segments)
    hint_level = hint_level_from_gap("implementation", codebert_sim)
    missing_concept = _infer_missing_concept(
        primary_issue, key_op_info, partial_segments, passing_segments
    )

    return {
        "gap_level": "implementation",
        "missing_concept": missing_concept,
        "evidence_partial": _summarize_ops(key_op_info["ops_partial"]),
        "evidence_passing": _summarize_ops(key_op_info["ops_passing"]),
        "primary_issue": primary_issue,
        "diverges_at_step": key_op_info["diverge_at_step"],
        "missing_steps_in_partial": key_op_info.get("missing_steps_in_partial", []),
        "op_match_scores": key_op_info.get("match_scores", []),
        "similarity_bucket": sim_bucket,
        "hint_level": hint_level,
        "ds_divergence": ds_gap,
        "output_logic_issue": _has_output_issue(partial_segments),
    }


def _summarize_ops(ops: list[str]) -> str:
    if not ops:
        return "nicio operație extrasă"
    if len(ops) <= 3:
        return " → ".join(ops)
    return " → ".join(ops[:3]) + f" (+ {len(ops)-3} mai multe)"


# ── Format for prompt injection ────────────────────────────────────────────────

def format_gap_for_prompt(gap: dict[str, Any]) -> str:
    """Format the gap_object as a Romanian prompt block."""
    lines = [
        "── ANALIZA DIFERENȚEI ──",
        f"Nivel gap:              {gap['gap_level']}",
        f"Concept lipsă:          {gap['missing_concept']}",
        f"Nivel hint:             {gap['hint_level']}",
        f"Similaritate cod:       {gap['similarity_bucket']}",
        "",
        f"Abordare parțială:      {gap.get('evidence_partial', 'N/A')}",
        f"Abordare corectă (100p):{gap.get('evidence_passing', 'N/A')}",
    ]
    if gap.get("primary_issue"):
        lines.append(f"Problemă principală:    {gap['primary_issue']}")
    if gap.get("diverges_at_step", -1) >= 0:
        lines.append(f"Divergență la pasul:    {gap['diverges_at_step'] + 1}")
    if gap.get("missing_steps_in_partial"):
        steps = gap["missing_steps_in_partial"]
        lines.append("Pași lipsă din parțială:")
        for s in steps[:4]:
            lines.append(f"  - {s}")
        if len(steps) > 4:
            lines.append(f"  - (+ {len(steps)-4} alți pași)")
    if gap.get("output_logic_issue"):
        lines.append("Logica ieșirii:         de asemenea incorectă")
    if gap.get("ds_divergence"):
        ds = gap["ds_divergence"]
        if ds.get("missing_in_partial"):
            lines.append(f"Structuri de date lipsă:{ds['missing_in_partial']}")
        if ds.get("extra_in_partial"):
            lines.append(f"Structuri de date în plus:{ds['extra_in_partial']}")
    lines.append("── SFÂRȘIT ANALIZĂ ──")
    return "\n".join(lines)


# ── Demo ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    partial_40 = {
        "intent": {
            "text": "calculează numărul maxim de monede pe oră după cheltuirea bugetului",
            "matches_problem": True,
            "confidence": "high",
        },
        "function": {
            "algorithm": "greedy",
            "data_structures": ["ArrayList"],
            "complexity": "O(n^2)",
            "confidence": "medium",
        },
        "implementation": {
            "key_operations": [
                "Sortează lista de calculatoare după capacitatea curentă (P).",
                "În fiecare iterație, identifică toate calculatoarele cu P minim, calculează suma costurilor de upgrade (U), și le crește P cu 1.",
                "Scade costul cumulat din bugetul rămas dacă este accesibil.",
                "După epuizarea bugetului, afișează P minim minus unu.",
            ],
            "potential_issues": [
                "Rezultatul final scade 1 din P minim, ceea ce este greșit.",
                "Bucla se oprește doar când suma necesară nu este strict mai mică decât bugetul; cazul de egalitate nu este tratat.",
                "Lista auxiliară subComputers este populată dar nu este folosită.",
                "Sortarea listei la fiecare iterație este ineficientă.",
            ],
            "confidence": "high",
        },
    }

    passing_100 = {
        "intent": {
            "text": "Maximizează minimul de monede pe oră peste toate calculatoarele",
            "matches_problem": True,
            "confidence": "high",
        },
        "function": {
            "algorithm": "greedy",
            "data_structures": ["ArrayList", "Collections.sort"],
            "complexity": "O(n*k)",
            "confidence": "medium",
        },
        "implementation": {
            "key_operations": [
                "ordonează calculatoarele după coinsPerHour curent",
                "determină coinsPerHour minim actual",
                "îmbunătățește toate calculatoarele care au acest minim cât timp bugetul permite",
                "deduce costul upgrade-ului din banii disponibili și incrementează coinsPerHour-ul lor",
                "repetă până când nu mai sunt posibile upgrade-uri",
                "afișează ultimul minim coinsPerHour atins",
            ],
            "potential_issues": [],
            "confidence": "medium",
        },
    }

    print("=" * 60)
    print("GAP: 40p vs 100p  (sim=0.72)")
    print("=" * 60)
    gap_40 = build_gap_object(partial_40, passing_100, codebert_sim=0.72)
    print(json.dumps(gap_40, indent=2, ensure_ascii=False))
    print()
    print("PROMPT BLOCK:")
    print(format_gap_for_prompt(gap_40))
