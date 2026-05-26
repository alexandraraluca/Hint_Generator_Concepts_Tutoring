"""Prompt building for the LLM bootstrap stage.

The system prompt encodes the **strict rubric** the user gave us
(criteria a-g + cap of 4 hints) and instructs the model to return JSON
matching `HINT_RUBRIC_SCHEMA`. The user prompt assembles the necessary
context: enunț, common_pitfalls, primary_concept, expected_complexity,
verdict-level info, and the failing code snippet.
"""

from __future__ import annotations

import textwrap
from typing import Any


HINT_RUBRIC_BULLETS = textwrap.dedent(
    """
    Reguli STRICTE pentru un hint bun:

    (a) Minimal information - dezvăluie exact cât e nevoie ca să deblocheze
        gândirea, NICIODATĂ să nu dea soluția.
    (b) Self-contained - fiecare hint citit izolat e util.
    (c) NO CODE - doar raționament/matematică, FĂRĂ secvențe de cod, FĂRĂ
        nume concrete de funcții/variabile preluate din codul utilizatorului.
        Niciodată '{', ';', for(, while(, identificatori cu paranteze.
    (d) Not a reformulation - nu repeta enunțul; dezvăluie structură ascunsă.
    (e) Strictly weaker than the solution - dă 30-60% din informația
        soluției, niciodată tot.
    (f) Short - 1-3 propoziții per hint.
    (g) Ordered by information density - hint 1 = cel mai macro / abstract,
        fiecare următor mai specific. Niciun hint fără să aducă info nouă.
    (h) Total: între 1 și 4 hinturi (fără să umpli artificial).
    """
).strip()

# Bloc comun cu build_system_prompt — același JSON și aceleași reguli ca la bootstrap.
_JSON_OUTPUT_INSTRUCTIONS = textwrap.dedent(
    """
    Întoarce STRICT JSON, fără text suplimentar înainte sau după, cu
    forma:

    {
      "hints": [
        { "level": "macro",      "text": "..."},
        { "level": "structural", "text": "..."},
        { "level": "specific",   "text": "..."}
      ],
      "concepts_targeted": ["concept_id_1", "concept_id_2"],
      "rationale_short": "1 propoziție: de ce hint-urile alese sunt cele potrivite"
    }

    - 'level' ∈ {"macro", "structural", "specific", "very_specific"}.
    - Numărul de hinturi între 1 și 4 (decide tu în funcție de complexitate).
    - IMPORTANT: array-ul "hints" poate avea cel mult 4 elemente — niciodată 5 sau mai multe;
      dacă ai mai multe idei, combină-le în mai puține hinturi.
    - Scrie hinturile în limba română.
    - 'concepts_targeted' folosește id-uri din lista 'concepts_dag'
      furnizată în user prompt.
    """
).strip()


def build_system_prompt(
    *,
    rubric_block: str = HINT_RUBRIC_BULLETS,
) -> str:
    return textwrap.dedent(
        f"""
        Ești un tutore expert la cursul Programarea Algoritmilor (PA),
        Universitatea Politehnica București. Sarcina ta: să formulezi 1-4
        HINTURI graduale care să ajute un student blocat la o temă, fără
        să-i dai soluția.

        {rubric_block}

        {_JSON_OUTPUT_INSTRUCTIONS}
        """
    ).strip()


def build_user_prompt(
    *,
    problem_meta: dict[str, Any],
    statement_excerpt: str,
    failing_code: str,
    verdict: str,
    issues: list[str],
    failing_test_size: str | None = None,
    valid_concept_ids: list[str] | None = None,
    error_l2: str | None = None,
    error_l3: str | None = None,
) -> str:
    pitfalls = "\n".join(f"- {p}" for p in problem_meta.get("common_pitfalls", []))
    if not pitfalls:
        pitfalls = "(fără capcane preadnotate)"

    concept_block = ""
    if valid_concept_ids:
        concept_block = (
            "concepts_dag (id-uri permise pentru concepts_targeted): "
            + ", ".join(valid_concept_ids)
        )

    err_block = ""
    if error_l2 or error_l3:
        err_block = (
            "Erori inferate (clasificator):\n"
            f"- L2: {error_l2 or 'n/a'}\n"
            f"- L3: {error_l3 or 'n/a'}\n"
        )

    failing_test_block = ""
    if failing_test_size:
        failing_test_block = (
            f"Testul picat este de tip '{failing_test_size}' "
            "(folosește această informație ca să decizi dacă problema este "
            "de complexitate sau de logică).\n"
        )

    return textwrap.dedent(
        f"""
        problem_id: {problem_meta["problem_id"]}
        title: {problem_meta.get("title", "")}
        primary_concept: {problem_meta.get("primary_concept", "")}
        concepts: {", ".join(problem_meta.get("concepts", []))}
        difficulty: {problem_meta.get("difficulty", "")}
        expected_complexity: {problem_meta.get("expected_complexity", "")}

        Capcane tipice ale problemei:
        {pitfalls}

        Enunț (extras):
        {statement_excerpt[:2500]}

        Verdict checker: {verdict}
        Issues: {", ".join(issues) if issues else "(none)"}
        {failing_test_block}{err_block}
        {concept_block}

        Codul studentului (limba inferată):
        ```code
        {failing_code[:5000]}
        ```

        Întoarce JSON cu hinturile graduale conform regulilor.
        """
    ).strip()


def build_silver_pair_user_prompt(
    *,
    problem_meta: dict[str, Any],
    statement_excerpt: str,
    failing_code: str,
    reference_passing_code: str,
    verdict: str,
    issues: list[str],
    valid_concept_ids: list[str] | None,
    codebert_similarity: float,
    diff_summary: str,
    diff_unified_excerpt: str,
    passing_file_hint: str,
    prefer_min_hints: int = 3,
) -> str:
    """User prompt for silver (failing vs 100p) pairs: grounds the LLM in both sources.

    Rubrica rămâne aceeași (fără cod în hinturi); modelul primește însă diferențele
    concrete între cele două surse ca să nu răspundă doar generic.
    """

    ref_cap = 4500
    # ref = reference_passing_code[:ref_cap]
    # dex = diff_unified_excerpt[:2200]


    # concept_block = ""
    # if valid_concept_ids:
    #     concept_block = (
    #         "concepts_dag (id-uri permise pentru concepts_targeted): "
    #         + ", ".join(valid_concept_ids)
    #     )

    # pitfalls = "\n".join(f"- {p}" for p in problem_meta.get("common_pitfalls", []))
    # if not pitfalls:
    #     pitfalls = "(fără capcane preadnotate)"

    return textwrap.dedent(
        f"""
        CONTEXT:
        # Ai un cod GREȘIT (student) și unul CORECT (referință).
        Ai un cod gresit si diferentele dintre el si codul de referinta (diff_unified_excerpt) sub forma unui unified diff.
        Scopul NU este să explici soluția corectă, ci să explici DE CE codul studentului este greșit.

        ────────────────────
        OBIECTIV:
        - Identifică eroarea principală din codul studentului
        - Ghidează studentul să o descopere singur
        - NU transforma soluția de referință în hinturi

        IMPORTANT:
        - Ignoră diferențele superficiale din diff
        - Caută diferențe de idee (algoritm, modelare, complexitate)
        - Dacă codul este conceptual greșit → concentrează-te DOAR pe asta
        -Nu furniza cod in hinturi, dar poți face referire la structura codului (ex: "bucla care iterează peste noduri" sau "partea care sortează array-ul")

        ────────────────────
        SIMILARITATE CODEBERT (cosinus, cod student vs. referință 100 pct):
        Scor: {codebert_similarity:.4f}
        Fișier referință: {passing_file_hint}
        Rezumat diff (linii normalizate): {diff_summary}

        INTERPRETARE SIMILARITATE CODEBERT:
        - Similaritate > 0.98 → codul este aproape corect; caută bug-uri locale sau detalii de implementare.
        - Similaritate între 0.90 și 0.98 → există diferențe structurale; concentrează-te pe organizarea logicii.
        - Similaritate < 0.90 → abordarea este probabil greșită; concentrează-te pe ideea algoritmică.

        Folosește această informație pentru a decide nivelul hinturilor.
        NU ignora acest semnal.


        ────────────────────
        ENUNȚ (extras):
        {statement_excerpt[:2000]}

        ────────────────────
        VERDICT: {verdict}
        Issues: {", ".join(issues) if issues else "(none)"}



        Codul studentului (varianta care NU are 100 pct):
        ```code
        {failing_code[:5000]}
        ```





        În JSON, lista "hints" are cel mult 4 elemente (inclusiv la probleme complexe).

        Întoarce JSON cu hinturile graduale conform regulilor.
        """
    ).strip()


def build_system_prompt_silver(
    *,
    rubric_block: str = HINT_RUBRIC_BULLETS,
) -> str:
    """System prompt pentru silver: aceeași schemă JSON ca `build_system_prompt` (bootstrap)."""
    return textwrap.dedent(
        f"""
        Ești un tutore expert la cursul Programarea Algoritmilor (PA),
        Universitatea Politehnica București.

        Scopul tău NU este să explici soluția corectă,
        ci să îl ajuți pe student să înțeleagă DE CE codul lui este greșit
        și cum să-și corecteze raționamentul.

        ────────────────────
        SIMILARITATE CODEBERT (indiciu primit în user prompt):
        Folosește similaritatea dintre coduri ca indiciu pentru tipul erorii:
        - similaritate mare → bug local / detaliu de implementare
        - similaritate mică → eroare conceptuală / abordare greșită

        Adaptează nivelul hinturilor (macro, structural, specific) în consecință.

        ────────────────────
        OBLIGATORIU (intern, dar reflectat în hinturi):
        - Identifică 1-2 erori principale din codul studentului (conceptuale sau de structură).
        - Dacă există o greșeală majoră de abordare, IGNORĂ bug-urile minore.
        - Hinturile trebuie să ghideze studentul către descoperirea acestor erori.

        ────────────────────

        NU sugera:
        - variabile noi
        - structuri de date noi
        - expresii concrete
        - formule
        - condiții exacte
        - pași de implementare

        Hinturile trebuie să descrie problema observată,
        nu modificarea exactă necesară.

        ────────────────────
        REGULI CRITICE:

        1. DEBUGGING > SOLUȚIE
           - Hinturile trebuie să explice DE CE codul studentului NU funcționează.
           - NU descrie pașii compleți ai soluției corecte.

        2. FĂRĂ SOLUTION LEAK
           - NU reformula soluția de referință.
           - Dacă hintul permite implementarea directă a soluției → este prea puternic.

        3. LEGĂTURĂ CU CODUL STUDENTULUI
           - Fiecare hint trebuie să fie justificabil prin ceva observabil în cod.
           - Evită hinturi care ar putea fi date fără a vedea codul.

        4. PRIORITIZARE ERORI
           - Dacă există o problemă de modelare (algoritm greșit),
             NU vorbi despre optimizări sau detalii minore.

        5. NIVELURI CORECTE
           - macro → ideea greșită
           - structural → unde în logică apare problema
           - specific → localizează precis regiunea logică problematică, fără a descrie modificarea necesară.

        6. CALITATE > CANTITATE
           - Dacă există o singură problemă majoră → 1-2 hinturi sunt suficiente.

        7. MAXIM 4 ELEMENTE în "hints"
           - În JSON, lista "hints" nu poate conține mai mult de 4 obiecte. Nu genera al 5-lea hint.

        ────────────────────
        {rubric_block}

        ────────────────────
        {_JSON_OUTPUT_INSTRUCTIONS}
        """
    ).strip()