# Tabel 1 — Compararea surselor de hinturi

Agregare pe datele existente din `data/hints/`. Metricile de similaritate
folosesc TF-IDF (1–2 grame) + cosinus față de enunț, respectiv codul
submisiei. Pragul rubricii pentru similaritate este 0,55.

| Sursă | Încercări | Valide | Rată validare | Probleme | Hinturi/medie | Cuvinte/hint | Mediană sim→enunț | Mediană sim→cod | Top violări |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| Bootstrap LLM | 556 | 489 | 87.9% | 35 | 3.15 | 30.2 | 0.142 | 0.009 | code_token_match (26), too_short_words (25), schema (17) |
| Silver (perechi) | 293 | 143 | 48.8% | 33 | 3.53 | 30.1 | 0.122 | 0.035 | too_short_words (36), llm_error (31), order_inversion_at_3 (16) |

**Note:**
- *Bootstrap* = `llm_bootstrap.py` (enunț + cod failing, fără pereche 100p).
- *Silver* = `silver_hints.py` (pereche failing→passing același student, CodeBERT + diff).
- Seturile de cazuri nu sunt identice; silver acoperă doar studenți cu traiectorie failing→100p.
- `sim→enunț` / `sim→cod` = mediană `max_sim_to_statement` / `max_sim_to_solution` per set.
