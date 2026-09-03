# ALIMA Workflow Definitions

YAML-driven workflows consumed by `WorkflowExecutor`.

## `status:` — operator-assigned maturity

Every workflow YAML carries a top-level `status:`. The loader ignores it (it
survives in `WorkflowDef.raw`); `alima_cli.py workflows list` prints it.

- `tested` — the operator has run it on real material. **Only the operator sets
  this**, never an agent, and never on the strength of a green test suite.
- `research` — everything else: usable, not signed off.

Currently `tested`: `alima_v51.yaml`, `alima_v51_105.yaml`. Adding a workflow
means adding `status: "research"`.

## Core Workflows

- **`alima_classic.yaml`** — Classic linear pipeline (5 rigid steps).
- **`alima_classic_v51.yaml`** — Classic pipeline with v5.1 prompt tuning.
- **`alima_v51.yaml`** — Default agentic v5.1 pipeline.

## Institution-Specific Variants

- **`alima_v51_105.yaml`** — UB Freiberg variant of `alima_v51.yaml`.
  - **Only difference:** classification prompt restricts RVK to economics/business (WiWi); non-WiWi subjects use DK only. DDC is not used in Freiberg.
  - Proof-of-concept for local, prompt-level institution customization.

## RVK via `rvk_lookup` tool (both v51 + v51_105)

- `dk_collect.config.rvk_inline: false` → the deterministic DK step collects
  DK/DDC only (no RVK anchor derivation, no RVK-API calls inline).
- `classification` is a tool-calling step: `tools: [rvk_lookup]`,
  `llm.max_iterations: 4`, and an extra `final_keywords` input passed to the
  tool. RVK is fetched on demand by the classification LLM.
  - `alima_v51`: LLM calls `rvk_lookup` whenever RVK fits thematically.
  - `alima_v51_105`: prompt instructs the LLM to call `rvk_lookup` **only for
    WiWi**; non-WiWi → no tool call, DK only.
- **Sync rule:** the non-prompt wiring (`rvk_inline`, `tools`,
  `max_iterations`, `final_keywords` input) is identical in both files; only
  the RVK prompt block differs (general vs. WiWi-only).

## Erschließungsregeln (both v51 + v51_105)

- **Gesamtdarstellung**: ein Fach in seiner Breite → Notation/Schlagwort des
  **Fachgebiets**, nicht je eines pro aufgezähltem Teilgebiet.
- **Zehn ist Obergrenze, nicht Ziel** — 3–5 Notationen die Regel, 2–3 bei
  Überblickswerken.
- **Formnotationen** (DK 378.245 & Co.) sind keine Sachnotationen: Prompt-Regel
  im `classification`-Step, Registry in `classification_systems.FORM_NOTATIONS`,
  Markierung in `build_structured_classifications`.
- **`core_keywords`/`form_keywords`**: der `selection`-Step benennt zusätzlich
  einen RSWK-Kern (2–5) und die Formschlagwörter; `keywords` bleibt die
  Retrieval-Liste. `verify_keywords` richtet beide am verifizierten Pool aus.
- **`rvk_lookup`** bekommt nur Sachschlagwörter — „Lehrbuch" holt sonst den
  Lehrbuch-Ast des falschen Fachs. Die Rückgabe trägt Label + Ancestor-Path
  (Plugin `rvk_api`, sonst RVK-API direkt) und ist eine Vorschlags-, keine
  Übernahmeliste.
- **`rvk_guard`** (Step 6b, deterministisch) verwirft RVK, die `rvk_lookup` in
  diesem Lauf nicht geliefert hat — Autorität ist `tool_log[].result_full`, nicht
  die Abschrift des Modells. Agentisches Gegenstück zu
  `_filter_final_rvk_classifications`. Kein Tool-Aufruf ⇒ kein autorisiertes RVK.
  `dk_postprocess` liest danach `${dk_classifications}`, nicht mehr die
  Step-Ausgabe.
- **`rank: core|additional`** je Notation, **je System getrennt** (DK, DDC und
  RVK haben je einen Kern). Vokabular + Sortierung in
  `classification_systems.normalize_rank`/`rank_sort_key`, getragen von
  `KeywordAnalysisState.classification_entries`. Fehlt das Feld, bleibt die
  Liste unsortiert — es wird kein Kern erfunden.

## MetaAgent: generic core + YAML domain rules

- The MetaAgent (PLAN→EXECUTE→REFLECT) is always active in agentic mode. It
  serializes the `steps:` workers (planner) and checks results (reflection).
- **`depends_on` gates the planner's pick**: a step whose prerequisite has not
  run is redirected to that prerequisite (`MetaAgent._unmet_dependency`). Before
  that gate a planner could run `classification` before `selection`, leaving
  `${extra.final_keywords}` empty.
- **The planner + reflection cores are GENERIC (ALIMA-branded) in code** — they
  carry no GND/DK/phase assumptions. Each prompt has a `{workflow_rules}` slot.
  Domain rules live in the **workflow YAML** and are injected there by
  `MetaAgent._inject_rules` (fills the slot, else appends).
  - `meta_agent.planning.rules` — phase/routing rules (which step needs GND/DK…).
  - `meta_agent.reflection.rules` — per-phase quality checks (DK depth, missing…).
  - `rules:` **augments** the generic base (preferred); `system_prompt:` would
    fully replace it. Keep planner + reflection rules in sync (same phase model).
- **Reflection is opt-in**: the gate runs **only** if a `meta_agent.reflection:`
  block exists. No block → pure PLAN→EXECUTE chain (planner alone finishes).
  `alima_classic`, `catalog_search`, … have no block → no reflection.
- The `reflection:` block is self-contained: `model`/`provider` (empty = main
  model), `temperature`/`top_p`/`max_tokens`, `rules`, optional
  `system_prompt`/`user_prompt`. The reflection **user prompt** (state dump,
  code default) feeds the **real DK codes** + a deterministic `has_deep_dk`
  flag — so the gate stops demanding "deeper" DK on already-deep codes.
- Planner and reflection stay **two separate prompts** on purpose (cross-check).
- Missing-concept reruns (`search_missing`) are reflection-driven: clean terms
  are searched targeted-and-merged; concepts flagged "kein GND" are skipped.

## Default Workflow

- Configurable via `SystemConfig.default_workflow` (`alima_v51` by default).
- GUI: Settings → System → Standard-Workflow.
- Used by `PipelineConfig.create_from_provider_preferences()` and respected by the Qt6 Pipeline-Tab and the webapp workflow dropdown.

## Token Budget

- Each step's `llm.max_tokens` in the YAML is the default (32768 throughout both v5.1 workflows, reflection included; code fallback is 32768 too).
- `PipelineConfig.global_max_tokens_override` outranks it for every step of a run: GUI toolbar "Budget", CLI `--max-tokens`, webapp select "Budget". Unset = the YAML decides.
- A reasoning model spends this budget on its thinking channel before the answer starts; measurements in [`src/llm/CLAUDE.md`](../src/llm/CLAUDE.md), probe: `scripts/probe_thinking.py`.

## Persönliche Zusatzregeln (nicht in dieser Datei)

- Regeln, die nur für einen Rechner oder eine Einrichtung gelten, gehören
  **nicht** in eine Workflow-YAML, sondern nach `~/.config/alima/rules.yaml`.
  Sie werden je nach Geltungsbereich (Workflow × Step) an die Systemprompts
  angehängt, ohne dass eine Repository-Datei geändert wird.
- Ein Lauf schreibt die verwendeten Regeln nach `applied_rules` ins Ergebnis —
  wichtig, weil dieselbe YAML auf zwei Rechnern damit anders läuft.
- Details: [`docs/user_rules.md`](../docs/user_rules.md).

## Maintenance Rule

- **`alima_v51_105.yaml` must stay in sync with `alima_v51.yaml`** except for the institution-specific classification prompt block.
- When updating `alima_v51.yaml`, apply the same non-prompt changes to `alima_v51_105.yaml`.
- The `meta_agent:` block (incl. `reflection:`) is identical in both files — keep it in sync.
- If the two files diverge beyond the classification prompt, that divergence is a bug.
