"""Claude Generated - Tests for group_catalog_hits_by_title and
format_duplicate_report_markdown (title_list_search workflow support).

Pure-function tests: no LLM, no network, no registry ceremony needed.
"""

from __future__ import annotations

import unittest

from src.core.agents.deterministic_functions import (
    apply_deterministic_overrides,
    group_catalog_hits_by_title,
)
from src.utils.duplicate_report_formatter import (
    apply_isbn_duplicate_override,
    format_duplicate_report_markdown,
)


class TestGroupCatalogHitsByTitle(unittest.TestCase):
    def test_groups_hits_by_matching_query(self):
        wishlist = [
            {"title": "Book A", "year": "2026"},
            {"title": "Book B", "year": "2025"},
        ]
        catalog_hits = [
            {"query": "Book A", "rsn": "1", "web_url": "http://x/1"},
            {"query": "Book A", "rsn": "2", "web_url": "http://x/2"},
            {"query": "Book B", "rsn": "3", "web_url": "http://x/3"},
        ]
        result = group_catalog_hits_by_title(wishlist, catalog_hits)
        combined = result["combined"]
        self.assertEqual(len(combined), 2)
        self.assertEqual(combined[0]["title"], "Book A")
        self.assertEqual(len(combined[0]["catalog_matches"]), 2)
        self.assertEqual(combined[1]["title"], "Book B")
        self.assertEqual(len(combined[1]["catalog_matches"]), 1)
        # Original wishlist fields are preserved alongside catalog_matches
        self.assertEqual(combined[0]["year"], "2026")

    def test_title_with_no_hits_gets_empty_matches(self):
        wishlist = [{"title": "Unfindable Book"}]
        result = group_catalog_hits_by_title(wishlist, [])
        self.assertEqual(result["combined"][0]["catalog_matches"], [])

    def test_max_matches_per_title_caps_list(self):
        wishlist = [{"title": "Popular Title"}]
        catalog_hits = [{"query": "Popular Title", "rsn": str(i)} for i in range(20)]
        result = group_catalog_hits_by_title(wishlist, catalog_hits, max_matches_per_title=3)
        self.assertEqual(len(result["combined"][0]["catalog_matches"]), 3)

    def test_string_wishlist_items_supported(self):
        # extract_titles normally emits dicts, but the function should
        # tolerate bare strings defensively. - Claude Generated
        result = group_catalog_hits_by_title(["Plain Title"], [{"query": "Plain Title", "rsn": "1"}])
        self.assertEqual(result["combined"][0]["title"], "Plain Title")
        self.assertEqual(len(result["combined"][0]["catalog_matches"]), 1)

    def test_none_inputs_do_not_raise(self):
        result = group_catalog_hits_by_title(None, None)
        self.assertEqual(result["combined"], [])

    def test_non_dict_hits_are_skipped(self):
        wishlist = [{"title": "A"}]
        result = group_catalog_hits_by_title(wishlist, ["not a dict", {"query": "A", "rsn": "1"}])
        self.assertEqual(len(result["combined"][0]["catalog_matches"]), 1)


class TestFormatDuplicateReportMarkdown(unittest.TestCase):
    def test_empty_analysis_renders_zero_counts(self):
        md = format_duplicate_report_markdown([])
        self.assertIn("0 Duplikat", md)
        self.assertIn("von 0 Titeln", md)

    def test_counts_computed_from_actual_statuses_not_llm_summary(self):
        analysis = [
            {"input_title": "A", "status": "duplicate", "matches": []},
            {"input_title": "B", "status": "different_edition", "matches": []},
            {"input_title": "C", "status": "different_edition", "matches": []},
            {"input_title": "D", "status": "new", "matches": []},
        ]
        md = format_duplicate_report_markdown(analysis)
        self.assertIn("1 Duplikat", md)
        self.assertIn("2 neue Auflage", md)
        self.assertIn("1 neu", md)
        self.assertIn("von 4 Titeln", md)

    def test_headline_rollup_answers_how_many_need_action(self):
        # Reported gap: the per-status breakdown forced the reader to
        # manually add up which numbers meant "needs acquisition". - Claude Generated
        analysis = (
            [{"input_title": "D", "status": "duplicate", "matches": []}]
            + [{"input_title": f"DE{i}", "status": "different_edition", "matches": []} for i in range(4)]
            + [{"input_title": f"N{i}", "status": "new", "matches": []} for i in range(15)]
            + [{"input_title": f"NM{i}", "status": "no_match", "matches": []} for i in range(37)]
        )
        md = format_duplicate_report_markdown(analysis)
        headline = md.splitlines()[0]
        self.assertIn("56 von 57 Titeln", headline)
        self.assertIn("1 bereits ausreichend im Bestand", headline)

    def test_likely_duplicate_counts_as_already_covered(self):
        analysis = [
            {"input_title": "A", "status": "duplicate", "matches": []},
            {"input_title": "B", "status": "likely_duplicate", "matches": []},
            {"input_title": "C", "status": "no_match", "matches": []},
        ]
        md = format_duplicate_report_markdown(analysis)
        headline = md.splitlines()[0]
        self.assertIn("1 von 3 Titeln", headline)
        self.assertIn("2 bereits ausreichend im Bestand", headline)

    def test_related_work_counts_as_needs_action(self):
        analysis = [
            {"input_title": "A", "status": "related_work", "matches": []},
        ]
        md = format_duplicate_report_markdown(analysis)
        headline = md.splitlines()[0]
        self.assertIn("1 von 1 Titeln", headline)
        self.assertIn("0 bereits ausreichend im Bestand", headline)

    def test_table_includes_catalog_and_fulltext_links(self):
        analysis = [{
            "input_title": "Cadmium Toxicity Mitigation",
            "status": "different_edition",
            "matches": [{
                "year": "2024", "publisher": "Springer",
                "web_url": "https://katalog.ub.tu-freiberg.de/Record/0-1878699474",
                "resource_url": "https://doi.org/10.1007/978-3-031-47390-6",
            }],
            "reasoning": "Neue Auflage, nicht im Bestand.",
        }]
        md = format_duplicate_report_markdown(analysis)
        self.assertIn("Cadmium Toxicity Mitigation", md)
        self.assertIn("[Katalog](https://katalog.ub.tu-freiberg.de/Record/0-1878699474)", md)
        self.assertIn("[Volltext](https://doi.org/10.1007/978-3-031-47390-6)", md)
        self.assertIn("2024 (Springer)", md)

    def test_no_matches_renders_em_dash(self):
        analysis = [{"input_title": "Not Found", "status": "no_match", "matches": []}]
        md = format_duplicate_report_markdown(analysis)
        lines = [l for l in md.splitlines() if l.startswith("| Not Found")]
        self.assertEqual(len(lines), 1)
        self.assertIn("—", lines[0])

    def test_new_status_suppresses_unrelated_matches_links_and_edition(self):
        # Real reported bug: status="new" means the LLM explicitly determined
        # the catalog hit is a DIFFERENT, unrelated work — showing its
        # year/publisher/links in this row misleadingly reads as "the
        # wishlist title is already held". - Claude Generated
        analysis = [{
            "input_title": "The Global Origins of Capitalism",
            "input_metadata": {"authors": ["Benkler, Yochai"], "isbn": ""},
            "status": "new",
            "matches": [{
                "title": "The Origins of Modern Racism in the United States and "
                         "Black Economic Dysphoria Under Global Corporate Crony "
                         "Capitalism and the COVID Economic Lockdown Shock",
                "year": "2020", "publisher": "SSRN",
                "web_url": "https://katalog.example/Record/0-999",
                "resource_url": "https://ssrn.com/abstract=999",
            }],
            "reasoning": (
                "Der Wunschlistentitel 'The Global Origins of Capitalism' stimmt "
                "nicht mit dem Katalogtreffer überein. Es handelt sich um "
                "unterschiedliche Werke."
            ),
        }]
        md = format_duplicate_report_markdown(analysis)
        data_row = next(l for l in md.splitlines() if l.startswith("| The Global Origins"))
        self.assertNotIn("2020 (SSRN)", data_row)
        self.assertNotIn("katalog.example", data_row)
        self.assertNotIn("ssrn.com", data_row)
        self.assertNotIn("[Katalog]", data_row)
        self.assertNotIn("[Volltext]", data_row)
        # Author/ISBN columns (wishlist's own metadata) are unaffected.
        self.assertIn("Benkler, Yochai", data_row)
        # Editions/latest-edition/catalog-link/fulltext-link cells are all em-dash.
        cells = [c.strip() for c in data_row.split("|")]
        # cells: ['', title, authors, isbn, status, editions, latest, catalog, fulltext, reasoning, '']
        self.assertEqual(cells[5], "—")  # Gefundene Auflage(n)
        self.assertEqual(cells[6], "—")  # Neueste Auflage im Bestand
        self.assertEqual(cells[7], "—")  # Katalog-Link
        self.assertEqual(cells[8], "—")  # Volltext-Link

    def test_different_edition_status_still_shows_matches(self):
        # Regression guard: only "new"/"no_match" suppress match columns —
        # duplicate/likely_duplicate/different_edition must still show them.
        analysis = [{
            "input_title": "Real Match", "status": "different_edition",
            "matches": [{"year": "2020", "publisher": "Springer", "web_url": "http://x/1"}],
        }]
        md = format_duplicate_report_markdown(analysis)
        data_row = next(l for l in md.splitlines() if l.startswith("| Real Match"))
        self.assertIn("2020 (Springer)", data_row)
        self.assertIn("[Katalog](http://x/1)", data_row)

    def test_related_work_status_still_shows_matches(self):
        # related_work = genuinely relevant literature ABOUT the wishlist
        # title (e.g. a review article) — informative enough to keep the
        # link, unlike new/no_match which are fully suppressed. - Claude Generated
        analysis = [{
            "input_title": "The Empire of Climate: A History of an Idea",
            "input_metadata": {"authors": ["Livingstone, David N."], "isbn": ""},
            "status": "related_work",
            "matches": [{
                "title": "David Livingstone's The Empire of Climate: A History of An Idea",
                "year": "2025", "publisher": "Elsevier BV",
                "web_url": "https://katalog.example/Record/ai-1",
                "resource_url": "https://doi.org/xyz",
                "formats": ["ElectronicArticle"],
            }],
            "reasoning": (
                "Rezensionsaufsatz über das Werk (Format: ElectronicArticle, "
                "Autor nur einer von mehreren Beitragenden) — keine Ausgabe "
                "des Werks selbst."
            ),
        }]
        md = format_duplicate_report_markdown(analysis)
        data_row = next(l for l in md.splitlines() if l.startswith("| The Empire"))
        self.assertIn("Sekundärliteratur zum Werk", data_row)
        self.assertIn("2025 (Elsevier BV)", data_row)
        self.assertIn("[Katalog](https://katalog.example/Record/ai-1)", data_row)
        self.assertIn("[Volltext](https://doi.org/xyz)", data_row)

    def test_pipe_characters_in_reasoning_are_escaped(self):
        analysis = [{
            "input_title": "Title | With Pipe", "status": "new", "matches": [],
            "reasoning": "Contains a | pipe character that could break the table",
        }]
        md = format_duplicate_report_markdown(analysis)
        # The escaped pipes must not increase the column count of the data row.
        data_row = next(l for l in md.splitlines() if l.startswith("| Title"))
        self.assertEqual(data_row.count(" | "), 8)  # 9 columns => 8 internal separators
        self.assertIn("\\|", data_row)

    def test_multiple_matches_joined_with_br(self):
        analysis = [{
            "input_title": "Multi Edition Book", "status": "different_edition",
            "matches": [
                {"year": "2016", "publisher": "OldPub", "web_url": "http://x/1"},
                {"year": "2026", "publisher": "NewPub", "web_url": "http://x/2"},
            ],
        }]
        md = format_duplicate_report_markdown(analysis)
        data_row = next(l for l in md.splitlines() if l.startswith("| Multi Edition Book"))
        self.assertIn("2016 (OldPub)<br>2026 (NewPub)", data_row)
        self.assertIn("[Katalog](http://x/1)<br>[Katalog](http://x/2)", data_row)

    def test_authors_and_isbn_columns_from_input_metadata(self):
        analysis = [{
            "input_title": "Some Book", "status": "new", "matches": [],
            "input_metadata": {"authors": ["Müller, Hans", "Schmidt, Anna"], "isbn": "978-3-123456-78-9"},
        }]
        md = format_duplicate_report_markdown(analysis)
        data_row = next(l for l in md.splitlines() if l.startswith("| Some Book"))
        self.assertIn("Müller, Hans, Schmidt, Anna", data_row)
        self.assertIn("978-3-123456-78-9", data_row)

    def test_missing_isbn_renders_em_dash_not_placeholder_text(self):
        analysis = [{"input_title": "No ISBN Book", "status": "new", "matches": [],
                      "input_metadata": {"authors": [], "isbn": ""}}]
        md = format_duplicate_report_markdown(analysis)
        data_row = next(l for l in md.splitlines() if l.startswith("| No ISBN Book"))
        # authors and isbn cells should both fall back to em-dash
        cells = [c.strip() for c in data_row.split("|")]
        self.assertIn("—", cells)

    def test_missing_input_metadata_does_not_raise(self):
        analysis = [{"input_title": "No Metadata At All", "status": "new", "matches": []}]
        md = format_duplicate_report_markdown(analysis)
        self.assertIn("No Metadata At All", md)

    def test_latest_edition_column_picks_highest_year(self):
        analysis = [{
            "input_title": "Thiel Einführung X", "status": "different_edition",
            "matches": [
                {"year": "2016", "publisher": "Springer", "web_url": "http://x/2016"},
                {"year": "2023", "publisher": "Springer", "web_url": "http://x/2023"},
                {"year": "2019", "publisher": "Springer", "web_url": "http://x/2019"},
            ],
        }]
        md = format_duplicate_report_markdown(analysis)
        data_row = next(l for l in md.splitlines() if l.startswith("| Thiel"))
        self.assertIn("[2023 (Springer)](http://x/2023)", data_row)
        # only the latest edition's link appears in that dedicated column —
        # not 2016/2019 (those still appear in "Gefundene Auflage(n)" though).
        self.assertNotIn("[2016", data_row.split("2023 (Springer)")[-1].split("|")[0])

    def test_latest_edition_no_matches_renders_em_dash(self):
        analysis = [{"input_title": "No Hits", "status": "no_match", "matches": []}]
        md = format_duplicate_report_markdown(analysis)
        data_row = next(l for l in md.splitlines() if l.startswith("| No Hits"))
        cells = [c.strip() for c in data_row.split("|")]
        self.assertTrue(all(c == "—" for c in cells if c and c not in ("No Hits", "nicht im Bestand")))

    def test_latest_edition_ignores_unparseable_years(self):
        analysis = [{
            "input_title": "Messy Years", "status": "different_edition",
            "matches": [
                {"year": "n.d.", "publisher": "Unknown", "web_url": "http://x/nd"},
                {"year": "2020", "publisher": "Springer", "web_url": "http://x/2020"},
            ],
        }]
        md = format_duplicate_report_markdown(analysis)
        data_row = next(l for l in md.splitlines() if l.startswith("| Messy Years"))
        self.assertIn("[2020 (Springer)](http://x/2020)", data_row)


class TestApplyIsbnDuplicateOverride(unittest.TestCase):
    """Determinism hardening: exact ISBN identity forces status='duplicate'
    regardless of what the LLM said, since that's the one genuinely
    unambiguous rule in the whole classification (already stated in the
    analyze_duplicates prompt, but an LLM can apply it inconsistently
    across runs/models). - Claude Generated"""

    def test_exact_isbn_match_overrides_non_duplicate_status(self):
        analysis = [{
            "input_title": "A", "status": "different_edition",
            "input_metadata": {"isbn": "978-3-16-148410-0"},
            "matches": [{"isbn": "978-3-16-148410-0", "year": "2020"}],
            "reasoning": "Andere Auflage laut Jahr.",
        }]
        result = apply_isbn_duplicate_override(analysis)
        self.assertEqual(result["overridden_count"], 1)
        entry = result["analysis"][0]
        self.assertEqual(entry["status"], "duplicate")
        self.assertIn("Deterministisch korrigiert", entry["reasoning"])
        self.assertIn("Andere Auflage laut Jahr.", entry["reasoning"])  # original preserved

    def test_isbn_match_ignores_hyphens_and_case(self):
        analysis = [{
            "input_title": "A", "status": "new",
            "input_metadata": {"isbn": "978-3-16-148410-0"},
            "matches": [{"isbn": "9783161484100"}],  # same ISBN, no hyphens
        }]
        result = apply_isbn_duplicate_override(analysis)
        self.assertEqual(result["analysis"][0]["status"], "duplicate")

    def test_already_duplicate_status_not_double_counted(self):
        analysis = [{
            "input_title": "A", "status": "duplicate",
            "input_metadata": {"isbn": "978-3-16-148410-0"},
            "matches": [{"isbn": "978-3-16-148410-0"}],
        }]
        result = apply_isbn_duplicate_override(analysis)
        self.assertEqual(result["overridden_count"], 0)  # already correct, no "correction" to log

    def test_no_isbn_on_wishlist_side_no_override(self):
        analysis = [{
            "input_title": "A", "status": "new",
            "input_metadata": {"isbn": ""},
            "matches": [{"isbn": "978-3-16-148410-0"}],
        }]
        result = apply_isbn_duplicate_override(analysis)
        self.assertEqual(result["overridden_count"], 0)
        self.assertEqual(result["analysis"][0]["status"], "new")

    def test_different_isbn_no_override(self):
        analysis = [{
            "input_title": "A", "status": "new",
            "input_metadata": {"isbn": "978-3-16-148410-0"},
            "matches": [{"isbn": "978-0-13-468599-1"}],
        }]
        result = apply_isbn_duplicate_override(analysis)
        self.assertEqual(result["overridden_count"], 0)

    def test_missing_input_metadata_does_not_raise(self):
        analysis = [{"input_title": "A", "status": "new", "matches": []}]
        result = apply_isbn_duplicate_override(analysis)
        self.assertEqual(result["overridden_count"], 0)
        self.assertEqual(result["analysis"][0]["status"], "new")

    def test_non_dict_entries_pass_through_unchanged(self):
        result = apply_isbn_duplicate_override(["not a dict", None])
        self.assertEqual(result["analysis"], ["not a dict", None])
        self.assertEqual(result["overridden_count"], 0)

    def test_empty_analysis(self):
        result = apply_isbn_duplicate_override([])
        self.assertEqual(result, {"analysis": [], "overridden_count": 0})

    def test_registered_tool_fn_streams_notice_when_overriding(self):
        analysis = [{
            "input_title": "A", "status": "new",
            "input_metadata": {"isbn": "123"},
            "matches": [{"isbn": "123"}],
        }]
        messages = []
        out = apply_deterministic_overrides(analysis, stream_callback=messages.append)
        self.assertEqual(out["analysis"][0]["status"], "duplicate")
        self.assertTrue(any("ISBN" in m for m in messages))

    def test_registered_tool_fn_silent_when_no_override_needed(self):
        analysis = [{"input_title": "A", "status": "new", "matches": []}]
        messages = []
        apply_deterministic_overrides(analysis, stream_callback=messages.append)
        self.assertEqual(messages, [])


if __name__ == "__main__":
    unittest.main()
