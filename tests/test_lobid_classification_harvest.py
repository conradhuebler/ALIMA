"""lobid classification harvest - Claude Generated (WP-D2 Phase 3).

Until now ``LobidSuggester.transform`` built the pool from the aggregation facet
alone and gave every entry ``classifications: {}`` — measured on three real runs,
the canonical field carried data for 0 of 5128 / 2364 / 1134 pool entries. lobid
ships the classifications all along, on the ``member`` records
(``subject[].notation`` with the system in ``source.label``): 4627 notations
across 248 cached responses, 2788 RVK / 761 DDC / 48 BK.

The attachment is a CO-OCCURRENCE heuristic — a record's notation classifies the
record, not each of its subject headings — so these tests pin the properties
that keep it honest: entries are weighted, marked ``cooccurrence``, capped, and
library-local systematics are dropped rather than mis-filed.
"""

from __future__ import annotations

import unittest

from src.core.search.providers.lobid.suggester import (
    _classifications_by_gnd_id,
    _entries_for_gnd_ids,
)


def _record(gnd_ids, notations):
    """A lobid member record: GND subjects in componentList, notations beside them."""
    subject = [
        {"componentList": [{"id": f"https://d-nb.info/gnd/{g}"} for g in gnd_ids]}
    ]
    for system_label, notation in notations:
        subject.append({"notation": notation, "source": {"label": system_label}})
    return {"subject": subject}


RVK = "RVK (Regensburger Verbundklassifikation)"
DDC = "Dewey-Dezimalklassifikation"
BK = "BK (Basisklassifikation)"
LOCAL = "Sachgruppen der DNB"


class TestHarvestFromMemberRecords(unittest.TestCase):
    def test_notations_attach_to_the_records_gnd_subjects(self):
        raw = {"member": [_record(["4035769-7"], [(RVK, "WI 4700")])]}
        harvested = _classifications_by_gnd_id(raw)
        self.assertEqual(harvested["4035769-7"][("RVK", "WI 4700")], 1)

    def test_repeated_pairing_across_records_is_counted(self):
        """The count IS the signal — it separates 13 observations from 1."""
        raw = {"member": [_record(["g1"], [(RVK, "WI 4700")]) for _ in range(13)]}
        self.assertEqual(_classifications_by_gnd_id(raw)["g1"][("RVK", "WI 4700")], 13)

    def test_duplicate_notation_within_one_record_counts_once(self):
        """One title is one observation, however often it repeats a notation."""
        raw = {"member": [_record(["g1"], [(RVK, "WI 4700"), (RVK, "WI 4700")])]}
        self.assertEqual(_classifications_by_gnd_id(raw)["g1"][("RVK", "WI 4700")], 1)

    def test_library_local_systematics_are_dropped(self):
        """An unregistered system must be dropped, not filed under a real one."""
        raw = {"member": [_record(["g1"], [(LOCAL, "720"), (RVK, "WI 4700")])]}
        pairs = set(_classifications_by_gnd_id(raw)["g1"])
        self.assertEqual(pairs, {("RVK", "WI 4700")})

    def test_records_without_both_halves_contribute_nothing(self):
        raw = {"member": [
            _record(["g1"], []),          # subjects, no notation
            _record([], [(RVK, "X")]),    # notation, no subject
        ]}
        self.assertEqual(_classifications_by_gnd_id(raw), {})

    def test_missing_member_key_is_not_an_error(self):
        self.assertEqual(_classifications_by_gnd_id({"aggregation": {}}), {})


class TestEntriesForOneSubject(unittest.TestCase):
    def _harvest(self):
        raw = {"member":
            [_record(["g1"], [(RVK, "WI 4700")]) for _ in range(13)]
            + [_record(["g1"], [(RVK, "WI 4800")]) for _ in range(6)]
            + [_record(["g1"], [(RVK, "AR 22480")])]
            + [_record(["g1"], [(DDC, "551.48")]) for _ in range(4)]
            + [_record(["g1"], [(BK, "38.00")])]
        }
        return _classifications_by_gnd_id(raw)

    def test_entries_are_ordered_strongest_first(self):
        entries = _entries_for_gnd_ids(self._harvest(), {"g1"})
        self.assertEqual(
            [e["code"] for e in entries["RVK"]], ["WI 4700", "WI 4800", "AR 22480"]
        )
        self.assertEqual(entries["RVK"][0]["count"], 13)

    def test_limit_applies_per_system_not_overall(self):
        """A cap across all systems would let a strong RVK crowd out the DDC."""
        entries = _entries_for_gnd_ids(self._harvest(), {"g1"}, limit=2)
        self.assertEqual([e["code"] for e in entries["RVK"]], ["WI 4700", "WI 4800"])
        self.assertEqual([e["code"] for e in entries["DDC"]], ["551.48"])
        self.assertEqual([e["code"] for e in entries["BK"]], ["38.00"])

    def test_everything_harvested_is_marked_cooccurrence(self):
        """It is evidence, never an authority statement — consumers must see that."""
        for entries in _entries_for_gnd_ids(self._harvest(), {"g1"}).values():
            for entry in entries:
                self.assertEqual(entry["origin"], "cooccurrence")
                self.assertIn("count", entry)

    def test_several_gnd_ids_combine_by_max_not_sum(self):
        """A subject can carry merged ids backed by the SAME records."""
        raw = {"member":
            [_record(["g1", "g2"], [(RVK, "WI 4700")]) for _ in range(5)]
        }
        entries = _entries_for_gnd_ids(_classifications_by_gnd_id(raw), {"g1", "g2"})
        self.assertEqual(entries["RVK"][0]["count"], 5, "counts were summed")

    def test_subject_without_harvested_ids_gets_nothing(self):
        self.assertEqual(_entries_for_gnd_ids(self._harvest(), {"unknown"}), {})


class TestTransformIntegration(unittest.TestCase):
    def test_pool_entries_receive_the_harvest(self):
        from src.core.search.providers.lobid.suggester import LobidSuggester

        suggester = LobidSuggester.__new__(LobidSuggester)
        suggester.gnd_subjects = {"4035769-7": "Limnologie"}
        suggester.debug = False

        raw = {
            "aggregation": {
                "subject.componentList.id": [
                    {"key": "https://d-nb.info/gnd/4035769-7", "doc_count": 9}
                ]
            },
            "member": [_record(["4035769-7"], [(RVK, "WI 4700")]) for _ in range(3)],
        }
        out = suggester.transform(raw)

        self.assertEqual(out["Limnologie"]["count"], 9)
        self.assertEqual(
            out["Limnologie"]["classifications"]["RVK"],
            [{"code": "WI 4700", "count": 3, "origin": "cooccurrence"}],
        )

    def test_response_without_members_still_yields_the_pool(self):
        """The aggregation-only path must keep working — it is the majority."""
        from src.core.search.providers.lobid.suggester import LobidSuggester

        suggester = LobidSuggester.__new__(LobidSuggester)
        suggester.gnd_subjects = {"4035769-7": "Limnologie"}
        suggester.debug = False
        out = suggester.transform({
            "aggregation": {
                "subject.componentList.id": [
                    {"key": "https://d-nb.info/gnd/4035769-7", "doc_count": 2}
                ]
            }
        })
        self.assertEqual(out["Limnologie"]["count"], 2)
        self.assertEqual(out["Limnologie"]["classifications"], {})


if __name__ == "__main__":
    unittest.main()
