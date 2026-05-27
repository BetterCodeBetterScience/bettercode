# Testing chapter conversion notes

Running log of conversion decisions, edge cases, and patterns discovered
during conversion of `book-testing.tex`. Becomes raw material for the
future `converting-chapter-listings` skill.

## Block count note

The implementation plan referred to "158 lstlisting blocks" but this
was a miscount. The grep command
`grep -n "begin{lstlisting}\|end{lstlisting}" latex/book-testing.tex`
produces **158 output lines** (79 `\begin` + 79 `\end`), representing
**79 distinct lstlisting blocks**. The catalog below covers all 79.

---

## Catalog

| # | Lines | Style | Type | Planned action | Notes |
|---|-------|-------|------|----------------|-------|
| ✅ 1 | 72–87 | Python | code | extract → `src/bettercode/testing/escape_velocity_v1.py` (full file) | Initial version, no validation. Shows `import math` and `import numpy as np`; v1 only needs `math`. Pedagogical import mismatch — hide `numpy` line via `firstline` or add preamble file. Pattern 1 (function + test). |
| ✅ 2 | 93–101 | Python | code | extract → `tests/testing_chapter/test_escape_velocity.py::test_escape_velocity` | Tests v1 (no-validation version); imports `escape_velocity` from v1. Pattern 1. |
| 3 | 109–109 | Pythonshort | inline-keep | keep inline | `assert <statement>, <message>` syntax pattern only. |
| 4 | 117–122 | shell | output | manifest entry: `pytest_escape_velocity_pass` | Passing pytest run for block 2 test. Normalizers: pytest_timings, file_paths, terminal_width. |
| ✅ 5 | 129–154 | shell | output | manifest entry: `pytest_escape_velocity_typo_fail` | Failing run — `ev_expected = 1186.0` typo. Shows full failure traceback. Normalizers: pytest_timings, file_paths, memory_addresses, terminal_width. Requires intentionally-failing test file `tests/testing_chapter/_demos/test_escape_velocity_typo.py`. |
| ✅ 6 | 160–163 | repl | output | manifest entry: `repl_float_assert_fail` (REPL driver) | `assert 0.1 + 0.2 == 0.3` raises AssertionError. Driver: `scripts/repl_drivers/testing/float_assert_fail.py`. Pattern 3a. |
| ✅ 7 | 169–170 | repl | output | manifest entry: `repl_float_value` (REPL driver) | Shows `0.1 + 0.2` = `0.30000000000000004`. Driver: `scripts/repl_drivers/testing/float_value.py`. Pattern 3a. Continues same conceptual session as block 6 but isolated driver. |
| ✅ 8 | 176–179 | repl | output | manifest entry: `repl_float_allclose` (REPL driver) | `import numpy as np; np.allclose(0.1 + 0.2, 0.3)` → True. Driver: `scripts/repl_drivers/testing/float_allclose.py`. Pattern 3a. |
| ✅ 9 | 191–204 | Python | code | extract → `src/bettercode/testing/escape_velocity_v2.py` (full file) | Adds `ValueError` for non-positive mass/radius. Pattern 1 (v2 function + test in block 10). |
| ✅ 10 | 211–216 | Python | code | extract → `tests/testing_chapter/test_escape_velocity.py::test_escape_velocity_negative` | Tests v2 ValueError guard. Imports from `escape_velocity_v2`. Pattern 1. |
| ✅ 11 | 234–263 | Python | code | extract → `src/bettercode/testing/find_outliers_v1_buggy.py` (full file) | `find_outliers` with divide-by-zero bug when all values equal. Pattern 1. |
| ✅ 12 | 269–272 | repl | output | manifest entry: `repl_find_outliers_normal` (REPL driver) | Normal call `find_outliers([1,2,3,1000,4,5,6])` → `[3]`. Driver: `scripts/repl_drivers/testing/find_outliers_normal.py`. Imports from `find_outliers_v1_buggy`. Pattern 3a. |
| ✅ 13 | 278–293 | repl | output | manifest entry: `repl_find_outliers_zero_div` (REPL driver) | All-equal data raises `ZeroDivisionError`. Driver: `scripts/repl_drivers/testing/find_outliers_zero_div.py`. Imports from `find_outliers_v1_buggy`. Pattern 3a. |
| ✅ 14 | 300–313 | Python | code | extract → `tests/testing_chapter/test_find_outliers.py` (both test functions) | `test_find_outliers_normal_case` and `test_find_outliers_identical_values`. Canonical file uses v2_fixed; demo file `_demos/test_find_outliers_v1_buggy.py` shows v1 buggy version that fails. Pattern 1. |
| ✅ 15 | 319–349 | shell | output | manifest entry: `pytest_find_outliers_bug_fail` | Failing pytest run for block 14 (ZeroDivisionError). `expected_exit_code: 1`. Normalizers: pytest_timings, file_paths, terminal_width. Run against `tests/testing_chapter/_demos/test_find_outliers_v1_buggy.py`. |
| ✅ 16 | 355–359 | Python | code | extract → `src/bettercode/testing/find_outliers_v2_fixed.py` (partial / fragment) | Only shows `if std == 0: return []` fix. This is a code fragment of v2; Pattern 2 — slice from full v2 file. Create `find_outliers_v2_fixed.py` and show via `firstline/lastline`. |
| 17 | 366–371 | shell | output | manifest entry: `pytest_find_outliers_fixed_pass` | Passing pytest run for both find_outliers tests against v2. Normalizers: pytest_timings, file_paths, terminal_width. Note: block references `pytest src/bettercode/bug_driven_testing.py` — likely meant `tests/test_find_outliers.py`; clarify path before generating output. |
| ✅ 18 | 408–422 | Python | code | extract → `src/bettercode/testing/simple_scaler.py` (full class) | `SimpleScaler` class definition. No version suffix needed (only one version shown). Pattern 1. |
| ✅ 19 | 428–435 | Python | code | extract → `tests/testing_chapter/test_simple_scaler.py::test_simple_scaler_interface` | Black-box interface test. Imports from `simple_scaler`. Pattern 1. |
| ✅ 20 | 441–449 | Python | code | extract → `tests/testing_chapter/test_simple_scaler.py::test_simple_scaler_internals` | Implementation-aware test (accesses `scaler.transformed_`). Same file as block 19 (Pattern 2 — both tests in one file, two blocks). |
| ✅ 21 | 469–474 | Python | code | extract → `src/bettercode/testing/dataframe_ops.py` (two functions) | `split_names(df)` and `get_initials(df)`. Pattern 2 — one file for the coupled functions. |
| ✅ 22 | 481–490 | Python | code | extract → `tests/testing_chapter/_demos/test_dataframe_ops_coupled.py` (coupled test block) | `people_df` global + `test_split_names` + `test_get_initials` (coupled/brittle tests demo). Placed in `_demos/` to keep out of normal collection (per Q3 rule: failure-demo goes in demos). Imports from `bettercode.testing.dataframe_ops`. |
| ✅ 23 | 498–511 | Python | code | extract → `tests/testing_chapter/test_dataframe_ops.py` (independent test block) | `get_people_df()` helper + `test_split_names_fullsetup` + `test_get_initials_fullsetup`. Canonical file; imports from `bettercode.testing.dataframe_ops`. |
| ✅ 24 | 526–535 | Python | code | extract → `src/bettercode/testing/distance.py` (full file) | `distance(p1, p2)` function. Pattern 1. |
| ✅ 25 | 542–558 | Python | code | extract → `tests/testing_chapter/_demos/test_distance_v1_failing.py` (initial AI-generated tests with wrong value) + `tests/testing_chapter/test_distance.py` (fixed version) | Separate files per Q4 resolution: v1_failing has `sqrt(125)` (wrong, in `_demos/`); canonical `test_distance.py` has `sqrt(89)` (correct). |
| 26 | 564–564 | shell | inline-keep | keep inline | Single `$ pytest tests/test_distance.py` command line with no output block — just the invocation. No output captured. |
| ✅ 27 | 571–577 | shell | output | manifest entry: `pytest_distance_fail` | Failure output for wrong `math.sqrt(125)` expected value. `expected_exit_code: 1`. Normalizers: pytest_timings, file_paths, terminal_width. Run against `tests/testing_chapter/_demos/test_distance_v1_failing.py`. |
| ✅ 28 | 586–591 | shell | output | manifest entry: `pytest_distance_pass` | Passing run after fixing the expected value to `math.sqrt(89)`. Normalizers: pytest_timings, file_paths, terminal_width. Run against `tests/testing_chapter/_demos/test_distance_v2_fixed.py` or canonical `tests/testing_chapter/test_distance.py`. |
| ✅ 29 | 603–618 | Python | code | extract → `tests/testing_chapter/_demos/test_escape_velocity_gpt4.py::test_escape_velocity_gpt4` | GPT-4-generated test with wrong Jupiter value; intentionally fails. Three-planet test. `expected_exit_code: 1` in manifest. Placed in `_demos/` per failing-demo convention. |
| 30 | 624–658 | shell | output | manifest entry: `pytest_escape_velocity_gpt4_fail` | Failure output for GPT-4 test (Jupiter value mismatch). `expected_exit_code: 1`. Normalizers: pytest_timings, file_paths, memory_addresses, terminal_width. |
| ✅ 31 | 694–701 | Python | code | extract → `tests/testing_chapter/test_textmining.py::test_get_PubMedIDs_for_query_check_valid` | First TDD test for PubMed query function. Requires live network. Pattern 1. |
| ✅ 32 | 707–714 | Python | code | extract → `tests/testing_chapter/test_textmining.py::test_get_PubMedIDs_for_query_check_empty` | Second TDD test for PubMed empty result. Same file as block 31 (Pattern 2). |
| 33 | 721–724 | Pythonshort | inline-keep | keep inline | Stub function skeleton `def get_PubMedIDs_for_query(...): return None` — TDD red-phase stub, not runnable as canonical code. |
| 34 | 730–761 | shell | output | manifest entry: `pytest_textmining_red_phase` | Failing pytest run for TDD red phase (both PubMed tests fail on stub). `expected_exit_code: 1`. Normalizers: pytest_timings, file_paths, terminal_width. Requires a separate stub file for the red-phase demo. Open question: is this worth generating or better kept inline? |
| ✅ 35 | 767–809 | Python | code | extract → `src/bettercode/testing/textmining.py` (full module, part 1) | `BASE_URL`, `get_PubMedIDs_for_query`, `format_pubmed_query_params`, `get_idlist_from_response`. This is the main textmining module. Pattern 2 — full file shown in one block. Note: existing `src/bettercode/textmining.py` may already have this code; check if rename/move is needed. |
| 36 | 823–828 | shell | output | manifest entry: `pytest_textmining_coverage` | Coverage report output (pytest-cov). Only shows the coverage table portion, not the test pass/fail. Normalizers: pytest_timings, file_paths, terminal_width. Requires `--cov` flag in command. |
| 37 | 835–837 | Pythonshort | inline-keep | keep inline | `else: raise ValueError("Bad request")` snippet — single block showing the uncovered line. |
| ✅ 38 | 845–852 | Python | code | extract → `tests/testing_chapter/test_textmining.py::test_get_PubMedIDs_for_query_check_badurl` | Bad-URL test to reach 100% coverage. Same file as blocks 31–32 (Pattern 2). Note: tex uses `base_url` kwarg; canonical uses `esearch_url` (consistent with function signature). |
| ✅ 39 | 873–877 | Python | code | extract → `tests/testing_chapter/test_textmining.py` (fixture `ids`) | `@pytest.fixture(scope="session") def ids()`. Part of same test file as blocks 31, 32, 38 (Pattern 2). |
| 40 | 883–885 | Pythonshort | inline-keep | keep inline | `def test_get_PubMedIDs_for_query_check_valid(ids): assert ...` — refactored form showing fixture usage. Syntax/concept pattern, not the final version of the test. |
| 41 | 905–917 | Python | code | keep inline (Q5 resolution: chapter does not show failure output for this block) | First `MockPubMedResponse` class — has bug (`def json()` missing `self`). Pattern 2 — fragment of same module file or a separate demo. Open question: separate `textmining_v1_bad_mock.py` or just show as inline example? |
| ✅ 42 | 924–939 | Python | code | superseded by block 43 — `ids_mocked` fixture not in canonical test file (collision resolved per open question). Block 42 is a pedagogical intermediate step; only block 43's version is in `test_textmining.py`. |
| ✅ 43 | 947–978 | Python | code | extract → `tests/testing_chapter/test_textmining.py` (mock fixture v2 + test, improved) | `mock_pubmed_api` fixture (setup only) + `test_get_PubMedIDs_for_query_check_valid_mocked` (action + assertion). Canonical version. |
| ✅ 44 | 995–999 | Python | code | extract → `src/bettercode/testing/textmining.py` (stub functions for TDD red phase) | `get_record_from_PubMedID` and `parse_year_from_PubMed_record` both `pass`. TDD red-phase stubs. Pattern 2 — fragment of same module. Stubs superseded by full implementations in block 46; shown via firstline/lastline in LaTeX. |
| ✅ 45 | 1006–1023 | Python | code | extract → `tests/testing_chapter/test_textmining.py` (fixtures + tests for get_record) | `valid_pmid` fixture + `pmid_record` fixture + `test_get_record_from_valid_PubMedID` + `test_get_record_from_invalid_PubMedID`. Pattern 2 — same test file. |
| ✅ 46 | 1029–1046 | Python | code | extract → `src/bettercode/testing/textmining.py` (`get_record_from_PubMedID` implementation) | Full implementation. Pattern 2 — same module file, line-range slice. |
| ✅ 47 | 1052–1065 | Python | code | extract → `tests/testing_chapter/test_textmining.py` (parse_year tests) | `test_parse_year_from_PubMed_record` + `test_parse_year_from_PubMed_record_empty`. Pattern 2 — same test file. |
| 48 | 1071–1073 | Pythonshort | inline-keep | keep inline | `parse_year_from_PubMed_record` one-liner implementation shown as short pattern. |
| ✅ 49 | 1080–1088 | Python | code | extract → `tests/testing_chapter/test_textmining.py` (testdata list for parametrize) | `testdata = [('17773841', 1944), ...]` list. Part of same test file (Pattern 2). |
| ✅ 50 | 1094–1099 | Python | code | extract → `tests/testing_chapter/test_textmining.py` (parametrized test) | `@pytest.mark.parametrize` test for `parse_year_from_pmid_parametric`. Pattern 2 — same test file. Live API call with `time.sleep`. |
| 51 | 1106–1113 | shell | output | manifest entry: `pytest_textmining_parametric` | Passing parametrized test run showing 7 parametric subtests. Normalizers: pytest_timings, file_paths, terminal_width. Requires live network. Open question: mock this or accept live-API dependency? |
| 52 | 1129–1131 | Pythonshort | inline-keep | keep inline | `linear_regression` one-liner using `np.c_[...]` and `np.linalg.inv(...)`. Syntax/concept pattern. |
| ✅ 53 | 1137–1147 | Python | code | extract → `src/bettercode/testing/linear_regression.py` (`linear_regression_verbose`) | Verbose version with `X_transpose`, `X_transpose_X`, etc. Pattern 1. |
| ✅ 54 | 1161–1180 | Python | code | extract → `src/bettercode/testing/linear_regression.py` (`_validate_input` + updated `linear_regression`) | Adds `_validate_input` and `validate` flag. Same file as block 53 (Pattern 2). Full file combining blocks 53 and 54. |
| ✅ 55 | 1186–1197 | Python | code | extract → `tests/testing_chapter/_demos/test_property_based_smoke.py` (`test_linear_regression_without_validation`) | `@given` hypothesis smoke test. Placed in `_demos/` because block 56 shows failure output (Q3 rule). |
| 56 | 1206–1247 | shell | output | manifest entry: `pytest_property_based_smoke_fail` | Failing hypothesis run (singular matrix). `expected_exit_code: 1`. Normalizers: pytest_timings, file_paths, memory_addresses, terminal_width. Note: hypothesis output includes `Falsifying example:` — may need additional normalizer for the numpy array repr. Open question: hypothesis database path may vary; needs `temp_paths` normalizer too. |
| ✅ 57 | 1257–1275 | Python | code | extract → `tests/testing_chapter/test_property_based.py` (`test_validate_input`) | `@given` test for `_validate_input`. Pattern 2. |
| ✅ 58 | 1284–1305 | Python | code | extract → `tests/testing_chapter/test_property_based.py` (`test_linear_regression_correctness`) | `@given` test comparing to scipy `linregress`. Same file as block 57 (Pattern 2). |
| 59 | 1340–1346 | yaml | inline-keep | keep inline | GitHub Actions `on:` trigger section — static config snippet. |
| 60 | 1353–1359 | yaml | inline-keep | keep inline | GitHub Actions `jobs:` section. |
| 61 | 1370–1376 | yaml | inline-keep | keep inline | GitHub Actions `steps:` checkout + install-uv section. |
| 62 | 1383–1391 | yaml | inline-keep | keep inline | GitHub Actions install + test steps. |
| 63 | 1423–1429 | shell | output | manifest entry: `pytest_cherry_pick_single` | Running a single named test from textmining suite. Normalizers: pytest_timings, file_paths, terminal_width. |
| ✅ 64 | 1448–1456 | Python | code | extract → `tests/testing_chapter/_demos/test_ordering_v1_natural.py` (unordered test pair) | `test_second` before `test_first` — pedagogical ordering example. In `_demos/` because block 65 shows natural-order output (Q9 rule). |
| 65 | 1462–1468 | shell | output | manifest entry: `pytest_ordering_natural_order` | Shows tests run in definition order (second before first). Normalizers: pytest_timings, file_paths, terminal_width. |
| ✅ 66 | 1474–1484 | Python | code | extract → `tests/testing_chapter/test_ordering.py` (with `@pytest.mark.order` marks) | Canonical file with `order(1)` / `order(2)` decorators. v1 unordered in `_demos/test_ordering_v1_natural.py` (Q9 rule). |
| 67 | 1490–1496 | shell | output | manifest entry: `pytest_ordering_controlled_order` | Shows tests now run first-before-second after adding marks. Normalizers: pytest_timings, file_paths, terminal_width. |
| 68 | 1516–1517 | toml | inline-keep | keep inline | `[tool.pytest.ini_options] addopts = " --ff"` — config snippet. |
| 69 | 1530–1534 | toml | inline-keep | keep inline | `[tool.pytest.ini_options] markers = [...]` — defines unit/integration markers. |
| ✅ 70 | 1540–1555 | Python | code | extract → `tests/testing_chapter/test_markers.py` | `test_unit1`, `test_unit2` with `@pytest.mark.unit`, `test_integration` with `@pytest.mark.integration`. Pattern 1. |
| 71 | 1561–1568 | shell | output | manifest entry: `pytest_markers_all` | All 3 tests passing. Normalizers: pytest_timings, file_paths, terminal_width. |
| 72 | 1574–1580 | shell | output | manifest entry: `pytest_markers_unit_only` | Only unit tests selected (`-m unit`). Normalizers: pytest_timings, file_paths, terminal_width. |
| 73 | 1586–1591 | shell | output | manifest entry: `pytest_markers_not_unit` | Integration test only (`-m "not unit"`). Normalizers: pytest_timings, file_paths, terminal_width. |
| ✅ 74 | 1603–1616 | Python | code | extract → `tests/testing_chapter/test_duration.py` (initial, unordered) | `test_duration_3`, `test_duration_5`, `test_duration_1` with `sleep()` calls. Pattern 1. No order marks (canonical shows natural definition order). |
| 75 | 1622–1632 | shell | output | manifest entry: `pytest_duration_report` | `--durations=0` output showing slowest durations. Normalizers: pytest_timings, file_paths, terminal_width. Note: total time ~9s — long-running output test; mark as slow in manifest. |
| 76 | 1645–1706 | transcript | inline-keep | keep inline | Claude Code agent interaction transcript (verbatim agentic session). This is a document artifact showing the AI session, not runnable code. Keep inline as `transcript` style. |
| ✅ 77 | 1733–1739 | Python | code | extract → `tests/testing_chapter/test_parallel.py` | `@pytest.mark.parametrize("x", range(10))` with `time.sleep(1)`. Pattern 1. |
| 78 | 1745–1750 | shell | output | manifest entry: `pytest_parallel_serial` | Serial run: 10 tests × 1s ≈ 10s. Normalizers: pytest_timings, file_paths, terminal_width. Long-running (~10s); mark as slow. |
| 79 | 1756–1760 | shell | output | manifest entry: `pytest_parallel_xdist` | Parallel run with `-n auto`: 10 tests in ~2s across 16 workers. Normalizers: pytest_timings, file_paths, terminal_width. Note: worker count (`16 workers`) is machine-specific — will need a `worker_count` normalizer or inline regex to strip the count. Open question: how to normalize `16 workers [10 items]` line? |

---

## Patterns observed during cataloging

- **Pattern 1 (function + test across two files):** ~20 occurrences. Resolution: separate `src/bettercode/testing/<concept>_vN.py` and `tests/testing_chapter/test_<concept>.py` files. The test file does a real `import`; both files are canonical runnable units.

- **Pattern 2 (one file, multiple LaTeX blocks via line ranges):** ~12 occurrences, particularly in the textmining section (blocks 35, 38, 39, 42–50) and linear_regression section (blocks 53–58). Resolution: one runnable file; LaTeX uses `firstline=/lastline=` to display slices. The textmining module (blocks 35, 41, 44, 46) is the most fragmented — the full file will contain all pieces, shown progressively.

- **Pattern 3a (independent REPL drivers per visual block):** 5 REPL sessions (blocks 6–8, 12–13), each assigned its own driver. Blocks 6–8 form a conceptual sequence (floating-point equality) but are shown as separate visual blocks in the chapter and should use independent drivers per the spec's preferred approach.

- **Pedagogical import mismatch (function block shows import not needed in canonical file):** 1 confirmed occurrence: block 1 shows `import numpy as np` in the `escape_velocity` function, but the function only uses `math`. Resolution choices: hide via `firstline=3`, add preamble file, or edit prose. Recommendation: hide via `firstline` (clean, no file duplication).

- **Multi-version code chains:** 3 chains observed:
  1. `escape_velocity_v1` (no validation, block 1) → `escape_velocity_v2` (with validation, block 9)
  2. `find_outliers_v1_buggy` (divide-by-zero, block 11) → `find_outliers_v2_fixed` (guards against zero std, block 16)
  3. `test_distance` initial (wrong expected value, block 25) → fixed (correct value, implied by block 28)

- **TDD red-phase stub blocks:** 2 instances (blocks 33 and 44) show stub implementations (`return None` / `pass`) for the purpose of demonstrating failing tests. These require careful handling — either kept inline (block 33 is `Pythonshort`, already inline-keep) or separated into a demo stub file used only for the failure-demo manifest entry.

- **Intentionally-failing test blocks:** 3 instances require `expected_exit_code: 1` in the manifest:
  - Block 5: `pytest_escape_velocity_typo_fail`
  - Block 15: `pytest_find_outliers_bug_fail`
  - Block 56: `pytest_property_based_smoke_fail`
  Plus blocks 27 and 30 (distance and GPT-4 test failures).

- **Non-Python static config blocks:** 6 YAML blocks (59–62) and 2 TOML blocks (68–69), all `inline-keep`. These are configuration examples without execution semantics.

- **Single transcript block:** Block 76 (lines 1645–1706) uses `transcript` style — a verbatim Claude Code agent session. Not runnable code; keep inline.

- **Long-running test outputs:** Blocks 75, 78, 79 (`test_duration`, `test_parallel` serial and xdist) include `sleep()` calls and will take 9–10 seconds to generate. Mark these manifest entries as slow and consider excluding from default CI runs.

---

## Open questions raised during cataloging

1. **Block count discrepancy:** The plan specifies "158 lstlisting blocks" but the actual count is 79. The grep command produces 158 *lines* (79 begin + 79 end). All 79 blocks are cataloged here. Confirm with author: does the number 158 appear anywhere in other documentation that needs updating?

2. **Block 17 — pytest command path mismatch:** The shell block at lines 366–371 shows `$ pytest src/bettercode/bug_driven_testing.py` but the tests were shown in `tests/test_find_outliers.py`. The correct command for the pass output should be `pytest tests/testing_chapter/test_find_outliers.py`. Clarify which path to use in the generated output.

3. **Blocks 22 and 23 — coupled vs. independent tests in the same file:** Block 22 shows pedagogically-bad coupled tests (global `people_df`); block 23 shows the fixed version. If both live in the same `test_dataframe_ops.py`, the coupled tests will break normal pytest runs. Options: (a) put the bad version in `_demos/` directory, (b) use `@pytest.mark.xfail`, (c) keep blocks 22–23 both inline. Recommend option (a) or (c).

4. **Block 25 — distance test with wrong value:** The AI-generated tests include `math.sqrt(125)` which is incorrect (should be `math.sqrt(89)`). The chapter shows this failure then the fix. Two options: (a) `test_distance_v1_failing.py` (wrong value, `expected_exit_code: 1`) and `test_distance_v2_fixed.py` (correct value, `expected_exit_code: 0`), or (b) use xfail on the wrong-value test within a single file. Decision needed before Tasks 12–13.

5. **Blocks 41–43 — MockPubMedResponse and mock fixture versions:** Block 41 shows a `MockPubMedResponse` class with a bug (`def json()` missing `self`). Block 42 shows a fixture with setup/execution conflation. Block 43 shows the correct version. Do blocks 41 and 42 need canonical files (for showing the bad version's output), or can they stay inline? If the chapter does not show the output of running block 42's code, they can be inline-keep.

6. **Block 34 — TDD red-phase pytest output for textmining:** The shell block shows failing tests against the stub. To generate this output canonically requires a stub version of the module to exist at test time. This is complex to arrange without creating a separate stub file that conflicts with the real module. Options: (a) keep block 34 inline, (b) create a `textmining_stub.py` used only for this demo. Recommend (a) — keep inline.

7. **Block 51 — parametrized textmining test requires live PubMed API:** `test_parse_year_from_pmid_parametric` makes 7 live API calls. For the manifest, this will need network access and a rate-limiting delay. Mark as integration/slow. Should this output be mocked or should the manifest entry accept a live-API dependency?

8. **Block 56 — hypothesis database and normalizers:** The property-based smoke test failure output includes a numpy array repr. This may vary slightly across runs (hypothesis minimization can find different minimal examples). Consider whether a `hypothesis_database` normalizer is needed, or whether the output should always show the same minimal example (achievable by seeding the hypothesis database). Also need `temp_paths` normalizer for any hypothesis temp paths.

9. **Block 66 — two versions of test_ordering:** Block 64 shows tests without `@pytest.mark.order`, block 66 shows them with the marks added. If both live in `tests/testing_chapter/test_ordering.py`, the file needs to be in the marked state (block 66 version) for the canonical test suite. The unmarked version (block 64) is only needed to generate the natural-order output. Options: `test_ordering_v1_natural.py` (no marks) and `test_ordering_v2_marked.py` (with marks), or keep block 64 inline.

10. **Block 79 — worker count normalization:** The xdist output line `16 workers [10 items]` is machine-specific. A `worker_count` normalizer is needed: replace `\d+ workers` with `N workers`. This normalizer is not in the spec's initial set and needs to be added to `scripts/normalize_output.py`.

---

## Test file collection convention (decided 2026-05-27)

Intentionally-failing test files (demos) are placed under `tests/testing_chapter/_demos/`.
Normal pytest collection is configured to skip this directory via `pyproject.toml`:

```toml
[tool.pytest.ini_options]
norecursedirs = ["_demos", ".git", ".tox", ".venv", "dist", "build", "*.egg", ".hypothesis"]
```

To run a demo file explicitly (for manifest output generation), pass its path directly:
```bash
uv run pytest tests/testing_chapter/_demos/test_escape_velocity_typo.py
```

**Files in `_demos/`:**
- `test_escape_velocity_typo.py` — ev_expected typo (block 5 manifest: `pytest_escape_velocity_typo_fail`)
- `test_escape_velocity_gpt4.py` — GPT-4 wrong Jupiter value (block 29/30 manifest: `pytest_escape_velocity_gpt4_fail`)
- `test_find_outliers_v1_buggy.py` — ZeroDivisionError against v1 (block 14/15 manifest: `pytest_find_outliers_bug_fail`)
- `test_distance_v1_failing.py` — sqrt(125) wrong value (block 25/27 manifest: `pytest_distance_fail`)
- `test_distance_v2_fixed.py` — sqrt(89) correct value (block 28 manifest: `pytest_distance_pass`)
- `test_dataframe_ops_coupled.py` — coupled tests with global DataFrame (block 22 pedagogical demo)
- `test_ordering_v1_natural.py` — unordered test_second/test_first (block 64/65 manifest: `pytest_ordering_natural_order`)
- `test_property_based_smoke.py` — hypothesis smoke test that fails on singular matrix (block 55/56 manifest: `pytest_property_based_smoke_fail`)

---

## Resolutions (decided 2026-05-27)

| Q | Decision |
|---|----------|
| 1 | Informational only — the plan's "158" was a miscount (158 grep lines = 79 begin + 79 end). 79 is correct. Note the correction here; no other docs need updating. |
| 2 | **Fix the book.** Update block 17 in `book-testing.tex` to show the correct path (`pytest tests/testing_chapter/test_find_outliers.py`). Manifest entry generates output matching the corrected text. Treat the original as a typo. |
| 3 | **Separate files when the failure run IS shown; keep inline when only the bad code is shown.** Apply to blocks 22–23 case-by-case. |
| 4 | **Separate files:** `test_distance_v1_failing.py` (wrong value, `expected_exit_code: 1`) and `test_distance_v2_fixed.py` (correct value, `expected_exit_code: 0`). |
| 5 | **Keep inline if the chapter does not show the failure output;** extract only if a manifest output references the bad version. |
| 6 | **Keep inline.** Generating output that depends on a non-canonical stub module is fragile. |
| 7 | **Mock the PubMed API.** Use `pytest-mock`/canned fixtures so CI is deterministic and offline-safe. The chapter's lesson is parametrization, not live API calls. Minor framing tweak to the chapter's prose may be needed. |
| 8 | Pin hypothesis with `@hypothesis.seed(42)` + `@settings(database=None)` on the canonical property test. Add `temp_paths` to the manifest entry's normalizers. |
| 9 | **Apply Q3 rule:** if block 64's natural-order output appears in the chapter, use separate files `test_ordering_v1_natural.py` / `test_ordering_v2_marked.py`. Otherwise keep block 64 inline. |
| 10 | **Add `worker_count` normalizer** to `scripts/normalize_output.py`: regex `\b\d+ workers\b` → `N workers`. Plus a unit test. Do this as part of Task 14 or 15 (before generating the affected output). |

---

## Summary counts

| Type | Count |
|------|-------|
| code | 40 |
| output | 25 |
| inline-keep | 14 |
| **Total** | **79** |

| Style | Count |
|-------|-------|
| Python | 40 |
| Pythonshort | 6 |
| shell | 21 |
| repl | 5 |
| yaml | 4 |
| toml | 2 |
| transcript | 1 |
| **Total** | **79** |

Style counts verified by `grep "begin{lstlisting}" latex/book-testing.tex | grep -o "style=[^,\]]*" | sort | uniq -c`.
Several blocks use `numbers=none` as an extra option; these are counted under their base style (shell or repl).

---

## External-importer pre-check (re-run 2026-05-27)

Result of grep over `notebooks/` and `../example-*/` for each module being migrated:

| Module | External importers | Safe to rename? |
|--------|---------------------|-----------------|
| escape_velocity | (none) | yes |
| find_outliers | (none) | yes |
| simple_scaler | (none) | yes |
| dataframe_ops | (none) | yes |
| distance | (none) | yes |
| textmining | (none) | yes |
| linear_regression | (none) | yes |
| narps | (none) | yes |

All modules are safe to rename/refactor. No external importers found in `notebooks/` or `example-*` directories.
