# Synthetic student data card (Annex E)

| Field | Value |
| --- | --- |
| Purpose | Exercise the assistant's tool paths (attendance, results, backlogs, eligibility) and every edge case the participant guide names, in the fixed Annex C schema. No real person appears in this data. |
| Files | `data/students.csv`, `data/courses.csv`, `data/attendance.csv`, `data/results.csv`, loader `scripts/load_students.py` |
| Generator | `llama3.2:latest` (Ollama, local, 3.2B, Q4_K_M) for the LLM path, with `temperature=0.4` and `seed=20261010+attempt`; deterministic seeded producer for the committed dataset (see below) |
| Verbatim prompt | [`prompts/generate_students.md`](../prompts/generate_students.md) — sent unchanged for every batch |
| Generator script | [`scripts/generate_students.py`](../scripts/generate_students.py) (`--source llm` or `--source deterministic`) |
| Validator | [`scripts/validate_students.py`](../scripts/validate_students.py) over [`src/assa/data_validation.py`](../src/assa/data_validation.py) |

## What was produced, and by what

| Producer | Command | Result |
| --- | --- | --- |
| LLM (evidence kept) | `python scripts/generate_students.py --source llm --count 24 --batch-size 4 --model llama3.2:latest` | Raw replies kept in `data/generated_raw/batch_*.json`. Two batches (8 students) completed before the run was stopped: the 3B model on CPU needs roughly 12 minutes per 4-student batch on this machine |
| Deterministic (committed) | `python scripts/generate_students.py --source deterministic --count 30` | The committed `data/*.csv`: 38 students (30 seeded + 8 exact edge-case students) |

**Be explicit about this:** the committed dataset was produced by the deterministic
producer, not by the model. A 3B local model is not reliable enough to hit exact
thresholds (attendance exactly at 80.0%, a total of 39 marks), which is what the
edge-case students must carry, and it takes ~12 minutes per batch on this CPU.
The LLM path is implemented, uses the committed verbatim prompt, enforces the
schema with code, and its real output is committed as evidence. Run the `--source llm`
command above (ideally with `llama3.1:8b` or a larger model) to regenerate the
seeded block from the model; the edge-case block is added on top in either case.

## Row counts and distributions (committed dataset)

| Table | Rows | Notes |
| --- | --- | --- |
| `students` | 38 | B.Tech CSE 2023: 11, B.Tech CSE 2024: 9, B.Tech ECE 2023: 10, B.Tech ECE 2024: 8 |
| `courses` | 6 | 3 per programme, semesters 3-4, 3-4 credits |
| `attendance` | 114 | 3 courses per student; percentage range 60.0-97.5; 49 of 114 rows below the 80% threshold |
| `results` | 97 | PASS 61, FAIL 34, ABSENT 1, DETAINED 1; total marks 13-97 |

CGPA range 4.70-9.44. Active backlogs: 0 for 10 students, 1 for 22, 2 for 5, 3 for 1.

## Schema enforcement

1. The prompt fixes the schema, the ranges, the identifier format and the pass rule.
2. Ollama is called with `format: "json"`, so the reply is a JSON object (not prose).
3. `repair_student()` recomputes `total_marks`, `max_marks`, `result` and
   `active_backlogs` from the marks instead of trusting them, clamps attendance to
   `0 <= attended <= held`, and logs every change.
4. `validate_dataset()` then applies the full checks (column names, formats, the
   judge-reserved ranges, referential integrity, `attended <= held`,
   `total = internal + external`, result consistent with marks,
   `active_backlogs` consistent with results, minimum coverage, presence of every
   edge case). A violation makes `validate_students.py` exit non-zero and makes
   `scripts/load_students.py` refuse to load.

## Validation results

`python scripts/validate_students.py` → **exit 0, no violations**, all seven edge
cases present.

Two defects were found by the checks during development and fixed:

1. **Pass rule applied to the wrong base.** The 30 percent paper requirement of
   EXAM-RULES-2025 clause 3.3 was applied to the 100-mark course total instead of
   the 70-mark paper (it is 21 marks, not 30). This made some rows inconsistent
   with the cited rule. Fixed in `data_validation.expected_result`, now also
   implemented as the `evaluate_course_outcome` tool, and covered by tests.
2. **Supplementary attempts compared as strings.** `2026-DEC` sorts before
   `2026-MAY` as text, so a cleared supplementary attempt could be read as an
   active backlog. Fixed with an explicit session ordering, with a regression test.

## What the LLM got wrong

Across the 8 students in the two committed raw batches, code found **9 defects**
that the model reported as valid (`data/generated_raw/`):

| Defect | Example |
| --- | --- |
| `PASS` recorded although the paper mark was below the 30% floor | `S1002/CS203: total=35, external=20 → FAIL`; `S1004/CS201: total=38, external=20 → FAIL` |
| `FAIL` recorded although the marks are a pass | `S1001/CS202: total=40, external=25 → PASS`; `S1006/CS202: total=60, external=40 → PASS` |
| `active_backlogs` wrong in 5 of 8 students | `S1004` declared 0, three failed courses |

What the model got right every time: valid JSON, the exact key names, the
programme and batch it was asked for, the student ID range (`S1001`-`S1008`),
attendance within range, and marks within the internal/external bounds. The
discipline the guide asks for is visible here: the model's arithmetic is not
trusted, it is recomputed and the corrections are logged.

## Edge cases and the student IDs that carry them

| Edge case | Student IDs |
| --- | --- |
| Attendance exactly at the 80% threshold | `S2001` (32/40), `S1011` |
| Attendance one class below the threshold (77.5%) | `S2002` (31/40) |
| Marks one mark below the pass mark (39) | `S2003` |
| `ABSENT` result | `S2004` |
| `DETAINED` (attendance 60%, below the 65% floor) | `S2005` |
| Multiple active backlogs | `S2006` (three), `S1011`, `S1016`, `S1020`, `S1028` |
| CGPA exactly at the 6.00 placement cut-off | `S2007` |
| Course cleared through a supplementary attempt | `S2008` |

## Known limitations of this data

* The names are invented combinations of common first and last names; they do not
  correspond to any real person and were checked for that only by construction.
* The distribution is deliberately pessimistic (more backlogs and attendance
  shortages than a real cohort) so the tool paths are exercised; it is not
  representative for statistics.
* Only two programmes, two batches and six courses, as the guide requires as a
  minimum; there are no laboratory-only courses, no part-time students and no
  semester-wise history beyond two exam sessions.
* Attendance is generated per course, not per class session, so there is no
  day-level record and no timetable.
