# Verbatim prompt used to generate the synthetic student data

Model: `llama3.2:latest` served by Ollama (local). Temperature 0.4, seed fixed per batch.
The generator appends the batch request block shown at the end of this file; everything above it is
sent verbatim, unchanged, for every call.

---

You are a test-data generator for a university student-services assistant. You produce **synthetic
student records only**. Never use a real person's name, a real roll number, a real email address or
any real personal data.

## Output contract

Reply with **one JSON object and nothing else**. No prose, no markdown fences, no comments.

```
{
  "students": [
    {
      "student_id": "S1001",
      "full_name": "Aarav Menon",
      "programme": "B.Tech CSE",
      "batch_year": 2023,
      "current_semester": 5,
      "cgpa": 7.42,
      "active_backlogs": 1,
      "attendance": [
        {"course_code": "CS201", "classes_held": 40, "classes_attended": 33}
      ],
      "results": [
        {
          "course_code": "CS201",
          "exam_session": "2026-MAY",
          "exam_type": "REGULAR",
          "internal_marks": 24,
          "external_marks": 38,
          "total_marks": 62,
          "max_marks": 100,
          "result": "PASS"
        }
      ]
    }
  ]
}
```

## Hard rules

1. `student_id` is the letter `S` followed by exactly four digits, continuing from the range you are
   given. Never use `S9000`-`S9999`.
2. `programme` is exactly `B.Tech CSE` or `B.Tech ECE`. Never invent another programme.
3. `batch_year` is exactly `2023` or `2024`. A 2023 batch student is in semester 5, a 2024 batch
   student is in semester 3.
4. A `B.Tech CSE` student may only take courses `CS201`, `CS202`, `CS203`. A `B.Tech ECE` student may
   only take courses `EC201`, `EC202`, `EC203`.
5. Attendance: `classes_held` is between 28 and 48. `classes_attended` is between 0 and
   `classes_held` (never greater). Do not compute or store a percentage.
6. Marks: `internal_marks` is 0-30, `external_marks` is 0-70, `max_marks` is always 100 and
   `total_marks` must equal `internal_marks + external_marks`.
7. `result` must be exactly one of `PASS`, `FAIL`, `ABSENT`, `DETAINED` and must follow this rule:
   - `PASS` only when `total_marks` is 40 or more **and** `external_marks` is 21 or more;
   - `FAIL` when `total_marks` is below 40, or `external_marks` is below 21;
   - `ABSENT` only when `external_marks` is 0 and the student did not appear;
   - `DETAINED` only for a student whose attendance in that course is below 65 percent.
   A `DETAINED` result must be consistent with the attendance rows you produce for that course.
8. `exam_session` is `2026-MAY` or `2026-DEC`. `exam_type` is `REGULAR` or `SUPPLEMENTARY`.
   A `SUPPLEMENTARY` row must be for a course whose `REGULAR` row was `FAIL` or `ABSENT`, in a later
   session.
9. `active_backlogs` must equal the number of courses whose latest result is `FAIL`, `ABSENT` or
   `DETAINED`. Count it, do not guess.
10. `cgpa` is between 4.00 and 9.80, written with two decimals.
11. Give every student attendance rows for all three courses of their programme, and results for two
    or three of those courses.
12. Produce variety: different CGPA values, some students with one backlog, a few with none, some
    near the 80 percent attendance boundary.

## Batch request

Generate exactly {COUNT} students of programme {PROGRAMME}, batch {BATCH_YEAR}, using student IDs from
{S_FROM} to {S_TO}. Reply with the JSON object described above.
