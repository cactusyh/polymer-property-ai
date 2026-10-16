# Phase 6T resume determinism

The exact interrupt/resume audit was not executed. The existing full trainer still
uses per-batch PSMILES tokenization and has no persistent-cache input path, so it
would not test the Phase 6T data path. The phase is fail-closed as
`BLOCKED_TRAINING`; no B_MLM checkpoint or Tg metrics are reported.
