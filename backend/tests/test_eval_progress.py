import threading
import time

from sotellme.eval_progress import EvalProgress


def _reporter(total: int) -> tuple[EvalProgress, list[str]]:
    lines: list[str] = []
    return EvalProgress(total, write=lines.append), lines


def test_start_announces_the_persona_and_its_position_in_the_run() -> None:
    reporter, lines = _reporter(8)

    reporter.start(1, "senior-strong")

    assert len(lines) == 1
    assert lines[0].startswith("[1/8] ")
    assert "senior-strong" in lines[0]
    assert lines[0].rstrip().endswith("running…")


def test_finish_reports_turns_per_persona_cost_and_the_running_total() -> None:
    reporter, lines = _reporter(8)

    reporter.finish(
        1, "senior-strong", turns=8, finished_reason="completed", persona_usd=1.234, total_usd=1.234
    )

    line = lines[0]
    assert line.startswith("[1/8] ")
    assert "senior-strong" in line
    assert "✓ done" in line
    assert "8 turns" in line
    assert "$1.23" in line
    assert "run total $1.23" in line


def test_finish_marks_a_terminated_persona_and_singularizes_one_turn() -> None:
    reporter, lines = _reporter(8)

    reporter.finish(
        3,
        "staff-injection",
        turns=1,
        finished_reason="terminated",
        persona_usd=0.05,
        total_usd=1.28,
    )

    line = lines[0]
    assert "⊘ terminated" in line
    assert "1 turn ·" in line and "1 turns" not in line
    assert "$0.05" in line
    assert "run total $1.28" in line


def test_finish_flags_a_persona_that_hit_the_turn_cap() -> None:
    reporter, lines = _reporter(8)

    reporter.finish(
        5, "mid-offtopic", turns=20, finished_reason="max_turns", persona_usd=2.10, total_usd=9.99
    )

    assert "⚠ hit turn cap" in lines[0]
    assert "20 turns" in lines[0]


def test_finish_aligns_the_cost_columns_across_personas_of_different_name_lengths() -> None:
    reporter, lines = _reporter(8)

    reporter.finish(
        1,
        "mid-blurred-ownership",
        turns=9,
        finished_reason="completed",
        persona_usd=1.0,
        total_usd=1.0,
    )
    reporter.finish(
        2, "junior-thin", turns=5, finished_reason="completed", persona_usd=0.5, total_usd=1.5
    )

    assert lines[0].index("✓") == lines[1].index("✓")


def test_concurrent_writes_never_overlap_so_lines_do_not_garble() -> None:
    # Personas finish on separate worker threads (asyncio.to_thread). The reporter must serialize
    # its writes; a writer that detects a second thread entering mid-write would otherwise garble
    # the line. The lock keeps at most one writer active at a time.
    active = 0
    max_active = 0
    guard = threading.Lock()

    def slow_write(_: str) -> None:
        nonlocal active, max_active
        with guard:
            active += 1
            max_active = max(max_active, active)
        time.sleep(0.001)
        with guard:
            active -= 1

    reporter = EvalProgress(16, write=slow_write)

    def run(index: int) -> None:
        reporter.start(index, f"persona-{index}")
        reporter.finish(
            index,
            f"persona-{index}",
            turns=4,
            finished_reason="completed",
            persona_usd=0.1,
            total_usd=0.1,
        )

    threads = [threading.Thread(target=run, args=(i,)) for i in range(16)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert max_active == 1
