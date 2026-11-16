import json

import pytest

from src.data.curriculum import example_difficulty, sort_by_curriculum


def _ex(num_tools: int, calls: list[tuple[str, int]]):
    """calls is a list of (name, num_args) pairs."""
    return {
        "tools": [{"name": f"tool_{i}"} for i in range(num_tools)],
        "ground_truth": [
            {"name": name, "arguments": {f"p{i}": i for i in range(n)}}
            for name, n in calls
        ],
    }


class TestExampleDifficulty:
    def test_zero_calls_zero_tools(self):
        assert example_difficulty({"tools": [], "ground_truth": []}) == 0.0

    def test_single_call_single_param(self):
        # 1 call + 1 param + 0.1 * 1 tool = 2.1
        assert example_difficulty(_ex(1, [("f", 1)])) == pytest.approx(2.1)

    def test_parallel_calls_harder_than_single(self):
        single = example_difficulty(_ex(3, [("f", 2)]))
        parallel = example_difficulty(_ex(3, [("f", 2), ("g", 2)]))
        assert parallel > single

    def test_more_params_harder(self):
        few = example_difficulty(_ex(3, [("f", 1)]))
        many = example_difficulty(_ex(3, [("f", 5)]))
        assert many > few

    def test_tool_count_has_smaller_impact_than_call_count(self):
        # An extra call with one param adds 1 + 1 = 2.
        # 10 extra tools add 0.1 * 10 = 1. Calls dominate.
        more_tools = example_difficulty(_ex(11, [("f", 1)]))
        more_calls = example_difficulty(_ex(1, [("f", 1), ("g", 1)]))
        assert more_calls > more_tools

    def test_handles_missing_keys(self):
        # Real xLAM examples sometimes omit tools or have no answers.
        assert example_difficulty({}) == 0.0
        assert example_difficulty({"tools": None, "ground_truth": None}) == 0.0
        assert example_difficulty({"ground_truth": [{"name": "f"}]}) == 1.0

    def test_handles_null_call_arguments(self):
        # Defensive: a malformed call with arguments=None should not crash.
        ex = {"tools": [], "ground_truth": [{"name": "f", "arguments": None}]}
        assert example_difficulty(ex) == 1.0


class TestSortByCurriculum:
    def test_ascending_order(self):
        easy = _ex(1, [("f", 0)])
        medium = _ex(2, [("f", 2)])
        hard = _ex(5, [("f", 3), ("g", 4), ("h", 1)])
        sorted_examples = sort_by_curriculum([hard, easy, medium])
        assert [example_difficulty(ex) for ex in sorted_examples] == sorted(
            [example_difficulty(easy), example_difficulty(medium), example_difficulty(hard)]
        )

    def test_stable_for_equal_difficulty(self):
        a = _ex(1, [("f", 1)])
        b = _ex(1, [("g", 1)])
        # Same difficulty - sort is stable, so order is preserved.
        sorted_examples = sort_by_curriculum([a, b])
        assert sorted_examples[0] is a
        assert sorted_examples[1] is b


class TestCurriculumCLI:
    def test_handles_empty_input(self, tmp_path, capsys):
        # An empty (or all-blank) input must not crash with IndexError.
        from src.data.curriculum import main as curriculum_main
        import sys

        in_path = tmp_path / "empty.jsonl"
        out_path = tmp_path / "out.jsonl"
        in_path.write_text("\n   \n")

        argv_backup = sys.argv
        try:
            sys.argv = ["curriculum", "--input", str(in_path), "--output", str(out_path)]
            curriculum_main()
        finally:
            sys.argv = argv_backup

        assert out_path.exists()
        assert out_path.read_text() == ""
        assert "empty" in capsys.readouterr().out.lower()

    def test_writes_sorted_output(self, tmp_path):
        from src.data.curriculum import main as curriculum_main
        import sys

        in_path = tmp_path / "in.jsonl"
        out_path = tmp_path / "out.jsonl"

        examples = [
            _ex(5, [("f", 3), ("g", 2)]),  # hard
            _ex(1, [("f", 0)]),  # easy
            _ex(2, [("f", 1)]),  # medium
        ]
        with open(in_path, "w") as f:
            for ex in examples:
                f.write(json.dumps(ex) + "\n")

        argv_backup = sys.argv
        try:
            sys.argv = ["curriculum", "--input", str(in_path), "--output", str(out_path)]
            curriculum_main()
        finally:
            sys.argv = argv_backup

        with open(out_path) as f:
            sorted_examples = [json.loads(line) for line in f]

        difficulties = [example_difficulty(ex) for ex in sorted_examples]
        assert difficulties == sorted(difficulties)
        assert len(sorted_examples) == 3
