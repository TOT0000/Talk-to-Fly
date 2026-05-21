from controller.minispec_interpreter import Statement
from controller.skillset import SkillSet


def _build_statement_with_skills():
    env = {}
    statement = Statement(env)
    calls = []

    class _FakeSkill:
        def __init__(self, name):
            self._name = name

        def get_name(self):
            return self._name

        def execute(self, args):
            calls.append((self._name, list(args)))
            return (None, False)

    class _FakeLowSkillset:
        def get_skill(self, name):
            if name in {"lo", "mb", "ml", "tu", "d", "gc"}:
                return _FakeSkill(name)
            return None

    low = _FakeLowSkillset()
    Statement.low_level_skillset = low
    Statement.high_level_skillset = SkillSet(level="high")
    return statement, env, calls


def _run_program(program: str):
    statement, env, calls = _build_statement_with_skills()
    chunks = [chunk.strip() for chunk in program.split(';') if chunk.strip()]
    for chunk in chunks:
        statement.eval_expr(chunk)
    return env, calls


def test_lo_string_with_equals_not_assignment():
    program = "lo('baseline3 replan triggered: predicted_collision_probability=0.909 > 0.5, dominant risky worker=worker_2');"
    env, calls = _run_program(program)
    assert env == {}
    assert calls == [(
        "lo",
        ["baseline3 replan triggered: predicted_collision_probability=0.909 > 0.5, dominant risky worker=worker_2"],
    )]


def test_lo_string_multiple_equals_not_assignment():
    env, calls = _run_program("lo('a=b=c');")
    assert env == {}
    assert calls == [("lo", ["a=b=c"])]


def test_lo_double_quote_equals_not_assignment():
    env, calls = _run_program('lo("worker=worker_2");')
    assert env == {}
    assert calls == [("lo", ["worker=worker_2"])]


def test_real_assignment_still_works():
    env, calls = _run_program("_x = 1; _y = 2.5; _z = _x + _y;")
    assert calls == []
    assert env["_x"] == 1
    assert env["_y"] == 2.5
    assert env["_z"] == 3.5


def test_assignment_rhs_string_with_equals():
    env, calls = _run_program("_msg = 'worker=worker_2';")
    assert calls == []
    assert env["_msg"] == "worker=worker_2"


def test_full_replan_program_executes_all_statements():
    program = (
        "lo('baseline3 replan triggered: predicted_collision_probability=0.909 > 0.5, dominant risky worker=worker_2');"
        "mb(1.0);"
        "ml(1.0);"
        "tu(30);"
        "d(0.5);"
        "gc('C5');"
        "d(2.0);"
        "gc('C1');"
        "d(2.0);"
    )
    env, calls = _run_program(program)
    assert env == {}
    assert [name for name, _ in calls] == ["lo", "mb", "ml", "tu", "d", "gc", "d", "gc", "d"]
    assert calls[0][1] == [
        "baseline3 replan triggered: predicted_collision_probability=0.909 > 0.5, dominant risky worker=worker_2"
    ]
