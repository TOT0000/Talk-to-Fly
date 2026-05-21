import os
import sys

import pytest

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from controller.abs.skill_item import SkillArg
from controller.abs.skill_item import SkillItem
from controller.minispec_interpreter import MiniSpecProgram, Statement
from controller.skillset import LowLevelSkillItem, SkillSet


def _install_skillset(call_log):
    SkillItem.abbr_dict = {}
    low = SkillSet(level="low")

    def _record(name, return_value=None):
        def _inner(*args):
            call_log.append((name, list(args)))
            return return_value, False

        return _inner

    low.add_skill(LowLevelSkillItem("lo", _record("lo"), args=[SkillArg("text", str)]))
    low.add_skill(LowLevelSkillItem("mb", _record("mb"), args=[SkillArg("distance", float)]))
    low.add_skill(LowLevelSkillItem("ml", _record("ml"), args=[SkillArg("distance", float)]))
    low.add_skill(LowLevelSkillItem("tu", _record("tu"), args=[SkillArg("degree", float)]))
    low.add_skill(LowLevelSkillItem("delay", _record("d"), args=[SkillArg("seconds", float)]))
    low.add_skill(LowLevelSkillItem("go_checkpoint", _record("gc"), args=[SkillArg("checkpoint_id", str)]))

    Statement.low_level_skillset = low
    Statement.high_level_skillset = SkillSet(level="high", lower_level_skillset=low)


def _run(program: str):
    call_log = []
    _install_skillset(call_log)
    env = {}
    minispec = MiniSpecProgram(env=env)
    minispec.parse([program], exec=False)
    minispec.finished = True
    ret = minispec.eval()
    return ret, call_log, env


def test_lo_string_with_equals_is_not_assignment():
    ret, call_log, _env = _run("lo('baseline3 replan triggered: predicted_collision_probability=0.719311 > 0.50, dominant risky worker=worker_3');")
    assert ret.replan is False
    assert call_log == [(
        "lo",
        ["baseline3 replan triggered: predicted_collision_probability=0.719311 > 0.50, dominant risky worker=worker_3"],
    )]


def test_quoted_equals_variants_and_comparison_text_not_assignment():
    _, call_log, _env = _run(
        "lo('a=b=c');"
        'lo("worker=worker_3");'
        "lo('risk >= 0.5');"
        "lo('risk <= 0.5');"
        "lo('risk == high');"
        "lo('risk != low');"
        "lo('predicted_collision_probability=0.719311 > 0.50');"
    )
    assert [args for name, args in call_log if name == "lo"] == [
        ["a=b=c"],
        ["worker=worker_3"],
        ["risk >= 0.5"],
        ["risk <= 0.5"],
        ["risk == high"],
        ["risk != low"],
        ["predicted_collision_probability=0.719311 > 0.50"],
    ]


def test_real_assignments_still_work_and_rhs_string_can_contain_equals():
    _ret, _call_log, env = _run("_x = 1;_y = 2.5;_z = _x + _y;_msg = 'worker=worker_3';")
    assert env["_x"] == 1
    assert env["_y"] == 2.5
    assert env["_z"] == pytest.approx(3.5)
    assert env["_msg"] == "worker=worker_3"


def test_full_replan_program_continues_after_lo_message_with_equals():
    _ret, call_log, _env = _run(
        "lo('baseline3 replan triggered: predicted_collision_probability=0.719311 > 0.50, dominant risky worker=worker_3');"
        "mb(1.5);"
        "ml(1.2);"
        "tu(45);"
        "d(0.6);"
        "gc('C6');"
        "d(2.0);"
        "gc('C5');"
        "d(2.0);"
        "gc('C1');"
        "d(2.0);"
    )
    assert [name for name, _args in call_log] == ["lo", "mb", "ml", "tu", "d", "gc", "d", "gc", "d", "gc", "d"]
