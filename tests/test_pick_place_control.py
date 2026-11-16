import importlib
import sys
import types
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np


def _module(name, *, package=False, **attributes):
    module = types.ModuleType(name)
    if package:
        module.__path__ = []
    for key, value in attributes.items():
        setattr(module, key, value)
    return module


def _load_pick_place_class():
    stub_types = {
        "isaacsim": _module("isaacsim", package=True),
        "isaacsim.core": _module("isaacsim.core", package=True),
        "isaacsim.core.experimental": _module("isaacsim.core.experimental", package=True),
        "isaacsim.core.experimental.utils": _module(
            "isaacsim.core.experimental.utils", package=True
        ),
        "isaacsim.core.experimental.utils.stage": _module(
            "isaacsim.core.experimental.utils.stage"
        ),
        "isaacsim.core.experimental.objects": _module(
            "isaacsim.core.experimental.objects",
            Cube=object,
            DomeLight=object,
            GroundPlane=object,
        ),
        "isaacsim.core.experimental.prims": _module(
            "isaacsim.core.experimental.prims",
            GeomPrim=object,
            RigidPrim=object,
        ),
        "hanoi_tower.franka": _module("hanoi_tower.franka", Franka=object),
    }
    with patch.dict(sys.modules, stub_types):
        sys.modules.pop("hanoi_tower.pick_place", None)
        controller_type = importlib.import_module("hanoi_tower.pick_place").FrankaPickPlace
    sys.modules.pop("hanoi_tower.pick_place", None)
    return controller_type


def _load_tower_class():
    class StubPickPlace:
        def is_done(self):
            return self._motion_failed

    stub_types = {
        "isaacsim": _module("isaacsim", package=True),
        "isaacsim.core": _module("isaacsim.core", package=True),
        "isaacsim.core.experimental": _module("isaacsim.core.experimental", package=True),
        "isaacsim.core.experimental.utils": _module(
            "isaacsim.core.experimental.utils", package=True
        ),
        "isaacsim.core.experimental.utils.stage": _module(
            "isaacsim.core.experimental.utils.stage"
        ),
        "isaacsim.core.experimental.objects": _module(
            "isaacsim.core.experimental.objects", Cube=object
        ),
        "isaacsim.core.experimental.materials": _module(
            "isaacsim.core.experimental.materials", RigidBodyMaterial=object
        ),
        "isaacsim.core.experimental.prims": _module(
            "isaacsim.core.experimental.prims", GeomPrim=object, RigidPrim=object
        ),
        "hanoi_tower.pick_place": _module(
            "hanoi_tower.pick_place", FrankaPickPlace=StubPickPlace
        ),
    }
    with patch.dict(sys.modules, stub_types):
        sys.modules.pop("hanoi_tower.tower", None)
        controller_type = importlib.import_module("hanoi_tower.tower").FrankaTowerOfHanoi
    sys.modules.pop("hanoi_tower.tower", None)
    return controller_type


FrankaPickPlace = _load_pick_place_class()
FrankaTowerOfHanoi = _load_tower_class()


class RecordingRobot:
    def __init__(self, position=(0.0, 0.0, 0.0)):
        self.commands = []
        self.position = np.asarray([position], dtype=np.float32)

    def set_end_effector_pose(self, **command):
        self.commands.append(command)

    def get_current_state(self):
        return np.zeros((1, 7)), self.position, np.array([[1.0, 0.0, 0.0, 0.0]])


class PickPlaceControlTests(unittest.TestCase):
    def test_default_transit_and_descent_durations_are_slower(self):
        controller = FrankaPickPlace()

        self.assertEqual(controller.events_dt, [60, 40, 80, 80, 33])
        self.assertEqual(controller.grasp_descent_steps, 80)
        self.assertEqual(controller.grasp_settle_timeout_steps, 40)

    def _controller(self, *, phase_step=0, phase_duration=8):
        controller = object.__new__(FrankaPickPlace)
        controller._event = 3
        controller._step = phase_step
        controller.events_dt = [1, 1, 1, phase_duration, 1]
        controller._phase_start_position = np.array([0.2, 0.1, 0.4], dtype=np.float32)
        controller.z_lead_fraction = 1 / 3
        controller.transit_tolerance = 0.008
        controller.ik_delta_scale = 0.5
        controller.robot = RecordingRobot()
        controller._motion_failed = False
        return controller

    def test_descent_aligns_xy_at_height_then_moves_only_z(self):
        controller = self._controller()
        target = np.array([0.3, 0.3, 0.1], dtype=np.float32)

        controller._set_phase_pose(target, np.array([1.0, 0.0, 0.0, 0.0]), "damped-least-squares")
        align_command = controller.robot.commands[-1]
        np.testing.assert_allclose(align_command["position"], [0.25, 0.2, 0.4])
        self.assertFalse(align_command["position_only"])

        controller._step = 3
        controller._set_phase_pose(target, np.array([1.0, 0.0, 0.0, 0.0]), "damped-least-squares")
        descent_command = controller.robot.commands[-1]
        np.testing.assert_allclose(descent_command["position"][:2], target[:2])
        self.assertLess(descent_command["position"][2], align_command["position"][2])
        self.assertFalse(descent_command["position_only"])

    def test_descent_timeout_fails_without_advancing_to_release(self):
        controller = self._controller(phase_step=3, phase_duration=2)
        controller.playback_speed = 1.0
        controller.descent_timeout_steps = 2
        controller.descent_tolerance = 0.005

        controller._advance_phase(np.array([1.0, 1.0, 1.0]))

        self.assertTrue(controller._motion_failed)
        self.assertEqual(controller._event, 3)
        self.assertTrue(controller.is_done())

    def test_transit_timeout_fails_without_entering_descent(self):
        controller = self._controller(phase_step=3, phase_duration=2)
        controller._event = 2
        controller.playback_speed = 1.0
        controller.transit_timeout_steps = 2

        controller._advance_phase(np.array([1.0, 1.0, 1.0]))

        self.assertTrue(controller._motion_failed)
        self.assertEqual(controller._event, 2)

    def test_reset_clears_descent_failure(self):
        controller = self._controller()
        controller.robot = None
        controller._motion_failed = True

        controller.reset_robot()

        self.assertFalse(controller._motion_failed)

    def test_ten_disk_carry_height_uses_one_clearance_margin(self):
        controller = object.__new__(FrankaTowerOfHanoi)
        controller._stacks = [list(range(10)), [], []]
        controller.disk_thickness = 0.025
        controller.approach_height = 0.0885
        controller.carry_reserve = 0.12

        self.assertAlmostEqual(controller._safe_transit_z(), 0.4835)

    def test_floor_disk_center_matches_contact_plane_offset(self):
        controller = object.__new__(FrankaTowerOfHanoi)
        controller.peg_positions = np.array([[0.5, 0.0]])
        controller.disk_thickness = 0.025

        np.testing.assert_allclose(controller._disk_position(0, 0, 0), [0.5, 0.0, 0.0125])

    def test_ten_disk_minimum_width_leaves_a_graspable_smallest_disk(self):
        width_span = min(0.024, 0.012 * (10 - 1))
        smallest_disk_width = np.linspace(0.056 - width_span, 0.056, 10)[0]

        self.assertAlmostEqual(smallest_disk_width, 0.032)

    def test_release_hand_target_places_disk_center_on_support(self):
        disk_target = np.array([0.5, 0.0, 0.0375])
        disk_hand_offset = np.array([0.0, 0.0, -0.091])

        hand_target = FrankaTowerOfHanoi._hand_target_for_disk(disk_target, disk_hand_offset)

        np.testing.assert_allclose(hand_target, [0.5, 0.0, 0.1285])

    def test_support_disk_motion_flags_a_double_grasp(self):
        controller = object.__new__(FrankaTowerOfHanoi)
        controller.double_grasp_tolerance = 0.005

        self.assertFalse(controller._support_disk_was_lifted(0.005))
        self.assertTrue(controller._support_disk_was_lifted(0.006))

    def test_motion_failure_does_not_commit_a_hanoi_move(self):
        controller = object.__new__(FrankaTowerOfHanoi)
        controller._failed = False
        controller._motion_failed = True
        controller._move_index = 0
        controller._moves = [(0, 0, 1)]

        self.assertFalse(controller.forward())
        self.assertTrue(controller._failed)
        self.assertEqual(controller._move_index, 0)

    def test_pickup_probes_are_limited_by_safe_hand_height(self):
        controller = object.__new__(FrankaTowerOfHanoi)
        controller._pickup_attempts = 0
        controller.max_pickup_attempts = 3
        controller.pickup_probe_step = 0.010
        controller.minimum_hand_target_z = 0.10
        controller.disk_thickness = 0.025
        controller.approach_height = 0.0785
        controller._event = 1
        controller._step = 8
        controller._phase_start_position = np.zeros(3)
        controller._pickup_start_z = None
        controller._grasp_object_offset = None
        controller._failed = False
        controller.robot = SimpleNamespace(open_gripper=lambda: None)
        controller.cube = SimpleNamespace(
            get_world_poses=lambda: (
                SimpleNamespace(numpy=lambda: np.array([[0.5, 0.0, 0.0375]])),
                None,
            )
        )

        self.assertTrue(controller._retry_failed_pickup())
        self.assertAlmostEqual(controller.approach_height, 0.0685)
        self.assertTrue(controller._retry_failed_pickup())
        self.assertAlmostEqual(controller.approach_height, 0.0585)
        self.assertFalse(controller._retry_failed_pickup())
        self.assertTrue(controller._failed)


if __name__ == "__main__":
    unittest.main()
