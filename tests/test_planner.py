import unittest

from hanoi_tower.planner import (
    plan_hanoi,
    plan_hanoi_from_state,
    required_lift_height,
    smoothstep_fraction,
    smooth_velocity_scale,
)


class HanoiPlannerTests(unittest.TestCase):
    def test_smoothstep_starts_and_ends_at_rest(self):
        self.assertEqual(smoothstep_fraction(0, 10), 0.0)
        self.assertEqual(smoothstep_fraction(10, 10), 1.0)
        self.assertEqual(smoothstep_fraction(8, 10), 1.0)
        self.assertLess(smoothstep_fraction(1, 10), 0.1)
        self.assertGreater(smoothstep_fraction(9, 10), 0.9)

    def test_smooth_velocity_scale_ramps_up_then_allows_full_correction(self):
        start = smooth_velocity_scale(0, 100)
        middle = smooth_velocity_scale(49, 100)
        end = smooth_velocity_scale(99, 100)

        self.assertGreater(start, 0.0)
        self.assertGreater(middle, start)
        self.assertEqual(end, 1.0)

    def test_lift_height_clears_the_taller_stack(self):
        lift_for_three_layers = required_lift_height(0.045, 3, reserve=0.08)
        lift_for_four_layers = required_lift_height(0.045, 4, reserve=0.08)

        self.assertAlmostEqual(lift_for_three_layers, 0.08 + 3 * 0.03 + 0.015 - 0.045)
        self.assertAlmostEqual(lift_for_four_layers, 0.08 + 4 * 0.03 + 0.015 - 0.045)
        self.assertGreater(lift_for_four_layers, lift_for_three_layers)

    def test_move_count_is_minimal(self):
        for disk_count in range(1, 11):
            with self.subTest(disk_count=disk_count):
                self.assertEqual(len(plan_hanoi(disk_count)), 2**disk_count - 1)
        # 驗證 10 層時最大步數正好為 1023 步 (2^10 - 1)
        self.assertEqual(len(plan_hanoi(10)), 1023)

    def test_plan_is_legal_and_reaches_destination(self):
        disk_count = 4
        stacks = [list(range(disk_count - 1, -1, -1)), [], []]

        for disk_index, source, destination in plan_hanoi(disk_count):
            self.assertTrue(stacks[source])
            self.assertEqual(stacks[source][-1], disk_index)
            if stacks[destination]:
                self.assertGreater(stacks[destination][-1], disk_index)
            stacks[source].pop()
            stacks[destination].append(disk_index)

        self.assertEqual(stacks, [[], [], list(range(disk_count - 1, -1, -1))])

    def test_plan_from_mixed_initial_state_is_legal_and_reaches_destination(self):
        stacks = [[2], [1, 0], []]

        moves = plan_hanoi_from_state(stacks, disk_count=3)

        for disk_index, source, destination in moves:
            self.assertTrue(stacks[source])
            self.assertEqual(stacks[source][-1], disk_index)
            if stacks[destination]:
                self.assertGreater(stacks[destination][-1], disk_index)
            stacks[source].pop()
            stacks[destination].append(disk_index)

        self.assertEqual(stacks, [[], [], [2, 1, 0]])

    def test_plan_from_goal_state_has_no_moves(self):
        self.assertEqual(plan_hanoi_from_state([[], [], [2, 1, 0]], disk_count=3), [])

    def test_plan_from_state_rejects_illegal_stack(self):
        with self.assertRaises(ValueError):
            plan_hanoi_from_state([[0, 1], [2], []], disk_count=3)

    def test_rejects_empty_puzzle(self):
        with self.assertRaises(ValueError):
            plan_hanoi(0)


if __name__ == "__main__":
    unittest.main()