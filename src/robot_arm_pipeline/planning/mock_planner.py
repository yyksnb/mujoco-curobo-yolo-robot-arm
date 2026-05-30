from robot_arm_pipeline.types import (
    CollisionScene,
    GraspTarget,
    PlannedTrajectory,
    RobotState,
    TrajectoryWaypoint,
)


class MockPlanner:
    name = "mock_planner"

    def plan(
        self,
        scene: CollisionScene,
        robot_state: RobotState,
        grasp_target: GraspTarget,
    ) -> PlannedTrajectory:
        if not scene.objects:
            raise ValueError("cannot plan without collision objects")
        if grasp_target.object_id not in {obj.object_id for obj in scene.objects}:
            raise ValueError(f"grasp target {grasp_target.object_id!r} is not in the scene")

        target_positions = self._target_joint_positions(robot_state, grasp_target)
        waypoints = self._interpolate(robot_state.joint_positions, target_positions, steps=5)
        return PlannedTrajectory(
            joint_names=robot_state.joint_names,
            waypoints=waypoints,
            planner_name=self.name,
            target_object_id=grasp_target.object_id,
        )

    def _target_joint_positions(
        self,
        robot_state: RobotState,
        grasp_target: GraspTarget,
    ) -> tuple[float, ...]:
        x, y, z = grasp_target.pose.position
        seed = (x, y, z, grasp_target.gripper_width_m, -z, x - y)
        return tuple(round(seed[index % len(seed)], 4) for index, _ in enumerate(robot_state.joint_positions))

    def _interpolate(
        self,
        start: tuple[float, ...],
        goal: tuple[float, ...],
        *,
        steps: int,
    ) -> tuple[TrajectoryWaypoint, ...]:
        if steps < 2:
            raise ValueError("steps must be at least 2")
        waypoints: list[TrajectoryWaypoint] = []
        for index in range(steps):
            ratio = index / (steps - 1)
            positions = tuple(
                round(start_value + (goal_value - start_value) * ratio, 4)
                for start_value, goal_value in zip(start, goal)
            )
            waypoints.append(TrajectoryWaypoint(time_s=round(index * 0.5, 3), joint_positions=positions))
        return tuple(waypoints)

