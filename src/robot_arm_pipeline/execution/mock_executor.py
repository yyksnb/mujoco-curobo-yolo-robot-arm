from robot_arm_pipeline.types import ExecutionResult, PlannedTrajectory, RobotState


class MockExecutor:
    name = "mock_executor"

    def execute(self, trajectory: PlannedTrajectory) -> ExecutionResult:
        if not trajectory.waypoints:
            return ExecutionResult(
                success=False,
                executor_name=self.name,
                duration_s=0.0,
                final_state=RobotState(
                    joint_names=trajectory.joint_names,
                    joint_positions=tuple(0.0 for _ in trajectory.joint_names),
                ),
                message="trajectory has no waypoints",
            )

        final_waypoint = trajectory.waypoints[-1]
        return ExecutionResult(
            success=True,
            executor_name=self.name,
            duration_s=final_waypoint.time_s,
            final_state=RobotState(
                joint_names=trajectory.joint_names,
                joint_positions=final_waypoint.joint_positions,
            ),
            message="mock execution completed",
        )
