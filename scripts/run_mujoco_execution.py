from _bootstrap import add_src_to_path

add_src_to_path()

from robot_arm_pipeline.execution.mujoco_executor import MujocoExecutor


def main() -> None:
    MujocoExecutor()


if __name__ == "__main__":
    main()
