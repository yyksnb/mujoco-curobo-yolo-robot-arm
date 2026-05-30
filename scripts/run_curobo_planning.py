from _bootstrap import add_src_to_path

add_src_to_path()

from robot_arm_pipeline.planning.curobo_planner import CuroboPlanner


def main() -> None:
    CuroboPlanner()


if __name__ == "__main__":
    main()
