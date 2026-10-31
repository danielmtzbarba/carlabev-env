from __future__ import annotations

import importlib
import sys
from collections.abc import Sequence


USAGE = """Usage:
  carlabev run sim [ARGS...]            # render/clock/scene/control options (see --help)
  carlabev run interactive [ARGS...]    # legacy manual viewer
  carlabev scene design [ARGS...]
  carlabev scene build-library [ARGS...]
  carlabev scene analyze-library [ARGS...]
  carlabev scene analyze-routes [ARGS...]
  carlabev graph build-town01-canonical [ARGS...]
  carlabev graph visualize-town01-canonical [ARGS...]
  carlabev graph validate-town01-joins [ARGS...]
  carlabev validate simulator-semantics [ARGS...]
  carlabev benchmark run|compare|baseline [ARGS...]
  carlabev viz semantic-channels [ARGS...]
  carlabev viz vehicle-temporal-fusion [ARGS...]
"""


def _invoke_module_main(module_name: str, argv: Sequence[str]) -> object:
    module = importlib.import_module(module_name)
    if not hasattr(module, "main"):
        raise ValueError(f"Module {module_name!r} does not expose a main() function.")

    old_argv = sys.argv
    sys.argv = [module_name, *argv]
    try:
        return module.main()
    finally:
        sys.argv = old_argv


def _pop_command(argv: list[str], context: str) -> str:
    if not argv:
        raise SystemExit(f"Missing {context}.\n\n{USAGE}")
    return argv.pop(0)


def main(argv: Sequence[str] | None = None) -> int:
    tokens = list(sys.argv[1:] if argv is None else argv)
    if not tokens or tokens[0] in {"-h", "--help"}:
        print(USAGE)
        return 0

    group = _pop_command(tokens, "command group")
    if group == "run":
        command = _pop_command(tokens, "run command")
        if command == "sim":
            from CarlaBEV.cli import run_sim

            return int(run_sim.main(tokens) or 0)
        if command == "interactive":
            return int(_invoke_module_main("CarlaBEV.cli.run_interactive", tokens) or 0)
    elif group == "scene":
        command = _pop_command(tokens, "scene command")
        if command == "design":
            return int(
                _invoke_module_main("CarlaBEV.tools.scene_designer", tokens) or 0
            )
        if command == "build-library":
            return int(
                _invoke_module_main("CarlaBEV.tools.build_scene_library", tokens) or 0
            )
        if command == "analyze-library":
            return int(
                _invoke_module_main("CarlaBEV.tools.analyze_scene_library", tokens) or 0
            )
        if command == "analyze-routes":
            return int(
                _invoke_module_main(
                    "CarlaBEV.tools.analyze_ego_route_distribution", tokens
                )
                or 0
            )
    elif group == "graph":
        command = _pop_command(tokens, "graph command")
        if command == "build-town01-canonical":
            return int(
                _invoke_module_main(
                    "CarlaBEV.tools.build_town01_canonical_graph", tokens
                )
                or 0
            )
        if command == "visualize-town01-canonical":
            return int(
                _invoke_module_main(
                    "CarlaBEV.tools.visualize_town01_canonical_graph", tokens
                )
                or 0
            )
        if command == "validate-town01-joins":
            return int(
                _invoke_module_main(
                    "CarlaBEV.tools.validate_town01_graph_joins", tokens
                )
                or 0
            )
    elif group == "validate":
        command = _pop_command(tokens, "validate command")
        if command == "simulator-semantics":
            return int(
                _invoke_module_main(
                    "CarlaBEV.tools.validate_simulator_semantics", tokens
                )
                or 0
            )
    elif group == "benchmark":
        from CarlaBEV.tools.benchmark import runner

        return int(runner.main(tokens) or 0)
    elif group == "viz":
        command = _pop_command(tokens, "viz command")
        if command == "semantic-channels":
            return int(
                _invoke_module_main(
                    "CarlaBEV.tools.visualize_semantic_channels", tokens
                )
                or 0
            )
        if command == "vehicle-temporal-fusion":
            return int(
                _invoke_module_main(
                    "CarlaBEV.tools.visualize_vehicle_temporal_fusion", tokens
                )
                or 0
            )

    raise SystemExit(f"Unsupported command.\n\n{USAGE}")


if __name__ == "__main__":
    raise SystemExit(main())
