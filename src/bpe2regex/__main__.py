import argparse
import logging
from pathlib import Path

from .encoding import Encoding

LOGGER = logging.getLogger("bpe2regex")
LOG_LEVELS = ("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL")


def _parse_encoding(value: str) -> Encoding:
    normalized = value.upper().replace("-", "_")
    try:
        return Encoding[normalized]
    except KeyError:
        try:
            return Encoding(value)
        except ValueError as error:
            names = ", ".join(item.name.lower() for item in Encoding)
            raise argparse.ArgumentTypeError(
                f"unknown encoding {value!r}; expected one of: {names}"
            ) from error


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="bpe2regex",
        description="Build compressed regex artifacts.",
    )
    commands = parser.add_subparsers(dest="command", required=True)
    build = commands.add_parser(
        "build",
        help="generate Python and ECMAScript binary artifacts",
    )
    build.add_argument("encoding", type=_parse_encoding)
    build.add_argument("directory", nargs="?", type=Path)
    build.add_argument(
        "--force",
        action="store_true",
        help="replace existing artifacts",
    )
    build.add_argument(
        "--log-level",
        choices=LOG_LEVELS,
        default="INFO",
        help="minimum build log level (default: INFO)",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, arguments.log_level),
        format="%(message)s",
    )
    if arguments.command != "build":
        raise AssertionError(arguments.command)

    from .build import build_regex_artifact

    result = build_regex_artifact(
        arguments.encoding,
        arguments.directory,
        overwrite=arguments.force,
        progress=LOGGER.info,
    )
    LOGGER.info("encoding=%s", result.encoding.name)
    LOGGER.info("directory=%s", result.directory)
    LOGGER.info("python_bytes=%d", result.metadata["python_artifact_bytes"])
    LOGGER.info("ecmascript_bytes=%d", result.metadata["ecmascript_artifact_bytes"])
    LOGGER.debug("build_metadata=%r", result.metadata)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
