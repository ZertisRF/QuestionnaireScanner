from __future__ import annotations

import sys

from questionnaire_scanner import main as scanner_main


def main(argv: list[str] | None = None) -> None:
    user_args = sys.argv[1:] if argv is None else argv
    scanner_main([
        "--crop-mode",
        "marker-box",
        "--output-width",
        "1700",
        "-o",
        "output_marker_crop",
        *user_args,
    ])


if __name__ == "__main__":
    main()
