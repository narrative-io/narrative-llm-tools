import argparse
from concurrent.futures import ThreadPoolExecutor
import sys

import tqdm

from narrative_llm_tools.state.conversation import validate_line


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate a .jsonl file against the Narrative Conversation File Format specification."
    )
    parser.add_argument("file", help="Path to the JSONL file to validate.")
    parser.add_argument("--threads", type=int, default=4, help="Number of threads to use")
    parser.add_argument("--quiet", "-q", action="store_true", help="Suppress progress bar")

    args = parser.parse_args()

    # Check for BOM (Byte-Order Mark) in the file.
    # If the file starts with a BOM, we'll report it before doing any other checks.
    with open(args.file, "rb") as f:
        start_bytes = f.read(4)
    if start_bytes.startswith(b"\xef\xbb\xbf"):
        print("Error: File contains a Byte-Order Mark (BOM). This is not allowed.")
        sys.exit(1)

    with open(args.file, "r", encoding="utf-8") as f:
        lines = f.readlines()

    if not lines:
        print("Error: File is empty (no lines present).")
        sys.exit(1)

    total_lines = len(lines)
    print(f"\nValidating {total_lines:,} lines using {args.threads} threads...")

     # Process lines in parallel
    errors = []
    with ThreadPoolExecutor(max_workers=args.threads) as executor:
        numbered_lines = [(line.rstrip("\n\r"), i + 1) for i, line in enumerate(lines)]
        
        # Create progress bar for all lines
        results = list(tqdm.tqdm(
            executor.map(validate_line, numbered_lines),
            total=len(numbered_lines),
            desc="Validating lines",
            disable=args.quiet,
            unit="lines"
        ))
        
        # Collect errors from results
        errors = [error for result in results for error in result.errors]

    if errors:
        print("Validation FAILED.\n")
        for err in sorted(errors, key=lambda x: int(x.split()[1].rstrip(':'))):  # Sort by line number
            print(err)
        sys.exit(1)
    else:
        print("Validation succeeded! No errors found.")

if __name__ == "__main__":
    main()