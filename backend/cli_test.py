"""
Quick offline sanity check — runs the detection pipeline on an image and
prints the result, without needing FastAPI/uvicorn installed.

Usage:
    python3 cli_test.py ../sample_images/sample_authentic_looking.jpg
"""

import sys
from PIL import Image
import detector


def main():
    if len(sys.argv) != 2:
        print("Usage: python3 cli_test.py <path-to-image>")
        sys.exit(1)

    path = sys.argv[1]
    image = Image.open(path)
    result = detector.analyze_image(image)

    print(f"\nFile:    {path}")
    print(f"Score:   {result.score}/100")
    print(f"Verdict: {result.verdict}\n")

    print("Breakdown:")
    print(f"  Metadata check : {result.metadata['score']}/100")
    print(f"  ELA forensics  : {result.ela['score']}/100  (hotspot ratio: {result.ela['hotspot_ratio']})")
    print(f"  Pattern check  : {result.pattern['score']}/100")

    print("\nEvidence:")
    for e in result.evidence:
        print(f"  - {e}")
    print()


if __name__ == "__main__":
    main()
