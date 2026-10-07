import sys
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.autonomous_qa.datasets.vietmdd.vietmdd_response_format import build_mixed_format_release

def main() -> int:
    res = build_mixed_format_release()
    print("=== RELEASE BUILD SUCCESSFUL ===")
    print(json.dumps(res["manifest"], indent=2, ensure_ascii=False))
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
