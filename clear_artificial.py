#!/usr/bin/env python3
import pathlib
TAG = "artif"
ROOT = pathlib.Path("./dataset/artificial")

def main():
    if not ROOT.exists():
        print("Nothing to clean.")
        return
    removed = 0
    for p in ROOT.rglob(f"{TAG}_*.wav"):
        try:
            p.unlink()
            removed += 1
        except Exception as e:
            print("Failed to remove", p, e)
    print(f"Removed {removed} files starting with '{TAG}_' under {ROOT}")

if __name__ == "__main__":
    main()
