"""CLI tool to download conducting-related datasets."""

from __future__ import annotations

import argparse
import sys

from .datasets import AUTO_DATASETS, DATASETS, VALID_NAMES


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Download conducting-related datasets to a local directory.",
    )
    p.add_argument(
        "--data_dir",
        default="~/data/conducting",
        help="Base directory for downloads (default: ~/data/conducting)",
    )
    group = p.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--dataset",
        nargs="+",
        choices=VALID_NAMES,
        metavar="NAME",
        help=f"Dataset(s) to download. Choices: {', '.join(VALID_NAMES)}",
    )
    group.add_argument("--all", action="store_true", help="Download all datasets")
    p.add_argument("--zenodo-token", help="Zenodo API token (for MOSA)")
    p.add_argument("--url", help="User-provided download URL (for URMP)")
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would be downloaded without downloading",
    )
    return p


def resolve_datasets(args: argparse.Namespace) -> list[str]:
    if args.all:
        return list(DATASETS.keys())
    return args.dataset


def validate_args(args: argparse.Namespace) -> list[str]:
    """Return list of validation errors, empty if valid."""
    errors = []
    names = resolve_datasets(args)
    if "mosa" in names and args.zenodo_token is None and not args.dry_run:
        errors.append("MOSA requires --zenodo-token (or use --dry-run)")
    if "urmp" in names and args.url is None and not args.dry_run:
        errors.append("URMP requires --url (or use --dry-run)")
    return errors


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    from pathlib import Path

    from .state import is_complete, load_state

    data_dir = Path(args.data_dir).expanduser()
    names = resolve_datasets(args)

    if args.dry_run:
        print(f"Data directory: {data_dir}")
        print(f"Datasets to process: {', '.join(names)}\n")
        state = load_state(data_dir) if data_dir.exists() else {}
        for name in names:
            cfg = DATASETS[name]
            status = "COMPLETE" if is_complete(state, name) else cfg.access_tier.upper()
            print(f"  [{status}] {cfg.display_name} -> {data_dir / cfg.target_dir}")
            if cfg.access_tier == "manual":
                print(f"         (manual download — instructions only)")
            elif cfg.access_tier == "semi-manual":
                token_ok = name != "mosa" or args.zenodo_token
                url_ok = name != "urmp" or args.url
                if not (token_ok and url_ok):
                    print(f"         (missing credentials — will print instructions)")
        return 0

    # Validate non-dry-run args
    warnings = validate_args(args)
    for w in warnings:
        print(f"Warning: {w}", file=sys.stderr)

    data_dir.mkdir(parents=True, exist_ok=True)

    from .state import load_state, mark_complete, mark_failed

    state = load_state(data_dir)

    for name in names:
        if is_complete(state, name):
            print(f"[SKIP] {DATASETS[name].display_name} — already complete")
            continue

        cfg = DATASETS[name]
        print(f"\n[DOWNLOAD] {cfg.display_name} ({cfg.access_tier})")
        dest = data_dir / cfg.target_dir
        dest.mkdir(parents=True, exist_ok=True)

        try:
            if name == "phenicx":
                from .download_auto import download_phenicx
                download_phenicx(cfg, dest)
            elif name == "edinburgh":
                from .download_auto import download_edinburgh
                download_edinburgh(cfg, dest)
            elif name == "aist-plusplus":
                from .download_auto import download_aist_plusplus
                download_aist_plusplus(cfg, dest)
            elif name == "mosa":
                from .download_semi import download_mosa
                download_mosa(cfg, dest, args.zenodo_token)
            elif name == "urmp":
                from .download_semi import download_urmp
                download_urmp(cfg, dest, args.url)
            elif name == "human36m":
                from .readme import generate_instructions
                generate_instructions(cfg, data_dir)
                mark_complete(data_dir, state, name)
                continue

            mark_complete(data_dir, state, name)
            print(f"[DONE] {cfg.display_name}")
        except Exception as e:
            mark_failed(data_dir, state, name, str(e))
            print(f"[FAIL] {cfg.display_name}: {e}", file=sys.stderr)

    # Always regenerate README
    from .readme import generate_readme
    generate_readme(data_dir, state)

    return 0


if __name__ == "__main__":
    sys.exit(main())
