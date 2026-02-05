#!/usr/bin/env python3

import argparse
import toml
from pathlib import Path
import ray

# ==============================
# Import pipeline entry functions
# ==============================
from scripts.segmentation import (
    segment_source as run_segment_source,
    segment_target as run_segment_target,
)

from scripts.embeddings import (
    embed_source as run_embed_source,
    embed_target as run_embed_target,
)

from scripts.tagging import (
    build_faiss_index as run_build_faiss_index,
    tag_target as run_tag_target,
)

# ==============================
# Config Loader
# ==============================
def load_config(config_path: str) -> dict:
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, "r") as f:
        return toml.load(f)


# ==============================
# Ray Initialization
# ==============================
def init_ray(pipeline_cfg: dict):
    if ray.is_initialized():
        return

    ray.init(
        num_cpus=pipeline_cfg["num_cpus"],
        num_gpus=pipeline_cfg["num_gpus"],
        object_store_memory=pipeline_cfg["object_store_gb"] * 1024**3,
        ignore_reinit_error=True,
    )


# ==============================
# Command Handlers
# ==============================
def cmd_segment_source(cfg):
    init_ray(cfg["tool"]["pipeline"])
    run_segment_source(cfg)


def cmd_embed_source(cfg):
    init_ray(cfg["tool"]["pipeline"])
    run_embed_source(cfg)


def cmd_segment_target(cfg, folder):
    init_ray(cfg["tool"]["pipeline"])
    run_segment_target(folder, cfg)


def cmd_embed_target(cfg, folder):
    init_ray(cfg["tool"]["pipeline"])
    run_embed_target(folder, cfg)


def cmd_build_faiss(cfg):
    init_ray(cfg["tool"]["pipeline"])
    run_build_faiss_index(cfg)


def cmd_tag_target(cfg, folder):
    init_ray(cfg["tool"]["pipeline"])
    run_tag_target(folder, cfg)


# ==============================
# CLI Definition
# ==============================
def build_parser():
    parser = argparse.ArgumentParser(
        description="Eros AutoTagging Pipeline (Ray-based)"
    )

    parser.add_argument(
        "--config",
        required=True,
        help="Path to config.toml",
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    # -------- Source --------
    subparsers.add_parser("segment-source")
    subparsers.add_parser("embed-source")

    # -------- Target --------
    p = subparsers.add_parser("segment-target")
    p.add_argument("--folder", required=True)

    p = subparsers.add_parser("embed-target")
    p.add_argument("--folder", required=True)

    # -------- Index / Tagging --------
    subparsers.add_parser("build-faiss")
    
    p = subparsers.add_parser("tag-target")
    p.add_argument("--folder", required=True)

    return parser


# ==============================
# Main Entrypoint
# ==============================
def main():
    parser = build_parser()
    args = parser.parse_args()

    cfg = load_config(args.config)

    print("Loaded config:")
    print(cfg)

    if args.command == "segment-source":
        cmd_segment_source(cfg)

    elif args.command == "embed-source":
        cmd_embed_source(cfg)

    elif args.command == "segment-target":
        cmd_segment_target(cfg, args.folder)

    elif args.command == "embed-target":
        cmd_embed_target(cfg, args.folder)

    elif args.command == "build-faiss":
        cmd_build_faiss(cfg)

    elif args.command == "tag-target":
        cmd_tag_target(cfg, args.folder)

    else:
        raise ValueError(f"Unknown command: {args.command}")

    # ✅ CLEAN RAY SHUTDOWN
    if ray.is_initialized():
        ray.shutdown()


if __name__ == "__main__":
    main()
