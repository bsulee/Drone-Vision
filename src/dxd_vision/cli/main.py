"""CLI entry point using Click."""

import sys
from pathlib import Path
from typing import Optional

import click

from dxd_vision import __version__
from dxd_vision.config.settings import load_config
from dxd_vision.utils.logging import setup_logging


@click.command()
@click.option(
    "--input", "-i",
    required=True,
    type=click.Path(exists=True, dir_okay=False, resolve_path=True),
    help="Path to input video file.",
)
@click.option(
    "--config", "-c",
    default="config/default.yaml",
    type=click.Path(resolve_path=True),
    help="Path to YAML configuration file.",
)
@click.option(
    "--output", "-o",
    default=None,
    type=click.Path(resolve_path=True),
    help="Output directory (overrides config).",
)
@click.option(
    "--fps",
    default=None,
    type=float,
    help="Target extraction FPS (overrides config).",
)
@click.option(
    "--verbose", "-v",
    is_flag=True,
    help="Enable debug logging.",
)
@click.version_option(version=__version__, prog_name="dxd-vision")
def main(input: str, config: str, output: Optional[str], fps: Optional[float], verbose: bool) -> None:
    """DXD Vision Engine -- AI-powered video analysis."""
    logger = setup_logging(verbose=verbose)

    # Load and apply config overrides.
    cfg = load_config(config)
    if fps is not None:
        cfg.extraction.target_fps = fps
    if output is not None:
        cfg.extraction.output_dir = output

    # Validate input format.
    input_path = Path(input)
    if input_path.suffix.lower() not in cfg.extraction.supported_formats:
        click.echo(
            f"Error: Unsupported format '{input_path.suffix}'. "
            f"Supported: {', '.join(cfg.extraction.supported_formats)}",
            err=True,
        )
        sys.exit(1)

    logger.info("DXD Vision Engine v%s", __version__)
    logger.info("Input: %s", input_path)
    logger.info("Target FPS: %s", cfg.extraction.target_fps)
    logger.debug("Config: %s", cfg.model_dump())

    # Import display here to allow --help to work before dependencies are installed.
    from dxd_vision.cli.display import DisplayManager

    display = DisplayManager()
    display.show_header()

    try:
        from dxd_vision.pipeline.pipeline import VisionPipeline
    except (ImportError, AttributeError):
        display.show_error(
            "Pipeline not yet implemented. "
            "Waiting on Back-End Engineer (Beads 4-6)."
        )
        sys.exit(1)

    try:
        pipeline = VisionPipeline(cfg)
        result = pipeline.process_video(input)
        display.show_results(result)
        logger.info("Extraction complete: %d frames", result.frames_extracted)
    except FileNotFoundError as exc:
        display.show_error(str(exc))
        sys.exit(1)
    except Exception as exc:
        logger.exception("Pipeline failed")
        display.show_error(f"Pipeline error: {exc}")
        sys.exit(1)
