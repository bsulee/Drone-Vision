"""CLI entry point using Click."""

import sys
from pathlib import Path
from typing import Optional

import click
import yaml
from pydantic import ValidationError

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
@click.option(
    "--detect", "-d",
    is_flag=True,
    help="Enable YOLO object detection.",
)
@click.option(
    "--model", "-m",
    default=None,
    type=click.Path(resolve_path=True),
    help="YOLO model path (overrides config).",
)
@click.option(
    "--confidence",
    default=None,
    type=float,
    help="Detection confidence threshold 0.0-1.0 (overrides config).",
)
@click.option(
    "--classes",
    default=None,
    help="Comma-separated target classes (e.g., 'person,vehicle').",
)
@click.version_option(version=__version__, prog_name="dxd-vision")
def main(
    input: str,
    config: str,
    output: Optional[str],
    fps: Optional[float],
    verbose: bool,
    detect: bool,
    model: Optional[str],
    confidence: Optional[float],
    classes: Optional[str],
) -> None:
    """DXD Vision Engine -- AI-powered video analysis."""
    logger = setup_logging(verbose=verbose)

    # Load and apply config overrides.
    try:
        cfg = load_config(config)
    except yaml.YAMLError as exc:
        click.echo(f"Error: Malformed YAML config file '{config}': {exc}", err=True)
        sys.exit(1)
    except ValidationError as exc:
        click.echo(f"Error: Invalid config values in '{config}': {exc}", err=True)
        sys.exit(1)
    except Exception as exc:
        click.echo(f"Error: Failed to load config '{config}': {exc}", err=True)
        sys.exit(1)

    # Validate and apply extraction CLI overrides.
    if fps is not None:
        if fps <= 0:
            click.echo(f"Error: FPS must be positive (got {fps})", err=True)
            sys.exit(1)
        cfg.extraction.target_fps = fps
    if output is not None:
        cfg.extraction.output_dir = output

    # Validate and apply detection CLI overrides.
    if detect:
        cfg.detection.enabled = True
    if model is not None:
        model_path = Path(model)
        if not model_path.exists():
            click.echo(f"Error: Model file not found: '{model}'", err=True)
            sys.exit(1)
        cfg.detection.model_path = model
    if confidence is not None:
        if not (0.0 <= confidence <= 1.0):
            click.echo(f"Error: Confidence must be between 0.0 and 1.0 (got {confidence})", err=True)
            sys.exit(1)
        cfg.detection.confidence_threshold = confidence
    if classes is not None:
        parsed_classes = [c.strip() for c in classes.split(",") if c.strip()]
        if not parsed_classes:
            click.echo("Error: --classes must contain at least one class name", err=True)
            sys.exit(1)
        cfg.detection.target_classes = parsed_classes

    # Ensure output directory exists and is writable.
    output_dir = Path(cfg.extraction.output_dir)
    try:
        output_dir.mkdir(parents=True, exist_ok=True)
    except PermissionError:
        click.echo(f"Error: Permission denied creating output directory '{output_dir}'", err=True)
        sys.exit(1)
    except Exception as exc:
        click.echo(f"Error: Failed to create output directory '{output_dir}': {exc}", err=True)
        sys.exit(1)

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

    # Detection logging.
    if cfg.detection.enabled:
        logger.info("Detection: ENABLED")
        logger.info("Model: %s", cfg.detection.model_path)
        logger.info("Confidence threshold: %.2f", cfg.detection.confidence_threshold)
        logger.info("Target classes: %s", ", ".join(cfg.detection.target_classes))
        logger.debug("Detection device: %s", cfg.detection.device)
    else:
        logger.info("Detection: DISABLED (extraction only)")

    logger.debug("Full config: %s", cfg.model_dump())

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

        # Log completion summary.
        logger.info("Extraction complete: %d frames", result.frames_extracted)
        if cfg.detection.enabled and hasattr(result, 'detection_summary') and result.detection_summary:
            summary = result.detection_summary
            logger.info(
                "Detection complete: %d objects in %d/%d frames",
                summary.total_detections,
                summary.frames_with_detections,
                result.frames_extracted,
            )
            logger.info("Avg confidence: %.2f", summary.avg_confidence)
            if summary.by_class:
                logger.info("Detections by class: %s", summary.by_class)
    except FileNotFoundError as exc:
        display.show_error(str(exc))
        sys.exit(1)
    except Exception as exc:
        logger.exception("Pipeline failed")
        display.show_error(f"Pipeline error: {exc}")
        sys.exit(1)
