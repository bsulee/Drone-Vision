"""Progress display and results formatting with Rich."""

from rich.console import Console
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)
from rich.table import Table

from dxd_vision import __version__
from dxd_vision.models.frame import ExtractionResult, VideoInfo
from dxd_vision.models.detection import (
    DetectionSummary,
    FrameDetections,
    ProcessingResult,
)


class DisplayManager:
    """Handles all terminal output formatting for DXD Vision Engine."""

    # Threat-level color coding for detection classes.
    CLASS_COLORS = {
        "weapon": "red",
        "person": "yellow",
        "vehicle": "green",
        "package": "blue",
    }

    def __init__(self, console: Console = None):
        self.console = console or Console(stderr=True)
        # Detect terminal width and adapt layouts accordingly.
        self.term_width = self.console.width

    def show_header(self) -> None:
        """Display application header."""
        # Adapt panel width to terminal, minimum 40 chars.
        panel_width = min(max(self.term_width - 4, 40), 60)
        self.console.print(
            Panel(
                f"[bold]DXD Vision Engine[/bold]  v{__version__}\n"
                "AI-Powered Video Analysis",
                border_style="blue",
                width=panel_width,
            )
        )

    def show_video_info(self, info: VideoInfo) -> None:
        """Display video metadata in a table."""
        # Adapt table width to terminal.
        table_width = max(self.term_width - 4, 30) if self.term_width < 40 else None
        table = Table(
            title="Video Information",
            show_header=False,
            border_style="dim",
            width=table_width,
        )
        table.add_column("Property", style="bold")
        table.add_column("Value")

        table.add_row("File", info.path)
        table.add_row("Resolution", f"{info.width}x{info.height}")
        table.add_row("FPS", f"{info.fps:.1f}")
        table.add_row("Duration", f"{info.duration_seconds:.1f}s")
        table.add_row("Total Frames", str(info.total_frames))
        table.add_row("Codec", info.codec)

        self.console.print(table)

    def create_progress(self, total_frames: int) -> Progress:
        """Create a progress bar for frame extraction.

        Usage:
            progress = display.create_progress(total)
            with progress:
                task = progress.add_task("Extracting", total=total)
                for frame in frames:
                    progress.update(task, advance=1)
        """
        # Adapt progress bar width to terminal (minimum 20).
        bar_width = max(min(self.term_width - 50, 40), 20)
        return Progress(
            SpinnerColumn(),
            TextColumn("[bold blue]{task.description}"),
            BarColumn(bar_width=bar_width),
            MofNCompleteColumn(),
            TimeElapsedColumn(),
            TimeRemainingColumn(),
            console=self.console,
        )

    def show_results(self, result: ExtractionResult | ProcessingResult) -> None:
        """Display extraction or processing results (supports both types)."""
        # Check if this is a ProcessingResult (Phase 2+) or ExtractionResult (Phase 1).
        if isinstance(result, ProcessingResult):
            self.show_processing_results(result)
        else:
            # Legacy Phase 1 extraction-only display.
            table_width = max(self.term_width - 4, 30) if self.term_width < 40 else None
            table = Table(
                title="Extraction Results",
                show_header=False,
                border_style="green",
                width=table_width,
            )
            table.add_column("Property", style="bold")
            table.add_column("Value")

            table.add_row("Frames Extracted", str(result.frames_extracted))
            table.add_row("Extraction FPS", f"{result.extraction_fps:.1f}")
            table.add_row(
                "Source",
                f"{result.video_info.path} ({result.video_info.duration_seconds:.1f}s)",
            )
            if result.sample_frame_path:
                table.add_row("Sample Frame", result.sample_frame_path)

            self.console.print(table)
            self.console.print("[green bold]Done.[/green bold]")

    def show_error(self, message: str) -> None:
        """Display an error message."""
        self.console.print(f"[red bold]Error:[/red bold] {message}")

    def show_detection_summary(self, summary: DetectionSummary) -> None:
        """Display detection summary with class counts and statistics."""
        table_width = max(self.term_width - 4, 30) if self.term_width < 40 else None
        table = Table(
            title="Detection Summary",
            show_header=False,
            border_style="cyan",
            width=table_width,
        )
        table.add_column("Property", style="bold")
        table.add_column("Value")

        # Class counts with threat-level colors.
        if summary.by_class:
            for class_name, count in sorted(summary.by_class.items()):
                color = self.CLASS_COLORS.get(class_name.lower(), "white")
                table.add_row(
                    class_name.capitalize(),
                    f"[{color}]{count}[/{color}]"
                )

        # Summary statistics.
        table.add_row("─" * 15, "─" * 15)  # Separator
        table.add_row("Total Detections", str(summary.total_detections))
        table.add_row("Avg Confidence", f"{summary.avg_confidence:.2f}")
        table.add_row(
            "Frames with Detections",
            f"{summary.frames_with_detections}/{summary.frames_with_detections + summary.frames_without_detections}"
        )

        self.console.print(table)

    def show_detection_details(self, detections: list[FrameDetections], top_n: int = 10) -> None:
        """Display top N highest-confidence detections (verbose mode)."""
        # Flatten all detections and sort by confidence.
        all_detections = []
        for frame_det in detections:
            for det in frame_det.detections:
                all_detections.append((frame_det.frame_number, det))

        # Sort by confidence descending.
        all_detections.sort(key=lambda x: x[1].confidence, reverse=True)
        top_detections = all_detections[:top_n]

        if not top_detections:
            self.console.print("[dim]No detections to display.[/dim]")
            return

        table_width = max(self.term_width - 4, 30) if self.term_width < 40 else None
        table = Table(
            title=f"Top {len(top_detections)} Detections",
            border_style="dim",
            width=table_width,
        )
        table.add_column("Frame", style="cyan")
        table.add_column("Class", style="bold")
        table.add_column("Confidence", justify="right")
        table.add_column("BBox (x,y,w,h)", style="dim")

        for frame_num, det in top_detections:
            color = self.CLASS_COLORS.get(det.class_name.lower(), "white")
            table.add_row(
                str(frame_num),
                f"[{color}]{det.class_name}[/{color}]",
                f"{det.confidence:.2f}",
                f"{det.bbox.x},{det.bbox.y},{det.bbox.width},{det.bbox.height}"
            )

        self.console.print(table)

    def show_processing_results(self, result: ProcessingResult) -> None:
        """Display combined extraction + detection results."""
        # Show extraction results first.
        table_width = max(self.term_width - 4, 30) if self.term_width < 40 else None
        table = Table(
            title="Processing Results",
            show_header=False,
            border_style="green",
            width=table_width,
        )
        table.add_column("Property", style="bold")
        table.add_column("Value")

        # Extraction info.
        table.add_row("Frames Extracted", str(result.frames_extracted))
        table.add_row("Extraction FPS", f"{result.extraction_fps:.1f}")
        table.add_row(
            "Source",
            f"{result.video_info.path} ({result.video_info.duration_seconds:.1f}s)",
        )
        if result.sample_frame_path:
            table.add_row("Sample Frame", result.sample_frame_path)

        self.console.print(table)

        # Show detection results if enabled.
        if result.detection_enabled and result.detection_summary:
            self.console.print()  # Blank line
            self.show_detection_summary(result.detection_summary)

            # Show output file paths.
            if result.detections_path or result.annotated_frame_path:
                self.console.print()
                output_table = Table(
                    title="Detection Output Files",
                    show_header=False,
                    border_style="dim",
                    width=table_width,
                )
                output_table.add_column("Type", style="bold")
                output_table.add_column("Path")

                if result.detections_path:
                    output_table.add_row("Detections JSON", result.detections_path)
                if result.annotated_frame_path:
                    output_table.add_row("Annotated Frame", result.annotated_frame_path)

                self.console.print(output_table)

        self.console.print("[green bold]Done.[/green bold]")
