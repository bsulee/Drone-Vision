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


class DisplayManager:
    """Handles all terminal output formatting for DXD Vision Engine."""

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

    def show_results(self, result: ExtractionResult) -> None:
        """Display extraction results summary."""
        # Adapt table width to terminal.
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
