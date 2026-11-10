"""Custom exceptions for the Breadfruit Dataset Engineering Pipeline."""


class PipelineError(Exception):
    """Base exception for pipeline errors."""
    pass


class ConfigurationError(PipelineError):
    """Raised when configuration is invalid or missing."""
    pass


class DatasetError(PipelineError):
    """Raised when dataset operations fail."""
    pass


class ImageProcessingError(PipelineError):
    """Raised when image processing fails."""
    pass


class ValidationError(PipelineError):
    """Raised when validation checks fail."""
    pass


class CorruptedImageError(ImageProcessingError):
    """Raised when an image file is corrupted or unreadable."""
    pass


class UnsupportedFormatError(ImageProcessingError):
    """Raised when image format is not supported."""
    pass


class DuplicateImageError(PipelineError):
    """Raised when duplicate image is detected."""
    pass


class QualityThresholdError(PipelineError):
    """Raised when image quality falls below threshold."""
    pass


class ObjectValidationError(PipelineError):
    """Raised when object validation fails."""
    pass


class FileOperationError(PipelineError):
    """Raised when file operations fail."""
    pass


class ReportGenerationError(PipelineError):
    """Raised when report generation fails."""
    pass


class PhaseExecutionError(PipelineError):
    """Raised when a pipeline phase fails."""
    def __init__(self, phase: str, message: str, original_error: Exception = None):
        self.phase = phase
        self.original_error = original_error
        super().__init__(f"Phase '{phase}' failed: {message}")


class InsufficientDataError(PipelineError):
    """Raised when dataset doesn't meet minimum requirements."""
    pass