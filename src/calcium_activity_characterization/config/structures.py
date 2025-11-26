# Usage example:
# --------------------------------------------------------------------
# from pathlib import Path
# from calcium_activity_characterization.config.structures import (
#     GlobalConfig,
#     SegmentationConfig,
#     ImageProcessingConfig,
# )
#
# # Load segmentation config from JSON
# seg_cfg = SegmentationConfig.from_json(Path("configs/segmentation.json"))
#
# # Manually construct a minimal GlobalConfig (usually done in presets)
# global_cfg = GlobalConfig(
#     debug=DebugConfig(),
#     data_dir="D:/data",
#     segmentation=seg_cfg,
#     image_processing_hoechst=ImageProcessingConfig(),
#     image_processing_fitc=ImageProcessingConfig(),
#     trace_extraction=TraceExtractionConfig(),
#     cell_filtering=CellFilteringConfig(),
#     cell_trace_processing=SignalProcessingConfig(),
#     cell_trace_peak_detection=PeakDetectionConfig(),
#     activity_trace_processing=SignalProcessingConfig(),
#     activity_trace_peak_detection=PeakDetectionConfig(),
#     event_extraction=EventExtractionConfig(),
#     export=ExportConfig(),
# )
# --------------------------------------------------------------------

"""
Configuration structures for the calcium activity characterization pipeline.

This module defines all configuration dataclasses and enums used throughout the
project, including:

- Debug and I/O paths
- Segmentation and image preprocessing
- Trace extraction and cell filtering
- Signal processing and peak detection
- Event extraction and export parameters
- (Deprecated/experimental) clustering, correlation, and causality parameters

Most configuration blocks are plain dataclasses and can be serialized to/from
JSON when appropriate.
"""

from __future__ import annotations

from abc import ABC
from dataclasses import dataclass, field, asdict
from enum import Enum
from pathlib import Path
import json

from calcium_activity_characterization.logger import get_logger


logger = get_logger(__name__)


# ===========================
# FLAGS
# ===========================
@dataclass
class DebugConfig:
    """
    Configuration for debugging and development paths.

    Attributes:
        debugging: Enable or disable debug mode.
        debugging_folder_path: Path to a dataset or experiment used for debugging.
    """

    debugging: bool = True
    debugging_folder_path: str = "C:/"


# ===========================
# SEGMENTATION PARAMETERS
# ===========================
class SegmentationMethod(str, Enum):
    """
    Enum for available segmentation methods.
    """

    MESMER = "mesmer"
    CELLPOSE = "cellpose"  # Not implemented yet
    WATERSHED = "watershed"  # Not implemented yet


@dataclass
class SegmentationParams(ABC):
    """
    Base class for segmentation parameters.
    """

    pass


@dataclass
class MesmerParams(SegmentationParams):
    """
    Parameters for MESMER segmentation.

    Attributes:
        image_mpp: Microns per pixel.
        maxima_threshold: Threshold for maxima detection.
        maxima_smooth: Smoothing for maxima.
        interior_threshold: Threshold for interior probability.
        interior_smooth: Smoothing for interior mask.
        small_objects_threshold: Minimum object size.
        fill_holes_threshold: Maximum hole size to fill.
        radius: Radius used for boundary refinement.
    """

    image_mpp: float = 0.5
    maxima_threshold: float = 0.2
    maxima_smooth: float = 2.5
    interior_threshold: float = 0.05
    interior_smooth: float = 1.0
    small_objects_threshold: int = 25
    fill_holes_threshold: int = 15
    radius: int = 2


@dataclass
class CellPoseParams(SegmentationParams):
    """
    Parameters for CellPose segmentation.

    Currently not implemented.
    """

    pass


@dataclass
class WatershedParams(SegmentationParams):
    """
    Parameters for Watershed segmentation.

    Currently not implemented.
    """

    pass


@dataclass
class SegmentationConfig:
    """
    Full segmentation configuration.

    Controls which segmentation method is used and its corresponding parameters.

    Attributes:
        method: Chosen segmentation method.
        params: Parameters associated with the selected method.
        save_overlay: Whether to save an overlay of the segmentation results.
    """

    method: SegmentationMethod = SegmentationMethod.MESMER
    params: SegmentationParams = field(default_factory=MesmerParams)
    save_overlay: bool = True

    @staticmethod
    def from_json(fp: Path) -> "SegmentationConfig":
        """
        Load segmentation configuration from a JSON file.

        The JSON file is expected to contain:
            - "method": one of the SegmentationMethod values
            - "params": a dict of parameters for the chosen method
            - "save_overlay": optional bool

        Args:
            fp: Path to the JSON configuration file.

        Returns:
            SegmentationConfig: Parsed segmentation configuration.

        Raises:
            ValueError: If the method is unknown or the file is malformed.
            OSError: If the file cannot be read.
            json.JSONDecodeError: If the file is not valid JSON.
        """
        try:
            payload = json.loads(fp.read_text())
            method = SegmentationMethod(payload["method"])
            params_data = payload["params"]

            if method is SegmentationMethod.MESMER:
                params = MesmerParams(**params_data)
            elif method is SegmentationMethod.CELLPOSE:
                params = CellPoseParams(**params_data)
            elif method is SegmentationMethod.WATERSHED:
                params = WatershedParams(**params_data)
            else:
                raise ValueError(f"Unknown segmentation method: {method}")

            save_overlay = payload.get("save_overlay", True)
            return SegmentationConfig(method=method, params=params, save_overlay=save_overlay)

        except Exception as exc:
            logger.error(f"Failed to load SegmentationConfig from '{fp}': {exc}")
            raise

    def to_json(self, fp: Path) -> None:
        """
        Serialize the segmentation configuration to a JSON file.

        Args:
            fp: Path where the JSON configuration will be written.

        Returns:
            None
        """
        try:
            payload = {
                "method": self.method.value,
                "params": asdict(self.params),
                "save_overlay": self.save_overlay,
            }
            fp.parent.mkdir(parents=True, exist_ok=True)
            fp.write_text(json.dumps(payload, indent=2))
        except Exception as exc:
            logger.error(f"Failed to write SegmentationConfig to '{fp}': {exc}")
            raise


# ===========================
# IMAGE PROCESSING PARAMETERS
# ===========================
class HotPixelMethod(str, Enum):
    """
    Enum for available hot pixel correction methods.
    """

    REPLACE = "replace"
    CLIP = "clip"


@dataclass
class HotPixelParameters:
    """
    Parameters for hot pixel correction.

    Attributes:
        method: Strategy for correction.
        use_auto_threshold: Use automatic thresholding.
        percentile: Percentile used for auto-threshold.
        mad_scale: Scale factor for MAD.
        static_threshold: Absolute intensity threshold for static mode.
        window_size: Window size for filtering.
    """

    method: HotPixelMethod = HotPixelMethod.REPLACE
    use_auto_threshold: bool = True
    percentile: float = 99.9
    mad_scale: float = 20.0
    static_threshold: int = 2000
    window_size: int = 3


@dataclass
class ImageProcessingPipeline:
    """
    Toggle steps in the image preprocessing pipeline.

    Attributes:
        padding: Apply zero-padding to images.
        cropping: Apply center-cropping to ROI.
        hot_pixel_cleaning: Apply hot pixel correction.
    """

    padding: bool = True
    cropping: bool = True
    hot_pixel_cleaning: bool = False


@dataclass
class ImageProcessingConfig:
    """
    Configuration for image preprocessing.

    Attributes:
        pipeline: Which steps to apply.
        padding_digits: Digits used in file naming (e.g., 5 -> t00001).
        roi_scale: Fraction of image to crop (centered). 1.0 = no crop.
        roi_centered: Whether the ROI crop is centered in the image.
        hot_pixel_cleaning: Parameters for hot pixel cleanup.
    """

    pipeline: ImageProcessingPipeline = field(default_factory=ImageProcessingPipeline)
    padding_digits: int = 5
    roi_scale: float = 0.75
    roi_centered: bool = True
    hot_pixel_cleaning: HotPixelParameters = field(default_factory=HotPixelParameters)

    @staticmethod
    def from_json(fp: Path) -> "ImageProcessingConfig":
        """
        Load image processing configuration from a JSON file.

        The JSON file may contain:
            - "pipeline": dict to initialize ImageProcessingPipeline
            - "padding_digits": int
            - "roi_scale": float
            - "roi_centered": bool
            - "hot_pixel_cleaning": dict to initialize HotPixelParameters

        Args:
            fp: Path to the JSON configuration file.

        Returns:
            ImageProcessingConfig: Parsed configuration object.

        Raises:
            OSError: If the file cannot be read.
            json.JSONDecodeError: If the file is not valid JSON.
            ValueError: If the hot pixel method is invalid.
        """
        try:
            payload = json.loads(fp.read_text())

            pipeline_data = payload.get("pipeline", {})
            pipeline = ImageProcessingPipeline(**pipeline_data)

            hot_pixel_data = payload.get("hot_pixel_cleaning", {})
            method_str = hot_pixel_data.get("method", HotPixelMethod.REPLACE)
            method = HotPixelMethod(method_str)
            hot_pixel_data["method"] = method
            hot_pixel_cleaning = HotPixelParameters(**hot_pixel_data)

            return ImageProcessingConfig(
                pipeline=pipeline,
                padding_digits=payload.get("padding_digits", 5),
                roi_scale=payload.get("roi_scale", 0.75),
                roi_centered=payload.get("roi_centered", True),
                hot_pixel_cleaning=hot_pixel_cleaning,
            )
        except Exception as exc:
            logger.error(f"Failed to load ImageProcessingConfig from '{fp}': {exc}")
            raise


# ===========================
# TRACE EXTRACTION PARAMETERS
# ===========================
@dataclass
class TraceExtractionConfig:
    """
    Configuration for how cell traces are extracted from image sequences.

    Attributes:
        parallelize: Whether to CPU-parallelize the extraction.
        trace_version_name: Name of the trace version (e.g., 'raw').
    """

    parallelize: bool = True
    trace_version_name: str = "raw"


# ===========================
# CELL DETECTION PARAMETERS
# ===========================
@dataclass
class ObjectSizeThresholds:
    """
    Size constraints for detected objects.

    Attributes:
        min: Minimum object area in pixels.
        max: Maximum object area in pixels.
    """

    min: int = 500
    max: int = 10000


@dataclass
class CellFilteringConfig:
    """
    Configuration for filtering segmented cells.

    Attributes:
        border_margin: Exclude cells within this many pixels from image border.
        object_size_thresholds: Min/max size thresholds for filtering.
    """

    border_margin: int = 20
    object_size_thresholds: ObjectSizeThresholds = field(default_factory=ObjectSizeThresholds)


# ===========================
# PEAK DETECTION PARAMETERS
# ===========================
class PeakDetectionMethod(str, Enum):
    """
    Enum of supported peak detection strategies.
    """

    SKIMAGE = "skimage"
    CUSTOM = "custom"
    THRESHOLD = "threshold"


@dataclass
class PeakDetectorParams(ABC):
    """
    Abstract base class for peak detection parameters.
    """

    pass


@dataclass
class SkimageParams(PeakDetectorParams):
    """
    Parameters for peak detection using skimage-like logic.

    Attributes:
        prominence: Minimum prominence.
        distance: Minimum distance between peaks in frames.
        height: Minimum peak height.
        threshold: Threshold for detecting a peak.
        width: Minimum peak width.
        scale_class_quantiles: Quantiles to assign peak scale class.
        full_half_width: Height relative to max for filtering.
        full_duration_threshold: Duration relative to full trace length.
    """

    prominence: float | None = None
    distance: int = 10
    height: float = 20
    threshold: float | None = None
    width: float | None = None
    scale_class_quantiles: tuple[float, float] = (0.33, 0.66)
    full_half_width: float = 0.3
    full_duration_threshold: float = 0.95


@dataclass
class CustomParams(PeakDetectorParams):
    """
    Parameters for custom peak detection.

    Currently not implemented.
    """

    pass


@dataclass
class ThresholdParams(PeakDetectorParams):
    """
    Parameters for threshold-based peak detection.

    Currently not implemented.
    """

    pass


@dataclass
class PeakGroupingParams:
    """
    Parameters controlling how closely-timed peaks are grouped.

    Attributes:
        overlap_margin: Allowed frame overlap between peaks.
    """

    overlap_margin: int = 0


@dataclass
class PeakDetectionConfig:
    """
    Full peak detection configuration.

    Attributes:
        verbose: Whether to print debug information.
        method: Detection method to use.
        params: Parameters associated with method.
        peak_grouping: Parameters for peak grouping.
        start_frame: Optional start frame to restrict detection.
        end_frame: Optional end frame to restrict detection.
        filter_overlapping_peaks: Whether to remove overlapping peaks.
        refine_durations: Whether to refine peak durations using local minima.
    """

    verbose: bool = False
    method: PeakDetectionMethod = PeakDetectionMethod.SKIMAGE
    params: PeakDetectorParams = field(default_factory=SkimageParams)
    peak_grouping: PeakGroupingParams = field(default_factory=PeakGroupingParams)
    start_frame: int | None = None
    end_frame: int | None = None
    filter_overlapping_peaks: bool = False
    refine_durations: bool = False


# ===========================
# NORMALIZATION METHODS PARAMETERS
# ===========================
class NormalizationMethod(str, Enum):
    """
    Enum of supported normalization methods.
    """

    DELTAF = "deltaf"
    ZSCORE = "zscore"
    MINMAX = "minmax"
    PERCENTILE = "percentile"


@dataclass
class NormalizationParams(ABC):
    """
    Abstract base class for normalization parameter sets.

    Attributes:
        epsilon: Small value to avoid division by zero.
    """

    epsilon: float = 1e-8


@dataclass
class ZScoreParams(NormalizationParams):
    """
    Parameters for Z-score normalization.

    Attributes:
        smoothing_sigma: Standard deviation for Gaussian smoothing.
        residuals_clip_percentile: Percentile for clipping residuals.
        residuals_min_number: Minimum number of residuals for analysis.
    """

    smoothing_sigma: float = 2.0
    residuals_clip_percentile: float = 80.0
    residuals_min_number: int = 20


@dataclass
class PercentileParams(NormalizationParams):
    """
    Parameters for percentile-based normalization.

    Attributes:
        percentile_baseline: Percentile used for baseline calculation.
    """

    percentile_baseline: float = 10.0


@dataclass
class MinMaxParams(NormalizationParams):
    """
    Parameters for min-max normalization.

    Attributes:
        min_range: Minimum range value.
    """

    min_range: float = 1e-2


@dataclass
class DeltaFParams(NormalizationParams):
    """
    Parameters for deltaF/F normalization.

    Attributes:
        percentile_baseline: Percentile used for baseline calculation.
        min_range: Minimum range value.
    """

    percentile_baseline: float = 10.0
    min_range: float = 1e-2


# ===========================
# DETRENDING METHODS PARAMETERS
# ===========================
class DetrendingMethod(str, Enum):
    """
    Enum of supported detrending methods.
    """

    LOCALMINIMA = "localminima"
    MOVINGAVERAGE = "movingaverage"
    POLYNOMIAL = "polynomial"
    ROBUSTPOLY = "robustpoly"
    EXPONENTIAL = "exponentialfit"
    SAVGOL = "savgol"
    BUTTERWORTH = "butterworth"
    FIR = "fir"
    WAVELET = "wavelet"
    DOUBLECURVE = "doublecurvefitting"


@dataclass
class DetrendingParams(ABC):
    """
    Abstract base class for detrending parameters.

    Attributes:
        cut_trace_num_points: Number of points to cut from the trace.
    """

    cut_trace_num_points: int = 100


@dataclass
class BaselineSubtractionDetrendingParams(DetrendingParams):
    """
    Parameters for baseline subtraction detrending.

    Attributes:
        baseline_detection_params: Parameters for baseline detection.
    """

    baseline_detection_params: PeakDetectionConfig = field(default_factory=PeakDetectionConfig)


@dataclass
class MovingAverageParams(BaselineSubtractionDetrendingParams):
    """
    Parameters for moving average detrending.

    Attributes:
        window_size: Size of the moving average window.
    """

    window_size: int = 201


@dataclass
class PolynomialParams(BaselineSubtractionDetrendingParams):
    """
    Parameters for polynomial detrending.

    Attributes:
        degree: Degree of the polynomial fit.
    """

    degree: int = 2


@dataclass
class RobustPolyParams(BaselineSubtractionDetrendingParams):
    """
    Parameters for robust polynomial detrending.

    Attributes:
        degree: Degree of the polynomial fit.
        method: Robust fitting method name.
    """

    degree: int = 2
    method: str = "huber"  # or "ransac"


@dataclass
class SavgolParams:
    """
    Parameters for Savitzky–Golay filter detrending.

    Attributes:
        window_length: Length of the filter window. Must be odd.
        polyorder: Order of the polynomial used to fit the samples.
    """

    window_length: int = 101
    polyorder: int = 2


@dataclass
class FilterDetrendingParams(DetrendingParams):
    """
    Parameters for filter-based detrending.

    Attributes:
        sampling_freq: Sampling frequency of the signal.
    """

    sampling_freq: float = 1.0


@dataclass
class ButterworthParams(FilterDetrendingParams):
    """
    Parameters for Butterworth filter detrending.

    Attributes:
        cutoff: Cutoff frequency for the filter.
        order: Order of the filter.
    """

    cutoff: float = 0.003
    order: int = 6


@dataclass
class FIRParams(FilterDetrendingParams):
    """
    Parameters for FIR filter detrending.

    Attributes:
        cutoff: Cutoff frequency for the filter.
        numtaps: Number of taps in the FIR filter.
    """

    cutoff: float = 0.001
    numtaps: int = 201


@dataclass
class WaveletParams(FilterDetrendingParams):
    """
    Parameters for wavelet detrending.

    Attributes:
        wavelet: Name of the wavelet to use.
        level: Level of decomposition.
    """

    wavelet: str = "db4"
    level: int = 3


@dataclass
class DoubleCurveFittingParams(DetrendingParams):
    """
    Parameters for double curve fitting detrending.

    Attributes:
        fit_method: Fitting method to use.
        window_size: Size of the moving average window.
        mask_method: Masking method to use.
        percentile_bounds: Percentile bounds for masking.
        max_iterations: Maximum number of iterations for fitting.
    """

    fit_method: str = "movingaverage"
    window_size: int = 121
    mask_method: str = "percentile"
    percentile_bounds: tuple[int, int] = (0, 75)
    max_iterations: int = 5


@dataclass
class LocalMinimaParams(DetrendingParams):
    """
    Parameters for local minima detrending.

    Attributes:
        verbose: Whether to print debug information.
        minima_detection_order: Order of the minima detection.
        edge_anchors_window: Window size for edge anchors.
        edge_anchors_delta: Delta for edge anchors.
        filtering_shoulder_neighbor_dist: Distance for shoulder neighbor filtering.
        filtering_shoulder_window: Window size for shoulder filtering.
        filtering_angle_thresh_deg: Angle threshold for filtering shoulders.
        crossing_correction_min_dist: Minimum distance for crossing correction.
        crossing_correction_max_iterations: Maximum iterations for crossing correction.
        fitting_method: Method for fitting the local minima.
        diagnostics_enabled: Whether to enable diagnostics.
        diagnostics_output_dir: Directory for diagnostics output.
    """

    verbose: bool = False
    minima_detection_order: int = 15

    edge_anchors_window: int = 50
    edge_anchors_delta: float = 0.03

    filtering_shoulder_neighbor_dist: int = 400
    filtering_shoulder_window: int = 100
    filtering_angle_thresh_deg: int = 10

    crossing_correction_min_dist: int = 10
    crossing_correction_max_iterations: int = 10

    fitting_method: str = "linear"

    diagnostics_enabled: bool = False
    diagnostics_output_dir: str = (
        "D:/Mateo/20250326/Output/IS1/debugging/detrending-diagnostics"
    )


# ===========================
# SIGNAL PROCESSING PARAMETERS
# ===========================
@dataclass
class SignalProcessingPipeline:
    """
    Configuration for the signal processing pipeline.

    Attributes:
        detrending: Whether to apply detrending.
        normalization: Whether to apply normalization.
        smoothing: Whether to apply smoothing.
    """

    detrending: bool = True
    normalization: bool = True
    smoothing: bool = True


@dataclass
class NormalizationConfig:
    """
    Configuration for normalization methods.

    Attributes:
        method: Normalization method to use.
        params: Parameters associated with the method.
    """

    method: NormalizationMethod = NormalizationMethod.ZSCORE
    params: NormalizationParams = field(default_factory=ZScoreParams)


@dataclass
class DetrendingConfig:
    """
    Configuration for detrending methods.

    Attributes:
        method: Detrending method to use.
        params: Parameters associated with the method.
    """

    method: DetrendingMethod = DetrendingMethod.LOCALMINIMA
    params: DetrendingParams = field(default_factory=LocalMinimaParams)


@dataclass
class SignalProcessingConfig:
    """
    Configuration for signal processing steps.

    Attributes:
        pipeline: Which steps to apply in the processing pipeline.
        smoothing_sigma: Standard deviation for Gaussian smoothing.
        normalization: Configuration for normalization methods.
        detrending: Configuration for detrending methods.
    """

    pipeline: SignalProcessingPipeline = field(default_factory=SignalProcessingPipeline)
    smoothing_sigma: float = 3.0
    normalization: NormalizationConfig = field(default_factory=NormalizationConfig)
    detrending: DetrendingConfig = field(default_factory=DetrendingConfig)


# ===========================
# EVENT DETECTION PARAMETERS
# ===========================
@dataclass
class ConvexHullParams:
    """
    Parameters for convex hull-based event detection.

    Attributes:
        min_points: Minimum number of points to form a convex hull.
        min_duration: Minimum duration (in frames) for an event.
    """

    min_points: int = 3
    min_duration: int = 1


@dataclass
class DirectionComputationParams:
    """
    Configuration for computing the dominant direction vector in global events.

    Attributes:
        num_time_bins: Number of bins to divide the event duration into.
        mad_filtering_multiplier: Multiplier applied to MAD for outlier filtering.
        min_net_displacement_ratio: Minimum ratio of net displacement to max extent
            required to consider the direction meaningful.
    """

    num_time_bins: int = 6
    mad_filtering_multiplier: float = 1.0
    min_net_displacement_ratio: float = 0.25


@dataclass
class EventExtractionConfig:
    """
    Configuration for event extraction from activity traces.

    Attributes:
        min_cell_count: Minimum number of cells required to form an event.
        threshold_ratio: Ratio of the maximum activity trace value to consider an event.
        radius: Radius for spatial clustering of events (in pixels or microns, depending on calibration).
        global_max_comm_time: Maximum communication time for global events.
        seq_max_comm_time: Maximum communication time for sequential events.
        convex_hull: Parameters for convex hull-based event detection.
        global_direction_computation: Configuration for computing dominant direction in global events.
    """

    min_cell_count: int = 2
    threshold_ratio: float = 0.4
    radius: float = 300.0
    global_max_comm_time: int = 10
    seq_max_comm_time: int = 10
    convex_hull: ConvexHullParams = field(default_factory=ConvexHullParams)
    global_direction_computation: DirectionComputationParams = field(
        default_factory=DirectionComputationParams
    )


# ===========================
# EXPORT CONFIG
# ===========================
@dataclass
class SpatialCalibrationParams:
    """
    Spatial calibration parameters for image data.

    Attributes:
        pixel_to_micron_x: Microns per pixel in the x direction.
        pixel_to_micron_y: Microns per pixel in the y direction.
    """

    pixel_to_micron_x: float = 0.325
    pixel_to_micron_y: float = 0.325


@dataclass
class ExportConfig:
    """
    Configuration for exporting results.

    Attributes:
        spatial_calibration_params: Spatial calibration parameters.
        frame_rate: Frame rate of the image sequence in frames per second.
    """

    spatial_calibration_params: SpatialCalibrationParams = field(
        default_factory=SpatialCalibrationParams
    )
    frame_rate: float = 1.0


# ===========================
# GLOBAL CONFIG
# ===========================
@dataclass
class GlobalConfig:
    """
    Master configuration for the calcium activity characterization pipeline.

    Attributes:
        debug: Debugging configuration.
        data_dir: Root directory for experimental data.
        segmentation: Segmentation configuration.
        image_processing_hoechst: Image processing configuration for Hoechst channel.
        image_processing_fitc: Image processing configuration for FITC channel.
        trace_extraction: Trace extraction configuration.
        cell_filtering: Cell filtering configuration.
        cell_trace_processing: Signal processing configuration for cell traces.
        cell_trace_peak_detection: Peak detection configuration for cell traces.
        activity_trace_processing: Signal processing configuration for population activity traces.
        activity_trace_peak_detection: Peak detection configuration for population activity traces.
        event_extraction: Event extraction configuration.
        export: Export configuration.
    """

    debug: DebugConfig
    data_dir: str
    segmentation: SegmentationConfig
    image_processing_hoechst: ImageProcessingConfig
    image_processing_fitc: ImageProcessingConfig
    trace_extraction: TraceExtractionConfig
    cell_filtering: CellFilteringConfig
    cell_trace_processing: SignalProcessingConfig
    cell_trace_peak_detection: PeakDetectionConfig
    activity_trace_processing: SignalProcessingConfig
    activity_trace_peak_detection: PeakDetectionConfig
    event_extraction: EventExtractionConfig
    export: ExportConfig


# ==================================================================
# UNUSED / DEPRECATED PARAMETERS (not used, retained for future use)
# ==================================================================
# Below are structures used in older versions of the pipeline or for
# experimental features (Granger causality, correlation, clustering,
# ARCOS-based event tracking). They are kept for reference and possible
# future reuse but are not relied on by the current core pipeline.
# ==================================================================


# ==========================
# SPATIAL CLUSTERING PARAMETERS
# ==========================
@dataclass
class SpatialClusteringParameters:
    trace: str = "impulse_trace"
    use_indirect_neighbors: bool = False
    indirect_neighbors_num: int = 1
    use_sequential: bool = True
    seq_max_comm_time: int = 10


# ==========================
# PEAK CLUSTERING PARAMETERS
# ==========================
@dataclass
class ScoreWeights:
    time: float = 0.7
    duration: float = 0.3


@dataclass
class PeakClusteringParams:
    method: str = "fixed"
    adaptive_window_factor: float = 0.5
    fixed_window_size: int = 20
    score_weights: ScoreWeights = field(default_factory=ScoreWeights)


# ==========================
# GRANGER CAUSALITY PARAMETERS
# ==========================
class GrangerCausalityMethod(str, Enum):
    PAIRWISE = "pairwise"
    MULTIVARIATE = "multivariate"


@dataclass
class GrangerCausalityParams(ABC):
    window_size: int = 150
    lag_order: int = 3
    min_cells: int = 1


@dataclass
class PairwiseParams(GrangerCausalityParams):
    threshold_links: bool = False
    pvalue_threshold: float = 0.001


@dataclass
class MultiVariateParams(GrangerCausalityParams):
    pass


@dataclass
class GrangerCausalityConfig:
    method: GrangerCausalityMethod = GrangerCausalityMethod.PAIRWISE
    trace: str = "binary_trace"
    parameters: GrangerCausalityParams = field(default_factory=PairwiseParams)


# ==========================
# CORRELATION PARAMETERS
# ==========================
class CorrelationMethod(str, Enum):
    CROSSCORRELATION = "crosscorrelation"
    JACCARD = "jaccard"
    PEARSON = "pearson"
    SPEARMAN = "spearman"


@dataclass
class CorrelationParams(ABC):
    pass


@dataclass
class CrossCorrelationParams(CorrelationParams):
    mode: str = "full"
    method: str = "direct"


@dataclass
class JaccardParams(CorrelationParams):
    pass


@dataclass
class PearsonParams(CorrelationParams):
    pass


@dataclass
class SpearmanParams(CorrelationParams):
    pass


@dataclass
class CorrelationConfig:
    parallelize: bool = True
    window_size: int = 100
    step_percent: float = 0.75
    lag_percent: float = 0.25
    method: CorrelationMethod = CorrelationMethod.CROSSCORRELATION
    params: CorrelationParams = field(default_factory=CorrelationParams)


# ==========================
# CLUSTERING PARAMETERS
# ==========================
class ClusteringMethod(str, Enum):
    DBSCAN = "dbscan"
    HDBSCAN = "hdbscan"
    AGGLOMERATIVE = "agglomerative"
    AFFINITYPROPAGATION = "affinitypropagation"
    GRAPHCOMMUNITY = "graphcommunity"


class AffinityMetric(str, Enum):
    PRECOMPUTED = "precomputed"
    EUCLIDEAN = "euclidean"
    MANHATTAN = "manhattan"


class LinkageType(str, Enum):
    COMPLETE = "complete"
    WARD = "ward"
    AVERAGE = "average"
    SINGLE = "single"


@dataclass
class ClusteringParams(ABC):
    pass


@dataclass
class DbscanParams(ClusteringParams):
    eps: float = 0.03
    min_samples: int = 3
    metric: AffinityMetric = AffinityMetric.PRECOMPUTED


@dataclass
class HdbscanParams(ClusteringParams):
    min_cluster_size: int = 3
    min_samples: int = 3
    metric: AffinityMetric = AffinityMetric.PRECOMPUTED
    clustering_method: str = "eom"  # or "leaf"
    probability_threshold: float = 0.85
    cluster_selection_epsilon: float = 0.5


@dataclass
class AgglomerativeParams(ClusteringParams):
    n_clusters: int | None = None
    distance_threshold: float = 0.5
    linkage: LinkageType = LinkageType.COMPLETE
    metric: AffinityMetric = AffinityMetric.PRECOMPUTED
    auto_threshold: bool = True


@dataclass
class AffinityPropagationParams(ClusteringParams):
    preference: float | None = None
    damping: float = 0.9
    max_iter: int = 200
    convergence_iter: int = 15
    affinity: AffinityMetric = AffinityMetric.PRECOMPUTED


@dataclass
class GraphCommunityParams(ClusteringParams):
    threshold: float = 0.7


@dataclass
class ClusteringConfig:
    method: ClusteringMethod = ClusteringMethod.AGGLOMERATIVE
    min_cluster_size: int = 3
    params: ClusteringParams = field(default_factory=ClusteringParams)


# ==========================
# ARCOS PARAMETERS
# ==========================
@dataclass
class ArcosBindataParameters:
    smooth_k: int = 3
    bias_k: int = 51
    peak_threshold: float = 0.2
    binarization_threshold: float = 0.1
    polynomial_degree: int = 1
    bias_method: str = "runmed"  # can be 'lm', 'runmed', or 'none'
    n_jobs: int = -1


@dataclass
class ArcosTrackingParameters:
    position_columns: list[str] = field(default_factory=lambda: ["x", "y"])
    frame_column: str = "frame"
    id_column: str = "trackID"
    binarized_measurement_column: str = "intensity.bin"
    clid_column: str = "event_id"
    eps: float = 50.0
    eps_prev: float = 150.0
    min_clustersize: int = 15
    allow_merges: bool = False
    allow_splits: bool = False
    stability_threshold: int = 30
    linking_method: str = "nearest"
    clustering_method: str = "dbscan"
    min_samples: int = 1
    remove_small_clusters: bool = False
    min_size_for_split: int = 1
    reg: int = 1
    reg_m: int = 10
    cost_threshold: int = 0
    n_prev: int = 1
    predictor: bool = False
    n_jobs: int = 10
    show_progress: bool = True
