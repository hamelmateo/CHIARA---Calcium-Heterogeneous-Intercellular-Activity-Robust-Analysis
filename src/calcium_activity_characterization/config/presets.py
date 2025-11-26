# Usage example:
# --------------------------------------------------------------------
# from calcium_activity_characterization.config.presets import GLOBAL_CONFIG
#
# # Access segmentation parameters
# seg_method = GLOBAL_CONFIG.segmentation.method
# mesmer_params = GLOBAL_CONFIG.segmentation.params
#
# # Access cell trace peak detection config
# cell_peak_cfg = GLOBAL_CONFIG.cell_trace_peak_detection
# --------------------------------------------------------------------

"""
Preset configuration instances for the calcium activity characterization pipeline.

This module builds a `GLOBAL_CONFIG` object using the configuration structures
defined in `config.structures`. It centralizes all default parameters for:

- Debug and data directories
- Segmentation and image preprocessing
- Trace extraction and cell filtering
- Signal processing and peak detection (cell and activity traces)
- Event extraction
- Export and spatial calibration
"""

from __future__ import annotations

from calcium_activity_characterization.config.structures import (
    DebugConfig,
    GlobalConfig,
    SegmentationConfig,
    SegmentationMethod,
    MesmerParams,
    HotPixelParameters,
    HotPixelMethod,
    ImageProcessingConfig,
    ImageProcessingPipeline,
    TraceExtractionConfig,
    CellFilteringConfig,
    ObjectSizeThresholds,
    SignalProcessingConfig,
    SignalProcessingPipeline,
    NormalizationConfig,
    NormalizationMethod,
    ZScoreParams,
    DetrendingConfig,
    DetrendingMethod,
    LocalMinimaParams,
    PeakDetectionConfig,
    PeakDetectionMethod,
    SkimageParams,
    PeakGroupingParams,
    EventExtractionConfig,
    ConvexHullParams,
    DirectionComputationParams,
    SpatialCalibrationParams,
    ExportConfig,
)


# ===========================
# FLAGS
# ===========================
DEBUG_CONFIG: DebugConfig = DebugConfig(
    debugging=False,
    debugging_folder_path="D:/Mateo/20250326/Data/IS1",
)

DATA_DIR: str = "D:/Mateo"


# ===========================
# SEGMENTATION CONFIG
# ===========================
SEGMENTATION_CONFIG: SegmentationConfig = SegmentationConfig(
    method=SegmentationMethod.MESMER,
    params=MesmerParams(
        image_mpp=0.5,
        maxima_threshold=0.2,
        maxima_smooth=2.5,
        interior_threshold=0.05,
        interior_smooth=1.0,
        small_objects_threshold=25,
        fill_holes_threshold=15,
        radius=2,
    ),
    save_overlay=True,
)


# ===========================
# IMAGE PROCESSING CONFIG
# ===========================
HOTPIXEL_CONFIG: HotPixelParameters = HotPixelParameters(
    method=HotPixelMethod.REPLACE,
    use_auto_threshold=True,
    percentile=99.9,
    mad_scale=20.0,
    static_threshold=2000,
    window_size=3,
)

HOECHST_IMAGE_PROCESSING_CONFIG: ImageProcessingConfig = ImageProcessingConfig(
    pipeline=ImageProcessingPipeline(
        padding=True,
        cropping=False,
        hot_pixel_cleaning=False,
    ),
    padding_digits=5,
    roi_scale=0.75,
    roi_centered=False,
    hot_pixel_cleaning=HotPixelParameters(
        method=HotPixelMethod.CLIP,
        use_auto_threshold=False,
        percentile=99.9,
        mad_scale=20.0,
        static_threshold=8000,
        window_size=3,
    ),
)

FITC_IMAGE_PROCESSING_CONFIG: ImageProcessingConfig = ImageProcessingConfig(
    pipeline=ImageProcessingPipeline(
        padding=True,
        cropping=True,
        hot_pixel_cleaning=True,
    ),
    padding_digits=5,
    roi_scale=0.75,
    roi_centered=False,
    hot_pixel_cleaning=HOTPIXEL_CONFIG,
)


# ===========================
# TRACE EXTRACTION PARAMETERS
# ===========================
TRACE_EXTRACTION_CONFIG: TraceExtractionConfig = TraceExtractionConfig(
    parallelize=True,
    trace_version_name="raw",
)


# ===========================
# CELL DETECTION PARAMETERS
# ===========================
CELL_FILTERING_CONFIG: CellFilteringConfig = CellFilteringConfig(
    border_margin=1,
    object_size_thresholds=ObjectSizeThresholds(
        min=500,
        max=10000,
    ),
)


# ===========================
# INDIVIDUAL CELLS SIGNAL PROCESSING CONFIG
# ===========================
STANDARD_ZSCORE_SIGNAL_PROCESSING: SignalProcessingConfig = SignalProcessingConfig(
    pipeline=SignalProcessingPipeline(
        detrending=True,
        normalization=True,
        smoothing=True,
    ),
    smoothing_sigma=3.0,
    normalization=NormalizationConfig(
        method=NormalizationMethod.ZSCORE,
        params=ZScoreParams(
            epsilon=1e-8,
            smoothing_sigma=2.0,
            residuals_clip_percentile=80.0,
            residuals_min_number=20,
        ),
    ),
    detrending=DetrendingConfig(
        method=DetrendingMethod.LOCALMINIMA,
        params=LocalMinimaParams(
            cut_trace_num_points=0,
            verbose=False,
            minima_detection_order=15,
            edge_anchors_window=50,
            edge_anchors_delta=0.03,
            filtering_shoulder_neighbor_dist=400,
            filtering_shoulder_window=100,
            filtering_angle_thresh_deg=10,
            crossing_correction_min_dist=10,
            crossing_correction_max_iterations=10,
            fitting_method="linear",
            diagnostics_enabled=False,
            diagnostics_output_dir=(
                "D:/Mateo/20250326/Output/IS1/signal-processing/detrending-diagnostics"
            ),
        ),
    ),
)


# ===========================
# INDIVIDUAL CELLS PEAK DETECTION CONFIG
# ===========================
CELL_PEAK_DETECTION_CONFIG: PeakDetectionConfig = PeakDetectionConfig(
    verbose=False,
    method=PeakDetectionMethod.SKIMAGE,
    params=SkimageParams(
        prominence=15.0,
        distance=10,
        height=None,
        threshold=None,
        width=None,
        scale_class_quantiles=(0.33, 0.66),
        full_half_width=0.5,
        full_duration_threshold=0.9,
    ),
    peak_grouping=PeakGroupingParams(
        overlap_margin=0,
    ),
    start_frame=None,
    end_frame=None,
    filter_overlapping_peaks=True,
    refine_durations=False,
)


# ===========================
# ACTIVITY TRACE PROCESSING & GLOBAL THRESHOLDING CONFIG
# ===========================
ACTIVITY_TRACE_PROCESSING_CONFIG: SignalProcessingConfig = SignalProcessingConfig(
    pipeline=SignalProcessingPipeline(
        detrending=False,
        normalization=False,
        smoothing=True,
    ),
    smoothing_sigma=5.0,
)

ACTIVITY_TRACE_PEAK_DETECTION_CONFIG: PeakDetectionConfig = PeakDetectionConfig(
    verbose=False,
    method=PeakDetectionMethod.SKIMAGE,
    params=SkimageParams(
        prominence=10.0,
        distance=20,
        height=None,
        threshold=None,
        width=None,
        scale_class_quantiles=(0.33, 0.66),
        full_half_width=0.5,
        full_duration_threshold=0.95,
    ),
    peak_grouping=PeakGroupingParams(
        overlap_margin=0,
    ),
    start_frame=None,
    end_frame=None,
    filter_overlapping_peaks=True,
    refine_durations=True,
)


# ===========================
# EVENT DETECTION CONFIG
# ===========================
EVENT_EXTRACTION_CONFIG: EventExtractionConfig = EventExtractionConfig(
    min_cell_count=2,
    threshold_ratio=0.5,
    radius=200,
    global_max_comm_time=30,
    seq_max_comm_time=15,
    convex_hull=ConvexHullParams(
        min_points=3,
        min_duration=1,
    ),
    global_direction_computation=DirectionComputationParams(
        num_time_bins=6,
        mad_filtering_multiplier=1.0,
        min_net_displacement_ratio=0.25,
    ),
)


# ===========================
# EXPORT CONFIG
# ===========================
EXPORT_CONFIG: ExportConfig = ExportConfig(
    spatial_calibration_params=SpatialCalibrationParams(
        pixel_to_micron_x=0.325,
        pixel_to_micron_y=0.325,
    ),
    frame_rate=1.0,
)


# ===========================
# GLOBAL CONFIG
# ===========================
GLOBAL_CONFIG: GlobalConfig = GlobalConfig(
    debug=DEBUG_CONFIG,
    data_dir=DATA_DIR,
    segmentation=SEGMENTATION_CONFIG,
    image_processing_hoechst=HOECHST_IMAGE_PROCESSING_CONFIG,
    image_processing_fitc=FITC_IMAGE_PROCESSING_CONFIG,
    trace_extraction=TRACE_EXTRACTION_CONFIG,
    cell_filtering=CELL_FILTERING_CONFIG,
    cell_trace_processing=STANDARD_ZSCORE_SIGNAL_PROCESSING,
    cell_trace_peak_detection=CELL_PEAK_DETECTION_CONFIG,
    activity_trace_processing=ACTIVITY_TRACE_PROCESSING_CONFIG,
    activity_trace_peak_detection=ACTIVITY_TRACE_PEAK_DETECTION_CONFIG,
    event_extraction=EVENT_EXTRACTION_CONFIG,
    export=EXPORT_CONFIG,
)