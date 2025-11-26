# Usage example:
# ---------------------------------------------------------
# from calcium_activity_characterization.main import main
# if __name__ == "__main__":
#     main()
# ---------------------------------------------------------

"""
Entry point to run the calcium activity characterization pipeline on one or more
image sequence (ISX) folders.

This module provides a PyQt-based folder selection dialog that allows the user
to select either:
- Date folders that contain a ``Data/`` directory with ISX subfolders, or
- Individual ISX folders directly.

All discovered ISX folders are then processed by the CalciumPipeline using
the global configuration.
"""

import sys
from pathlib import Path

from PyQt5.QtWidgets import QApplication, QFileDialog, QWidget

from calcium_activity_characterization.logger import logger
from calcium_activity_characterization.config.presets import GLOBAL_CONFIG
from calcium_activity_characterization.core.pipeline import CalciumPipeline



def find_isx_folders(folder: Path) -> list[Path]:
    """
    Recursively find all ISX folders under ``Data/``-like directories.

    A folder is considered an ISX folder if its name starts with ``"IS"``.
    Any path that contains an ``"Output"`` segment in its components is skipped
    to avoid re-processing exported results.

    Args:
        folder: Root folder from which to search recursively.

    Returns:
        list[Path]: List of paths to ISX folders discovered under ``folder``.
    """
    isx_folders: list[Path] = []
    try:
        for subpath in folder.rglob("*"):
            if "Output" in subpath.parts:
                continue
            if subpath.is_dir() and subpath.name.startswith("IS"):
                isx_folders.append(subpath)
    except Exception as e:
        logger.error(f"Error while searching for ISX folders in '{folder}': {e}")
    return isx_folders

def main() -> None:
    """
    Launch the Qt folder selection dialog and run the analysis pipeline.

    The user can select one or more directories:
    - If a selected directory name starts with ``"IS"``, it is treated as an
      ISX folder and processed directly.
    - Otherwise, the directory is searched recursively for ISX folders using
      :func:`find_isx_folders`.

    The pipeline is run on each discovered ISX folder, and corresponding
    outputs are written into an ``Output/<ISX_NAME>/`` directory at the same
    experiment level.

    Returns:
        None
    """
    try:
        app = QApplication(sys.argv)

        if GLOBAL_CONFIG.debug.debugging:
            logger.info("[DEBUGGING MODE] Using test folder from config.")
            selected: list[Path] = [Path(GLOBAL_CONFIG.debug.debugging_folder_path)]
        else:
            folder_dialog = QFileDialog()
            folder_dialog.setDirectory(str(GLOBAL_CONFIG.data_dir))
            folder_dialog.setFileMode(QFileDialog.Directory)
            folder_dialog.setOption(QFileDialog.DontUseNativeDialog, True)
            folder_dialog.setOption(QFileDialog.ShowDirsOnly, True)
            folder_dialog.setWindowTitle(
                "Select One or More Folders (Date folders or ISX)"
            )

            # Allow selecting multiple directories in non-native dialog
            view = folder_dialog.findChild(QWidget, "listView")
            if view is not None:
                view.setSelectionMode(view.ExtendedSelection)

            tree_view = folder_dialog.findChild(QWidget, "treeView")
            if tree_view is not None:
                tree_view.setSelectionMode(tree_view.ExtendedSelection)

            if not folder_dialog.exec_():
                logger.info("No folder selected. Exiting.")
                return

            selected = [Path(folder_str) for folder_str in folder_dialog.selectedFiles()]

        all_isx_folders: list[Path] = []
        for folder in selected:
            if folder.name.startswith("IS"):
                all_isx_folders.append(folder)
            else:
                all_isx_folders.extend(find_isx_folders(folder))

        if not all_isx_folders:
            logger.warning("No ISX folders found in selected path(s). Exiting.")
            return

        pipeline = CalciumPipeline(GLOBAL_CONFIG)
        logger.info(f"Found {len(all_isx_folders)} ISX folders to process.")

        for isx_folder in sorted(all_isx_folders):
            try:
                output_folder = isx_folder.parents[1] / "Output" / isx_folder.name
                logger.info(f"Processing ISX folder: {isx_folder} -> {output_folder}")
                pipeline.run(isx_folder, output_folder)
            except Exception as e:
                logger.error(
                    f"Failed to process ISX folder '{isx_folder}': {e}",
                )

    except Exception as e:
        logger.error(f"Unexpected error in main(): {e}")

if __name__ == "__main__":
    main()
