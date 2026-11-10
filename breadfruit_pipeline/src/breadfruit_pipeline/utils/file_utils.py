"""File and path utilities for the pipeline."""
import os
import shutil
import hashlib
from pathlib import Path
from typing import List, Optional, Generator, Union
import logging

logger = logging.getLogger(__name__)


def get_image_files(
    directory: Union[str, Path],
    extensions: Optional[List[str]] = None,
    recursive: bool = True
) -> Generator[Path, None, None]:
    """Yield image files from directory with given extensions."""
    if extensions is None:
        extensions = [".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp"]
    
    extensions = [ext.lower() for ext in extensions]
    directory = Path(directory)
    
    if not directory.exists():
        logger.warning(f"Directory does not exist: {directory}")
        return
    
    if recursive:
        pattern = "**/*"
    else:
        pattern = "*"
    
    for file_path in directory.glob(pattern):
        if file_path.is_file() and file_path.suffix.lower() in extensions:
            yield file_path


def get_all_files(
    directory: Union[str, Path],
    recursive: bool = True
) -> Generator[Path, None, None]:
    """Yield all files from directory."""
    directory = Path(directory)
    if not directory.exists():
        return
    
    if recursive:
        for file_path in directory.rglob("*"):
            if file_path.is_file():
                yield file_path
    else:
        for file_path in directory.iterdir():
            if file_path.is_file():
                yield file_path


def get_file_hash(file_path: Union[str, Path], algorithm: str = "sha256") -> str:
    """Calculate hash of file contents."""
    hash_func = hashlib.new(algorithm)
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            hash_func.update(chunk)
    return hash_func.hexdigest()


def get_file_size(file_path: Union[str, Path]) -> int:
    """Get file size in bytes."""
    return Path(file_path).stat().st_size


def is_zero_byte_file(file_path: Union[str, Path]) -> bool:
    """Check if file is zero bytes."""
    return get_file_size(file_path) == 0


def safe_move(
    src: Union[str, Path],
    dst: Union[str, Path],
    overwrite: bool = False
) -> bool:
    """Safely move file with error handling."""
    src = Path(src)
    dst = Path(dst)
    
    try:
        dst.parent.mkdir(parents=True, exist_ok=True)
        
        if dst.exists() and not overwrite:
            logger.warning(f"Destination exists, skipping: {dst}")
            return False
        
        shutil.move(str(src), str(dst))
        logger.debug(f"Moved: {src} -> {dst}")
        return True
    except Exception as e:
        logger.error(f"Failed to move {src} to {dst}: {e}")
        return False


def safe_copy(
    src: Union[str, Path],
    dst: Union[str, Path],
    overwrite: bool = False
) -> bool:
    """Safely copy file with error handling."""
    src = Path(src)
    dst = Path(dst)
    
    try:
        dst.parent.mkdir(parents=True, exist_ok=True)
        
        if dst.exists() and not overwrite:
            logger.warning(f"Destination exists, skipping: {dst}")
            return False
        
        shutil.copy2(str(src), str(dst))
        logger.debug(f"Copied: {src} -> {dst}")
        return True
    except Exception as e:
        logger.error(f"Failed to copy {src} to {dst}: {e}")
        return False


def ensure_dir(path: Union[str, Path]) -> Path:
    """Ensure directory exists."""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def get_relative_path(file_path: Union[str, Path], base: Union[str, Path]) -> Path:
    """Get relative path of file from base directory."""
    return Path(file_path).relative_to(Path(base))


def count_files(directory: Union[str, Path], extensions: Optional[List[str]] = None) -> int:
    """Count files in directory."""
    return sum(1 for _ in get_image_files(directory, extensions))


def get_directory_stats(directory: Union[str, Path]) -> dict:
    """Get directory statistics."""
    directory = Path(directory)
    stats = {
        "total_files": 0,
        "total_size": 0,
        "extensions": {},
        "subdirs": 0
    }
    
    for file_path in directory.rglob("*"):
        if file_path.is_file():
            stats["total_files"] += 1
            stats["total_size"] += file_path.stat().st_size
            ext = file_path.suffix.lower()
            stats["extensions"][ext] = stats["extensions"].get(ext, 0) + 1
        elif file_path.is_dir():
            stats["subdirs"] += 1
    
    return stats


def create_directory_structure(base_dir: Union[str, Path], structure: dict) -> dict:
    """Create directory structure from config."""
    base = Path(base_dir)
    created = {}
    
    for key, value in structure.items():
        if isinstance(value, dict):
            if "base" in value and "subdirs" in value:
                base_path = base / value["base"]
                base_path.mkdir(parents=True, exist_ok=True)
                created[value["base"]] = base_path
                for subdir in value["subdirs"]:
                    sub_path = base_path / subdir
                    sub_path.mkdir(parents=True, exist_ok=True)
                    created[f"{value['base']}/{subdir}"] = sub_path
            else:
                sub_created = create_directory_structure(base / key, value)
                created.update(sub_created)
        else:
            path = base / value
            path.mkdir(parents=True, exist_ok=True)
            created[key] = path
    
    return created


def list_directories(path: Union[str, Path]) -> List[Path]:
    """List immediate subdirectories."""
    path = Path(path)
    if not path.exists():
        return []
    return [d for d in path.iterdir() if d.is_dir()]


def get_directory_tree(path: Union[str, Path], max_depth: int = 3) -> dict:
    """Get directory tree structure."""
    path = Path(path)
    
    def _scan(dir_path: Path, depth: int) -> dict:
        if depth > max_depth:
            return {"truncated": True}
        
        result = {"files": 0, "dirs": {}}
        try:
            for item in dir_path.iterdir():
                if item.is_file():
                    result["files"] += 1
                elif item.is_dir():
                    result["dirs"][item.name] = _scan(item, depth + 1)
        except PermissionError:
            result["error"] = "Permission denied"
        return result
    
    return {path.name: _scan(path, 0)}


def find_duplicate_filenames(directory: Union[str, Path]) -> dict:
    """Find duplicate filenames (same name, different paths)."""
    directory = Path(directory)
    name_map = {}
    
    for file_path in directory.rglob("*"):
        if file_path.is_file():
            name = file_path.name
            if name not in name_map:
                name_map[name] = []
            name_map[name].append(file_path)
    
    return {name: paths for name, paths in name_map.items() if len(paths) > 1}


def clean_filename(filename: str) -> str:
    """Clean filename for safe filesystem operations."""
    import re
    # Remove or replace problematic characters
    cleaned = re.sub(r'[<>:"/\\|?*]', '_', filename)
    # Remove leading/trailing spaces and dots
    cleaned = cleaned.strip(' .')
    return cleaned