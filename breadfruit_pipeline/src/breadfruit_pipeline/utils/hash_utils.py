"""
Hash utilities for perceptual hashing and duplicate detection.
"""
import cv2
import numpy as np
import imagehash
from PIL import Image
from typing import Dict, List, Tuple
from collections import defaultdict


def compute_phash(image: np.ndarray, hash_size: int = 8) -> str:
    """Compute perceptual hash (pHash) of an image."""
    if len(image.shape) == 3:
        pil_img = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    else:
        pil_img = Image.fromarray(image)
    return str(imagehash.phash(pil_img, hash_size=hash_size))


def compute_dhash(image: np.ndarray, hash_size: int = 8) -> str:
    """Compute difference hash (dHash) of an image."""
    if len(image.shape) == 3:
        pil_img = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    else:
        pil_img = Image.fromarray(image)
    return str(imagehash.dhash(pil_img, hash_size=hash_size))


def compute_ahash(image: np.ndarray, hash_size: int = 8) -> str:
    """Compute average hash (aHash) of an image."""
    if len(image.shape) == 3:
        pil_img = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    else:
        pil_img = Image.fromarray(image)
    return str(imagehash.average_hash(pil_img, hash_size=hash_size))


def compute_whash(image: np.ndarray, hash_size: int = 8) -> str:
    """Compute wavelet hash (wHash) of an image."""
    if len(image.shape) == 3:
        pil_img = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    else:
        pil_img = Image.fromarray(image)
    return str(imagehash.whash(pil_img, hash_size=hash_size))


def compute_colorhash(image: np.ndarray, binbits: int = 3) -> str:
    """Compute color hash of an image."""
    pil_img = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    return str(imagehash.colorhash(pil_img, binbits=binbits))


def compute_all_hashes(image: np.ndarray) -> Dict[str, str]:
    """Compute all perceptual hashes for an image."""
    return {
        "phash": compute_phash(image),
        "dhash": compute_dhash(image),
        "ahash": compute_ahash(image),
        "whash": compute_whash(image),
        "colorhash": compute_colorhash(image)
    }


def hamming_distance(hash1: str, hash2: str) -> int:
    """Calculate Hamming distance between two hash strings."""
    if len(hash1) != len(hash2):
        raise ValueError("Hashes must be of equal length")
    
    # Convert hex strings to integers
    int1 = int(hash1, 16)
    int2 = int(hash2, 16)
    
    # XOR and count bits
    return bin(int1 ^ int2).count('1')


def hamming_distance_int(hash1: int, hash2: int) -> int:
    """Calculate Hamming distance between two integer hashes."""
    return bin(hash1 ^ hash2).count('1')


def hash_to_int(hash_str: str) -> int:
    """Convert hex hash string to integer."""
    return int(hash_str, 16)


def find_duplicate_groups(
    hashes: Dict[str, str],
    threshold: int = 10
) -> List[List[str]]:
    """Find groups of duplicate images based on hash distance."""
    # Build adjacency list
    adj = defaultdict(set)
    paths = list(hashes.keys())
    
    for i, path1 in enumerate(paths):
        for path2 in paths[i+1:]:
            dist = hamming_distance(hashes[path1], hashes[path2])
            if dist <= threshold:
                adj[path1].add(path2)
                adj[path2].add(path1)
    
    # Find connected components
    visited = set()
    groups = []
    
    for path in paths:
        if path in visited or path not in adj:
            continue
        
        group = []
        stack = [path]
        
        while stack:
            current = stack.pop()
            if current in visited:
                continue
            visited.add(current)
            group.append(current)
            stack.extend(adj[current] - visited)
        
        if len(group) > 1:
            groups.append(group)
    
    return groups


def find_duplicate_groups_multi(
    hash_dicts: Dict[str, Dict[str, str]],
    thresholds: Dict[str, int],
    consensus: int = 2
) -> List[List[str]]:
    """Find duplicate groups using multiple hash methods with consensus."""
    all_paths = set()
    for method_hashes in hash_dicts.values():
        all_paths.update(method_hashes.keys())
    paths = list(all_paths)
    
    # Count agreements for each pair
    pair_votes = defaultdict(int)
    
    for method, hashes in hash_dicts.items():
        threshold = thresholds.get(method, 10)
        
        for i, path1 in enumerate(paths):
            for path2 in paths[i+1:]:
                if path1 in hashes and path2 in hashes:
                    dist = hamming_distance(hashes[path1], hashes[path2])
                    if dist <= threshold:
                        pair = tuple(sorted([path1, path2]))
                        pair_votes[pair] += 1
    
    # Build adjacency from consensus
    adj = defaultdict(set)
    for (p1, p2), votes in pair_votes.items():
        if votes >= consensus:
            adj[p1].add(p2)
            adj[p2].add(p1)
    
    # Find connected components
    visited = set()
    groups = []
    
    for path in paths:
        if path in visited or path not in adj:
            continue
        
        group = []
        stack = [path]
        
        while stack:
            current = stack.pop()
            if current in visited:
                continue
            visited.add(current)
            group.append(current)
            stack.extend(adj[current] - visited)
        
        if len(group) > 1:
            groups.append(group)
    
    return groups