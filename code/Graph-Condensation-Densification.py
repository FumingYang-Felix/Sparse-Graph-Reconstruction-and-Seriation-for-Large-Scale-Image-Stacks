#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Graph-Condensation-Densification (GCD) Algorithm Overview:
  • Phase A: Short-edge Borůvka tree construction - O(N log N) complexity
  • Phase B: Farthest-point condensation - densifies connectivity  
  • Phase C: Double-sweep BFS for diameter-based ordering
  • Phase D: K-window densification - adds local connections
  • Phase E: Spectral ordering with Fiedler vector (optional)
  • Phase F: Final K-densification verification - validates ordering quality

Performance Optimizations:
  • SIFT caching: 99% reduction in feature extraction overhead
  • Fast FLANN matching: 2-3x speedup in feature matching
  • Combined optimizations: Up to 24x performance improvement
  • Sub-quadratic complexity: O(N(log N + K)) vs O(N²) exhaustive

Examples:
  # Recommended: BUILD-AND-ORDER with optimizations
  python Graph-Condensation-Densification.py --build-and-order --cache-sift --fast-flann --verbose
  
  # Baseline test with specific parameters  
  python Graph-Condensation-Densification.py --build-and-order --samples-per-phase 7 --window-k 7
  
  # Auto-tune SSIM threshold for optimal connectivity
  python Graph-Condensation-Densification.py --build-and-order --auto-tune-threshold --verbose
  
  # Two-phase analysis: relaxed → strict with proximity mask
  python Graph-Condensation-Densification.py --build-and-order --two-phase --proximity-window 15
  
  # Pre-compute all SIFT features upfront
  python Graph-Condensation-Densification.py --build-and-order --precompute-sift --verbose
  python Graph-Condensation-Densification.py --window-k 10                # Override K (window half-width) for K-window densification
  
Performance comparison:
  Complete (no optimizations):    ~11,130 SIFT computations, standard FLANN (4+ hours)
  Complete (SIFT caching only):   ~106 SIFT computations, standard FLANN (~15 minutes)
  Complete (Fast FLANN only):     ~11,130 SIFT computations, fast FLANN (~1.5 hours)
  Complete (both optimizations): ~106 SIFT computations, fast FLANN (~3 minutes)
  BUILD-AND-ORDER:               ~N(log N + K) edge tests, 5-phase algorithm (~5-10 minutes, RECOMMENDED)
  Sublinear (Randomized Borůvka): ~N log²N edge tests instead of N²/2 (10-50x reduction)
  
The sublinear approach uses Randomized Borůvka algorithm to build a spanning tree
of the strongest connections, achieving O(N log²N) complexity instead of O(N²).

The BUILD-AND-ORDER approach provides sub-quadratic 1-D ordering using theoretical algorithm:
1. Phase A: Short-edge Borůvka tree construction
2. Phase B: Farthest-point condensation (3 rounds)
3. Phase C: Double-sweep BFS for diameter-based rough ordering
4. Phase D: K-window densification for local connectivity
5. Phase E: Final spectral ordering (enabled by default, can be disabled)
6. Phase F: Final K-densification verification - validates ordering and finds missed connections
This achieves O(N(log N + K)) complexity and is adapted for biological sections.

Strong Connection Criteria (Pipeline Quality):
  - SSIM > threshold (default 0.25, recommend 0.3 for high quality)
  - Rotation: ±90° (biological sections shouldn't be flipped)
  - Scale: 0.8 to 1.25 (both X and Y)
  - Inlier ratio: > 8% (sufficient feature correspondence)

Two-Phase Analysis (--two-phase):
  Phase 1: Relaxed threshold (auto-tuned for ~10 connections/node) → get linear order
  Phase 2: Strict threshold + proximity mask (nodes close in order are permissible)
  Benefits: Reduces biological pipeline calls by focusing on locally plausible connections
  
Auto-Tune Threshold (--auto-tune-threshold):
  Analyzes degree of random nodes to find optimal SSIM threshold
  Target: ~10 connections per node for good connectivity without noise
  Method: Tests thresholds 0.1-0.5 and selects closest to target degree
"""

import cv2
import numpy as np
import time
import pandas as pd
from itertools import combinations
from skimage.metrics import structural_similarity as ssim
import os
import random
import glob
import argparse
import pickle
from pathlib import Path
import math
import matplotlib.pyplot as plt
import networkx as nx
import heapq
from collections import defaultdict, deque
import contextlib, io  # added for suppressing prints

# ============================================================================
# DEFAULT PIPELINE PARAMETERS
# ============================================================================
# These values define the default thresholds for strong connection criteria
# They can be overridden via command-line arguments or function parameters

DEFAULT_SSIM_THRESHOLD = 0.4      # SSIM threshold for strong connections
DEFAULT_INLIER_THRESHOLD = 0.08    # Inlier ratio threshold (8%)

# Fixed biological constraints (always applied regardless of input parameters)
ROTATION_LIMIT_DEGREES = 90        # Maximum rotation: ±90°
SCALE_MIN = 0.95                    # Minimum scale factor
SCALE_MAX = 1.05                   # Maximum scale factor

# ============================================================================

# Global SIFT cache
SIFT_CACHE = {}

# Helper functions for keypoint serialization
def keypoints_to_serializable(keypoints, descriptors):
    """Convert OpenCV keypoints to a serializable format"""
    if keypoints is None or descriptors is None:
        return None, None
    
    # Convert keypoints to a list of dictionaries
    kp_data = []
    for kp in keypoints:
        kp_dict = {
            'pt': kp.pt,
            'angle': kp.angle,
            'class_id': kp.class_id,
            'octave': kp.octave,
            'response': kp.response,
            'size': kp.size
        }
        kp_data.append(kp_dict)
    
    return kp_data, descriptors

def serializable_to_keypoints(kp_data, descriptors):
    """Convert serializable format back to OpenCV keypoints"""
    if kp_data is None or descriptors is None:
        return None, None
    
    # Convert list of dictionaries back to keypoints
    keypoints = []
    for kp_dict in kp_data:
        kp = cv2.KeyPoint(
            x=kp_dict['pt'][0],
            y=kp_dict['pt'][1],
            size=kp_dict['size'],
            angle=kp_dict['angle'],
            response=kp_dict['response'],
            octave=kp_dict['octave'],
            class_id=kp_dict['class_id']
        )
        keypoints.append(kp)
    
    return keypoints, descriptors

def calculate_tree_diameter(tree_edges, all_sections):
    """
    Calculate tree diameter using double-sweep BFS
    Returns (diameter, diameter_path, center_nodes)
    """
    if not tree_edges:
        return 0, [], []
    
    # Build adjacency list from tree edges
    graph = defaultdict(list)
    for u, v, weight in tree_edges:
        graph[u].append(v)
        graph[v].append(u)
    
    def bfs_farthest(start):
        """BFS to find farthest node and distance from start"""
        visited = set()
        queue = deque([(start, 0, [start])])
        visited.add(start)
        farthest_node = start
        max_distance = 0
        farthest_path = [start]
        
        while queue:
            node, distance, path = queue.popleft()
            
            if distance > max_distance:
                max_distance = distance
                farthest_node = node
                farthest_path = path[:]
            
            for neighbor in graph[node]:
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append((neighbor, distance + 1, path + [neighbor]))
        
        return farthest_node, max_distance, farthest_path
    
    # Double-sweep BFS to find diameter
    # First sweep: find one end of diameter
    arbitrary_node = tree_edges[0][0]  # Start from any node
    end1, _, _ = bfs_farthest(arbitrary_node)
    
    # Second sweep: find actual diameter
    end2, diameter, diameter_path = bfs_farthest(end1)
    
    # Find center node(s) - middle of diameter path
    center_idx = len(diameter_path) // 2
    if len(diameter_path) % 2 == 1:
        # Odd length: single center
        center_nodes = [diameter_path[center_idx]]
    else:
        # Even length: two centers
        center_nodes = [diameter_path[center_idx-1], diameter_path[center_idx]]
    
    return diameter, diameter_path, center_nodes

def analyze_tree_structure(tree_edges, all_sections, verbose=False):
    """
    Comprehensive tree structure analysis
    Returns dictionary with structural information
    """
    if not tree_edges:
        return {
            'diameter': 0,
            'diameter_path': [],
            'center_nodes': [],
            'connected_components': len(all_sections),
            'total_nodes': len(all_sections),
            'total_edges': 0,
            'edge_list': []
        }
    
    # Calculate diameter
    diameter, diameter_path, center_nodes = calculate_tree_diameter(tree_edges, all_sections)
    
    # Build connected components
    graph = defaultdict(list)
    edge_set = set()
    for u, v, weight in tree_edges:
        graph[u].append(v)
        graph[v].append(u)
        edge_set.add((min(u, v), max(u, v)))
    
    # Find connected components
    visited = set()
    components = []
    
    for section in all_sections:
        if section not in visited:
            # BFS to find component
            component = []
            queue = deque([section])
            visited.add(section)
            
            while queue:
                node = queue.popleft()
                component.append(node)
                
                for neighbor in graph[node]:
                    if neighbor not in visited:
                        visited.add(neighbor)
                        queue.append(neighbor)
            
            components.append(sorted(component))
    
    # Create detailed edge list with section numbers
    edge_list = [(u, v, weight) for u, v, weight in tree_edges]
    edge_list.sort()  # Sort for consistent output
    
    result = {
        'diameter': diameter,
        'diameter_path': diameter_path,
        'center_nodes': center_nodes,
        'connected_components': len(components),
        'largest_component_size': max(len(comp) for comp in components) if components else 0,
        'component_sizes': sorted([len(comp) for comp in components], reverse=True),
        'total_nodes': len(all_sections),
        'total_edges': len(tree_edges),
        'edge_list': edge_list
    }
    
    if verbose:
        print(f"\n📊 TREE STRUCTURE ANALYSIS:")
        print(f"   Diameter: {diameter} (longest path in tree)")
        print(f"   Diameter path: {' → '.join(map(str, diameter_path))}")
        print(f"   Center node(s): {center_nodes}")
        print(f"   Connected components: {len(components)}")
        if len(components) > 1:
            print(f"   Component sizes: {result['component_sizes']}")
        print(f"   Total edges: {len(tree_edges)}")
        print(f"   Tree density: {len(tree_edges)/(len(all_sections)-1):.1%} of minimum spanning tree")
    
    return result

def auto_tune_threshold(images, all_sections, data_manager, cache_sift=False, fast_flann=False, 
                       target_degree=10, sample_nodes=3, inlier_threshold=None, verbose=False):
    """
    Auto-tune SSIM threshold to achieve target degree for sample nodes
    
    Args:
        images: Dictionary of {section_num: image}
        all_sections: List of all section identifiers
        data_manager: PairwiseDataManager instance
        cache_sift: Whether to use SIFT caching
        fast_flann: Whether to use fast FLANN parameters
        target_degree: Target number of connections per node
        sample_nodes: Number of nodes to sample for tuning
        inlier_threshold: Inlier ratio threshold for strong connections (None = use DEFAULT_INLIER_THRESHOLD)
        verbose: Whether to print detailed progress
    
    Returns:
        Tuned SSIM threshold
    """
    
    # Apply default values if not provided
    if inlier_threshold is None:
        inlier_threshold = DEFAULT_INLIER_THRESHOLD
    if verbose:
        print(f"\n AUTO-TUNING SSIM THRESHOLD")
        print(f"   Target degree: {target_degree} connections per node")
        print(f"   Sample nodes: {sample_nodes}")
    
    # Sample random nodes
    sample_sections = random.sample(all_sections, min(sample_nodes, len(all_sections)))
    
    # Test different thresholds
    thresholds = [0.3, 0.35, 0.4, 0.45, 0.5]
    threshold_degrees = []
    
    for threshold in thresholds:
        total_degree = 0
        
        for sample_sec in sample_sections:
            degree = 0
            
            # Test connections to all other sections
            for other_sec in all_sections:
                if other_sec == sample_sec:
                    continue
                
                # Check if already computed
                if data_manager.is_computed(sample_sec, other_sec):
                    ssim_score = data_manager.get_ssim(sample_sec, other_sec)
                    pipeline_valid = data_manager.is_pipeline_valid(sample_sec, other_sec)
                else:
                    # Compute alignment
                    img1 = images[sample_sec]
                    img2 = images[other_sec]
                    
                    transformation_matrix, _, _, stats = find_rigid_sift_alignment(
                        img1, img2, f"section_{sample_sec}", f"section_{other_sec}",
                        use_cache=cache_sift, fast_flann=fast_flann, verbose=False, inlier_threshold=inlier_threshold
                    )
                    
                    # Calculate SSIM
                    ssim_score = 0.0
                    if transformation_matrix is not None and stats['success']:
                        aligned_img1, overlap_region1, overlap_region2, bbox = apply_rigid_transformation_and_find_overlap(
                            img1, img2, transformation_matrix)
                        
                        if overlap_region1 is not None and overlap_region2 is not None:
                            ssim_score = calculate_ssim_score(overlap_region1, overlap_region2)
                    
                    # Apply pipeline validation
                    pipeline_valid = False
                    if stats['success'] and ssim_score > 0:
                        rotation = abs(stats.get('rotation_deg', 0))
                        scale_x = stats.get('scale_x', 1)
                        scale_y = stats.get('scale_y', 1)
                        inlier_ratio = stats.get('inlier_ratio', 0)
                        
                        pipeline_valid = (
                            rotation <= ROTATION_LIMIT_DEGREES and
                            SCALE_MIN <= scale_x <= SCALE_MAX and
                            SCALE_MIN <= scale_y <= SCALE_MAX and
                            inlier_ratio > inlier_threshold
                        )
                    
                    # Store result
                    result = {
                        'section1': sample_sec,
                        'section2': other_sec,
                        'ssim_score': ssim_score,
                        'pipeline_valid': pipeline_valid,
                        'features1': stats['features1'],
                        'features2': stats['features2'],
                        'initial_matches': stats['matches'],
                        'ransac_inliers': stats['inliers'],
                        'inlier_ratio': stats['inlier_ratio'],
                        'rotation_degrees': stats.get('rotation_deg', 0),
                        'translation_x': stats.get('translation_x', 0),
                        'translation_y': stats.get('translation_y', 0),
                        'scale_x': stats.get('scale_x', 1),
                        'scale_y': stats.get('scale_y', 1),
                        'detection_time_sec': stats['detection_time'],
                        'matching_time_sec': stats['matching_time'],
                        'alignment_success': stats['success'],
                        'error_message': stats.get('error', ''),
                        'overlap_area_pixels': 0,
                        'overlap_bbox': ''
                    }
                    
                    data_manager.store_result(sample_sec, other_sec, result, ssim_score, pipeline_valid)
                
                # Count connection if meets threshold and pipeline valid
                if pipeline_valid and ssim_score > threshold:
                    degree += 1
            
            total_degree += degree
        
        avg_degree = total_degree / len(sample_sections)
        threshold_degrees.append((threshold, avg_degree))
        
        if verbose:
            print(f"   Threshold {threshold:.2f}: avg degree {avg_degree:.1f}")
    
    # Find threshold closest to target degree
    best_threshold = min(threshold_degrees, key=lambda x: abs(x[1] - target_degree))[0]
    
    if verbose:
        print(f"   Auto-tuned threshold: {best_threshold:.2f}")
    
    return best_threshold

def create_proximity_mask(linear_order, proximity_window=10):
    """
    Create proximity mask from linear order - nodes close in order are permissible
    
    Args:
        linear_order: List of sections in linear order
        proximity_window: Maximum distance in order to consider permissible
    
    Returns:
        Set of permissible (sec1, sec2) pairs
    """
    permissible_pairs = set()
    
    # Create position mapping
    position = {sec: i for i, sec in enumerate(linear_order)}
    
    # Add all pairs within proximity window
    for i, sec1 in enumerate(linear_order):
        for j, sec2 in enumerate(linear_order):
            if i != j and abs(i - j) <= proximity_window:
                permissible_pairs.add((sec1, sec2))
    
    return permissible_pairs

class PairwiseDataManager:
    """
    Efficient management of pairwise distance computations for BUILD-AND-ORDER algorithm.
    Uses NxN matrices to track computed pairs and store distance/similarity data.
    """
    
    def __init__(self, all_sections):
        self.all_sections = sorted(all_sections)
        self.N = len(all_sections)
        self.section_to_idx = {section: i for i, section in enumerate(self.all_sections)}
        
        # NxN matrices for efficient storage
        self.computed = np.zeros((self.N, self.N), dtype=bool)      # Track what's been computed
        self.distance_matrix = np.full((self.N, self.N), np.inf)   # 1-SSIM distances
        self.ssim_matrix = np.zeros((self.N, self.N))              # Raw SSIM scores
        self.pipeline_valid = np.zeros((self.N, self.N), dtype=bool)  # Pipeline validation results
        
        # Detailed results storage for analysis/visualization
        self.detailed_results = {}  # Key: (min_idx, max_idx), Value: full result dict
        
        # Statistics
        self.unique_pairs_computed = 0
        self.total_oracle_calls = 0
        self.cache_hits = 0
        
        # Initialize diagonal (self-similarity)
        for i in range(self.N):
            self.computed[i, i] = True
            self.distance_matrix[i, i] = 0.0
            self.ssim_matrix[i, i] = 1.0
            self.pipeline_valid[i, i] = True
    
    def get_indices(self, sec1, sec2):
        """Convert section numbers to matrix indices"""
        return self.section_to_idx[sec1], self.section_to_idx[sec2]
    
    def is_computed(self, sec1, sec2):
        """Check if this pair has already been computed"""
        i, j = self.get_indices(sec1, sec2)
        return self.computed[i, j]
    
    def get_distance(self, sec1, sec2):
        """Get 1-SSIM distance between sections (returns inf if not computed)"""
        i, j = self.get_indices(sec1, sec2)
        return self.distance_matrix[i, j]
    
    def get_ssim(self, sec1, sec2):
        """Get SSIM score between sections (returns 0 if not computed)"""
        i, j = self.get_indices(sec1, sec2)
        return self.ssim_matrix[i, j]
    
    def is_pipeline_valid(self, sec1, sec2):
        """Check if pair meets pipeline validation criteria"""
        i, j = self.get_indices(sec1, sec2)
        return self.pipeline_valid[i, j]
    
    def get_detailed_result(self, sec1, sec2):
        """Get full result dictionary for a pair"""
        i, j = self.get_indices(sec1, sec2)
        min_idx, max_idx = min(i, j), max(i, j)
        return self.detailed_results.get((min_idx, max_idx), None)
    
    def store_result(self, sec1, sec2, result_dict, ssim_score, pipeline_valid_flag):
        """Store computation result in matrices"""
        i, j = self.get_indices(sec1, sec2)
        
        # Check if already computed
        if self.computed[i, j]:
            self.cache_hits += 1
            return False  # Already computed
        
        # Store in symmetric matrices
        self.computed[i, j] = self.computed[j, i] = True
        
        distance = max(0.0, 1.0 - ssim_score)  # Ensure non-negative distance
        self.distance_matrix[i, j] = self.distance_matrix[j, i] = distance
        self.ssim_matrix[i, j] = self.ssim_matrix[j, i] = ssim_score
        self.pipeline_valid[i, j] = self.pipeline_valid[j, i] = pipeline_valid_flag
        
        # Store detailed result (use min/max indices for consistent key)
        min_idx, max_idx = min(i, j), max(i, j)
        self.detailed_results[(min_idx, max_idx)] = result_dict
        
        self.unique_pairs_computed += 1
        return True  # New computation
    
    def record_oracle_call(self):
        """Record that an oracle call was made (for statistics)"""
        self.total_oracle_calls += 1
    
    def get_statistics(self):
        """Get computation statistics"""
        theoretical_pairs = self.N * (self.N - 1) // 2
        return {
            'total_sections': self.N,
            'theoretical_pairs': theoretical_pairs,
            'unique_pairs_computed': self.unique_pairs_computed,
            'total_oracle_calls': self.total_oracle_calls,
            'cache_hits': self.cache_hits,
            'computation_ratio': self.unique_pairs_computed / theoretical_pairs,
            'redundancy_ratio': self.total_oracle_calls / max(1, self.unique_pairs_computed),
            'cache_hit_rate': self.cache_hits / max(1, self.total_oracle_calls)
        }
    
    def get_computed_pairs_list(self):
        """Get list of all computed pairs for iteration"""
        pairs = []
        for i in range(self.N):
            for j in range(i + 1, self.N):  # Only upper triangle
                if self.computed[i, j]:
                    pairs.append((self.all_sections[i], self.all_sections[j]))
        return pairs
    
    def get_pipeline_valid_pairs(self):
        """Get list of pipeline-valid pairs"""
        pairs = []
        for i in range(self.N):
            for j in range(i + 1, self.N):
                if self.computed[i, j] and self.pipeline_valid[i, j]:
                    pairs.append((self.all_sections[i], self.all_sections[j]))
        return pairs
    
    def export_to_dataframe(self):
        """Export all computed results to pandas DataFrame"""
        import pandas as pd
        
        rows = []
        for i in range(self.N):
            for j in range(i + 1, self.N):
                if self.computed[i, j]:
                    sec1, sec2 = self.all_sections[i], self.all_sections[j]
                    detailed = self.get_detailed_result(sec1, sec2)
                    
                    row = {
                        'section1': sec1,
                        'section2': sec2,
                        'ssim_score': self.ssim_matrix[i, j],
                        'distance': self.distance_matrix[i, j],
                        'pipeline_valid': self.pipeline_valid[i, j]
                    }
                    
                    # Add detailed fields if available
                    if detailed:
                        row.update(detailed)
                    
                    rows.append(row)
        
        return pd.DataFrame(rows)

# Union-Find (Disjoint Set Union) class for Randomized Borůvka
class UnionFind:
    """Union-Find data structure with path compression and union by rank"""
    def __init__(self, n):
        self.parent = list(range(n))
        self.rank = [0] * n
        self.num_components = n
    
    def find(self, x):
        """Find with path compression"""
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]
    
    def union(self, x, y):
        """Union by rank"""
        px, py = self.find(x), self.find(y)
        if px == py:
            return False
        
        if self.rank[px] < self.rank[py]:
            px, py = py, px
        
        self.parent[py] = px
        if self.rank[px] == self.rank[py]:
            self.rank[px] += 1
        
        self.num_components -= 1
        return True
    
    def connected(self, x, y):
        """Check if two elements are in the same component"""
        return self.find(x) != self.find(y)

def randomized_boruvka_sublinear(all_sections, edge_oracle, verbose=False):
    """
    Randomized Borůvka algorithm for sublinear spanning tree construction
    
    Parameters:
    - all_sections: List of section numbers
    - edge_oracle: Function that takes (u, v) and returns True if edge should be included
    - verbose: Whether to print detailed progress
    
    Returns:
    - List of (u, v) edge pairs forming the spanning tree
    - Statistics about the process
    """
    
    if verbose:
        print("\n" + "="*60)
        print(" SUBLINEAR APPROACH: Randomized Borůvka Algorithm")
        print("="*60)
        print("⚠  WARNING: This feature is UNDER CONSTRUCTION")
        print("  The sublinear approach is experimental and may not be fully optimized.")
        print("   Please use --verbose for detailed algorithmic walkthrough.\n")
        
        print(" ALGORITHMIC OVERVIEW:")
        print("   Using Randomized Borůvka (\"random-hook\") algorithm")
        print("   Works with nothing more than an EdgeExists(u,v) oracle and Union-Find structure.\n")
        
        print(" DETAILED APPROACH:")
        print("   1. Forest view: Treat every vertex as one-node \"component\"")
        print("   2. Hooking phase: Each component tries to attach to different component")
        print("      by probing just a few random edges instead of scanning all pairs")
        print("   3. Component merge: When edge found between different components,")
        print("      union them and keep that edge in output tree")
        print("   4. Repeat: After one phase, many components merge. Run another phase")
        print("      on now-smaller forest. Continue until only one component remains")
        print("   5. Output: The N-1 accepted edges form spanning tree (no cycles)\n")
    
    N = len(all_sections)
    
    # Create mapping from section numbers to indices
    section_to_idx = {section: i for i, section in enumerate(all_sections)}
    idx_to_section = {i: section for i, section in enumerate(all_sections)}
    
    # Parameters
    k = max(4, math.ceil(2 * math.log2(N)))  # samples per vertex per phase
    max_rounds = math.ceil(math.log2(N))     # maximum number of rounds
    
    if verbose:
        print(f" PARAMETERS:")
        print(f"   N (vertices): {N}")
        print(f"   k (samples per vertex per phase): {k}")
        print(f"   max_rounds (maximum phases): {max_rounds}")
        print(f"   Theoretical edge tests: N × k × log₂(N) = {N} × {k} × {max_rounds} ≈ {N * k * max_rounds:,}")
        print(f"   vs. Complete pairwise: N(N-1)/2 = {N*(N-1)//2:,}")
        print(f"   Reduction factor: {(N*(N-1)//2) / (N * k * max_rounds):.1f}x fewer edge tests\n")
    
    # Initialize Union-Find
    uf = UnionFind(N)
    spanning_tree = []
    
    # Statistics
    total_edge_tests = 0
    total_successful_hooks = 0
    
    # Main algorithm loop
    phase = 0
    while uf.num_components > 1 and phase < max_rounds:
        phase += 1
        phase_edge_tests = 0
        phase_successful_hooks = 0
        
        if verbose:
            print(f"🔄 PHASE {phase}:")
            print(f"   Components at start: {uf.num_components}")
        
        # For each vertex, try to find a hook to another component
        for u_idx in range(N):
            u_section = idx_to_section[u_idx]
            
            # Try k random samples
            for trial in range(k):
                # Random sample
                v_idx = random.randint(0, N - 1)
                v_section = idx_to_section[v_idx]
                
                # Skip self-loops
                if u_idx == v_idx:
                    continue
                
                phase_edge_tests += 1
                
                # Check if they're in different components
                if uf.find(u_idx) != uf.find(v_idx):
                    # Call the oracle to check if edge should exist
                    if edge_oracle(u_section, v_section):
                        # Union the components
                        if uf.union(u_idx, v_idx):
                            spanning_tree.append((u_section, v_section))
                            phase_successful_hooks += 1
                            
                            if verbose:
                                print(f"      ✓ Hook found: {u_section} ↔ {v_section}")
                            
                            # One successful hook per vertex per phase is enough
                            break
        
        total_edge_tests += phase_edge_tests
        total_successful_hooks += phase_successful_hooks
        
        if verbose:
            print(f"   Edge tests this phase: {phase_edge_tests}")
            print(f"   Successful hooks this phase: {phase_successful_hooks}")
            print(f"   Components at end: {uf.num_components}")
            
            if uf.num_components == 1:
                print(f"   🎉 Single component achieved!")
                break
            print()
    
    # Final statistics
    stats = {
        'total_phases': phase,
        'total_edge_tests': total_edge_tests,
        'total_successful_hooks': total_successful_hooks,
        'spanning_tree_edges': len(spanning_tree),
        'theoretical_complete_pairs': N * (N - 1) // 2,
        'reduction_factor': (N * (N - 1) // 2) / total_edge_tests if total_edge_tests > 0 else 0,
        'final_components': uf.num_components
    }
    
    if verbose:
        print(f" FINAL STATISTICS:")
        print(f"   Total phases completed: {phase}")
        print(f"   Total edge tests: {total_edge_tests:,}")
        print(f"   Total successful hooks: {total_successful_hooks}")
        print(f"   Spanning tree edges: {len(spanning_tree)}")
        print(f"   Final components: {uf.num_components}")
        print(f"   Reduction vs complete: {stats['reduction_factor']:.1f}x fewer edge tests")
        print(f"   Success rate: {(total_successful_hooks / total_edge_tests * 100):.1f}% of tested edges were useful")
        
        if uf.num_components == 1:
            print(f"   ✅ SUCCESS: Connected spanning tree achieved!")
        else:
            print(f"   ⚠  WARNING: {uf.num_components} disconnected components remain")
            print(f"   This may indicate sparse connectivity in the underlying graph")
    
    return spanning_tree, stats


def final_k_densification_verification(all_sections, final_order, edge_oracle, get_edge_data, K, data_manager, verbose=False):
    """
    Phase F: Final K-densification verification on the spectral ordering output
    
    This phase performs a K-window densification on the final ordering to:
    1. Verify that the ordering is K-neighbor dense
    2. Discover any additional local connections that may have been missed
    3. Provide validation of the final ordering quality
    
    Parameters:
    - all_sections: List of section numbers
    - final_order: Final ordering from spectral phase
    - edge_oracle: Function to test if edge should be included
    - get_edge_data: Function to get edge data
    - K: Window half-width for densification
    - data_manager: PairwiseDataManager to track which edges have been computed
    - verbose: Whether to print detailed progress
    
    Returns:
    - Dictionary with verification statistics and any new edges found
    """
    N = len(final_order)
    new_edges = []
    total_tests = 0
    newly_computed_tests = 0
    connections_found = 0
    newly_computed_connections = 0
    
    # Create position mapping for efficiency
    position = {section: i for i, section in enumerate(final_order)}
    
    if verbose:
        print(f"   K-window size: ±{K} positions")
        print(f"   Testing {final_order[0]} → {final_order[1]} → ... → {final_order[-1]}")
    
    # Test K-window around each position in the final ordering
    for i, section_u in enumerate(final_order):
        # Test forward window: positions i+1 to i+K
        for offset in range(1, K + 1):
            j = i + offset
            if j >= N:
                break
                
            section_v = final_order[j]
            total_tests += 1
            
            # Check if this edge has been computed before
            was_previously_computed = data_manager.is_computed(section_u, section_v)
            if not was_previously_computed:
                newly_computed_tests += 1
            
            # Test if this edge should be included
            if edge_oracle(section_u, section_v):
                edge_data = get_edge_data(section_u, section_v)
                if edge_data and edge_data.get('ssim_score', 0) > 0:
                    ssim_score = edge_data['ssim_score']
                    connections_found += 1
                    new_edges.append((section_u, section_v, ssim_score))
                    
                    if not was_previously_computed:
                        newly_computed_connections += 1
                    
                    if verbose and connections_found <= 10:  # Show first 10 discoveries
                        rotation = edge_data.get('rotation_degrees', 0)
                        inlier_ratio = edge_data.get('inlier_ratio', 0) * 100
                        cache_status = "(cached)" if was_previously_computed else "(NEW)"
                        print(f"      ✅ section_{section_u} ↔ section_{section_v} (gap={offset}) {cache_status}: "
                              f"SSIM={ssim_score:.3f}, Rot={rotation:.1f}°, Inliers={inlier_ratio:.1f}%")
    
    success_rate = connections_found / max(1, total_tests)
    newly_computed_success_rate = newly_computed_connections / max(1, newly_computed_tests)
    
    if verbose:
        if connections_found > 10:
            print(f"      ... and {connections_found - 10} more connections")
        print(f"      Phase F Stats: {total_tests:,} tests, {connections_found:,} connections")
        print(f"      NEW tests: {newly_computed_tests:,}, NEW connections: {newly_computed_connections:,}")
        print(f"      Overall success rate: {success_rate:.1%}")
        print(f"      NEW computations success rate: {newly_computed_success_rate:.1%}")
        
        if newly_computed_connections == 0:
            print(f"      ✅ Perfect K-densification: No new local connections discovered!")
        else:
            print(f"      Found {newly_computed_connections} NEW additional local connections")
            print(f"      This suggests the ordering could be further improved")
    
    return {
        'total_tests': total_tests,
        'newly_computed_tests': newly_computed_tests,
        'connections_found': connections_found,
        'newly_computed_connections': newly_computed_connections,
        'success_rate': success_rate,
        'newly_computed_success_rate': newly_computed_success_rate,
        'new_edges': new_edges,
        'average_gap': sum(abs(position[u] - position[v]) for u, v, _ in new_edges) / max(1, len(new_edges)),
        'max_gap': max((abs(position[u] - position[v]) for u, v, _ in new_edges), default=0)
    }


def build_and_order_sublinear(all_sections, edge_oracle, get_edge_data, verbose=False, use_spectral=True, two_phase=False, ssim_threshold=None, inlier_threshold=None, images=None, data_manager=None, cache_sift=False, fast_flann=False, K_override=None, samples_per_phase=None, condensation_rounds=None):
    """
    BUILD-AND-ORDER algorithm for sub-quadratic 1-D ordering with binary graph
    Adapted for biological section alignment using binary oracle (strong connections only)
    
    Parameters:
    - all_sections: List of section numbers
    - edge_oracle: Function that takes (u, v) and returns True if edge should be included
    - get_edge_data: Function that returns edge data (SSIM, rotation, etc.)
    - verbose: Whether to print detailed progress
    - use_spectral: Whether to use spectral ordering in Phase E (optional, as in theory)
    - ssim_threshold: SSIM threshold for strong connections (None = use DEFAULT_SSIM_THRESHOLD)
    - inlier_threshold: Inlier ratio threshold for strong connections (None = use DEFAULT_INLIER_THRESHOLD)
    
    Returns:
    - final_order: List of sections in linear order
    - stats: Algorithm statistics
    """
    
    # Apply default values if not provided
    if ssim_threshold is None:
        ssim_threshold = DEFAULT_SSIM_THRESHOLD
    if inlier_threshold is None:
        inlier_threshold = DEFAULT_INLIER_THRESHOLD
    
    if verbose:
        print("\n" + "="*70)
        print("   BUILD-AND-ORDER: Sub-quadratic 1-D Ordering")
        print("="*70)
        print("   BINARY GRAPH ALGORITHM FOR BIOLOGICAL SECTION ALIGNMENT:")
        print("   Using binary oracle (strong connections only)")
        print("   Phase A: Short-edge Borůvka tree")
        print("   Phase B: Farthest-point condensation (3 rounds)")
        print("   Phase C: Double-sweep BFS ordering")
        print("   Phase D: K-window densification")
        if use_spectral:
            print("   Phase E: Final spectral ordering (OPTIONAL)")
        else:
            print("   Phase E: Skip spectral ordering (use BFS result)")
        print()
    
    N = len(all_sections)
    
    # Binary graph parameters (no λ or ρ needed!)
    if K_override is not None:
        K = K_override
    else:
        K = min(15, max(5, N // 20))  # Window half-width
    
    if samples_per_phase is not None and samples_per_phase > 0:
        s = samples_per_phase
    else:
        s = 12  # default
    
    if condensation_rounds is not None and condensation_rounds > 0:
        t = condensation_rounds
    else:
        t = 3   # Fixed condensation rounds
    
    if verbose:
        print(f"   ALGORITHM PARAMETERS:")
        print(f"   Algorithm: Binary Graph BUILD-AND-ORDER")
        print(f"   K (window half-width): {K}")
        print(f"   s (samples per phase): {s}")
        print(f"   t (condensation rounds): {t} (FIXED)")
        print(f"   Spectral ordering: {'ENABLED' if use_spectral else 'DISABLED'}")
        print(f"   Theoretical complexity: O(N(log N + K)) = O({N}({math.ceil(math.log2(N))} + {K}))\n")
    
    # Phase A: Short-edge Borůvka tree
    if verbose:
        print(" PHASE A: Short-edge Borůvka Tree")
        if two_phase:
            print("   Using two-phase weak→strong approach")
    
    if two_phase:
        tree_edges, phase_a_stats = two_phase_boruvka_bio_binary(all_sections, edge_oracle, get_edge_data, s, verbose)
    else:
        tree_edges, phase_a_stats = short_edge_boruvka_bio_binary(all_sections, edge_oracle, get_edge_data, s, ssim_threshold, inlier_threshold, images, data_manager, cache_sift, fast_flann, verbose)
    
    if verbose:
        comps = N - len(tree_edges)
        print(f"   ✅ Found {len(tree_edges)} tree edges (components remaining: {comps})")
        print(f"      Phase A Stats: {phase_a_stats['edge_tests']:,} tests, {phase_a_stats['connections_found']:,} connections, {phase_a_stats['success_rate']:.1%} success rate")
    
    # Phase B: Farthest-point condensation (exactly 3 rounds)
    if verbose:
        print(f"\n   PHASE B: Farthest-point Condensation ({t} rounds)")
    
    condensed_graph, phase_b_stats = farthest_point_condense_bio_binary(all_sections, tree_edges, edge_oracle, get_edge_data, t, K, verbose)
    
    if verbose:
        print(f"     Phase B Stats: {phase_b_stats['total_tests']:,} tests, {phase_b_stats['connections_found']:,} connections, {phase_b_stats['success_rate']:.1%} success rate")
        for round_num, round_stats in enumerate(phase_b_stats['rounds'], 1):
            print(f"      Round {round_num}: {round_stats['tests']:,} tests, {round_stats['connections']:,} connections")
    
    # Phase C: Double-sweep BFS ordering
    if verbose:
        print("\n  PHASE C: Double-sweep BFS Ordering")
    
    rough_order = double_sweep_order_bio(all_sections, condensed_graph, verbose)
    
    # Phase D: K-window densification
    if verbose:
        print("\n  PHASE D: K-window Densification")
    
    window_edges, phase_d_stats = k_window_edges_bio_binary(all_sections, rough_order, edge_oracle, get_edge_data, K, verbose)
    
    if verbose:
        print(f"      Phase D Stats: {phase_d_stats['window_tests']:,} tests, {phase_d_stats['connections_found']:,} connections")
        print(f"      Window success rate: {phase_d_stats['success_rate']:.1%}")
        print(f"      Average connections per vertex: {phase_d_stats['average_connections_per_vertex']:.1f}")
    
    # Add window edges to graph
    for u, v, d in window_edges:
        u_idx = all_sections.index(u)
        v_idx = all_sections.index(v)
        condensed_graph[u_idx].append((v_idx, d))
        condensed_graph[v_idx].append((u_idx, d))
    
    # Phase E: Final ordering (spectral is optional)
    if use_spectral:
        if verbose:
            print("\n   PHASE E: Final Spectral Ordering")
        final_order = spectral_order_bio(all_sections, condensed_graph, rough_order, verbose)
    else:
        if verbose:
            print("\n   PHASE E: Using BFS ordering (spectral disabled)")
        final_order = rough_order
    
    # Phase F: Final K-densification verification on spectral output
    if verbose:
        print("\n   PHASE F: Final K-densification Verification")
        print(f"   Verifying K-neighbor density on final ordering...")
    
    phase_f_stats = final_k_densification_verification(
        all_sections, final_order, edge_oracle, get_edge_data, K, data_manager, verbose
    )
    
    # Update final order with any additional edges found
    if phase_f_stats['connections_found'] > 0:
        # Add new edges to the graph for potential ordering improvement
        for u, v, d in phase_f_stats['new_edges']:
            u_idx = all_sections.index(u)
            v_idx = all_sections.index(v)
            condensed_graph[u_idx].append((v_idx, d))
            condensed_graph[v_idx].append((u_idx, d))
        
        if verbose:
            print(f"   Found {phase_f_stats['connections_found']} additional edges")
            if phase_f_stats['connections_found'] > 0:
                print(f"    Consider re-running spectral ordering with these additional edges")
    
    # Compile comprehensive statistics
    # Combine tree structure information from Phase A
    combined_stats = {
        'algorithm': 'BUILD-AND-ORDER (Binary Graph)',
        'parameters': {
            'K': K,
            's': s,
            't': t,
            'use_spectral': use_spectral
        },
        'phase_a': phase_a_stats,
        'phase_b': phase_b_stats,
        'phase_d': phase_d_stats,
        'phase_f': phase_f_stats,
        'tree_edges': len(tree_edges),
        'window_edges': len(window_edges),
        'rough_order_quality': evaluate_order_quality(rough_order, get_edge_data),
        'final_order_quality': evaluate_order_quality(final_order, get_edge_data)
    }
    
    # Add tree structure analysis from Phase A if available
    if 'tree_edges_list' in phase_a_stats:
        combined_stats['tree_edges_list'] = phase_a_stats['tree_edges_list']
    if 'tree_diameter' in phase_a_stats:
        combined_stats['tree_diameter'] = phase_a_stats['tree_diameter']
    if 'connected_components' in phase_a_stats:
        combined_stats['connected_components'] = phase_a_stats['connected_components']
    
    stats = combined_stats
    
    return final_order, stats


def estimate_minimal_spacing(all_sections, edge_oracle, get_edge_data, verbose=False):
    """Estimate minimal spacing λ from successful alignments"""
    
    # Sample some edges to estimate typical "good" distances
    sample_size = min(100, len(all_sections) * 2)
    distances = []
    
    for _ in range(sample_size):
        u = random.choice(all_sections)
        v = random.choice(all_sections)
        if u != v and edge_oracle(u, v):
            edge_data = get_edge_data(u, v)
            if edge_data and edge_data.get('ssim_score', 0) > 0:
                distance = 1 - edge_data['ssim_score']  # Convert SSIM to distance
                distances.append(distance)
    
    if not distances:
        # Fallback if no good edges found
        return 0.5
    
    # Use 20th percentile as estimate of minimal spacing
    distances.sort()
    λ = distances[len(distances) // 5] if len(distances) >= 5 else distances[0]
    
    if verbose:
        print(f"   Sampled {len(distances)} good edges")
        print(f"   Distance range: {min(distances):.3f} to {max(distances):.3f}")
        print(f"   Estimated λ: {λ:.3f}")
    
    return λ


def short_edge_boruvka_bio_binary(all_sections, edge_oracle, get_edge_data, s, ssim_threshold=None, inlier_threshold=None, images=None, data_manager=None, cache_sift=False, fast_flann=False, verbose=False):
    """Phase A: Short-edge Borůvka tree for binary graph (no distance thresholds)
    
    CORRECTED IMPLEMENTATION: Following user's exact specification.
            D(u,v) called exactly once per edge - no duplicate biological pipeline calls.
    """
    
    # Apply default values if not provided
    if ssim_threshold is None:
        ssim_threshold = DEFAULT_SSIM_THRESHOLD
    if inlier_threshold is None:
        inlier_threshold = DEFAULT_INLIER_THRESHOLD
    
    N = len(all_sections)
    
    class DSU:
        def __init__(self, n):
            self.parent = list(range(n))
            self.rank = [0] * n
            self.components = n
        
        def find(self, x):
            if self.parent[x] != x:
                self.parent[x] = self.find(self.parent[x])
            return self.parent[x]
        
        def union(self, a, b):
            root_a = self.find(a)
            root_b = self.find(b)
            
            if root_a == root_b:
                return False
            
            if self.rank[root_a] < self.rank[root_b]:
                self.parent[root_a] = root_b
            elif self.rank[root_a] > self.rank[root_b]:
                self.parent[root_b] = root_a
            else:
                self.parent[root_b] = root_a
                self.rank[root_a] += 1
            
            self.components -= 1
            return True
    
    dsu = DSU(N)
    tree_edges = []
    phase = 0
    distance_calls = 0
    connections_found = 0
    
    # Set random seed for reproducibility
    random.seed(42)
    
    # Define single distance function that does biological alignment computation once
    def D(u_sec, v_sec):
        """Single distance function - full biological pipeline alignment happens here"""
        nonlocal distance_calls
        distance_calls += 1
        
        # Always do the full biological pipeline alignment and get actual distance
        # Don't do threshold comparison here - let the algorithm handle it
        if data_manager.is_computed(u_sec, v_sec):
            # Use cached result
            edge_data = data_manager.get_detailed_result(u_sec, v_sec)
            if edge_data and edge_data.get('ssim_score', 0) > 0:
                distance = 1 - edge_data['ssim_score']
                return distance
            else:
                return float('inf')
        else:
            # Do full biological pipeline alignment
            img1 = images[u_sec]
            img2 = images[v_sec]
            
            # Try to find alignment
            transformation_matrix, _, _, stats = find_rigid_sift_alignment(
                img1, img2, f"section_{u_sec}", f"section_{v_sec}",
                use_cache=cache_sift, fast_flann=fast_flann, verbose=verbose, inlier_threshold=inlier_threshold
            )
            
            # Calculate SSIM if basic alignment was successful
            ssim_score = 0.0
            if transformation_matrix is not None and stats['success']:
                # Pre-validate transformation parameters before attempting SSIM calculation
                rotation = abs(stats.get('rotation_deg', 0))
                scale_x = stats.get('scale_x', 1)
                scale_y = stats.get('scale_y', 1)
                
                # Check if transformation parameters are biologically plausible
                transformation_valid = (
                    rotation <= ROTATION_LIMIT_DEGREES and             # Rotation within ±90°
                    SCALE_MIN <= scale_x <= SCALE_MAX and               # Scale X within range
                    SCALE_MIN <= scale_y <= SCALE_MAX and               # Scale Y within range
                    scale_x > 0.1 and scale_y > 0.1                    # Avoid degenerate scales
                )
                
                if transformation_valid:
                    aligned_img1, overlap_region1, overlap_region2, bbox = apply_rigid_transformation_and_find_overlap(
                        img1, img2, transformation_matrix)
                    
                    if overlap_region1 is not None and overlap_region2 is not None:
                        ssim_score = calculate_ssim_score(overlap_region1, overlap_region2)
                else:
                    # Mark as failed due to invalid transformation
                    stats['success'] = False
            
            # Store result in data manager
            result = {
                'section1': u_sec,
                'section2': v_sec,
                'ssim_score': ssim_score,
                'features1': stats['features1'],
                'features2': stats['features2'],
                'initial_matches': stats['matches'],
                'ransac_inliers': stats['inliers'],
                'inlier_ratio': stats['inlier_ratio'],
                'rotation_degrees': stats.get('rotation_deg', 0),
                'translation_x': stats.get('translation_x', 0),
                'translation_y': stats.get('translation_y', 0),
                'scale_x': stats.get('scale_x', 1),
                'scale_y': stats.get('scale_y', 1),
                'detection_time_sec': stats['detection_time'],
                'matching_time_sec': stats['matching_time'],
                'alignment_success': stats['success'],
                'error_message': stats.get('error', ''),
                'overlap_area_pixels': bbox[2] * bbox[3] if 'bbox' in locals() else 0,
                'overlap_bbox': f"{bbox[0]},{bbox[1]},{bbox[2]},{bbox[3]}" if 'bbox' in locals() else '',
                'pipeline_valid': False
            }
            
            # Apply pipeline validation
            pipeline_valid = False
            if stats['success'] and ssim_score > 0:
                rotation = abs(stats.get('rotation_deg', 0))
                scale_x = stats.get('scale_x', 1)
                scale_y = stats.get('scale_y', 1)
                inlier_ratio = stats.get('inlier_ratio', 0)
                
                pipeline_valid = (
                    rotation <= ROTATION_LIMIT_DEGREES and
                    SCALE_MIN <= scale_x <= SCALE_MAX and
                    SCALE_MIN <= scale_y <= SCALE_MAX and
                    inlier_ratio > inlier_threshold
                )
            
            result['pipeline_valid'] = pipeline_valid
            data_manager.store_result(u_sec, v_sec, result, ssim_score, pipeline_valid)
            
            # Return 1-SSIM distance
            if ssim_score > 0:
                return 1 - ssim_score
            else:
                return float('inf')
    
    while dsu.components > 1:
        phase += 1
        if verbose:
            print(f"   Borůvka phase {phase}: {dsu.components} components remaining")
        # >>>>>>>>>>>>>>>>>>>> here to insert <<<<<<<<<<<<<<<<<<
        if dsu.components <= 20:        # only print when components are few, too many will flood the screen
            from collections import defaultdict
            comp_map = defaultdict(list)
            for idx, sec in enumerate(all_sections):
                root = dsu.find(idx)     # find root in DSU
                comp_map[root].append(sec)

            # sort each connected component by section number and output
            print("      🔍 Current components:")
            for cid, members in comp_map.items():
                members.sort()
                print(f"        • {members}")
                # also write to file:
                # with open("remaining_components.txt", "a") as f:
                #     f.write(f"Phase {phase} | Component {cid}: {members}\n")        
        phase_connections = 0
        
        for u in range(N):
            if dsu.components == 1:
                break
                
            # Try s random vertices to find a connection
            for _ in range(s):
                v = random.randint(0, N - 1)
                if dsu.find(u) == dsu.find(v):
                    continue  # Same component, skip
                
                # SINGLE call to distance function (full biological pipeline)
                u_sec = all_sections[u]
                v_sec = all_sections[v]
                d = D(u_sec, v_sec)
                
                # Check if connection passed ALL pipeline criteria (not just SSIM threshold)
                if not data_manager.is_pipeline_valid(u_sec, v_sec):
                    continue  # Failed pipeline validation (SSIM, rotation, scale, or inlier threshold)
                
                # SUCCESS: accept and stop sampling
                if dsu.union(u, v):
                    tree_edges.append((u_sec, v_sec, d))
                    connections_found += 1
                    phase_connections += 1
                    
                    if verbose:
                        ssim_score = 1 - d
                        print(f"      Connected {u_sec} ↔ {v_sec} (SSIM: {ssim_score:.3f})")
                    break  # ← EARLY EXIT: u hooked this phase, stop sampling
        
        if verbose:
            print(f"      Phase {phase}: {phase_connections:,} connections added")
    
    stats = {
        'distance_calls': distance_calls,
        'connections_found': connections_found,
        'success_rate': connections_found / max(1, distance_calls),
        'phases': phase,
        'tree_edges': len(tree_edges)
    }
    
    # Add alias for backward compatibility with build_and_order_sublinear
    stats['edge_tests'] = stats['distance_calls']
    
    if verbose:
        print(f"   Phase A (Borůvka): {distance_calls:,} distance calls, {connections_found:,} connections in {phase} phases")
        print(f"   Success rate: {connections_found/max(1, distance_calls)*100:.1f}%")
    
    # Calculate tree diameter and structure analysis
    diameter, diameter_path, center_nodes = calculate_tree_diameter(tree_edges, all_sections)
    
    # Count connected components
    all_nodes = set()
    for u, v, weight in tree_edges:
        all_nodes.add(u)
        all_nodes.add(v)
    connected_components = 1 if len(all_nodes) == len(all_sections) else len(all_sections) - len(all_nodes) + 1
    
    # Enhanced statistics with tree structure analysis
    stats.update({
        'tree_edges_list': [(u, v, weight) for u, v, weight in tree_edges],
        'tree_diameter': {
            'diameter': diameter,
            'endpoints': (diameter_path[0], diameter_path[-1]) if diameter_path else (None, None),
            'center_nodes': center_nodes,
            'diameter_path': diameter_path
        },
        'connected_components': connected_components
    })
    
    return tree_edges, stats


def max_true_gap(order, short_neighbors):
    """
    Compute the maximum gap between any vertex and its short neighbors in the current ordering.
    
    Args:
        order: List of sections in current ordering
        short_neighbors: Dict mapping section -> list of sections with distance <= ρ
    
    Returns:
        Maximum gap (in positions) between any vertex and its short neighbors
    """
    N = len(order)
    pos = {}
    for i, v in enumerate(order):
        pos[v] = i
    
    worst_gap = 0
    for v in order:
        for w in short_neighbors.get(v, []):
            gap = abs(pos[v] - pos[w])
            if gap > worst_gap:
                worst_gap = gap
    
    return worst_gap


def farthest_point_condense_bio_binary(all_sections, tree_edges, edge_oracle, get_edge_data, t, K, verbose=False):
    """Phase B: Farthest-point condensation for binary graph"""
    
    N = len(all_sections)
    
    # Build initial graph from tree edges
    G = [[] for _ in range(N)]
    for u, v, d in tree_edges:
        u_idx = all_sections.index(u)
        v_idx = all_sections.index(v)
        G[u_idx].append((v_idx, d))
        G[v_idx].append((u_idx, d))
    
    # Track short neighbors for early stopping
    short_neighbors = [[] for _ in range(N)]
    for u, v, _ in tree_edges:
        u_idx = all_sections.index(u)
        v_idx = all_sections.index(v)
        short_neighbors[u_idx].append(v_idx)
        short_neighbors[v_idx].append(u_idx)
    
    seed = []
    dist = [float('inf')] * N
    
    total_tests = 0
    connections_found = 0
    round_stats = []
    
    for round_num in range(t):
        if verbose:
            print(f"   Condensation round {round_num + 1}/{t}")
        
        round_tests = 0
        round_connections = 0
        
        # 1. Pick farthest vertex w.r.t. current seed set
        if not seed:
            w = 0  # arbitrary first seed
        else:
            w = max(range(N), key=lambda x: dist[x])
        seed.append(w)
        
        if verbose:
            print(f"      Selected seed: {all_sections[w]}")
        
        # 2. Single-source BFS from w (no distance threshold in binary version)
        queue = [(0, w)]
        visited = [False] * N
        visited[w] = True
        dist[w] = 0
        
        while queue:
            d_u, u = queue.pop(0)
            if d_u < dist[u]:
                dist[u] = d_u
            
            for v_idx, edge_dist in G[u]:
                if not visited[v_idx]:
                    visited[v_idx] = True
                    dist[v_idx] = min(dist[v_idx], d_u + edge_dist)
                    queue.append((d_u + edge_dist, v_idx))
        
        # 3. Add edges from w to all vertices with strong connections
        for v_idx in range(N):
            if v_idx != w:
                u_sec = all_sections[w]
                v_sec = all_sections[v_idx]
                
                round_tests += 1
                total_tests += 1
                
                if edge_oracle(u_sec, v_sec):
                    round_connections += 1
                    connections_found += 1
                    
                    # Get edge data
                    edge_data = get_edge_data(u_sec, v_sec)
                    distance = 1 - edge_data.get('ssim_score', 0) if edge_data else 0.5
                    
                    G[w].append((v_idx, distance))
                    G[v_idx].append((w, distance))
                    
                    # Update short neighbors tracking
                    if v_idx not in short_neighbors[w]:
                        short_neighbors[w].append(v_idx)
                    if w not in short_neighbors[v_idx]:
                        short_neighbors[v_idx].append(w)
        
        round_stats.append({
            'tests': round_tests,
            'connections': round_connections
        })
        
        if verbose:
            print(f"      Round {round_num + 1}: {round_tests:,} tests, {round_connections:,} connections")
        
        # 4. Early stop check (after at least 1 round)
        if round_num >= 1:
            # Quick ordering check
            def bfs_order():
                visited = [False] * N
                queue = [0]
                visited[0] = True
                order = []
                
                while queue:
                    u = queue.pop(0)
                    order.append(u)
                    for v, _ in G[u]:
                        if not visited[v]:
                            visited[v] = True
                            queue.append(v)
                
                return order
            
            current_order = bfs_order()
            max_gap = max_true_gap_binary(current_order, short_neighbors)
            
            if verbose:
                print(f"      Max gap: {max_gap}, K: {K}")
            
            if max_gap <= K:
                if verbose:
                    print(f"      ✅ Early stop: gap {max_gap} ≤ K={K}")
                break
    
    stats = {
        'total_tests': total_tests,
        'connections_found': connections_found,
        'success_rate': connections_found / max(1, total_tests),
        'rounds': round_stats,
        'early_stop': round_num + 1 < t
    }
    
    return G, stats


def double_sweep_order_bio(all_sections, G, verbose=False):
    """Phase C: Double-sweep BFS ordering"""
    
    N = len(all_sections)
    
    def bfs(src):
        Q = [src]
        dist = [-1] * N
        dist[src] = 0
        for u in Q:
            for v, _ in G[u]:
                if dist[v] < 0:
                    dist[v] = dist[u] + 1
                    Q.append(v)
        farthest = max(range(N), key=lambda x: dist[x])
        return farthest, dist
    
    # Double sweep to find diameter
    a, _ = bfs(0)
    b, distA = bfs(a)
    
    # Order by distance from a
    order_indices = sorted(range(N), key=lambda x: (distA[x], x))
    rough_order = [all_sections[i] for i in order_indices]
    
    if verbose:
        print(f"   Diameter endpoints: {all_sections[a]} ↔ {all_sections[b]}")
        print(f"   Diameter length: {distA[b]} hops")
        print(f"   Rough order: {rough_order[:5]}...{rough_order[-5:]}")
    
    return rough_order


def k_window_edges_bio_binary(all_sections, rough_order, edge_oracle, get_edge_data, K, verbose=False):
    """Phase D: K-window densification for binary graph with recovery rate analysis"""
    
    N = len(rough_order)
    extra_edges = []
    window_tests = 0
    connections_found = 0
    
    # K-window densification (no exhaustive recovery analysis!)
    for rank, u in enumerate(rough_order):
        for off in range(1, K + 1):
            v_pos = rank + off
            if v_pos >= N:
                break
            v = rough_order[v_pos]
            
            window_tests += 1
            if edge_oracle(u, v):
                connections_found += 1
                edge_data = get_edge_data(u, v)
                distance = 1 - edge_data.get('ssim_score', 0) if edge_data else 0.5
                extra_edges.append((u, v, distance))
    
    # Calculate statistics based on window performance only
    # (No exhaustive analysis to avoid O(N²) oracle calls)
    success_rate = connections_found / max(1, window_tests)
    
    stats = {
        'window_tests': window_tests,
        'connections_found': connections_found,
        'success_rate': success_rate,
        'average_connections_per_vertex': connections_found / N,
        'window_density': connections_found / max(1, window_tests)
    }
    
    if verbose:
        print(f"   Window tests: {window_tests:,}")
        print(f"   Window edges found: {len(extra_edges):,}")
        print(f"   Window success rate: {success_rate:.1%}")
    
    return extra_edges, stats


def max_true_gap_binary(order, short_neighbors):
    """Calculate maximum gap between consecutive vertices in binary graph"""
    max_gap = 0
    N = len(order)
    
    for i in range(N):
        u = order[i]
        
        # Find the farthest neighbor of u in the ordering
        max_neighbor_pos = i
        for neighbor in short_neighbors[u]:
            try:
                neighbor_pos = order.index(neighbor)
                max_neighbor_pos = max(max_neighbor_pos, neighbor_pos)
            except ValueError:
                continue
        
        gap = max_neighbor_pos - i
        max_gap = max(max_gap, gap)
    
    return max_gap


def spectral_order_bio(all_sections, G, rough_order, verbose=False):
    """Phase E: Final spectral ordering using Fiedler vector"""
    
    N = len(all_sections)
    section_to_idx = {section: i for i, section in enumerate(all_sections)}
    
    # Build adjacency matrix
    W = np.zeros((N, N))
    for u_idx in range(N):
        for v_idx, distance in G[u_idx]:
            # Convert distance back to similarity weight
            similarity = max(0, 1 - distance)
            W[u_idx, v_idx] = similarity
    
    # Ensure symmetry
    W = (W + W.T) / 2
    
    # Compute Laplacian
    D = np.diag(np.sum(W, axis=1))
    L = D - W
    
    # Add small regularization for numerical stability
    L += 1e-10 * np.eye(N)
    
    try:
        from scipy.sparse.linalg import eigsh
        eigenvals, eigenvecs = eigsh(L, k=2, which='SM')
        fiedler_vector = eigenvecs[:, 1]
    except:
        eigenvals, eigenvecs = np.linalg.eigh(L)
        fiedler_vector = eigenvecs[:, 1]
    
    # Order by Fiedler vector
    order_indices = np.argsort(fiedler_vector)
    spectral_order = [all_sections[i] for i in order_indices]
    
    # Check orientation
    rough_cost = compute_arrangement_cost_bio(rough_order, W, all_sections)
    spectral_cost = compute_arrangement_cost_bio(spectral_order, W, all_sections)
    spectral_cost_rev = compute_arrangement_cost_bio(spectral_order[::-1], W, all_sections)
    
    if spectral_cost_rev < spectral_cost:
        spectral_order = spectral_order[::-1]
        spectral_cost = spectral_cost_rev
    
    if verbose:
        print(f"   Rough order cost: {rough_cost:.3f}")
        print(f"   Spectral order cost: {spectral_cost:.3f}")
        print(f"   Improvement: {((rough_cost - spectral_cost) / rough_cost * 100):.1f}%")
    
    return spectral_order


def compute_arrangement_cost_bio(order, W, all_sections):
    """Compute arrangement cost for biological section ordering"""
    
    section_to_idx = {section: i for i, section in enumerate(all_sections)}
    cost = 0.0
    
    for i, sec1 in enumerate(order):
        for j, sec2 in enumerate(order[i+1:], i+1):
            idx1 = section_to_idx[sec1]
            idx2 = section_to_idx[sec2]
            weight = W[idx1, idx2]
            distance = abs(i - j)
            cost += weight * distance
    
    return cost


def evaluate_order_quality(order, get_edge_data):
    """Evaluate the quality of a linear ordering"""
    
    if len(order) < 2:
        return 0.0
    
    # Check sequential connectivity
    sequential_scores = []
    for i in range(len(order) - 1):
        edge_data = get_edge_data(order[i], order[i + 1])
        if edge_data and edge_data.get('ssim_score', 0) > 0:
            sequential_scores.append(edge_data['ssim_score'])
    
    if not sequential_scores:
        return 0.0
    
    return np.mean(sequential_scores)

def texture_rich_color_invariant_preprocess(image):
    """Apply texture-rich color-invariant preprocessing"""
    if len(image.shape) == 3:
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        gray = lab[:, :, 0]
    else:
        gray = image.copy()
    
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    blurred = cv2.GaussianBlur(enhanced, (3, 3), 0)
    return blurred

def get_sift_features(image, section_name, use_cache=False):
    """
    Get SIFT features for an image, with optional caching
    
    Args:
        image: Input image
        section_name: Name/identifier for the section
        use_cache: Whether to use cached features
    
    Returns:
        tuple: (keypoints, descriptors)
    """
    if use_cache and section_name in SIFT_CACHE:
        return SIFT_CACHE[section_name]
    
    # Apply preprocessing
    processed = texture_rich_color_invariant_preprocess(image)
    
    # Initialize SIFT
    sift = cv2.SIFT_create(nfeatures=2500, contrastThreshold=0.02, edgeThreshold=20, sigma=1.6)
    
    # Detect features
    kp, des = sift.detectAndCompute(processed, None)
    
    # Cache if requested
    if use_cache:
        SIFT_CACHE[section_name] = (kp, des)
    
    return kp, des

def precompute_all_sift_features(images, cache_file=None):
    """
    Pre-compute SIFT features for all images and cache them
    
    Args:
        images: Dictionary of {section_num: image}
        cache_file: Optional file to save/load cache from
    
    Returns:
        dict: SIFT cache with features for all sections
    """
    global SIFT_CACHE
    
    # Try to load from cache file if it exists
    if cache_file and os.path.exists(cache_file):
        print(f"Loading SIFT cache from {cache_file}...")
        try:
            with open(cache_file, 'rb') as f:
                serialized_cache = pickle.load(f)
            
            # Convert back to OpenCV keypoints
            SIFT_CACHE = {}
            for section_name, (kp_data, descriptors) in serialized_cache.items():
                keypoints, desc = serializable_to_keypoints(kp_data, descriptors)
                SIFT_CACHE[section_name] = (keypoints, desc)
            
            print(f"  Loaded cached features for {len(SIFT_CACHE)} sections")
            
            # Check if we have all sections cached
            missing_sections = []
            for section_num in images.keys():
                section_name = f"section_{section_num}"
                if section_name not in SIFT_CACHE:
                    missing_sections.append(section_num)
            
            if missing_sections:
                print(f"  Missing cached features for {len(missing_sections)} sections: {missing_sections}")
            else:
                print("  All sections found in cache!")
                return SIFT_CACHE
                
        except Exception as e:
            print(f"  Error loading cache: {e}")
            print("  Will compute features from scratch")
            SIFT_CACHE = {}
    
    # Compute features for sections not in cache
    sections_to_compute = []
    for section_num in images.keys():
        section_name = f"section_{section_num}"
        if section_name not in SIFT_CACHE:
            sections_to_compute.append(section_num)
    
    if sections_to_compute:
        print(f"Computing SIFT features for {len(sections_to_compute)} sections...")
        print("This is a one-time computation that will be cached for future runs.")
        
        start_time = time.time()
        
        for i, section_num in enumerate(sections_to_compute):
            section_name = f"section_{section_num}"
            img = images[section_num]
            
            print(f"  [{i+1}/{len(sections_to_compute)}] Computing features for section {section_num}...")
            
            # Compute features
            kp, des = get_sift_features(img, section_name, use_cache=True)
            
            feature_count = len(kp) if kp else 0
            print(f"    Features detected: {feature_count}")
        
        computation_time = time.time() - start_time
        print(f"SIFT feature computation completed in {computation_time:.1f} seconds")
        print(f"Average time per section: {computation_time/len(sections_to_compute):.2f} seconds")
        
        # Save cache to file if requested
        if cache_file:
            print(f"Saving SIFT cache to {cache_file}...")
            try:
                # Only create directory if cache_file has a directory component
                cache_dir = os.path.dirname(cache_file)
                if cache_dir:  # Only create directory if there is one
                    os.makedirs(cache_dir, exist_ok=True)
                
                # Convert keypoints to serializable format before saving
                serializable_cache = {}
                for section_name, (keypoints, descriptors) in SIFT_CACHE.items():
                    kp_data, desc = keypoints_to_serializable(keypoints, descriptors)
                    serializable_cache[section_name] = (kp_data, desc)
                
                with open(cache_file, 'wb') as f:
                    pickle.dump(serializable_cache, f)
                print(f"  Cache saved successfully")
            except Exception as e:
                print(f"  Warning: Could not save cache: {e}")
    
    return SIFT_CACHE

def find_rigid_sift_alignment(img1, img2, section1_name, section2_name, use_cache=False, fast_flann=False, verbose=False, inlier_threshold=None):
    """Find SIFT-based RIGID alignment between two images"""
    # Apply default inlier threshold if not provided
    if inlier_threshold is None:
        inlier_threshold = DEFAULT_INLIER_THRESHOLD
        
    if verbose:
        print(f"    Processing {section1_name} vs {section2_name}...")
    
    # Get SIFT features (cached or fresh)
    start_time = time.time()
    kp1, des1 = get_sift_features(img1, section1_name, use_cache=use_cache)
    kp2, des2 = get_sift_features(img2, section2_name, use_cache=use_cache)
    detection_time = time.time() - start_time
    
    # Only show feature details for failed alignments or when explicitly verbose
    if verbose:
        if use_cache:
            print(f"      Features retrieved from cache: {len(kp1)} vs {len(kp2)} ({detection_time:.3f}s)")
        else:
            print(f"      Features computed: {len(kp1)} vs {len(kp2)} ({detection_time:.2f}s)")
    
    if des1 is None or des2 is None or len(des1) < 10 or len(des2) < 10:
        return None, None, None, {
            'features1': len(kp1) if kp1 else 0,
            'features2': len(kp2) if kp2 else 0,
            'matches': 0,
            'inliers': 0,
            'inlier_ratio': 0.0,
            'detection_time': detection_time,
            'matching_time': 0,
            'success': False,
            'error': 'Insufficient features'
        }
    
    # FLANN matching with optional optimization
    FLANN_INDEX_KDTREE = 1
    
    if fast_flann:
        # Fast FLANN parameters: ~2-3x faster with minimal accuracy loss
        index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=4)  # Reduced from 8
        search_params = dict(checks=50)  # Reduced from 100
        flann_mode = "fast"
    else:
        # Standard FLANN parameters: more thorough but slower
        index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=8)
        search_params = dict(checks=100)
        flann_mode = "standard"
    
    if verbose:
        print(f"      FLANN mode: {flann_mode} (trees={index_params['trees']}, checks={search_params['checks']})")
    
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    
    start_time = time.time()
    matches = flann.knnMatch(des1, des2, k=2)
    matching_time = time.time() - start_time
    
    if verbose:
        print(f"      FLANN matching: {matching_time:.3f}s")
    
    # Apply ratio test
    good_matches = []
    for match_pair in matches:
        if len(match_pair) == 2:
            m, n = match_pair
            if m.distance < 0.85 * n.distance:
                good_matches.append(m)
    
    if len(good_matches) < 10:
        return None, good_matches, None, {
            'features1': len(kp1),
            'features2': len(kp2),
            'matches': len(good_matches),
            'inliers': 0,
            'inlier_ratio': 0.0,
            'detection_time': detection_time,
            'matching_time': matching_time,
            'success': False,
            'error': 'Insufficient matches'
        }
    
    # Extract points for RIGID transformation estimation
    src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches])
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches])
    
    # Use estimateAffinePartial2D for RIGID transformation
    transformation_matrix, inlier_mask = cv2.estimateAffinePartial2D(
        src_pts, dst_pts, method=cv2.RANSAC, ransacReprojThreshold=25.0,
        maxIters=5000, confidence=0.99)
    
    if transformation_matrix is None:
        return None, good_matches, None, {
            'features1': len(kp1),
            'features2': len(kp2),
            'matches': len(good_matches),
            'inliers': 0,
            'inlier_ratio': 0.0,
            'detection_time': detection_time,
            'matching_time': matching_time,
            'success': False,
            'error': 'RANSAC failed'
        }
    
    # Count inliers
    inliers = inlier_mask.ravel().tolist()
    num_inliers = sum(inliers)
    inlier_ratio = num_inliers / len(good_matches)
    
    # Extract transformation parameters
    angle = np.arctan2(transformation_matrix[1, 0], transformation_matrix[0, 0]) * 180 / np.pi
    translation_x = transformation_matrix[0, 2]
    translation_y = transformation_matrix[1, 2]
    scale_x = np.sqrt(transformation_matrix[0, 0]**2 + transformation_matrix[1, 0]**2)
    scale_y = np.sqrt(transformation_matrix[0, 1]**2 + transformation_matrix[1, 1]**2)
    
    # Create inlier matches list
    inlier_matches = [good_matches[i] for i, is_inlier in enumerate(inliers) if is_inlier]
    
    stats = {
        'features1': len(kp1),
        'features2': len(kp2),
        'matches': len(good_matches),
        'inliers': num_inliers,
        'inlier_ratio': inlier_ratio,
        'rotation_deg': angle,
        'translation_x': translation_x,
        'translation_y': translation_y,
        'scale_x': scale_x,
        'scale_y': scale_y,
        'detection_time': detection_time,
        'matching_time': matching_time,
        'success': inlier_ratio > inlier_threshold,  # Use configurable threshold
        'error': None
    }
    
    # Report successful alignments when not explicitly verbose
    if not verbose and stats['success']:
        print(f"    ✅ {section1_name} ↔ {section2_name}: {num_inliers}/{len(good_matches)} inliers ({inlier_ratio:.1%})")
    
    return transformation_matrix, good_matches, inlier_matches, stats

def apply_rigid_transformation_and_find_overlap(img1, img2, transformation_matrix):
    """Apply RIGID transformation and find overlapping region"""
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]
    
    # Apply affine transformation
    aligned_img1 = cv2.warpAffine(img1, transformation_matrix, (w2, h2))
    
    # Create masks to find valid regions
    mask1 = cv2.warpAffine(np.ones((h1, w1), dtype=np.uint8) * 255, transformation_matrix, (w2, h2))
    mask2 = np.ones((h2, w2), dtype=np.uint8) * 255
    
    # Find overlap
    overlap_mask = cv2.bitwise_and(mask1, mask2)
    
    # Find bounding box of overlap
    contours, _ = cv2.findContours(overlap_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None, None, None, None
    
    # Get largest contour
    largest_contour = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(largest_contour)
    
    # Extract overlapping regions
    overlap_region1 = aligned_img1[y:y+h, x:x+w]
    overlap_region2 = img2[y:y+h, x:x+w]
    
    return aligned_img1, overlap_region1, overlap_region2, (x, y, w, h)

def calculate_ssim_score(region1, region2):
    """Calculate SSIM between two image regions"""
    if region1.shape != region2.shape:
        return 0.0
    
    # Convert to grayscale if needed
    if len(region1.shape) == 3:
        gray1 = cv2.cvtColor(region1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(region2, cv2.COLOR_BGR2GRAY)
    else:
        gray1, gray2 = region1, region2
    
    # Check if regions are large enough for SSIM calculation
    min_dim = min(gray1.shape[0], gray1.shape[1])
    if min_dim < 7:
        # For very small regions, use simple correlation coefficient
        try:
            corr_matrix = np.corrcoef(gray1.flatten(), gray2.flatten())
            if np.isnan(corr_matrix[0, 1]):
                return 0.0
            return float(corr_matrix[0, 1])
        except:
            return 0.0
    
    # Calculate SSIM with appropriate window size
    win_size = min(7, min_dim)
    if win_size % 2 == 0:  # Ensure odd window size
        win_size -= 1
    
    try:
        score = ssim(gray1, gray2, win_size=win_size, data_range=255)
        return float(score)
    except Exception as e:
        print(f"        SSIM calculation failed: {e}")
        return 0.0

def get_available_sections():
    """Get list of all available section numbers"""
    sections = []
    pattern = "w7_png_4k/section_*_r01_c01.png"
    
    for filepath in glob.glob(pattern):
        basename = os.path.basename(filepath)
        # Format: section_X_r01_c01.png
        parts = basename.split('_')
        if len(parts) >= 2:
            try:
                section_num = int(parts[1])
                sections.append(section_num)
            except ValueError:
                continue
    return sorted(sections)

def comprehensive_pairwise_analysis(cache_sift=False, precompute_sift=False, fast_flann=False, sublinear=False, build_and_order=False, verbose=False, visualize=False, ssim_threshold=None, inlier_threshold=None, use_spectral=True, two_phase=False, edge_mask=None, auto_tune_threshold=False, proximity_window=10, window_k=None, samples_per_phase=None, condensation_rounds=None, save_tree_edges=True):
    """
    Analyze ALL available sections with comprehensive pairwise analysis
    
    Args:
        cache_sift: Enable SIFT feature caching for performance
        fast_flann: Use fast FLANN parameters for speed
        sublinear: Use sublinear Borůvka algorithm  
        build_and_order: Use BUILD-AND-ORDER algorithm
        verbose: Enable verbose output
        visualize: Create visualization plots
        ssim_threshold: SSIM threshold for strong connections (None = use DEFAULT_SSIM_THRESHOLD)
        inlier_threshold: Inlier ratio threshold for strong connections (None = use DEFAULT_INLIER_THRESHOLD)
        use_spectral: Enable spectral ordering in BUILD-AND-ORDER
        two_phase: Use two-phase analysis (Phase 1: relaxed → Phase 2: strict + proximity mask)
        edge_mask: Set of permissible (sec1, sec2) pairs (None = all permissible)
        auto_tune_threshold: Auto-tune SSIM threshold based on node degree analysis
        proximity_window: Window size for proximity mask in two-phase mode (default: 10)
        window_k: Override K (window half-width) for K-window densification
        samples_per_phase: s (random samples per vertex per Borůvka phase)
        condensation_rounds: Number of condensation rounds in Phase B
        save_tree_edges: Save spanning tree edges to CSV
    """
    
    # Apply default values if not provided
    if ssim_threshold is None:
        ssim_threshold = DEFAULT_SSIM_THRESHOLD
    if inlier_threshold is None:
        inlier_threshold = DEFAULT_INLIER_THRESHOLD
    
    print("=== COMPREHENSIVE PAIRWISE ANALYSIS (ALL SECTIONS) ===")
    if build_and_order:
        print(" BUILD-AND-ORDER MODE: Using theoretical algorithm")
        print(" THEORETICAL ALGORITHM: 5-phase sub-quadratic ordering")
        if use_spectral:
            print("   Phase E: Spectral ordering ENABLED")
        else:
            print("   Phase E: Spectral ordering DISABLED (use BFS result)")
        if verbose:
            print("   Verbose mode enabled - detailed algorithmic walkthrough will follow")
    elif sublinear:
        print(" SUBLINEAR MODE: Using Randomized Borůvka algorithm")
        print("⚠️  This approach is UNDER CONSTRUCTION")
        if verbose:
            print("   Verbose mode enabled - detailed algorithmic walkthrough will follow")
    else:
        print("🔬 STANDARD MODE: Complete pairwise analysis")
        
    if cache_sift:
        if precompute_sift:
            print(" SIFT caching + pre-computation enabled - all features computed upfront with progress tracking")
        else:
            print(" SIFT caching enabled - features computed lazily as needed")
    if fast_flann:
        print("⚡ Fast FLANN enabled - 2-3x faster matching with minimal accuracy loss")
    if two_phase:
        print(f" Two-phase analysis enabled - relaxed → strict with proximity mask (window: {proximity_window})")
    if auto_tune_threshold:
        print(" Auto-tune threshold enabled - will analyze node degrees to find optimal threshold")
    if edge_mask is not None:
        print(f" Edge mask provided - {len(edge_mask):,} permissible pairs")
    if visualize:
        print(" Visualization enabled - spanning tree and results will be visualized")
    
    print(f" Strong connection criteria (similarity thresholds):")
    print(f"   • SSIM > {ssim_threshold}")
    print(f"   • Rotation: -90° to +90°")
    print(f"   • Scale: 0.8 to 1.25 (both X and Y)")
    print(f"   • Inlier ratio > {inlier_threshold}")
    
    # Get ALL available sections
    all_sections = get_available_sections()
    print(f"Total available sections: {len(all_sections)}")
    print(f"Section range: {min(all_sections)} to {max(all_sections)}")
    print(f"Sections: {all_sections}")
    
    # Calculate number of pairs for comparison
    num_complete_pairs = len(list(combinations(all_sections, 2)))
    print(f"Complete pairwise analysis would process: {num_complete_pairs:,} pairs")
    
    # Load all images (this might use significant memory)
    images = {}
    print("\nLoading all images...")
    for section in all_sections:
        img_path = f"w7_png_4k/section_{section}_r01_c01.png"
        img = cv2.imread(img_path)
        if img is not None:
            images[section] = img
        else:
            print(f"Warning: Could not load image for section {section}")
    
    print(f"Successfully loaded {len(images)} images")
    
    # Create output directory early for SIFT caching
    output_dir = f"comprehensive_all_{len(all_sections)}_sections"
    if build_and_order:
        output_dir += "_build_order"
    elif sublinear:
        output_dir += "_sublinear"
    else:
        output_dir += "_complete"
    if cache_sift:
        output_dir += "_cached"
    if fast_flann:
        output_dir += "_fastflann"
    if use_spectral and build_and_order:
        output_dir += "_spectral"
    if edge_mask is not None:
        output_dir += f"_masked_{len(edge_mask)}pairs"
    if auto_tune_threshold:
        output_dir += "_autotuned"
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Optional upfront SIFT computation with proper cache file path
    if cache_sift and precompute_sift:
        print("\n Pre-computing SIFT features for all sections (one-time step)…")
        cache_file = os.path.join(output_dir, "sift_features_cache.pkl")
        precompute_all_sift_features(images, cache_file=cache_file)
    elif cache_sift:
        print("\n SIFT caching enabled - features will be computed lazily as needed")
    
    # Handle two-phase mode
    if two_phase:
        if verbose:
            print(f"\n TWO-PHASE ANALYSIS MODE")
            print(f"   Phase 1: Relaxed threshold → linear order")
            print(f"   Phase 2: Strict threshold + proximity mask")
        
        # Phase 1: Run with relaxed threshold (auto-tuned for ~10 connections per node)
        if verbose:
            print(f"\n🔄 PHASE 1: Relaxed Analysis")
        
        phase1_result = comprehensive_pairwise_analysis(
            cache_sift=cache_sift,
            fast_flann=fast_flann,
            sublinear=sublinear,
            build_and_order=build_and_order,
            verbose=verbose,
            visualize=False,  # Skip visualization in Phase 1
            ssim_threshold=ssim_threshold,
            use_spectral=use_spectral,
            two_phase=False,  # Prevent recursive two-phase
            edge_mask=edge_mask,  # Use provided edge mask if any
            auto_tune_threshold=True,  # Auto-tune for Phase 1
            proximity_window=proximity_window
        )
        
        # Extract linear order from Phase 1 results
        # For BUILD-AND-ORDER, we should get a linear order
        # For other methods, we'll use the pipeline-valid connections to create an order
        if build_and_order:
            # TODO: Extract linear order from BUILD-AND-ORDER results
            phase1_pairs = phase1_result[1] if isinstance(phase1_result, tuple) else []
            # For now, create a simple order from the connected pairs
            linear_order = list(all_sections)  # Fallback
        else:
            # Use pipeline-valid pairs to create order
            phase1_pairs = phase1_result[1] if isinstance(phase1_result, tuple) else []
            linear_order = list(all_sections)  # Fallback
        
        if verbose:
            print(f"   Phase 1 completed, found {len(phase1_pairs) if phase1_pairs else 0} connections")
        
        # Create proximity mask from Phase 1 linear order
        proximity_mask = create_proximity_mask(linear_order, proximity_window)
        
        if verbose:
            print(f"\n PHASE 2: Strict Analysis with Proximity Mask")
            print(f"   Proximity mask: {len(proximity_mask):,} permissible pairs")
        
        # Phase 2: Run with strict threshold + proximity mask
        phase2_result = comprehensive_pairwise_analysis(
            cache_sift=cache_sift,
            fast_flann=fast_flann,
            sublinear=sublinear,
            build_and_order=build_and_order,
            verbose=verbose,
            visualize=visualize,
            ssim_threshold=ssim_threshold,  # Use original strict threshold
            use_spectral=use_spectral,
            two_phase=False,  # Prevent recursive two-phase
            edge_mask=proximity_mask,  # Use proximity mask
            auto_tune_threshold=False,  # Don't auto-tune in Phase 2
            proximity_window=proximity_window
        )
        
        return phase2_result
    
    # Initialize data manager for efficient pair tracking
    data_manager = PairwiseDataManager(all_sections)
    
    # Results list for backward compatibility
    results = []
    
    # Auto-tune threshold if requested
    if auto_tune_threshold:
        ssim_threshold = auto_tune_threshold(
            images, all_sections, data_manager, 
            cache_sift=cache_sift, fast_flann=fast_flann,
            target_degree=10, sample_nodes=3, inlier_threshold=inlier_threshold, verbose=verbose
        )
    
    # Define edge oracle for sublinear approach with proper validation criteria
    def edge_oracle_base(sec1, sec2):
        """
        Base oracle function that determines if an edge should exist between two sections
        Returns True if sections can be successfully aligned with pipeline-quality criteria
        Uses data_manager to avoid duplicate computations
        """
        data_manager.record_oracle_call()
        
        # Check edge mask first (if provided)
        if edge_mask is not None:
            if (sec1, sec2) not in edge_mask:
                if verbose:
                    print(f"       Edge mask filtered: {sec1} ↔ {sec2} (not in permissible set)")
                return False
        
        # Check if already computed
        if data_manager.is_computed(sec1, sec2):
            if verbose:
                pipeline_valid = data_manager.is_pipeline_valid(sec1, sec2)
                ssim_score = data_manager.get_ssim(sec1, sec2)
                if pipeline_valid:
                    print(f"      ✅ Strong connection (cached): SSIM={ssim_score:.3f}")
                else:
                    print(f"      ❌ Weak connection (cached): SSIM={ssim_score:.3f}")
            return data_manager.is_pipeline_valid(sec1, sec2)
        
        img1 = images[sec1]
        img2 = images[sec2]
        
        # Try to find alignment
        transformation_matrix, _, _, stats = find_rigid_sift_alignment(
            img1, img2, f"section_{sec1}", f"section_{sec2}", 
            use_cache=cache_sift, fast_flann=fast_flann, verbose=False, inlier_threshold=inlier_threshold
        )
        
        # Store result for later analysis
        result = {
            'section1': sec1,
            'section2': sec2,
            'features1': stats['features1'],
            'features2': stats['features2'],
            'initial_matches': stats['matches'],
            'ransac_inliers': stats['inliers'],
            'inlier_ratio': stats['inlier_ratio'],
            'rotation_degrees': stats.get('rotation_deg', 0),
            'translation_x': stats.get('translation_x', 0),
            'translation_y': stats.get('translation_y', 0),
            'scale_x': stats.get('scale_x', 1),
            'scale_y': stats.get('scale_y', 1),
            'detection_time_sec': stats['detection_time'],
            'matching_time_sec': stats['matching_time'],
            'alignment_success': stats['success'],
            'error_message': stats.get('error', ''),
            'ssim_score': 0.0,
            'overlap_area_pixels': 0,
            'overlap_bbox': '',
            'pipeline_valid': False  # Will be set based on pipeline criteria
        }
        
        # Calculate SSIM if basic alignment was successful
        ssim_score = 0.0
        if transformation_matrix is not None and stats['success']:
            # Pre-validate transformation parameters before attempting SSIM calculation
            rotation = abs(stats.get('rotation_deg', 0))
            scale_x = stats.get('scale_x', 1)
            scale_y = stats.get('scale_y', 1)
            
            # Check if transformation parameters are biologically plausible
            transformation_valid = (
                rotation <= ROTATION_LIMIT_DEGREES and              # Rotation within ±90°
                SCALE_MIN <= scale_x <= SCALE_MAX and               # Scale X within range
                SCALE_MIN <= scale_y <= SCALE_MAX and               # Scale Y within range
                scale_x > 0.1 and scale_y > 0.1                    # Avoid degenerate scales
            )
            
            if transformation_valid:
                aligned_img1, overlap_region1, overlap_region2, bbox = apply_rigid_transformation_and_find_overlap(
                    img1, img2, transformation_matrix)
                
                if overlap_region1 is not None and overlap_region2 is not None:
                    ssim_score = calculate_ssim_score(overlap_region1, overlap_region2)
                    result['ssim_score'] = ssim_score
                    result['overlap_area_pixels'] = bbox[2] * bbox[3]
                    result['overlap_bbox'] = f"{bbox[0]},{bbox[1]},{bbox[2]},{bbox[3]}"
            else:
                # Mark as failed due to invalid transformation
                stats['success'] = False
                if verbose:
                    print(f"      ❌ Invalid transformation: Rot={rotation:.1f}°, Scale=({scale_x:.3f},{scale_y:.3f})")
        
        # Apply pipeline validation criteria for "strong connection" 
        pipeline_valid = False
        if stats['success'] and ssim_score > 0:
            rotation = abs(stats.get('rotation_deg', 0))
            scale_x = stats.get('scale_x', 1)
            scale_y = stats.get('scale_y', 1)
            inlier_ratio = stats.get('inlier_ratio', 0)
            
            # Pipeline criteria (matching create_alignment_graph.py and multi_branch_alignment.py)
            pipeline_valid = (
                ssim_score > ssim_threshold and                     # SSIM threshold
                rotation <= ROTATION_LIMIT_DEGREES and             # Rotation within ±90°
                SCALE_MIN <= scale_x <= SCALE_MAX and               # Scale X within range
                SCALE_MIN <= scale_y <= SCALE_MAX and               # Scale Y within range
                inlier_ratio > inlier_threshold                     # Inlier ratio threshold
            )
            
            if verbose and pipeline_valid:
                print(f"      ✅ Strong connection: SSIM={ssim_score:.3f}, Rot={rotation:.1f}°, Scale=({scale_x:.2f},{scale_y:.2f}), Inliers={inlier_ratio:.1%}")
            elif verbose:
                reasons = []
                if ssim_score <= ssim_threshold:
                    reasons.append(f"SSIM={ssim_score:.3f}<={ssim_threshold}")
                if rotation > ROTATION_LIMIT_DEGREES:
                    reasons.append(f"Rot={rotation:.1f}°>{ROTATION_LIMIT_DEGREES}°")
                if not (SCALE_MIN <= scale_x <= SCALE_MAX):
                    reasons.append(f"ScaleX={scale_x:.2f}")
                if not (SCALE_MIN <= scale_y <= SCALE_MAX):
                    reasons.append(f"ScaleY={scale_y:.2f}")
                if inlier_ratio <= inlier_threshold:
                    reasons.append(f"Inliers={inlier_ratio:.1%}<={inlier_threshold}")
                print(f"      ❌ Weak connection: {', '.join(reasons)}")
        
        result['pipeline_valid'] = pipeline_valid
        
        # Store in data manager (this will handle deduplication)
        data_manager.store_result(sec1, sec2, result, ssim_score, pipeline_valid)
        
        # Also add to results list for backward compatibility
        results.append(result)
        
        # Return pipeline validation result (not just basic alignment success)
        return pipeline_valid
    
    # Use the edge oracle with built-in edge mask support
    edge_oracle = edge_oracle_base
    
    # Choose analysis approach
    if build_and_order:
        print(f"\n Running BUILD-AND-ORDER analysis...")
        start_time = time.time()
        
        # Create edge data retrieval function using data_manager
        def get_edge_data(sec1, sec2):
            """Get edge data for a specific pair of sections"""
            return data_manager.get_detailed_result(sec1, sec2)
        
        # Run BUILD-AND-ORDER analysis
        final_order, build_order_stats = build_and_order_sublinear(
            all_sections, edge_oracle, get_edge_data, verbose=verbose, use_spectral=use_spectral, two_phase=two_phase, ssim_threshold=ssim_threshold, inlier_threshold=inlier_threshold, images=images, data_manager=data_manager, cache_sift=cache_sift, fast_flann=fast_flann, K_override=window_k, samples_per_phase=samples_per_phase, condensation_rounds=condensation_rounds
        )
        
        elapsed_time = time.time() - start_time
        
        # Output directory already created earlier for SIFT caching
        
        # Get accurate statistics from data_manager
        dm_stats = data_manager.get_statistics()
        
        print(f"\n  BUILD-AND-ORDER ANALYSIS COMPLETE!")
        print(f"   Algorithm: {build_order_stats['algorithm']}")
        print(f"   K (window size): {build_order_stats['parameters']['K']}")
        print(f"   t (condensation rounds): {build_order_stats['parameters']['t']}")
        print(f"   Spectral ordering: {'ENABLED' if build_order_stats['parameters']['use_spectral'] else 'DISABLED'}")
        print(f"   Tree edges: {build_order_stats['tree_edges']}")
        print(f"   Window edges: {build_order_stats['window_edges']}")
        print(f"   Rough order quality: {build_order_stats['rough_order_quality']:.3f}")
        print(f"   Final order quality: {build_order_stats['final_order_quality']:.3f}")
        print(f"   Total computation time: {elapsed_time:.1f} seconds")
        print(f"\n  DETAILED PHASE STATISTICS:")
        print(f"   Phase A (Borůvka): {build_order_stats['phase_a']['edge_tests']:,} tests, {build_order_stats['phase_a']['connections_found']:,} connections ({build_order_stats['phase_a']['success_rate']:.1%} success)")
        print(f"   Phase B (Condensation): {build_order_stats['phase_b']['total_tests']:,} tests, {build_order_stats['phase_b']['connections_found']:,} connections ({build_order_stats['phase_b']['success_rate']:.1%} success)")
        print(f"   Phase D (K-window): {build_order_stats['phase_d']['window_tests']:,} tests, {build_order_stats['phase_d']['connections_found']:,} connections")
        print(f"      Window success rate: {build_order_stats['phase_d']['success_rate']:.1%}")
        print(f"      Average connections per vertex: {build_order_stats['phase_d']['average_connections_per_vertex']:.1f}")
        print(f"   Phase F (Final verification): {build_order_stats['phase_f']['total_tests']:,} tests, {build_order_stats['phase_f']['connections_found']:,} connections")
        print(f"      NEW computations: {build_order_stats['phase_f']['newly_computed_tests']:,} tests, {build_order_stats['phase_f']['newly_computed_connections']:,} connections")
        print(f"      Overall success rate: {build_order_stats['phase_f']['success_rate']:.1%}")
        print(f"      NEW computations success rate: {build_order_stats['phase_f']['newly_computed_success_rate']:.1%}")
        if build_order_stats['phase_f']['newly_computed_connections'] > 0:
            print(f"       Found {build_order_stats['phase_f']['newly_computed_connections']} NEW additional local connections")
            print(f"      Average gap: {build_order_stats['phase_f']['average_gap']:.1f}, Max gap: {build_order_stats['phase_f']['max_gap']}")
        else:
            print(f"      ✅ Perfect K-densification: No new local connections discovered!")
        if build_order_stats['phase_d']['success_rate'] < 0.15:
            print(f"       Consider increasing K from {build_order_stats['parameters']['K']} to find more connections")
        elif build_order_stats['phase_d']['success_rate'] > 0.8:
            print(f"       Consider decreasing K from {build_order_stats['parameters']['K']} to reduce computation")
        print(f"\n EFFICIENCY STATISTICS:")
        print(f"   Unique pairs computed: {dm_stats['unique_pairs_computed']:,}")
        print(f"   Total oracle calls: {dm_stats['total_oracle_calls']:,}")
        print(f"   Cache hits: {dm_stats['cache_hits']:,}")
        print(f"   True reduction factor: {1/dm_stats['computation_ratio']:.1f}x")
        print(f"   Redundancy ratio: {dm_stats['redundancy_ratio']:.1f}x")
        print(f"   Cache hit rate: {dm_stats['cache_hit_rate']:.1%}")
        
        # Print edge mask statistics if provided
        if edge_mask is not None:
            print(f"\n EDGE MASK STATISTICS:")
            print(f"   Permissible pairs: {len(edge_mask):,}")
            print(f"   Total possible pairs: {len(all_sections) * (len(all_sections) - 1) // 2:,}")
            print(f"   Mask density: {len(edge_mask) / (len(all_sections) * (len(all_sections) - 1) // 2):.1%}")
        
        # Save results
        # 1. Final order
        final_order_df = pd.DataFrame({'section': final_order, 'order': range(len(final_order))})
        final_order_file = f"{output_dir}/final_order.csv"
        final_order_df.to_csv(final_order_file, index=False)
        
        # 2. Save spanning tree edges if available
        if 'tree_edges_list' in build_order_stats and save_tree_edges:
            tree_edges_df = pd.DataFrame(build_order_stats['tree_edges_list'], 
                                       columns=['section_1', 'section_2', 'ssim_score'])
            tree_edges_file = f"{output_dir}/spanning_tree_edges.csv"
            tree_edges_df.to_csv(tree_edges_file, index=False)
            print(f"   Tree edges: {tree_edges_file}")
        
        # 3. Export comprehensive pairwise connections CSV (all successful alignments)
        print(f"   Exporting comprehensive pairwise connections...")
        comprehensive_df = data_manager.export_to_dataframe()
        comprehensive_file = f"{output_dir}/comprehensive_pairwise_results.csv"
        comprehensive_df.to_csv(comprehensive_file, index=False)
        print(f"   Comprehensive results: {comprehensive_file}")
        
        # 4. Tree structure analysis
        if 'tree_diameter' in build_order_stats:
            diameter_info = build_order_stats['tree_diameter']
            print(f"\n TREE STRUCTURE ANALYSIS:")
            print(f"   Tree diameter: {diameter_info['diameter']} hops")
            print(f"   Diameter endpoints: {diameter_info['endpoints'][0]} ↔ {diameter_info['endpoints'][1]}")
            print(f"   Center nodes: {diameter_info['center_nodes']}")
            print(f"   Connected components: {build_order_stats.get('connected_components', 1)}")
        
        # 5. Algorithm statistics  
        stats_file = f"{output_dir}/build_order_statistics.txt"
        with open(stats_file, 'w') as f:
            f.write("=== BUILD-AND-ORDER ALGORITHM STATISTICS ===\n\n")
            f.write(f"Algorithm: {build_order_stats['algorithm']}\n\n")
            f.write(f"Parameters:\n")
            f.write(f"  K (window half-width): {build_order_stats['parameters']['K']}\n")
            f.write(f"  s (samples per phase): {build_order_stats['parameters']['s']}\n")
            f.write(f"  t (condensation rounds): {build_order_stats['parameters']['t']}\n")
            f.write(f"  use_spectral: {build_order_stats['parameters']['use_spectral']}\n\n")
            f.write(f"Results:\n")
            f.write(f"  Tree edges: {build_order_stats['tree_edges']}\n")
            f.write(f"  Window edges: {build_order_stats['window_edges']}\n")
            f.write(f"  Rough order quality: {build_order_stats['rough_order_quality']:.3f}\n")
            f.write(f"  Final order quality: {build_order_stats['final_order_quality']:.3f}\n")
            f.write(f"  Total computation time: {elapsed_time:.1f} seconds\n\n")
            f.write(f"Detailed Phase Statistics:\n")
            f.write(f"  Phase A (Borůvka Tree):\n")
            f.write(f"    Edge tests: {build_order_stats['phase_a']['edge_tests']:,}\n")
            f.write(f"    Connections found: {build_order_stats['phase_a']['connections_found']:,}\n")
            f.write(f"    Success rate: {build_order_stats['phase_a']['success_rate']:.1%}\n")
            f.write(f"    Phases completed: {build_order_stats['phase_a']['phases']}\n\n")
            f.write(f"  Phase B (Farthest-Point Condensation):\n")
            f.write(f"    Total tests: {build_order_stats['phase_b']['total_tests']:,}\n")
            f.write(f"    Connections found: {build_order_stats['phase_b']['connections_found']:,}\n")
            f.write(f"    Success rate: {build_order_stats['phase_b']['success_rate']:.1%}\n")
            f.write(f"    Early stop: {build_order_stats['phase_b']['early_stop']}\n")
            for i, round_stats in enumerate(build_order_stats['phase_b']['rounds'], 1):
                f.write(f"      Round {i}: {round_stats['tests']:,} tests, {round_stats['connections']:,} connections\n")
            f.write(f"\n  Phase D (K-Window Densification):\n")
            f.write(f"    Window tests: {build_order_stats['phase_d']['window_tests']:,}\n")
            f.write(f"    Connections found: {build_order_stats['phase_d']['connections_found']:,}\n")
            f.write(f"    Window success rate: {build_order_stats['phase_d']['success_rate']:.1%}\n")
            f.write(f"    Average connections per vertex: {build_order_stats['phase_d']['average_connections_per_vertex']:.1f}\n")
            f.write(f"    Window density: {build_order_stats['phase_d']['window_density']:.1%}\n\n")
            f.write(f"  Phase F (Final K-densification Verification):\n")
            f.write(f"    Verification tests: {build_order_stats['phase_f']['total_tests']:,}\n")
            f.write(f"    NEW computations: {build_order_stats['phase_f']['newly_computed_tests']:,}\n")
            f.write(f"    Additional connections found: {build_order_stats['phase_f']['connections_found']:,}\n")
            f.write(f"    NEW connections discovered: {build_order_stats['phase_f']['newly_computed_connections']:,}\n")
            f.write(f"    Overall verification success rate: {build_order_stats['phase_f']['success_rate']:.1%}\n")
            f.write(f"    NEW computations success rate: {build_order_stats['phase_f']['newly_computed_success_rate']:.1%}\n")
            if build_order_stats['phase_f']['newly_computed_connections'] > 0:
                f.write(f"    Average gap: {build_order_stats['phase_f']['average_gap']:.1f}\n")
                f.write(f"    Maximum gap: {build_order_stats['phase_f']['max_gap']}\n")
            f.write(f"\n")
            
            # Add tree structure analysis to the file
            if 'tree_diameter' in build_order_stats:
                diameter_info = build_order_stats['tree_diameter']
                f.write(f"Tree Structure Analysis:\n")
                f.write(f"  Tree diameter: {diameter_info['diameter']} hops\n")
                f.write(f"  Diameter endpoints: {diameter_info['endpoints'][0]} ↔ {diameter_info['endpoints'][1]}\n")
                f.write(f"  Center nodes: {diameter_info['center_nodes']}\n")
                f.write(f"  Connected components: {build_order_stats.get('connected_components', 1)}\n\n")
            
            f.write(f"Final Order:\n")
            for i, section in enumerate(final_order):
                f.write(f"  {i+1:3d}: Section {section}\n")
        
        # Create visualization if requested
        if visualize:
            print(f"\n Creating BUILD-AND-ORDER visualization...")
            # For now, reuse existing visualization with adapted stats
            boruvka_style_stats = {
                'algorithm': 'BUILD-AND-ORDER',
                'total_edge_tests': build_order_stats['tree_edges'] + build_order_stats['window_edges'],
                'total_successful_hooks': build_order_stats['tree_edges'],
                'reduction_factor': 1.0,  # Placeholder
                'theoretical_complete_pairs': len(all_sections) * (len(all_sections) - 1) // 2,
                'tree_edges': build_order_stats['tree_edges'],
                'window_edges': build_order_stats['window_edges'],
                'rough_order_quality': build_order_stats['rough_order_quality'],
                'final_order_quality': build_order_stats['final_order_quality']
            }
            create_spanning_tree_visualization(
                [(final_order[i], final_order[i+1]) for i in range(len(final_order)-1)],
                all_sections, results, output_dir, boruvka_style_stats
            )
        
        print(f"\n Results saved to: {output_dir}/")
        print(f"   Final order: {final_order_file}")
        print(f"   Statistics: {stats_file}")
        if visualize:
            print(f"   Visualization: {output_dir}/spanning_tree_visualization.png")
        
        # Return the data manager for further analysis
        return data_manager.export_to_dataframe(), data_manager.get_pipeline_valid_pairs()
    
    elif sublinear:
        print(f"\n Running SUBLINEAR analysis...")
        start_time = time.time()
        
        # Run sublinear analysis
        spanning_tree, boruvka_stats = randomized_boruvka_sublinear(
            all_sections, edge_oracle, verbose=verbose
        )
        
        elapsed_time = time.time() - start_time
        
        # Output directory already created earlier for SIFT caching
        
        # Get accurate statistics from data_manager
        dm_stats = data_manager.get_statistics()
        
        print(f"\n SUBLINEAR ANALYSIS COMPLETE!")
        print(f"   Algorithm: Randomized Borůvka")
        print(f"   Spanning tree edges: {len(spanning_tree)}")
        print(f"   Total phases: {boruvka_stats['total_phases']}")
        print(f"   Total edge tests: {boruvka_stats['total_edge_tests']:,}")
        print(f"   Successful hooks: {boruvka_stats['total_successful_hooks']}")
        print(f"   Reduction factor: {boruvka_stats['reduction_factor']:.1f}x")
        print(f"   Success rate: {(boruvka_stats['total_successful_hooks'] / boruvka_stats['total_edge_tests'] * 100):.1f}%")
        print(f"   Total computation time: {elapsed_time:.1f} seconds")
        print(f"\n EFFICIENCY STATISTICS:")
        print(f"   Unique pairs computed: {dm_stats['unique_pairs_computed']:,}")
        print(f"   Total oracle calls: {dm_stats['total_oracle_calls']:,}")
        print(f"   Cache hits: {dm_stats['cache_hits']:,}")
        print(f"   True reduction factor: {1/dm_stats['computation_ratio']:.1f}x")
        print(f"   Redundancy ratio: {dm_stats['redundancy_ratio']:.1f}x")
        print(f"   Cache hit rate: {dm_stats['cache_hit_rate']:.1%}")
        
        # Print edge mask statistics if provided
        if edge_mask is not None:
            print(f"\n EDGE MASK STATISTICS:")
            print(f"   Permissible pairs: {len(edge_mask):,}")
            print(f"   Total possible pairs: {len(all_sections) * (len(all_sections) - 1) // 2:,}")
            print(f"   Mask density: {len(edge_mask) / (len(all_sections) * (len(all_sections) - 1) // 2):.1%}")
        
        # Save results
        # 1. Spanning tree
        spanning_tree_df = pd.DataFrame(spanning_tree, columns=['section1', 'section2'])
        spanning_tree_file = f"{output_dir}/spanning_tree.csv"
        spanning_tree_df.to_csv(spanning_tree_file, index=False)
        
        # 2. Borůvka statistics
        stats_file = f"{output_dir}/boruvka_statistics.txt"
        with open(stats_file, 'w') as f:
            f.write("=== RANDOMIZED BORŮVKA ALGORITHM STATISTICS ===\n\n")
            f.write(f"Input:\n")
            f.write(f"  Total sections: {len(all_sections)}\n")
            f.write(f"  Complete pairwise comparisons: {len(all_sections) * (len(all_sections) - 1) // 2:,}\n\n")
            f.write(f"Algorithm Parameters:\n")
            f.write(f"  k (samples per vertex per phase): {boruvka_stats.get('k', 'N/A')}\n")
            f.write(f"  max_rounds (maximum phases): {boruvka_stats.get('max_rounds', 'N/A')}\n\n")
            f.write(f"Validation Criteria (Strong Connections):\n")
            f.write(f"  SSIM threshold: > {ssim_threshold}\n")
            f.write(f"  Rotation limits: ±90°\n")
            f.write(f"  Scale limits: 0.95 to 1.05\n")
            f.write(f"  Inlier ratio: > {inlier_threshold}\n\n")
            f.write(f"Results:\n")
            f.write(f"  Total phases: {boruvka_stats['total_phases']}\n")
            f.write(f"  Total edge tests: {boruvka_stats['total_edge_tests']:,}\n")
            f.write(f"  Successful hooks: {boruvka_stats['total_successful_hooks']}\n")
            f.write(f"  Spanning tree edges: {len(spanning_tree)}\n")
            f.write(f"  Final components: {boruvka_stats['final_components']}\n\n")
            f.write(f"Performance:\n")
            f.write(f"  Reduction factor: {boruvka_stats['reduction_factor']:.1f}x\n")
            f.write(f"  Success rate: {(boruvka_stats['total_successful_hooks'] / boruvka_stats['total_edge_tests'] * 100):.1f}%\n")
            f.write(f"  Computation time: {elapsed_time:.1f} seconds\n")
        
        # Create visualization if requested
        if visualize:
            print(f"\n📊 Creating spanning tree visualization...")
            create_spanning_tree_visualization(
                spanning_tree, all_sections, results, output_dir, boruvka_stats
            )
        
        print(f"\n📁 Results saved to: {output_dir}/")
        print(f"   Spanning tree: {spanning_tree_file}")
        print(f"   Statistics: {stats_file}")
        if visualize:
            print(f"   Visualization: {output_dir}/spanning_tree_visualization.png")
        
        # Return the data manager for further analysis
        return data_manager.export_to_dataframe(), data_manager.get_pipeline_valid_pairs()
    
    else:
        # Standard complete pairwise analysis
        print(f"\n🔬 Running complete pairwise analysis...")
        start_time = time.time()
        
        # Process all pairs
        pair_count = 0
        successful_count = 0
        
        for sec1, sec2 in combinations(all_sections, 2):
            pair_count += 1
            elapsed_time = time.time() - start_time
            
            # Print progress every 100 pairs
            if pair_count % 100 == 0:
                avg_time_per_pair = elapsed_time / pair_count
                remaining_pairs = num_complete_pairs - pair_count
                eta_seconds = remaining_pairs * avg_time_per_pair
                eta_hours = eta_seconds / 3600
                
                print(f"\n[{pair_count:,}/{num_complete_pairs:,}] Progress: {pair_count/num_complete_pairs:.1%}")
                print(f"  Elapsed time: {elapsed_time/3600:.1f} hours")
                print(f"  Average time per pair: {avg_time_per_pair:.2f} seconds")
                print(f"  ETA: {eta_hours:.1f} hours")
            
            print(f"[{pair_count:,}/{num_complete_pairs:,}] Analyzing sections {sec1} vs {sec2}")
            
            img1 = images[sec1]
            img2 = images[sec2]
            
            # Find RIGID alignment
            transformation_matrix, all_matches, inlier_matches, stats = find_rigid_sift_alignment(
                img1, img2, f"section_{sec1}", f"section_{sec2}", use_cache=cache_sift, fast_flann=fast_flann, verbose=False, inlier_threshold=inlier_threshold)
            
            # Initialize result record
            result = {
                'section1': sec1,
                'section2': sec2,
                'features1': stats['features1'],
                'features2': stats['features2'],
                'initial_matches': stats['matches'],
                'ransac_inliers': stats['inliers'],
                'inlier_ratio': stats['inlier_ratio'],
                'rotation_degrees': stats.get('rotation_deg', 0),
                'translation_x': stats.get('translation_x', 0),
                'translation_y': stats.get('translation_y', 0),
                'scale_x': stats.get('scale_x', 1),
                'scale_y': stats.get('scale_y', 1),
                'detection_time_sec': stats['detection_time'],
                'matching_time_sec': stats['matching_time'],
                'alignment_success': stats['success'],
                'error_message': stats.get('error', ''),
                'ssim_score': 0.0,
                'overlap_area_pixels': 0,
                'overlap_bbox': '',
                'pipeline_valid': False  # Will be set based on pipeline criteria
            }
            
            # Calculate SSIM if basic alignment was successful
            ssim_score = 0.0
            if transformation_matrix is not None and stats['success']:
                # Pre-validate transformation parameters before attempting SSIM calculation
                rotation = abs(stats.get('rotation_deg', 0))
                scale_x = stats.get('scale_x', 1)
                scale_y = stats.get('scale_y', 1)
                
                # Check if transformation parameters are biologically plausible
                transformation_valid = (
                    rotation <= ROTATION_LIMIT_DEGREES and             # Rotation within ±90°
                    SCALE_MIN <= scale_x <= SCALE_MAX and               # Scale X within range
                    SCALE_MIN <= scale_y <= SCALE_MAX and               # Scale Y within range
                    scale_x > 0.1 and scale_y > 0.1                    # Avoid degenerate scales
                )
                
                if transformation_valid:
                    aligned_img1, overlap_region1, overlap_region2, bbox = apply_rigid_transformation_and_find_overlap(
                        img1, img2, transformation_matrix)
                    
                    if overlap_region1 is not None and overlap_region2 is not None:
                        ssim_score = calculate_ssim_score(overlap_region1, overlap_region2)
                        result['ssim_score'] = ssim_score
                        result['overlap_area_pixels'] = bbox[2] * bbox[3]
                        result['overlap_bbox'] = f"{bbox[0]},{bbox[1]},{bbox[2]},{bbox[3]}"
                else:
                    # Mark as failed due to invalid transformation
                    stats['success'] = False
            
            # Apply pipeline validation criteria for "strong connection" 
            pipeline_valid = False
            if stats['success'] and ssim_score > 0:
                rotation = abs(stats.get('rotation_deg', 0))
                scale_x = stats.get('scale_x', 1)
                scale_y = stats.get('scale_y', 1)
                inlier_ratio = stats.get('inlier_ratio', 0)
                
                # Pipeline criteria (matching create_alignment_graph.py and multi_branch_alignment.py)
                pipeline_valid = (
                    ssim_score > ssim_threshold and                     # SSIM threshold
                    rotation <= ROTATION_LIMIT_DEGREES and             # Rotation within ±90°
                    SCALE_MIN <= scale_x <= SCALE_MAX and               # Scale X within range
                    SCALE_MIN <= scale_y <= SCALE_MAX and               # Scale Y within range
                    inlier_ratio > inlier_threshold                     # Inlier ratio threshold
                )
                
                if pipeline_valid:
                    successful_count += 1
                    print(f"  ✅ Strong connection: SSIM={ssim_score:.3f}, Rot={rotation:.1f}°, Scale=({scale_x:.2f},{scale_y:.2f}), Inliers={inlier_ratio:.1%}")
                else:
                    reasons = []
                    if ssim_score <= ssim_threshold:
                        reasons.append(f"SSIM={ssim_score:.3f}<={ssim_threshold}")
                    if rotation > ROTATION_LIMIT_DEGREES:
                        reasons.append(f"Rot={rotation:.1f}°>{ROTATION_LIMIT_DEGREES}°")
                    if not (SCALE_MIN <= scale_x <= SCALE_MAX):
                        reasons.append(f"ScaleX={scale_x:.2f}")
                    if not (SCALE_MIN <= scale_y <= SCALE_MAX):
                        reasons.append(f"ScaleY={scale_y:.2f}")
                    if inlier_ratio <= inlier_threshold:
                        reasons.append(f"Inliers={inlier_ratio:.1%}<={inlier_threshold}")
                    print(f"  ❌ Weak connection: {', '.join(reasons)}")
            else:
                print(f"  ❌ No alignment found")
            
            result['pipeline_valid'] = pipeline_valid
            results.append(result)
        
        elapsed_time = time.time() - start_time
        
        print(f"\n🔬 COMPLETE PAIRWISE ANALYSIS FINISHED!")
        print(f"   Total pairs processed: {pair_count:,}")
        print(f"   Strong connections found: {successful_count:,}")
        print(f"   Success rate: {(successful_count / pair_count * 100):.1f}%")
        print(f"   Total computation time: {elapsed_time/3600:.1f} hours")
        print(f"   Average time per pair: {elapsed_time/pair_count:.2f} seconds")
        
        # Print edge mask statistics if provided
        if edge_mask is not None:
            print(f"\n📊 EDGE MASK STATISTICS:")
            print(f"   Permissible pairs: {len(edge_mask):,}")
            print(f"   Total possible pairs: {len(all_sections) * (len(all_sections) - 1) // 2:,}")
            print(f"   Mask density: {len(edge_mask) / (len(all_sections) * (len(all_sections) - 1) // 2):.1%}")
        
        # Output directory already created earlier for SIFT caching
        
        # Save results
        df = pd.DataFrame(results)
        results_file = f"{output_dir}/comprehensive_pairwise_results.csv"
        df.to_csv(results_file, index=False)
        
        # Save analysis summary
        summary_file = f"{output_dir}/analysis_summary.txt"
        with open(summary_file, 'w') as f:
            f.write("=== COMPREHENSIVE PAIRWISE ANALYSIS SUMMARY ===\n\n")
            f.write(f"Analysis Mode: COMPLETE\n")
            f.write(f"Configuration:\n")
            f.write(f"  SIFT Caching: {'Enabled' if cache_sift else 'Disabled'}\n")
            f.write(f"  Fast FLANN: {'Enabled' if fast_flann else 'Disabled'}\n")
            f.write(f"  Visualization: {'Enabled' if visualize else 'Disabled'}\n")
            f.write(f"  SSIM Threshold: {ssim_threshold}\n")
            f.write(f"  Inlier Threshold: {inlier_threshold}\n\n")
            f.write(f"Pipeline Validation Criteria (Strong Connections):\n")
            f.write(f"  SSIM > {ssim_threshold}\n")
            f.write(f"  Rotation: ±90°\n")
            f.write(f"  Scale: 0.8 to 1.25 (both X and Y)\n")
            f.write(f"  Inlier ratio: > {inlier_threshold}\n\n")
            f.write(f"Total sections analyzed: {len(all_sections)}\n")
            f.write(f"Section range: {min(all_sections)} to {max(all_sections)}\n")
            f.write(f"All sections: {all_sections}\n")
            f.write(f"Total pairs processed: {pair_count:,}\n")
            f.write(f"Total computation time: {elapsed_time/3600:.2f} hours\n")
            f.write(f"Average time per pair: {elapsed_time/pair_count:.2f} seconds\n\n")
            f.write(f"ALIGNMENT RESULTS:\n")
            f.write(f"Basic successful alignments: {len([r for r in results if r['alignment_success']])}\n")
            f.write(f"Pipeline valid alignments (strong connections): {successful_count}\n")
            f.write(f"Failed alignments: {len([r for r in results if not r['alignment_success']])}\n")
            f.write(f"Basic success rate: {(len([r for r in results if r['alignment_success']]) / pair_count * 100):.1f}%\n")
            f.write(f"Pipeline valid rate: {(successful_count / pair_count * 100):.1f}%\n\n")
            successful_results = [r for r in results if r['alignment_success']]
            if successful_results:
                f.write(f"SSIM STATISTICS (All successful):\n")
                ssim_scores = [r['ssim_score'] for r in successful_results if r['ssim_score'] > 0]
                if ssim_scores:
                    f.write(f"Mean SSIM: {np.mean(ssim_scores):.3f}\n")
                    f.write(f"Std SSIM: {np.std(ssim_scores):.3f}\n")
                    f.write(f"Min SSIM: {np.min(ssim_scores):.3f}\n")
                    f.write(f"Max SSIM: {np.max(ssim_scores):.3f}\n\n")
            
            pipeline_valid_results = [r for r in results if r['pipeline_valid']]
            if pipeline_valid_results:
                f.write(f"SSIM STATISTICS (Pipeline valid only):\n")
                ssim_scores = [r['ssim_score'] for r in pipeline_valid_results if r['ssim_score'] > 0]
                if ssim_scores:
                    f.write(f"Mean SSIM: {np.mean(ssim_scores):.3f}\n")
                    f.write(f"Std SSIM: {np.std(ssim_scores):.3f}\n")
                    f.write(f"Min SSIM: {np.min(ssim_scores):.3f}\n")
                    f.write(f"Max SSIM: {np.max(ssim_scores):.3f}\n")
        
        print(f"\n Results saved to: {output_dir}/")
        print(f"   Complete results: {results_file}")
        print(f"   Summary: {summary_file}")
        
        # Create pipeline valid dataframe
        pipeline_valid_df = pd.DataFrame([r for r in results if r['pipeline_valid']])
        if len(pipeline_valid_df) > 0:
            pipeline_valid_file = f"{output_dir}/pipeline_valid_connections.csv"
            pipeline_valid_df.to_csv(pipeline_valid_file, index=False)
            print(f"   Pipeline valid connections: {pipeline_valid_file}")
            return df, pipeline_valid_df
        else:
            print(f"   ⚠️  No pipeline valid connections found")
            return df, None

def create_spanning_tree_visualization(spanning_tree, all_sections, results, output_dir, boruvka_stats):
    """
    Create a visualization of the spanning tree found by Randomized Borůvka algorithm
    """
    # Create NetworkX graph
    G = nx.Graph()
    
    # Add all sections as nodes
    G.add_nodes_from(all_sections)
    
    # Create mapping from results for edge attributes
    edge_data = {}
    for result in results:
        if result['pipeline_valid']:
            pair = (result['section1'], result['section2'])
            edge_data[pair] = result
    
    # Add spanning tree edges
    edges_added = 0
    for sec1, sec2 in spanning_tree:
        if (sec1, sec2) in edge_data:
            data = edge_data[(sec1, sec2)]
        elif (sec2, sec1) in edge_data:
            data = edge_data[(sec2, sec1)]
        else:
            # Edge in spanning tree but no result data (shouldn't happen)
            data = {'ssim_score': 0.5, 'rotation_degrees': 0, 'scale_x': 1, 'scale_y': 1}
        
        G.add_edge(sec1, sec2, 
                   ssim=data['ssim_score'],
                   rotation=data.get('rotation_degrees', 0),
                   scale_x=data.get('scale_x', 1),
                   scale_y=data.get('scale_y', 1))
        edges_added += 1
    
    # Create figure
    plt.figure(figsize=(16, 12))
    
    # Use spring layout for better visualization
    pos = nx.spring_layout(G, k=2, iterations=50, seed=42)
    
    # Find connected components for coloring
    connected_components = list(nx.connected_components(G))
    
    # Create color mapping
    import matplotlib.cm as cm
    if len(connected_components) <= 10:
        colors = plt.cm.tab10(np.linspace(0, 1, 10))
    else:
        colors = plt.cm.tab20(np.linspace(0, 1, 20))
    
    # Color nodes by connected component
    node_colors = []
    for node in G.nodes():
        for i, component in enumerate(connected_components):
            if node in component:
                node_colors.append(colors[i % len(colors)])
                break
        else:
            node_colors.append('lightgray')  # Isolated nodes
    
    # Draw nodes
    nx.draw_networkx_nodes(G, pos, 
                          node_color=node_colors,
                          node_size=500,
                          alpha=0.8,
                          edgecolors='black',
                          linewidths=1)
    
    # Draw node labels
    nx.draw_networkx_labels(G, pos, 
                           font_size=10,
                           font_weight='bold',
                           font_color='white')
    
    # Draw edges with thickness proportional to SSIM
    if G.edges():
        edge_ssims = [G[u][v]['ssim'] for u, v in G.edges()]
        edge_widths = [1 + 4 * ssim for ssim in edge_ssims]  # 1-5 range
        
        nx.draw_networkx_edges(G, pos,
                              width=edge_widths,
                              edge_color='darkblue',
                              alpha=0.7)
        
        # Add edge labels (SSIM values)
        edge_labels = {(u, v): f"{G[u][v]['ssim']:.3f}" for u, v in G.edges()}
        nx.draw_networkx_edge_labels(G, pos, edge_labels,
                                    font_size=8,
                                    font_color='red',
                                    bbox=dict(boxstyle='round,pad=0.1',
                                             facecolor='white',
                                             alpha=0.8))
    
    # Add title and statistics (handle both formats)
    if boruvka_stats.get('algorithm') == 'BUILD-AND-ORDER':
        plt.title(f"BUILD-AND-ORDER Spanning Tree\n"
                  f"Sections: {len(all_sections)}, Strong Connections: {edges_added}, "
                  f"Components: {len(connected_components)}\n"
                  f"Tree Edges: {boruvka_stats['tree_edges']}, "
                  f"Window Edges: {boruvka_stats['window_edges']}",
                  fontsize=14, fontweight='bold')
    else:
        plt.title(f"Randomized Borůvka Spanning Tree\n"
                  f"Sections: {len(all_sections)}, Strong Connections: {edges_added}, "
                  f"Components: {len(connected_components)}\n"
                  f"Edge Tests: {boruvka_stats['total_edge_tests']:,} "
                  f"(vs {boruvka_stats['theoretical_complete_pairs']:,} complete), "
                  f"Reduction: {boruvka_stats['reduction_factor']:.1f}x",
                  fontsize=14, fontweight='bold')
    
    # Add legend
    legend_elements = []
    for i, component in enumerate(connected_components[:10]):  # Show first 10 components
        legend_elements.append(
            plt.Line2D([0], [0], marker='o', color='w', 
                      markerfacecolor=colors[i % len(colors)], markersize=10,
                      label=f'Component {i+1} ({len(component)} sections)')
        )
    
    if len(connected_components) <= 10:
        plt.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(1.05, 1))
    
    # Add statistics text box (handle both Borůvka and BUILD-AND-ORDER formats)
    if boruvka_stats.get('algorithm') == 'BUILD-AND-ORDER':
        # BUILD-AND-ORDER format
        stats_text = (f"Algorithm Statistics:\n"
                      f"• Algorithm: BUILD-AND-ORDER\n"
                      f"• Tree Edges: {boruvka_stats['tree_edges']}\n"
                      f"• Window Edges: {boruvka_stats['window_edges']}\n"
                      f"• Rough Order Quality: {boruvka_stats['rough_order_quality']:.3f}\n"
                      f"• Final Order Quality: {boruvka_stats['final_order_quality']:.3f}")
    else:
        # Original Borůvka format
        stats_text = (f"Algorithm Statistics:\n"
                      f"• Phases: {boruvka_stats['total_phases']}\n"
                      f"• Edge Tests: {boruvka_stats['total_edge_tests']:,}\n"
                      f"• Success Rate: {boruvka_stats['total_successful_hooks']/boruvka_stats['total_edge_tests']*100:.1f}%\n"
                      f"• Theoretical Reduction: {boruvka_stats['reduction_factor']:.1f}x")
    
    plt.text(0.02, 0.98, stats_text, transform=plt.gca().transAxes,
             fontsize=10, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
    
    plt.axis('off')
    plt.tight_layout()
    
    # Save visualization
    viz_file = f"{output_dir}/spanning_tree_visualization.png"
    plt.savefig(viz_file, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"📊 Spanning tree visualization saved: {viz_file}")
    
    # Create a second visualization: adjacency matrix heatmap
    plt.figure(figsize=(12, 10))
    
    # Create adjacency matrix for visualization
    section_to_idx = {sec: i for i, sec in enumerate(sorted(all_sections))}
    n = len(all_sections)
    adj_matrix = np.zeros((n, n))
    
    for sec1, sec2 in spanning_tree:
        i, j = section_to_idx[sec1], section_to_idx[sec2]
        # Find SSIM score for this edge
        ssim_score = 0.5  # default
        for result in results:
            if ((result['section1'] == sec1 and result['section2'] == sec2) or
                (result['section1'] == sec2 and result['section2'] == sec1)) and result['pipeline_valid']:
                ssim_score = result['ssim_score']
                break
        
        adj_matrix[i, j] = ssim_score
        adj_matrix[j, i] = ssim_score  # Symmetric
    
    # Create heatmap
    plt.imshow(adj_matrix, cmap='RdYlBu_r', vmin=0, vmax=1, aspect='auto')
    plt.colorbar(label='SSIM Score')
    
    # Set ticks and labels
    sorted_sections = sorted(all_sections)
    tick_indices = range(0, len(sorted_sections), max(1, len(sorted_sections)//20))
    plt.xticks(tick_indices, [sorted_sections[i] for i in tick_indices], rotation=45)
    plt.yticks(tick_indices, [sorted_sections[i] for i in tick_indices])
    
    algorithm_name = "BUILD-AND-ORDER" if boruvka_stats.get('algorithm') == 'BUILD-AND-ORDER' else "Randomized Borůvka"
    plt.title(f"{algorithm_name} Spanning Tree Adjacency Matrix\nStrong Connections (SSIM > threshold)", 
              fontsize=14, fontweight='bold')
    plt.xlabel('Section Number')
    plt.ylabel('Section Number')
    
    # Save heatmap
    heatmap_file = f"{output_dir}/spanning_tree_heatmap.png"
    plt.savefig(heatmap_file, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"📊 Spanning tree heatmap saved: {heatmap_file}")

def weak_connectivity_oracle(sec1, sec2):
    """Fast weak connectivity oracle using basic image statistics"""
    # This is a placeholder - in practice, you'd use:
    # - Simple correlation
    # - Histogram similarity  
    # - Coarse SIFT features
    # - Basic edge detection
    
    # For now, use a relaxed SSIM threshold as proxy for "weak" connection
    edge_data = data_manager.get_edge_data(sec1, sec2)
    if edge_data is None:
        return False
    
    # Much more permissive criteria for weak connectivity
    ssim_score = edge_data.get('ssim_score', 0)
    rotation_deg = abs(edge_data.get('rotation_deg', 180))
    
    # Weak connection: SSIM > 0.15 (vs 0.25 for strong) and rotation < 120° (vs 90° for strong)
    return ssim_score > 0.15 and rotation_deg < 120

def two_phase_boruvka_bio_binary(all_sections, edge_oracle, get_edge_data, s, verbose=False):
    """Two-phase Borůvka: weak connectivity first, then strong purification"""
    
    N = len(all_sections)
    
    class DSU:
        def __init__(self, n):
            self.parent = list(range(n))
            self.rank = [0] * n
            self.components = n
        
        def find(self, x):
            if self.parent[x] != x:
                self.parent[x] = self.find(self.parent[x])
            return self.parent[x]
        
        def union(self, a, b):
            root_a = self.find(a)
            root_b = self.find(b)
            
            if root_a == root_b:
                return False
            
            if self.rank[root_a] < self.rank[root_b]:
                self.parent[root_a] = root_b
            elif self.rank[root_a] > self.rank[root_b]:
                self.parent[root_b] = root_a
            else:
                self.parent[root_b] = root_a
                self.rank[root_a] += 1
            
            self.components -= 1
            return True
    
    # PHASE 1: Weak connectivity with fast oracle
    if verbose:
        print("   Phase 1: Fast weak connectivity")
    
    dsu = DSU(N)
    weak_edges = []
    weak_tests = 0
    weak_connections = 0
    
    phase = 0
    while dsu.components > 1:
        phase += 1
        if verbose:
            print(f"      Weak phase {phase}: {dsu.components} components remaining")
        
        phase_tests = 0
        phase_connections = 0
        
        for u_idx in range(N):
            if dsu.components == 1:
                break
                
            u = all_sections[u_idx]
            
            # Try s random vertices to find a weak connection
            for _ in range(s):
                v_idx = random.randint(0, N - 1)
                if dsu.find(u_idx) == dsu.find(v_idx):
                    continue
                
                v = all_sections[v_idx]
                phase_tests += 1
                weak_tests += 1
                
                # Fast weak oracle
                if weak_connectivity_oracle(u, v):
                    weak_connections += 1
                    phase_connections += 1
                    
                    if dsu.union(u_idx, v_idx):
                        weak_edges.append((u, v))
                        if verbose:
                            edge_data = get_edge_data(u, v)
                            ssim = edge_data.get('ssim_score', 0) if edge_data else 0
                            print(f"         Weak connection: {u} ↔ {v} (SSIM: {ssim:.3f})")
                        break  # u hooked this phase
        
        if verbose:
            print(f"         Phase {phase}: {phase_tests:,} tests, {phase_connections:,} connections")
    
    # PHASE 2: Strong purification on weak connected component
    if verbose:
        print("   Phase 2: Strong purification")
    
    strong_edges = []
    strong_tests = 0
    strong_connections = 0
    
    # Test all weak edges with strong oracle
    for u, v in weak_edges:
        strong_tests += 1
        if edge_oracle(u, v):  # Strong oracle
            strong_connections += 1
            edge_data = get_edge_data(u, v)
            distance = 1 - edge_data.get('ssim_score', 0) if edge_data else 0.5
            strong_edges.append((u, v, distance))
            if verbose:
                print(f"      Strong connection: {u} ↔ {v} (SSIM: {edge_data.get('ssim_score', 0):.3f})")
    
    total_tests = weak_tests + strong_tests
    total_connections = strong_connections
    
    if verbose:
        print(f"   📊 Two-phase summary:")
        print(f"      Weak phase: {weak_tests:,} tests → {weak_connections:,} weak connections")
        print(f"      Strong phase: {strong_tests:,} tests → {strong_connections:,} strong connections")
        print(f"      Total: {total_tests:,} tests → {total_connections:,} final connections")
        print(f"      Purification rate: {strong_connections/max(1, weak_connections):.1%}")
    
    stats = {
        'edge_tests': total_tests,
        'connections_found': total_connections,
        'success_rate': total_connections / max(1, total_tests),
        'phases': phase,
        'weak_tests': weak_tests,
        'weak_connections': weak_connections,
        'strong_tests': strong_tests,
        'strong_connections': strong_connections,
        'purification_rate': strong_connections / max(1, weak_connections)
    }
    
    return strong_edges, stats

def main():
    """Main function with command-line argument parsing"""
    parser = argparse.ArgumentParser(
        description="BUILD-AND-ORDER: Sub-quadratic wafer section alignment and ordering pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
BUILD-AND-ORDER Algorithm Overview:
  • Phase A: Short-edge Borůvka tree construction - O(N log N) complexity
  • Phase B: Farthest-point condensation - densifies connectivity  
  • Phase C: Double-sweep BFS for diameter-based ordering
  • Phase D: K-window densification - adds local connections
  • Phase E: Spectral ordering with Fiedler vector (optional)
  • Phase F: Final K-densification verification - validates ordering quality

Performance Optimizations:
  • SIFT caching: 99% reduction in feature extraction overhead
  • Fast FLANN matching: 2-3x speedup in feature matching
  • Combined optimizations: Up to 24x performance improvement
  • Sub-quadratic complexity: O(N(log N + K)) vs O(N²) exhaustive

Examples:
  # Recommended: BUILD-AND-ORDER with optimizations
  python Graph-Condensation-Densification.py --build-and-order --cache-sift --fast-flann --verbose
  
  # Baseline test with specific parameters  
  python Graph-Condensation-Densification.py --build-and-order --samples-per-phase 7 --window-k 7
  
  # Auto-tune SSIM threshold for optimal connectivity
  python Graph-Condensation-Densification.py --build-and-order --auto-tune-threshold --verbose
  
  # Two-phase analysis: relaxed → strict with proximity mask
  python Graph-Condensation-Densification.py --build-and-order --two-phase --proximity-window 15
  
  # Pre-compute all SIFT features upfront
  python Graph-Condensation-Densification.py --build-and-order --precompute-sift --verbose
  python Graph-Condensation-Densification.py --window-k 10                # Override K (window half-width) for K-window densification
  
Performance comparison:
  Complete (no optimizations):    ~11,130 SIFT computations, standard FLANN (4+ hours)
  Complete (SIFT caching only):   ~106 SIFT computations, standard FLANN (~15 minutes)
  Complete (Fast FLANN only):     ~11,130 SIFT computations, fast FLANN (~1.5 hours)
  Complete (both optimizations): ~106 SIFT computations, fast FLANN (~3 minutes)
  BUILD-AND-ORDER:               ~N(log N + K) edge tests, 5-phase algorithm (~5-10 minutes, RECOMMENDED)
  Sublinear (Randomized Borůvka): ~N log²N edge tests instead of N²/2 (10-50x reduction)
  
The sublinear approach uses Randomized Borůvka algorithm to build a spanning tree
of the strongest connections, achieving O(N log²N) complexity instead of O(N²).

The BUILD-AND-ORDER approach provides sub-quadratic 1-D ordering using theoretical algorithm:
1. Phase A: Short-edge Borůvka tree construction
2. Phase B: Farthest-point condensation (3 rounds)
3. Phase C: Double-sweep BFS for diameter-based rough ordering
4. Phase D: K-window densification for local connectivity
5. Phase E: Final spectral ordering (enabled by default, can be disabled)
6. Phase F: Final K-densification verification - validates ordering and finds missed connections
This achieves O(N(log N + K)) complexity and is adapted for biological sections.

Strong Connection Criteria (Pipeline Quality):
  - SSIM > threshold (default 0.25, recommend 0.3 for high quality)
  - Rotation: ±90° (biological sections shouldn't be flipped)
  - Scale: 0.8 to 1.25 (both X and Y)
  - Inlier ratio: > 8% (sufficient feature correspondence)

Two-Phase Analysis (--two-phase):
  Phase 1: Relaxed threshold (auto-tuned for ~10 connections/node) → get linear order
  Phase 2: Strict threshold + proximity mask (nodes close in order are permissible)
  Benefits: Reduces biological pipeline calls by focusing on locally plausible connections
  
Auto-Tune Threshold (--auto-tune-threshold):
  Analyzes degree of random nodes to find optimal SSIM threshold
  Target: ~10 connections per node for good connectivity without noise
  Method: Tests thresholds 0.1-0.5 and selects closest to target degree
        """
    )
    
    parser.add_argument('--cache-sift', action='store_true',
                        help='Enable SIFT feature caching for dramatic speedup')
    parser.add_argument('--precompute-sift', action='store_true',
                        help='Pre-compute all SIFT features upfront (requires --cache-sift, provides progress tracking)')
    parser.add_argument('--fast-flann', action='store_true',
                        help='Use fast FLANN parameters (trees=4, checks=50) for 2-3x speedup')
    parser.add_argument('--sublinear', action='store_true',
                        help='Use sublinear Randomized Borůvka algorithm (EXPERIMENTAL)')
    parser.add_argument('--build-and-order', action='store_true',
                        help='Use BUILD-AND-ORDER theoretical algorithm (RECOMMENDED)')
    parser.add_argument('--no-spectral', action='store_true',
                        help='Disable spectral ordering in Phase E (enabled by default)')
    parser.add_argument('--two-phase', action='store_true',
                        help='Use two-phase weak→strong connectivity in Phase A (experimental)')
    parser.add_argument('--verbose', action='store_true',
                        help='Enable verbose output with detailed progress')
    parser.add_argument('--visualize', action='store_true',
                        help='Create visualization plots (spanning tree, results)')
    parser.add_argument('--ssim-threshold', type=float, default=DEFAULT_SSIM_THRESHOLD,
                        help=f'SSIM threshold for strong connections (default: {DEFAULT_SSIM_THRESHOLD})')
    parser.add_argument('--inlier-threshold', type=float, default=DEFAULT_INLIER_THRESHOLD,
                        help=f'Inlier ratio threshold for strong connections (default: {DEFAULT_INLIER_THRESHOLD} = {DEFAULT_INLIER_THRESHOLD*100:.0f}%%)')
    parser.add_argument('--auto-tune-threshold', action='store_true',
                        help='Auto-tune SSIM threshold based on node degree analysis')
    parser.add_argument('--proximity-window', type=int, default=10,
                        help='Window size for proximity mask in two-phase mode (default: 10)')
    parser.add_argument('--window-k', type=int, default=18,
                        help='K (window half-width) for K-window densification (default: 7)')
    parser.add_argument('--samples-per-phase', type=int, default=15,
                        help='s (random samples per vertex per Borůvka phase, default: 7)')
    parser.add_argument('--condensation-rounds', type=int, default=3,
                        help='Number of condensation rounds in Phase B (default: 3)')
    
    # Edge masking
    parser.add_argument('--edge-mask', type=str, default=None,
                        help='Path to CSV file with permissible edge pairs (i1,i2 format)')
    
    # Output options
    parser.add_argument('--save-tree-edges', action='store_true', default=True,
                        help='Save spanning tree edges to CSV (default: True)')
    parser.add_argument('--save-alignments', action='store_true',
                        help='Save successful alignment visualizations')
    parser.add_argument('--output-dir', type=str, default=None,
                        help='Custom output directory name')
    
    args = parser.parse_args()
    
    # Validate arguments
    mode_count = sum([args.sublinear, args.build_and_order])
    if mode_count > 1:
        print("ERROR: Please specify only one analysis mode (--sublinear, --build-and-order, or neither for complete analysis)")
        return
    
    if args.no_spectral and not args.build_and_order:
        print("ERROR: --no-spectral can only be used with --build-and-order")
        return
    
    print("🎯 BUILD-AND-ORDER: Sub-quadratic Wafer Section Analysis Pipeline")
    print("=" * 80)
    print(f"Analysis Mode: {'BUILD-AND-ORDER' if args.build_and_order else 'Sublinear Borůvka' if args.sublinear else 'Complete Pairwise'}")
    print(f"Performance:   SIFT Cache: {'✅' if args.cache_sift else '❌'} | Fast FLANN: {'✅' if args.fast_flann else '❌'} | Precompute: {'✅' if args.precompute_sift else '❌'}")
    print(f"Output:        Verbose: {'✅' if args.verbose else '❌'} | Visualize: {'✅' if args.visualize else '❌'} | Tree Edges: {'✅' if args.save_tree_edges else '❌'}")
    print(f"Parameters:    SSIM: {args.ssim_threshold} | Inlier: {args.inlier_threshold} | Auto-tune: {'✅' if args.auto_tune_threshold else '❌'} | Two-phase: {'✅' if args.two_phase else '❌'}")
    
    if args.build_and_order:
        print(f"\n🎯 BUILD-AND-ORDER ALGORITHM CONFIGURATION:")
        print(f"   Phase A: Borůvka tree (s={args.samples_per_phase} samples/phase)")
        print(f"   Phase B: Condensation ({args.condensation_rounds} rounds)")
        print(f"   Phase C: BFS diameter ordering") 
        print(f"   Phase D: K-window densification (K={args.window_k})")
        print(f"   Phase E: Spectral ordering {'✅ ENABLED' if not args.no_spectral else '❌ DISABLED'}")
        if args.two_phase:
            print(f"   Two-phase: Proximity window = {args.proximity_window}")
        if args.edge_mask:
            print(f"   Edge mask: {args.edge_mask}")
    
    print("=" * 80)
    
    if args.build_and_order and not args.no_spectral:
        pass  # Already handled above
    elif args.build_and_order and args.no_spectral:
        print("   Note: Spectral ordering disabled - will use BFS diameter ordering")
    
    if args.sublinear:
        print("\n SUBLINEAR MODE SELECTED:")
        print("   Algorithm: Randomized Borůvka spanning tree construction")
        print("   ⚠️  EXPERIMENTAL: This approach is under development")
        print("   ✅ O(N log²N) complexity instead of O(N²)")
        if args.verbose:
            print("   Detailed algorithmic progress will be provided")
    else:
        print("\n🔬 COMPLETE PAIRWISE MODE SELECTED:")
        print("   Algorithm: Exhaustive pairwise alignment analysis")
        print("   ⚠️  SLOW: O(N²) complexity, suitable for small datasets")
        print("   ✅ Comprehensive coverage of all possible alignments")
    
    print(f"\nStrong Connection Criteria:")
    print(f"   SSIM > {args.ssim_threshold}")
    print(f"   Rotation: ±90°")
    print(f"   Scale: 0.95 to 1.05 (both X and Y)")
    print(f"   Inlier ratio: > {args.inlier_threshold}")
    
    # Validate argument combinations
    if args.precompute_sift and not args.cache_sift:
        print("Error: --precompute-sift requires --cache-sift to be enabled")
        return
    
    # Parse edge mask if provided
    edge_mask = None
    if args.edge_mask:
        print(f"\n Loading edge mask from: {args.edge_mask}")
        try:
            edge_mask_df = pd.read_csv(args.edge_mask)
            edge_mask = set(zip(edge_mask_df.iloc[:, 0], edge_mask_df.iloc[:, 1]))
            print(f"   ✅ Loaded {len(edge_mask):,} permissible edge pairs")
        except Exception as e:
            print(f"   ❌ Failed to load edge mask: {e}")
            return
    
    # Start timing
    start_time = time.perf_counter()
    
    # Run comprehensive analysis
    result = comprehensive_pairwise_analysis(
        cache_sift=args.cache_sift, 
        precompute_sift=args.precompute_sift,
        fast_flann=args.fast_flann,
        sublinear=args.sublinear,
        build_and_order=args.build_and_order,
        verbose=args.verbose,
        visualize=args.visualize,
        ssim_threshold=args.ssim_threshold,
        inlier_threshold=args.inlier_threshold,
        use_spectral=not args.no_spectral,
        two_phase=args.two_phase,
        edge_mask=edge_mask,
        auto_tune_threshold=args.auto_tune_threshold,
        proximity_window=args.proximity_window,
        window_k=args.window_k,
        samples_per_phase=args.samples_per_phase,
        condensation_rounds=args.condensation_rounds,
        save_tree_edges=args.save_tree_edges
    )
    
    # Calculate total runtime
    total_time = time.perf_counter() - start_time
    
    # Generate comprehensive final report
    print(f"\n PIPELINE EXECUTION COMPLETE!")
    print(f"=" * 80)
    print(f" FINAL PERFORMANCE SUMMARY:")
    print(f"   Total runtime: {total_time:.1f} seconds ({total_time/60:.1f} minutes)")
    
    # Unpack and analyze results
    if isinstance(result, tuple):
        df, pipeline_valid_df = result
        if pipeline_valid_df is not None and len(pipeline_valid_df) > 0:
            print(f"   Strong connections: {len(pipeline_valid_df):,}")
            if hasattr(pipeline_valid_df, 'iloc') and 'ssim_score' in pipeline_valid_df.columns:
                avg_ssim = pipeline_valid_df['ssim_score'].mean()
                print(f"   Average SSIM score: {avg_ssim:.3f}")
            print(f"     Pipeline validation SUCCESS!")
            print(f"   These connections form the basis for spectral clustering and linear ordering")
        else:
            print(f"       No strong connections found")
            print(f"   Consider lowering --ssim-threshold or checking input data quality")
    else:
        print(f"     Analysis completed - check output directory for detailed results")
    
    print(f"=" * 80)

if __name__ == "__main__":
    main() 
