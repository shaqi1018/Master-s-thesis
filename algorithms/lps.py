# -*- coding: utf-8 -*-
"""
Local Peak Search (LPS) based instantaneous frequency tracking.
"""

import numpy as np
from scipy.signal import find_peaks

from .time_freq import compute_stft
from .Entropy.penalty_factory import get_strategy


def _normalize_by_max(values):
    """Normalize an array to [0, 1] using its maximum value."""
    values = np.asarray(values, dtype=float)
    max_value = np.max(values) if len(values) > 0 else 0.0
    if max_value <= 1e-12:
        return np.zeros_like(values)
    return values / max_value


def _extract_initial_candidates(power_spectrum, frequencies, search_indices,
                                num_candidates=8, peak_prominence_ratio=0.05):
    """Extract strong local peaks from the initialization window."""
    if power_spectrum.ndim == 2:
        avg_power = np.mean(power_spectrum, axis=1)
    else:
        avg_power = power_spectrum

    power_in_range = avg_power[search_indices]
    max_power = np.max(power_in_range)
    prominence = max_power * peak_prominence_ratio if max_power > 1e-12 else 0.0

    local_peaks, _ = find_peaks(power_in_range, prominence=prominence)

    # Keep edge peaks and the global maximum even if find_peaks misses them.
    extra_peaks = [int(np.argmax(power_in_range))]
    if len(power_in_range) > 1:
        if power_in_range[0] >= power_in_range[1]:
            extra_peaks.append(0)
        if power_in_range[-1] >= power_in_range[-2]:
            extra_peaks.append(len(power_in_range) - 1)

    peak_positions = np.unique(np.concatenate([local_peaks, np.asarray(extra_peaks, dtype=int)]))
    peak_indices = search_indices[peak_positions]

    candidate_powers = avg_power[peak_indices]
    order = np.argsort(candidate_powers)[::-1][:num_candidates]
    peak_indices = peak_indices[order]
    candidate_powers = candidate_powers[order]

    if power_spectrum.ndim == 2:
        stability_raw = []
        for idx in peak_indices:
            peak_series = power_spectrum[idx, :]
            stability_raw.append(np.mean(peak_series) / (np.std(peak_series) + 1e-12))
        stability_raw = np.asarray(stability_raw, dtype=float)
    else:
        stability_raw = np.ones(len(peak_indices), dtype=float)

    power_scores = _normalize_by_max(candidate_powers)
    stability_scores = _normalize_by_max(stability_raw)

    freq_span = max(frequencies[search_indices[-1]] - frequencies[search_indices[0]], 1e-12)
    low_freq_scores = 1.0 - (frequencies[peak_indices] - frequencies[search_indices[0]]) / freq_span

    candidates = []
    for idx, power_value, power_score, stability_score, low_freq_score in zip(
        peak_indices, candidate_powers, power_scores, stability_scores, low_freq_scores
    ):
        candidates.append({
            'idx': int(idx),
            'freq': float(frequencies[idx]),
            'power': float(power_value),
            'power_score': float(power_score),
            'stability_score': float(stability_score),
            'low_freq_score': float(np.clip(low_freq_score, 0.0, 1.0)),
        })

    return candidates


def _select_initial_candidate(candidates, candidate_power_ratio=0.35,
                              harmonic_tolerance=0.08, score_margin=0.05):
    """
    Prefer a strong lower-frequency candidate when it has clear 2f support.

    The heuristic is:
    1. Find several strong local peaks in the initialization frames.
    2. Score them by strength, temporal stability, and low-frequency preference.
    3. If a lower-frequency peak has a matching 2f peak, treat it as a likely
       fundamental and prefer it over the stronger harmonic.
    """
    if not candidates:
        raise ValueError("No initialization candidates found in the search range.")

    for candidate in candidates:
        candidate['selection_score'] = (
            0.70 * candidate['power_score'] +
            0.20 * candidate['stability_score'] +
            0.10 * candidate['low_freq_score']
        )
        candidate['harmonic_support'] = 0.0

    strong_candidates = [
        candidate for candidate in candidates
        if candidate['power_score'] >= candidate_power_ratio
    ]
    if not strong_candidates:
        strong_candidates = [candidates[0]]

    harmonic_supported = []
    for candidate in strong_candidates:
        for other in candidates:
            if other['idx'] == candidate['idx']:
                continue

            harmonic_error = abs(other['freq'] - 2.0 * candidate['freq'])
            if harmonic_error <= harmonic_tolerance * candidate['freq']:
                candidate['harmonic_support'] = max(candidate['harmonic_support'], other['power_score'])

        if candidate['harmonic_support'] > 0:
            candidate['selection_score'] += 0.20 * candidate['harmonic_support']
            harmonic_supported.append(candidate)

    if harmonic_supported:
        best_score = max(candidate['selection_score'] for candidate in harmonic_supported)
        near_best = [
            candidate for candidate in harmonic_supported
            if candidate['selection_score'] >= best_score - score_margin
        ]
        return min(near_best, key=lambda candidate: candidate['freq'])

    best_score = max(candidate['selection_score'] for candidate in strong_candidates)
    near_best = [
        candidate for candidate in strong_candidates
        if candidate['selection_score'] >= best_score - score_margin
    ]
    return min(near_best, key=lambda candidate: candidate['freq'])


def find_initial_if(power_spectrum, frequencies, min_freq, max_freq,
                    num_candidates=8, peak_prominence_ratio=0.05,
                    candidate_power_ratio=0.35, harmonic_tolerance=0.08):
    """Find the initialization IF from several peak candidates in the search band."""
    freq_mask = (frequencies >= min_freq) & (frequencies <= max_freq)
    search_indices = np.where(freq_mask)[0]

    if len(search_indices) == 0:
        raise ValueError(f"Search range [{min_freq}, {max_freq}] Hz contains no valid bins")

    candidates = _extract_initial_candidates(
        power_spectrum, frequencies, search_indices,
        num_candidates=num_candidates,
        peak_prominence_ratio=peak_prominence_ratio
    )
    best_candidate = _select_initial_candidate(
        candidates,
        candidate_power_ratio=candidate_power_ratio,
        harmonic_tolerance=harmonic_tolerance
    )

    return best_candidate['freq'], best_candidate['idx']


def parabolic_interpolation(power_spectrum, peak_idx, frequencies):
    """Refine the peak frequency with a simple three-point parabola."""
    if peak_idx <= 0 or peak_idx >= len(power_spectrum) - 1:
        return frequencies[peak_idx]

    y_left = power_spectrum[peak_idx - 1]
    y_center = power_spectrum[peak_idx]
    y_right = power_spectrum[peak_idx + 1]

    denominator = 2 * (2 * y_center - y_left - y_right)
    if abs(denominator) < 1e-10:
        return frequencies[peak_idx]

    delta = np.clip((y_left - y_right) / denominator, -0.5, 0.5)
    freq_resolution = frequencies[1] - frequencies[0]
    return frequencies[peak_idx] + delta * freq_resolution


def _track_if_single_direction(power, f, start_idx, end_idx, step,
                               min_freq, max_freq, c1, c2,
                               adaptive_range, lambda_smooth, use_interpolation,
                               compute_penalty_func, strategy_params=None,
                               init_frames=10, init_num_candidates=8,
                               init_peak_prominence_ratio=0.05,
                               init_candidate_power_ratio=0.35,
                               init_harmonic_tolerance=0.08):
    """Track the IF path in a single direction."""
    if strategy_params is None:
        strategy_params = {}

    num_time_points = power.shape[1]
    if_path = np.zeros(num_time_points)

    if step > 0:
        init_end = min(start_idx + init_frames, num_time_points)
        init_power = power[:, start_idx:init_end]
    else:
        init_start = max(start_idx - init_frames + 1, 0)
        init_power = power[:, init_start:start_idx + 1]

    initial_if, _ = find_initial_if(
        init_power, f, min_freq, max_freq,
        num_candidates=init_num_candidates,
        peak_prominence_ratio=init_peak_prominence_ratio,
        candidate_power_ratio=init_candidate_power_ratio,
        harmonic_tolerance=init_harmonic_tolerance
    )
    if_path[start_idx] = initial_if

    for k in range(start_idx + step, end_idx + step, step):
        prev_if = if_path[k - step]

        if adaptive_range:
            search_min = max(c1 * prev_if, min_freq)
            search_max = min(c2 * prev_if, max_freq)
        else:
            search_min = min_freq
            search_max = max_freq

        search_indices = np.where((f >= search_min) & (f <= search_max))[0]
        if len(search_indices) == 0:
            if_path[k] = prev_if
            continue

        best_freq, _, _ = compute_penalty_func(
            power_spectrum=power[:, k],
            freq_indices=search_indices,
            frequencies=f,
            prev_if=prev_if,
            lambda_smooth=lambda_smooth,
            use_interpolation=use_interpolation,
            parabolic_interpolation_func=parabolic_interpolation,
            **strategy_params
        )

        best_freq = np.clip(best_freq, min_freq, max_freq)
        if_path[k] = best_freq

    return if_path


def _compute_path_smoothness(if_path):
    """Use the standard deviation of the first difference as a smoothness metric."""
    return np.std(np.diff(if_path))


def _fuse_bidirectional_paths(if_forward, if_backward, verbose=False):
    """Fuse forward and backward IF paths with a simple smoothness rule."""
    smoothness_fwd = _compute_path_smoothness(if_forward)
    smoothness_bwd = _compute_path_smoothness(if_backward)
    mean_diff = np.mean(np.abs(if_forward - if_backward))

    if verbose:
        print(f"  Forward smoothness: {smoothness_fwd:.4f}, backward smoothness: {smoothness_bwd:.4f}")
        print(f"  Mean path difference: {mean_diff:.4f} Hz")

    if mean_diff < 0.5:
        if_fused = (if_forward + if_backward) / 2
        fusion_method = "average"
    elif smoothness_fwd < smoothness_bwd * 0.8:
        if_fused = if_forward.copy()
        fusion_method = "forward"
    elif smoothness_bwd < smoothness_fwd * 0.8:
        if_fused = if_backward.copy()
        fusion_method = "backward"
    else:
        if_fused = np.minimum(if_forward, if_backward)
        fusion_method = "pointwise_min"

    if verbose:
        print(f"  Fusion method: {fusion_method}")

    return if_fused


def entropy_based_lps(signal_data, fs, nperseg=131072, noverlap=None,
                      min_freq=5, max_freq=50, c1=0.9, c2=1.1,
                      adaptive_range=True, lambda_smooth=0.0,
                      use_interpolation=True, bidirectional=True, verbose=True,
                      strategy='baseline', strategy_params=None,
                      init_num_candidates=8, init_peak_prominence_ratio=0.05,
                      init_candidate_power_ratio=0.35,
                      init_harmonic_tolerance=0.08):
    """Estimate the IF ridge with LPS on the STFT magnitude."""
    compute_penalty = get_strategy(strategy)
    if strategy_params is None:
        strategy_params = {}

    if verbose:
        print(f"LPS: fs={fs}Hz, nperseg={nperseg}, freq=[{min_freq},{max_freq}]Hz")
        print(f"  Strategy: {strategy}")

    f, t, Zxx = compute_stft(signal_data, fs, nperseg=nperseg, noverlap=noverlap)
    power = np.abs(Zxx) ** 2
    num_time_points = len(t)

    if verbose:
        print(f"  Time frames: {num_time_points}, freq resolution: {f[1] - f[0]:.4f}Hz")

    if_forward = _track_if_single_direction(
        power, f, 0, num_time_points - 1, +1,
        min_freq, max_freq, c1, c2, adaptive_range, lambda_smooth, use_interpolation,
        compute_penalty, strategy_params,
        init_num_candidates=init_num_candidates,
        init_peak_prominence_ratio=init_peak_prominence_ratio,
        init_candidate_power_ratio=init_candidate_power_ratio,
        init_harmonic_tolerance=init_harmonic_tolerance
    )

    if not bidirectional:
        if_estimated = if_forward
    else:
        if_backward = _track_if_single_direction(
            power, f, num_time_points - 1, 0, -1,
            min_freq, max_freq, c1, c2, adaptive_range, lambda_smooth, use_interpolation,
            compute_penalty, strategy_params,
            init_num_candidates=init_num_candidates,
            init_peak_prominence_ratio=init_peak_prominence_ratio,
            init_candidate_power_ratio=init_candidate_power_ratio,
            init_harmonic_tolerance=init_harmonic_tolerance
        )
        if_estimated = _fuse_bidirectional_paths(if_forward, if_backward, verbose=verbose)

    if verbose:
        print(f"  Result: mean IF={np.mean(if_estimated):.2f}Hz, RPM={np.mean(if_estimated) * 60:.1f}")

    return t, if_estimated, f, Zxx


def convert_if_to_rpm(if_array):
    """Convert instantaneous frequency in Hz to RPM."""
    return if_array * 60
