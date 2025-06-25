"""
Shared radar parameter utilities for mmWave to Soli-format pipeline.
"""
from typing import Dict
import sys
import logging

# Constants (should match those in mmwave_to_soli.py)
MAX_TX = 3
MAX_RX = 4
logger = logging.getLogger(__name__)

def calculate_derived_parameters(params: Dict) -> Dict:
    """
    Calculate derived radar parameters from parsed configuration.
    Args:
        params: Dictionary containing parsed radar parameters
    Returns:
        Dictionary containing derived radar parameters
    """
    derived = {}
    derived['sampling_rate_msps'] = params['profile']['sample_rate_msps']
    derived['adc_samples_per_chirp'] = params['profile']['adc_samples']
    if 'freq_slope_mhz_us' in params['chirp']:
        derived['chirp_slope_mhz_us'] = params['chirp']['freq_slope_mhz_us']
    elif 'freq_slope_mhz_us' in params['profile']:
        derived['chirp_slope_mhz_us'] = params['profile']['freq_slope_mhz_us']
    else:
        logger.error("Error: freq_slope_mhz_us not found in config. Check your profileCfg and chirpCfg lines.")
        sys.exit(1)
    derived['num_tx'] = params['chirp']['tx0_enable'] + params['chirp']['tx1_enable'] + params['chirp']['tx2_enable']
    derived['num_rx'] = MAX_RX
    c = 299792458
    ramp_time_s = params['profile']['ramp_end_time_us'] * 1e-6
    chirp_slope_mhz_us = derived['chirp_slope_mhz_us']
    chirp_slope_hz_s = chirp_slope_mhz_us * 1e12
    bandwidth_hz = chirp_slope_hz_s * ramp_time_s
    if bandwidth_hz == 0:
        logger.error(f"Error: Calculated bandwidth is zero. Check your configuration file for valid 'ramp_end_time_us' and 'freq_slope_mhz_us'. Values: ramp_end_time_us={params['profile']['ramp_end_time_us']}, freq_slope_mhz_us={chirp_slope_mhz_us}")
        sys.exit(1)
    derived['range_resolution_mm'] = (c / (2 * bandwidth_hz)) * 1000
    derived['max_range_mm'] = (derived['sampling_rate_msps'] * 1e6 * c) / (2 * chirp_slope_hz_s) * 1000
    chirp_time_s = (params['profile']['idle_time_us'] + params['profile']['ramp_end_time_us']) * 1e-6
    derived['prf_hz'] = 1 / chirp_time_s
    derived['frame_period_ms'] = params['frame']['periodicity_ms']
    derived['fps'] = 1000 / derived['frame_period_ms']
    chirps_per_loop = params['frame']['chirp_end_idx'] - params['frame']['chirp_start_idx'] + 1
    frame_source_line = params['frame'].get('frame_source_line', 'N/A')
    if params['frame']['num_loops'] <= 0:
        logger.error(
            "Error: num_loops in frameCfg is {}, but must be > 0.\n  frameCfg source line: {}\n  Parsed: chirp_start_idx={}, chirp_end_idx={}, num_frames={}, num_loops={}, periodicity_ms={}\n  chirps_per_loop={}".format(
                params['frame'].get('num_loops', 'N/A'),
                frame_source_line,
                params['frame'].get('chirp_start_idx', 'N/A'),
                params['frame'].get('chirp_end_idx', 'N/A'),
                params['frame'].get('num_frames', 'N/A'),
                params['frame'].get('num_loops', 'N/A'),
                params['frame'].get('periodicity_ms', 'N/A'),
                chirps_per_loop
            )
        )
        sys.exit(1)
    derived['chirps_per_frame'] = chirps_per_loop * params['frame']['num_loops']
    if derived['chirps_per_frame'] <= 0:
        logger.error(
            "Error: chirps_per_frame is {}, but must be > 0.\n  frameCfg source line: {}\n  Parsed: chirp_start_idx={}, chirp_end_idx={}, num_frames={}, num_loops={}, periodicity_ms={}\n  chirps_per_loop={}".format(
                derived['chirps_per_frame'],
                frame_source_line,
                params['frame'].get('chirp_start_idx', 'N/A'),
                params['frame'].get('chirp_end_idx', 'N/A'),
                params['frame'].get('num_frames', 'N/A'),
                params['frame'].get('num_loops', 'N/A'),
                params['frame'].get('periodicity_ms', 'N/A'),
                chirps_per_loop
            )
        )
        sys.exit(1)
    derived['expected_raw_size_bytes'] = (
        params['frame']['num_frames'] *
        derived['chirps_per_frame'] *
        derived['num_rx'] *
        derived['adc_samples_per_chirp'] *
         2 * 2  # 2 (I/Q) * 2 bytes (16-bit)
    )
    return derived
