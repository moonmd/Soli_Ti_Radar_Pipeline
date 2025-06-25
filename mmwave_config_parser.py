"""
Shared mmWaveStudio configuration parser for both generator and consumer.
"""
import sys
import logging
from mmwave_shared import calculate_derived_parameters
import pprint
import json

# Ensure parse_config_file is available for import
__all__ = ['parse_config_file']

def parse_config_file(config_path):
    """
    Parse mmWaveStudio TXT configuration file to extract radar parameters.
    Args:
        config_path: Path to the configuration file
    Returns:
        Dictionary containing parsed radar parameters
    """
    # Use logger if available, else fallback to print
    logger = logging.getLogger(__name__)
    def log_error(msg):
        if logger.handlers:
            logger.error(msg)
        else:
            print(msg)

    params = {
        'profile': {},
        'chirp': {},
        'frame': {}
    }
    profile_count = 0
    chirp_count = 0
    with open(config_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('%'):
                continue
            if line.startswith('profileCfg'):
                params['profile']['_source_line_'] = line
                profile_count += 1
                if profile_count > 1:
                    print("EEEEEEEEEE   Profile Parameters   EEEEEEEEEE")
                    print(json.dumps(params, indent=4))
                    print("EEEEEEEEEE   Profile Parameters   EEEEEEEEEE")
                    log_error("Error: Multiple profileCfg blocks not supported.")
                    sys.exit(1)
                # Robust split: handle both whitespace and commas as delimiters
                parts = [p.strip() for p in line.replace(',', ' ').split()]
                if len(parts) < 18:
                    print("EEEEEEEEEE   Profile Parameters   EEEEEEEEEE")
                    print(json.dumps(params, indent=4))
                    print("EEEEEEEEEE   Profile Parameters   EEEEEEEEEE")
                    log_error(f"Error: profileCfg has too few arguments ({len(parts)}), expected 18\n  Source line: {line}\n  Parsed parts: {parts}")
                    sys.exit(1)
                try:
                    params['profile']['profile_id'] = int(parts[1])
                    params['profile']['start_freq_ghz'] = float(parts[2])
                    params['profile']['idle_time_us'] = float(parts[3])
                    params['profile']['adc_start_time_us'] = float(parts[4])
                    params['profile']['ramp_end_time_us'] = float(parts[5])
                    params['profile']['freq_slope_mhz_us'] = float(parts[12])
                    params['profile']['tx_start_time_us'] = float(parts[13])
                    params['profile']['adc_samples'] = int(parts[14])
                    # digOutSampleRate is in ksps, convert to Msps for consistency
                    params['profile']['sample_rate_msps'] = float(parts[15]) / 1000.0
                except Exception as e:
                    print("EEEEEEEEEE   Profile Parameters   EEEEEEEEEE")
                    print(json.dumps(params, indent=4))
                    print("EEEEEEEEEE   Profile Parameters   EEEEEEEEEE")
                    log_error(f"Error parsing profileCfg line: {line}\n  Parsed parts: {parts}\n  Exception: {e}\n  Profile params parsed so far: {params['profile']}")
                    sys.exit(1)
            elif line.startswith('chirpCfg'):
                params['chirp']['_source_line_'] = line
                chirp_count += 1
                if chirp_count > 1:
                    log_error("Error: Multiple chirpCfg blocks not supported.")
                    sys.exit(1)
                parts = [p.strip().rstrip(',') for p in line.split()]
                if len(parts) < 9:
                    log_error(f"Error: chirpCfg has too few arguments ({len(parts)}), expected at least 9")
                    sys.exit(1)
                try:
                    params['chirp']['start_idx'] = int(parts[1])
                    params['chirp']['end_idx'] = int(parts[2])
                    params['chirp']['profile_id'] = int(parts[3])
                    params['chirp']['start_freq_var'] = float(parts[4])
                    params['chirp']['freq_slope_var'] = float(parts[5])
                    params['chirp']['idle_time_var'] = float(parts[6])
                    params['chirp']['adc_start_time_var'] = float(parts[7])
                    params['chirp']['tx0_enable'] = int(parts[8])
                    params['chirp']['tx1_enable'] = int(parts[9])
                    params['chirp']['tx2_enable'] = int(parts[10])
                    # Copy freq_slope_mhz_us from profile for valid mmWaveStudio configs
                    if 'freq_slope_mhz_us' in params['profile']:
                        params['chirp']['freq_slope_mhz_us'] = params['profile']['freq_slope_mhz_us']
                except Exception as e:
                    print("EEEEEEEEEE   Profile Parameters   EEEEEEEEEE")
                    print(json.dumps(params, indent=4))
                    print("EEEEEEEEEE   Profile Parameters   EEEEEEEEEE")
                    log_error(f"Error parsing chirpCfg line: {line}\n  Parsed parts: {parts}\n  Exception: {e}")
                    sys.exit(1)
            elif line.startswith('frameCfg'):
                params['frame']['_source_line_'] = line
                parts = [p.strip().rstrip(',') for p in line.split()]
                expected_args = 6
                if len(parts) < expected_args:
                    print("EEEEEEEEEE   Profile Parameters   EEEEEEEEEE")
                    print(json.dumps(params, indent=4))
                    print("EEEEEEEEEE   Profile Parameters   EEEEEEEEEE")
                    log_error(f"Error: frameCfg has too few arguments ({len(parts)}), expected at least {expected_args}")
                    sys.exit(1)
                try:
                    params['frame']['chirp_start_idx'] = int(parts[1])
                    params['frame']['chirp_end_idx'] = int(parts[2])
                    params['frame']['num_frames'] = int(parts[3])
                    params['frame']['num_loops'] = int(parts[4])
                    params['frame']['periodicity_ms'] = float(parts[5]) if len(parts) > 5 else 33.33
                    params['frame']['trigger_delay_us'] = int(parts[6])
                    params['frame']['dummy_chirps'] = int(parts[7])
                    params['frame']['trigger_select'] = int(parts[8])
                except Exception as e:
                    log_error(f"Error parsing frameCfg line: {line}\n  Parsed parts: {parts}\n  Exception: {e}")
                    sys.exit(1)
    # Calculate chirps per frame
    chirps_per_loop = params['frame']['chirp_end_idx'] - params['frame']['chirp_start_idx'] + 1
    params['chirps_per_frame'] = chirps_per_loop * params['frame']['num_loops']

    # Calculate derived parameters if not already present
    try:
        if 'derived' not in params:
            params['derived'] = calculate_derived_parameters(params)

    except Exception as e:
        print("==== Parsed config parameters ==== DERIVED PARAMS EXCEPTION ERROR! ====")
        print(json.dumps(params, indent=4))
        print("==== END config parameters ==== DERIVED PARAMS EXCEPTION ERROR! ====")
        raise

    return params
