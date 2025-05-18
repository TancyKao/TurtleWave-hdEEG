"""
Test example for ParalEvents class with parallel processing of multi-channel EEG data.
This script demonstrates spindle detection and analysis across multiple channels.
"""

import os
import time
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from concurrent.futures import ProcessPoolExecutor

# Import necessary modules - adjust paths as needed
from wonambi.dataset import Dataset as WonambiDataset
from wonambi.attr import Annotations as WonambiAnnotations
from wonambi_hdEEG import ParalEvents, CustomAnnotations

# Set up logging
import logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("SpindleTest")

def run_spindle_detection_test(data_file, annot_file=None, output_dir=None):
    """
    Run a complete test of spindle detection and analysis.
    
    Parameters
    ----------
    data_file : str
        Path to the EEG data file
    annot_file : str or None
        Path to the annotation file, if any
    output_dir : str or None
        Directory to save results, defaults to current directory
    """
    # Configure output directory
    if output_dir is None:
        output_dir = os.path.dirname(data_file)
    os.makedirs(output_dir, exist_ok=True)
    
    # Start timing
    total_start_time = time.time()
    
    # Step 1: Load dataset and annotations
    logger.info(f"Loading dataset: {data_file}")
    try:
        data = WonambiDataset(data_file)
        logger.info(f"Dataset loaded successfully")
        
        # Log basic dataset info
        if hasattr(data, 'header'):
            logger.info(f"Sampling rate: {data.header.get('s_freq', 'unknown')} Hz")
            logger.info(f"Number of channels: {len(data.header.get('chan_name', []))}")
            logger.info(f"Duration: {data.header.get('n_samples', 0) / data.header.get('s_freq', 1):.2f} seconds")
    except Exception as e:
        logger.error(f"Error loading dataset: {e}")
        return None
    
    # Load annotations if available
    #from wonambi import Dataset
    
    
    #annot = WonambiAnnotations(annot_file)
    annot = CustomAnnotations(annot_file)

    # Step 2: Create ParalEvents instance
    logger.info("Creating event processor")
    event_processor = ParalEvents(dataset=data, annotations=annot)
    
    # Step 3: Select channels to analyze
    # For testing, we'll limit to a small set of central channels
    all_channels = data.header.get('chan_name', [])
    logger.info(f"Available channels: {len(all_channels)}")
    
    # Select a subset of channels for testing (adjust based on your data)
    # For qEEG: standard sleep channels
    psg_channels = ['C3', 'C4', 'F3', 'F4', 'O1', 'O2']
    # For EEG: focus on central channels
    eeg_channels = ['E101']
    
    # Find available channels in the data
    test_channels = []
    for channel in psg_channels + eeg_channels:
        if channel in all_channels:
            test_channels.append(channel)

    
    logger.info(f"Selected {len(test_channels)} channels for analysis: {test_channels}")


    # Step 4: Run spindle detection with parallel processing
    logger.info("Starting spindle detection")
    try:
        # Define concatenation settings
        cat = (1, 1, 1, 0)  # concatenate within and between stages, cycles separate
        logger.info(f"Using concatenation settings: {cat} (cycle, stage, discontinuous, event)")
        # Define stages to analyze
        stages = ['NREM2']  # Default to NREM2 for spindle detection
        logger.info(f"Analyzing sleep stages: {stages}")

        spindle_results = event_processor.detect_spindles_multichannel(
        method='Ferrarelli2007',  # Commonly used method
        channels=test_channels,
        frequency=(11, 16),       # Standard spindle frequency range
        duration=(0.5, 3),        # Standard spindle duration range (0.5-3 seconds)
        reject_artifacts=True,
        reject_arousals=True,
        n_workers=1,#min(4, os.cpu_count() - 1),  # Use at most 4 workers
        chunk_size=1)              # Process 3 channels per worker)
        
        logger.info(f"Spindle detection complete. Total spindles: {spindle_results['total']}")
        
        # Save detection results to CSV
        detection_file = os.path.join(output_dir, "spindle_detection_results.csv")
        detection_df = pd.DataFrame({
            'Channel': list(spindle_results['by_channel'].keys()),
            'Spindle_Count': list(spindle_results['by_channel'].values())
        })
        detection_df.to_csv(detection_file, index=False)
        logger.info(f"Saved detection results to {detection_file}")
        
    except Exception as e:
        logger.error(f"Error in spindle detection: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return None
    
    # Step 5: Analyze detected spindles
    logger.info("Starting spindle analysis")
    try:
        analysis_results = event_processor.analyze_spindles(
            spindle_results=spindle_results,
            window_size=5,            # 5 seconds around each spindle
            n_workers=min(8, os.cpu_count() - 1),  # Use more workers for analysis
            max_spindles=5000         # Limit for very large datasets
        )
        
        logger.info(f"Analysis complete. Analyzed {analysis_results['overall']['analyzed_spindles']} spindles")
        
        # Save analysis results to CSV
        channel_stats = []
        for channel, stats in analysis_results['analysis'].items():
            if stats['count'] > 0:
                channel_stats.append({
                    'Channel': channel,
                    'Spindle_Count': stats['count'],
                    'Frequency_Mean': stats.get('frequency_mean'),
                    'Frequency_Std': stats.get('frequency_std'),
                    'Duration_Mean': stats.get('duration_mean'),
                    'Duration_Std': stats.get('duration_std'),
                    'Amplitude_Mean': stats.get('amplitude_mean'),
                    'Amplitude_Std': stats.get('amplitude_std'),
                    'Power_Mean': stats.get('power_mean'),
                    'Power_Std': stats.get('power_std')
                })
        
        if channel_stats:
            analysis_file = os.path.join(output_dir, "spindle_analysis_results.csv")
            analysis_df = pd.DataFrame(channel_stats)
            analysis_df.to_csv(analysis_file, index=False)
            logger.info(f"Saved analysis results to {analysis_file}")
            
            # Generate summary plots
            try:
                plot_file = os.path.join(output_dir, "spindle_analysis_summary.png")
                
                fig, axes = plt.subplots(2, 2, figsize=(12, 10))
                
                # Plot 1: Spindle counts by channel
                axes[0, 0].bar(analysis_df['Channel'], analysis_df['Spindle_Count'])
                axes[0, 0].set_title('Spindle Count by Channel')
                axes[0, 0].set_ylabel('Count')
                axes[0, 0].set_xticklabels(analysis_df['Channel'], rotation=45)
                
                # Plot 2: Spindle frequency by channel
                if 'Frequency_Mean' in analysis_df:
                    axes[0, 1].bar(analysis_df['Channel'], analysis_df['Frequency_Mean'])
                    axes[0, 1].set_title('Spindle Frequency by Channel')
                    axes[0, 1].set_ylabel('Frequency (Hz)')
                    axes[0, 1].set_xticklabels(analysis_df['Channel'], rotation=45)
                    
                    # Add error bars if available
                    if 'Frequency_Std' in analysis_df:
                        axes[0, 1].errorbar(
                            x=range(len(analysis_df)), 
                            y=analysis_df['Frequency_Mean'],
                            yerr=analysis_df['Frequency_Std'],
                            fmt='none', 
                            ecolor='black', 
                            capsize=5
                        )
                
                # Plot 3: Spindle duration by channel
                if 'Duration_Mean' in analysis_df:
                    # Convert to milliseconds for easier reading
                    durations = analysis_df['Duration_Mean'] * 1000
                    axes[1, 0].bar(analysis_df['Channel'], durations)
                    axes[1, 0].set_title('Spindle Duration by Channel')
                    axes[1, 0].set_ylabel('Duration (ms)')
                    axes[1, 0].set_xticklabels(analysis_df['Channel'], rotation=45)
                    
                    # Add error bars if available
                    if 'Duration_Std' in analysis_df:
                        axes[1, 0].errorbar(
                            x=range(len(analysis_df)), 
                            y=durations,
                            yerr=analysis_df['Duration_Std'] * 1000,
                            fmt='none', 
                            ecolor='black', 
                            capsize=5
                        )
                
                # Plot 4: Spindle amplitude/power by channel
                if 'Power_Mean' in analysis_df:
                    axes[1, 1].bar(analysis_df['Channel'], analysis_df['Power_Mean'])
                    axes[1, 1].set_title('Spindle Power by Channel')
                    axes[1, 1].set_ylabel('Power')
                    axes[1, 1].set_xticklabels(analysis_df['Channel'], rotation=45)
                    
                    # Add error bars if available
                    if 'Power_Std' in analysis_df:
                        axes[1, 1].errorbar(
                            x=range(len(analysis_df)), 
                            y=analysis_df['Power_Mean'],
                            yerr=analysis_df['Power_Std'],
                            fmt='none', 
                            ecolor='black', 
                            capsize=5
                        )
                
                plt.tight_layout()
                fig.savefig(plot_file)
                plt.close(fig)
                logger.info(f"Saved summary plots to {plot_file}")
                
            except Exception as e:
                logger.error(f"Error generating plots: {e}")
        
    except Exception as e:
        logger.error(f"Error in spindle analysis: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return None
    
    # Step 6: Extract and save individual spindle data (optional for detailed analysis)
    try:
        # Extract a sample of spindles for validation
        spindle_samples = []
        
        for channel in test_channels:
            if channel in analysis_results['results'] and analysis_results['results'][channel]:
                # Get up to 5 spindles from each channel
                channel_results = analysis_results['results'][channel][:5]
                
                for result in channel_results:
                    if 'data' in result:
                        # Save key spindle info
                        spindle = {
                            'channel': channel,
                            'start_time': result['start_time'],
                            'end_time': result['end_time']
                        }
                        
                        if 'analysis' in result:
                            spindle.update(result['analysis'])
                            
                        spindle_samples.append(spindle)
        
        # Save spindle samples to CSV
        if spindle_samples:
            samples_file = os.path.join(output_dir, "spindle_samples.csv")
            samples_df = pd.DataFrame(spindle_samples)
            samples_df.to_csv(samples_file, index=False)
            logger.info(f"Saved {len(spindle_samples)} spindle samples to {samples_file}")
        
    except Exception as e:
        logger.error(f"Error extracting spindle samples: {e}")
    
    # Log total runtime
    total_time = time.time() - total_start_time
    logger.info(f"Complete test finished in {total_time:.2f} seconds")
    
    return {
        'detection': spindle_results,
        'analysis': analysis_results,
        'runtime': total_time
    }


def test_performance_comparison(data_file, annot_file=None):
    """
    Compare performance between sequential and parallel processing.
    
    Parameters
    ----------
    data_file : str
        Path to the EEG data file
    annot_file : str or None
        Path to the annotation file, if any
    """
    logger.info("=== PERFORMANCE COMPARISON TEST ===")
    
    # Load dataset and annotations
    logger.info(f"Loading dataset: {data_file}")
    data = LargeDataset(data_file)
    
    annotations = None
    if annot_file and os.path.exists(annot_file):
        logger.info(f"Loading annotations: {annot_file}")
        annotations = XLAnnotations(data, annot_file)
    
    # Create ParalEvents instance
    event_processor = ParalEvents(dataset=data, annotations=annotations)
    
    # Select channels for testing
    all_channels = data.header.get('chan_name', [])
    test_channels = all_channels[:min(10, len(all_channels))]
    
    # Test 1: Sequential processing (1 worker)
    logger.info("Running sequential processing test (1 worker)")
    sequential_start = time.time()
    seq_results = event_processor.detect_spindles_multichannel(
        method='Ferrarelli2007',
        channels=test_channels[:3],  # Use fewer channels for quick test
        frequency=(11, 16),
        duration=(0.5, 3),
        n_workers=1,
        chunk_size=1
    )
    sequential_time = time.time() - sequential_start
    logger.info(f"Sequential processing took {sequential_time:.2f} seconds")
    
    # Test 2: Parallel processing (max workers)
    logger.info("Running parallel processing test (max workers)")
    max_workers = min(8, os.cpu_count() - 1)
    parallel_start = time.time()
    par_results = event_processor.detect_spindles_multichannel(
        method='Ferrarelli2007',
        channels=test_channels[:3],  # Same channels for fair comparison
        frequency=(11, 16),
        duration=(0.5, 3),
        n_workers=max_workers,
        chunk_size=1
    )
    parallel_time = time.time() - parallel_start
    logger.info(f"Parallel processing ({max_workers} workers) took {parallel_time:.2f} seconds")
    
    # Calculate speedup
    if sequential_time > 0:
        speedup = sequential_time / parallel_time
        logger.info(f"Speedup factor: {speedup:.2f}x")
    
    # Test 3: Analyze a set of spindles with different worker counts
    if seq_results['total'] > 0:
        logger.info("Testing analysis phase with different worker counts")
        
        # Generate test data once
        test_spindle_results = seq_results
        
        worker_counts = [1, 2, 4, min(8, os.cpu_count() - 1)]
        analysis_times = []
        
        for workers in worker_counts:
            logger.info(f"Testing with {workers} workers")
            start_time = time.time()
            event_processor.analyze_spindles(
                spindle_results=test_spindle_results,
                window_size=5,
                n_workers=workers,
                max_spindles=100  # Limit for quick testing
            )
            elapsed = time.time() - start_time
            analysis_times.append(elapsed)
            logger.info(f"Analysis with {workers} workers took {elapsed:.2f} seconds")
        
        # Report speedups
        base_time = analysis_times[0]  # Sequential time
        for i, workers in enumerate(worker_counts[1:], 1):
            speedup = base_time / analysis_times[i]
            logger.info(f"Speedup with {workers} workers: {speedup:.2f}x")
    
    return {
        'sequential_time': sequential_time,
        'parallel_time': parallel_time,
        'speedup': sequential_time / parallel_time if sequential_time > 0 else 0,
        'worker_count': max_workers
    }


if __name__ == "__main__":
    # Example usage with your data
    data_file = "/Users/tancykao/Dropbox/05_Woolcock_DS/AnalyzeTools/Coupling_python/CFC_080920/individual/01js/ses-1/sub-001js_ses-1_task-psg_run-1_desc-avg1_eeg.set"
    annot_file = "/Users/tancykao/Dropbox/05_Woolcock_DS/AnalyzeTools/Coupling_python/CFC_080920/individual/01js/ses-1/sub-001js_ses-1_task-psg_run-1_desc-avg1_eeg.xml"
    
    # Create output directory based on subject ID
    subject_id = os.path.basename(data_file).split('_')[0]
    output_dir = f"spindle_results_{subject_id}"
    
    # Run the complete test
    logger.info("=== STARTING COMPLETE SPINDLE DETECTION TEST ===")
    results = run_spindle_detection_test(data_file, annot_file, output_dir)
    
    # Run performance comparison if time permits
    # test_performance_comparison(data_file, annot_file)
    
    logger.info("=== TEST COMPLETE ===")



    # Now add spindle detection
# Example 1: Basic spindle detection
# spindles = annotations.detect_spindles(
#     method='Ferrarelli2007',  # Choose your preferred method
#     channels=['Cz', 'C3', 'C4'],  # Central channels are good for spindles
#     frequency=(11, 16),  # Standard spindle frequency range
#     duration=(0.5, 3), # Min and max spindle duration in seconds
#     stage=[2]    
# )