"""Module for plotting Power Spectral Density (PSD) comparisons and BPM time
series with user controls. This module is called inside the measure() function.
Use green_avg_psd_plot.py as an example of how to call this module.
"""

import numpy as np
import matplotlib.pyplot as plt


class PlotState:
    """Class to hold plotting state. This allows variables to be modified
    within matplotlib event handler functions."""
    def __init__(self):
        self.continue_plotting = True
        self.skip_acquisition = False


def on_key(event, plotting_state: PlotState):
    """Handle key press events for the plots. Default matplotlib keys still 
    work.
    
    Keys:
        a: Skip remaining acquisition period plots  
        x/Escape: Stop all further plots for current video

    Args:
        event: Matplotlib key press event
        plotting_state (PlotState): Object holding the plotting state
    Returns:
        None
    """
    if event.key == 'a' and not plotting_state.skip_acquisition:
        print("       Skipping acquisition period plots...")
        plotting_state.skip_acquisition = True
        plt.close(event.canvas.figure)
    elif event.key in ['x', 'escape']:
        print("       Stopping all further plots for current video...")
        plotting_state.continue_plotting = False
        plt.close(event.canvas.figure)


def psd_plot(bpm_timeseries: np.ndarray, 
                 signal_comparison_data: list, test_name: str,
                 state: PlotState, frame_idx: int, acquisition_time: float):
    """Plots the Power Spectral Density (PSD) vs. frequency for different 
    stages of the signal processing and the time series of the output's BPM
    estimates. The PSD values have been offset by 1e-2 for visualising any
    overlapping plots.

    Args:
        bpm_timeseries (np.ndarray): 2D array of the output heart rate with 
            columns [timestamps, bpm_series].
        signal_comparison_data (list): List of signal data to compare. The list
            contains np.nan values during the acquisition period, otherwise it
            also contains nested dictionaries for the signal stages with
            their BPM estimates and PSD frequency data. Dictionary structure:
                {signal_stage_name: {'bpm': float or np.nan,
                                    'psd_data': np.ndarray or np.nan}}
                where psd_data is a 2D array with columns [freqs, magnitudes] 
                or np.nan if that signalstage's estimation failed.
        test_name (str): Name of the video-degradation-degradation_level being 
            processed. 
        state (PlotState): Object holding the plotting state (allows user 
            controls).
        frame_idx (int): Current index in the time series
        acquisition_time (float): the time when acquisition period ends in
            seconds. Acquisition period is the initial period where the
            algorithm is stabilising and no valid bpm estimates are produced.
        

    Returns:
        None
    """
    ts = bpm_timeseries[:, 0]
    bpm_series = bpm_timeseries[:, 1]

    fig, ax = plt.subplots(2, 1, figsize=(10, 6))
    fig.canvas.mpl_connect('key_press_event', lambda e: on_key(e, state))
    fig.canvas.manager.set_window_title(f'{test_name} at frame {frame_idx} '
                                        f'({ts[frame_idx]:.2f}s)')

    # get the signal comparison data for the current frame
    current_frame_data = signal_comparison_data[frame_idx]

    # ============= First plot: PSD vs. frequency at current time =============
    has_legend_items = False

    # First, check if we have data (not in acquisition period)
    if isinstance(current_frame_data, float):  # np.nan case
        # Show acquisition period message
        ax[0].text(0.5, 0.5, 'Acquisition Period\n(No PSD data available)', 
                   transform=ax[0].transAxes, ha='center', va='center',
                   fontsize=14, color='gray', style='italic')
    else:
        # Plot PSD data for each signal stage
        plotted_any = False
        for i, (signal_stage, data) in enumerate(current_frame_data.items()):
            # Skip if estimation failed for this method
            if np.isnan(data['bpm']) or np.isnan(data['psd_data']).any():
                print(f"Warning: {signal_stage} estimation failed for frame "
                      f"{frame_idx}")
                continue

            # Extract frequencies and magnitudes from psd_data
            freqs = data['psd_data'][:, 0]
            mags = data['psd_data'][:, 1]

            # Normalize magnitudes for better visual comparison
            mags = mags / np.max(mags)
            
            # Offset y-values slightly for visibility and plot
            offset = i * 1e-2
            ax[0].plot(freqs, mags + offset, alpha=0.7, linewidth=2,
                    label=f'{signal_stage} ({data["bpm"]:.1f} BPM)')
            
            plotted_any = True
            has_legend_items = True

        # Show message if no methods succeeded
        if not plotted_any:
            ax[0].text(0.5, 0.5, 'All estimation methods failed\nfor this frame', 
                    transform=ax[0].transAxes, ha='center', va='center',
                    fontsize=14, color='red', style='italic')
            
    # Add vertical line for BPM estimate returned by measurement function
    if not np.isnan(bpm_series[frame_idx]):
        ax[0].axvline(x=bpm_series[frame_idx]/60, color='gray', linestyle='--', 
                      label=f'output BPM estimate '
                      f'({bpm_series[frame_idx]:.1f} BPM)')
        has_legend_items = True


    # Secondary x-axis that converts frequency to BPM
    ax2 = ax[0].twiny()
    ax2.set_xlim(ax[0].get_xlim())
    ax2.set_xticks(np.arange(0, 3.6, 0.25))
    ax2.set_xticklabels([f"{int(tick*60)}" for tick in ax2.get_xticks()])
    ax2.set_xlabel('Heart Rate (BPM)')

    ax[0].set_title(f'Power Spectral Density Comparison at '
                    f'{ts[frame_idx]:.2f}s into Video')
    ax[0].set_xlabel('Frequency (Hz)')
    ax[0].set_ylabel('Normalised PSD')
    ax[0].set_xlim(0, 3.5)
    ax[0].grid(True, alpha=0.3)

    if has_legend_items:
        ax[0].legend(fontsize='small')
    # =========================================================================

   # =============== Second plot: Time series of BPM estimates ================
    # Plot the output BPM time series
    ax[1].plot(ts, bpm_series, label='Output BPM')

    # --------------- Plot each signal stage's BPM time series ----------------
    # # NOTE: This causes too much clutter if many stages are plotted
    #
    # # Find acquisition end index
    # try:
    #     acquisition_end_idx = next(
    #         i for i, frame_data in enumerate(signal_comparison_data) 
    #         if not isinstance(frame_data, float))
    # except StopIteration:
    #     acquisition_end_idx = None  # All entries are np.nan

    # # Plot all signal stage BPM values after acquisition period
    # if acquisition_end_idx is not None:
    #     post_acq_timestamps = ts[acquisition_end_idx:]
        
    #     # Get all signal stage data
    #     post_acq_data = signal_comparison_data[acquisition_end_idx:]
    #     signal_stages = list(post_acq_data[0].keys())

    #     for i, signal_stage in enumerate(signal_stages):
    #         # Extract all BPM values for this stage using list comprehension
    #         stage_bpm_series = np.array([frame[signal_stage]['bpm']
    #                                      for frame in post_acq_data])

    #         # Offset y-values slightly for visibility and plot
    #         offset = i * 1
    #         ax[1].plot(post_acq_timestamps, stage_bpm_series + offset,
    #                    alpha=0.5, linewidth=1, label=f'{signal_stage}')
    # ax[1].legend(fontsize='small')
    # -------------------------------------------------------------------------

    # Vertical line at current time with label above it
    ax[1].axvline(x=ts[frame_idx], color='gray', linestyle='--')
    ax[1].text(ts[frame_idx], ax[1].get_ylim()[1], f'{ts[frame_idx]:.2f}s',
               color='gray', ha='center', va='bottom')
    
    # Shade the acquisition period
    ax[1].axvspan(0, acquisition_time, alpha=0.1, color='gray')

    # Add "Acquisition Period" text at the center of the shaded region
    ax[1].text(acquisition_time / 2, 
               (ax[1].get_ylim()[0]+ax[1].get_ylim()[1])/2,
               'Acquisition Period', ha='center', va='center',
               color='darkgray')

    ax[1].set_xlabel("Time (s)")
    ax[1].set_ylabel("BPM")
    ax[1].set_title("Output BPM over Time", pad=15)
    ax[1].grid(True, alpha=0.3)
    # =========================================================================

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.15, hspace=0.75)

     # Add instruction text at the bottom of the figure
    fig.text(0.5, 0.02, 'Controls:  q=Next plot,  a=Skip acquisition '
             'period,  x=Exit all plots',
             ha='center', va='bottom', fontsize=10, color='gray',
             bbox=dict(boxstyle='round,pad=0.3', 
                       facecolor='lightgray', alpha=0.8))
    
    plt.show()
    return
