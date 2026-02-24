import numpy as np
import spikeinterface as si
from spikeinterface import preprocessing


def CatGT(recording):
    recording = si.preprocessing.phase_shift(recording)
    recording = si.preprocessing.common_reference(
        recording, operator="median", reference="global"
    )
    return recording


def IBLdestriping(recording):
    # From International Brain Laboratory. “Spike sorting pipeline for the International Brain Laboratory”. 4 May 2022. 9 Jun 2022.
    recording = si.preprocessing.highpass_filter(recording, freq_min=400.0)
    bad_channel_ids, channel_labels = si.preprocessing.detect_bad_channels(recording)
    # For IBL destriping interpolate bad channels
    recording = si.preprocessing.interpolate_bad_channels(bad_channel_ids)
    recording = si.preprocessing.phase_shift(recording)
    # For IBL destriping use highpass_spatial_filter used instead of common reference
    recording = si.preprocessing.highpass_spatial_filter(
        recording, operator="median", reference="global"
    )
    return recording


def IBLdestriping_modified(recording):
    # From SpikeInterface Implementation (https://spikeinterface.readthedocs.io/en/latest/how_to/analyse_neuropixels.html)
    recording = si.preprocessing.highpass_filter(recording, freq_min=400.0)
    bad_channel_ids, channel_labels = si.preprocessing.detect_bad_channels(recording)
    # For IBL destriping interpolate bad channels
    recording = recording.remove_channels(bad_channel_ids)
    recording = si.preprocessing.phase_shift(recording)
    recording = si.preprocessing.common_reference(
        recording, operator="median", reference="global"
    )
    return recording


def NienborgLab_preproc(recording):
    """Preprocessing pipeline for 32chn ephys data from Trellis."""
    recording = si.preprocessing.bandpass_filter(
        recording=recording, freq_min=300, freq_max=6000
    )
    recording = si.preprocessing.common_reference(
        recording=recording, operator="median"
    )
    return recording


def MBA_infer_map(recording, **infer_map_kwargs):
    """Preprocessing pipeline to infer channel map for microwire brush array data"""
    from element_array_ephys.spike_sorting.infer_map import infer_map

    recording = si.preprocessing.bandpass_filter(
        recording=recording, freq_min=300, freq_max=6000
    )
    if recording.get_num_channels() <= 32:
        recording = si.preprocessing.common_reference(
            recording=recording, operator="median"
        )
    else:
        # do common average referencing on each group of 32 channels
        group_ids = np.arange(recording.get_num_channels(), dtype=int) // 32
        recording.set_property("group", group_ids)
        split_recording_dict = recording.split_by("group")
        split_recording_dict = si.preprocessing.common_reference(split_recording_dict)
        recording = si.aggregate_channels(split_recording_dict)

    fs = recording.get_sampling_frequency()
    if recording.get_duration() > 120:
        # extract second minute of recording (arbitrary)
        start_frame = 60 * fs
        end_frame = 120 * fs
    else:
        # extract first minute or whole recording
        start_frame = 0
        end_frame = int(min(60, recording.get_duration()) * fs)

    signal = recording.get_traces(start_frame=start_frame, end_frame=end_frame).astype(
        np.float32
    )

    channel_map = infer_map(signal, **infer_map_kwargs)

    # modify electrode positions within SI object
    # TODO: eventually figure out a better way to do this
    si_probe = recording.get_probe()
    assert (
        channel_map.shape == si_probe.contact_positions.shape
    ), f"Inferred coordinates dimensions: {channel_map.shape} do not match target dimensions: {si_probe.contact_positions.shape}"
    si_probe.set_contacts(positions=channel_map)
    recording.set_probe(si_probe, in_place=True)

    return recording
