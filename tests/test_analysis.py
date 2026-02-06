import pytest
import numpy as np
import mne
from core import Study, Subject, ConditionBlob, ConditionView
from core_types import ConditionProperties
from power_specra_analyzable import PowerSpectcraAnalyzable

def create_mock_info(sfreq=100.0, ch_names=None):
    if ch_names is None:
        ch_names = ["T5", "T6", "P3", "P4"]
    info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types='eeg')
    return info

def test_snr_at_target_auto_implementation():
    # Setup a mock ConditionView
    # Use larger F to avoid edge effects in into_SNR
    # sfreq=100, F=101 -> window_size=200. Freqs: [0, 0.5, 1, ..., 50]
    # Target 5Hz is at index 10.
    props = ConditionProperties(target_frequency=np.float64(5.0), carrier_frequency=np.float64(10.0))
    info = create_mock_info(sfreq=100.0)
    
    data = np.zeros((2, 3, 4, 1, 101), dtype=np.complex64)
    
    # Power at target (index 10)
    data[0, :, :, :, 10] = np.sqrt(10.0) 
    data[1, :, :, :, 10] = np.sqrt(20.0) 
    
    # Set all other frequencies to 1.0 to ensure noise=1.0
    # but keep target at its value
    data[:, :, :, :, :] = 1.0
    data[0, :, :, :, 10] = np.sqrt(10.0)
    data[1, :, :, :, 10] = np.sqrt(20.0)
        
    subjects = [Subject(0, "S0"), Subject(1, "S1")]
    blob = ConditionBlob(data, props, info, subjects)
    view = ConditionView(blob)
    
    # Check inheritance
    assert isinstance(view, PowerSpectcraAnalyzable)
    
    # Target 5Hz should be at index 10
    freqs = view.frequencies()
    assert freqs[10] == 5.0
    
    mean, sem_val = view.snr_at_target()
    
    # We expect 15.0 and 5.0
    assert mean == pytest.approx(15.0)
    assert sem_val == pytest.approx(5.0)

    # Test Power API
    p_mean, p_sem = view.power_at_target()
    # In Subject 0, power was 10. In Subject 1, power was 20.
    # Mean = 15, SEM = 5
    assert p_mean == pytest.approx(15.0)
    assert p_sem == pytest.approx(5.0)

    # Test Carrier API
    # Carrier is at index 20 (10Hz)
    data[0, :, :, :, 20] = np.sqrt(100.0)
    data[1, :, :, :, 20] = np.sqrt(200.0)
    # Refresh blob/view
    blob_c = ConditionBlob(data, props, info, subjects)
    view_c = ConditionView(blob_c)
    
    snr_c_mean, snr_c_sem = view_c.snr_at_carrier()
    assert snr_c_mean == pytest.approx(150.0) # (100+200)/2 assuming noise=1
    assert snr_c_sem == pytest.approx(50.0)

def test_subject_level_sem_vs_trial_level():
    # Verify that SEM is calculated across subjects and NOT across trials
    props = ConditionProperties(target_frequency=np.float64(5.0), carrier_frequency=np.float64(10.0))
    info = create_mock_info(sfreq=100.0)
    
    # S=2, T=10, E=1, W=1, F=101
    data = np.zeros((2, 10, 1, 1, 101), dtype=np.complex64)
    data[:, :, :, :, :] = 1.0
    data[0, :, :, :, 10] = np.sqrt(10.0)
    data[1, :, :, :, 10] = np.sqrt(20.0)
    
    subjects = [Subject(0, "S0"), Subject(1, "S1")]
    blob = ConditionBlob(data, props, info, subjects)
    view = ConditionView(blob)
    
    mean, sem_val = view.snr_at_target()
    
    assert mean == pytest.approx(15.0)
    assert sem_val == pytest.approx(5.0)
    
    # If it was across trials (20 trials), SEM would be different.
    assert sem_val != pytest.approx(1.17669, rel=1e-3)

def test_study_conditions_and_get_group_view():
    def mock_loader():
        info = create_mock_info()
        data = np.zeros((3, 4, 1, 11), dtype=np.complex64)
        
        yield "S1", ConditionProperties(np.float64(5.0), np.float64(10.0)), info, data
        yield "S1", ConditionProperties(np.float64(10.0), np.float64(20.0)), info, data
        yield "S2", ConditionProperties(np.float64(5.0), np.float64(10.0)), info, data
        yield "S2", ConditionProperties(np.float64(10.0), np.float64(20.0)), info, data

    study = Study(mock_loader())
    
    conditions = list(study.conditions())
    assert len(conditions) == 2
    assert conditions[0].target_frequency == 5.0
    assert conditions[1].target_frequency == 10.0
    
    view = study.get_group_view(conditions[0])
    assert view.data.shape[0] == 2 # 2 subjects
