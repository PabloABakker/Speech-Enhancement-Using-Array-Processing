"""
Utility functions for the project. Made by Pablo Bakker and Jochem Groenenberg
"""

# Import libraries
import numpy as np
from scipy.signal import stft, istft
from pystoi import stoi
from pysiib import SIIB


# Create data using data model from class
def make_mixture(target_sig, interferer_sigs, h_target, h_inters,
                 ref_mic=0, snr_db=0.0, sensor_snr_db=40.0,
                 seed=0):
    """
    Build a 4-mic mixture from a target + point-source interferers.

    Parameters:
    target_sig      : 1-D array, the clean target speech
    interferer_sigs : list of 1-D arrays, one per interferer
    h_target        : (n_mics, n_taps) RIR for the target
    h_inters        : list of (n_mics, n_taps) RIRs, same length as interferer_sigs
    ref_mic         : reference microphone index (0-based)
    snr_db          : desired input SNR (target vs summed interferers) at ref mic
    sensor_snr_db   : SNR of target vs independent per-mic sensor noise at ref mic
    seed            : RNG seed for reproducible sensor noise

    Returns
    dict:
      mix           : (n_mics, N)  the microphone mixture x(t)
      target_image  : (n_mics, N)  clean target at the mics  (ground truth)
      noise_images  : (n_mics, N)  summed interferers + sensor noise
    """
    rng = np.random.default_rng(seed)
    n_mics = h_target.shape[0]

    # Trim signals to the same length
    L = min([len(target_sig)] + [len(s) for s in interferer_sigs])
    target_sig = target_sig[:L]
    interferer_sigs = [s[:L] for s in interferer_sigs]

    # Convolve a single signal through each mics RIR: (n_mics, L)
    def image(sig, H):
        out = np.stack([np.convolve(sig, H[m])[:L] for m in range(n_mics)])
        return out

    # Target image 
    target_image = image(target_sig, h_target)

    # Interferer images summed 
    noise_image = np.zeros((n_mics, L))
    for sig, H in zip(interferer_sigs, h_inters):
        noise_image += image(sig, H)

    # Scale acoustic noise to hit snr_db at the reference mic 
    P_target = np.mean(target_image[ref_mic] ** 2)
    P_noise  = np.mean(noise_image[ref_mic] ** 2)
    if P_noise > 0:
        alpha = np.sqrt(P_target / (P_noise * 10 ** (snr_db / 10)))
        noise_image *= alpha

    # Add independent sensor noise per mic 
    P_sensor = P_target / (10 ** (sensor_snr_db / 10))
    sensor = rng.standard_normal((n_mics, L)) * np.sqrt(P_sensor)
    noise_image_total = noise_image + sensor

    mix = target_image + noise_image_total

    return {
        "mix": mix,
        "target_image": target_image,
        "noise_image": noise_image_total,
        "length": L,
    }


# Function for performing multichannel STFT
def multichannel_stft(x, fs=16000, nperseg=512, noverlap=None, window="hann"):
    """
    STFT of a multichannel signal.
    x : (n_mics, N)
    returns
      f : (n_freq,)            frequency bins (Hz)
      t : (n_frames,)          frame times (s)
      X : (n_mics, n_freq, n_frames)  complex STFT
    """
    if noverlap is None:
        noverlap = nperseg // 2

    n_mics = x.shape[0]

    Xs = []
    for m in range(n_mics):
        f, t, Xm = stft(x[m], fs=fs, window=window
                        nperseg=nperseg, noverlap=noverlap)
        Xs.append(Xm)

    X = np.stack(Xs)  

    return f, t, X

# Function for performing multichannel ISTFT
def multichannel_istft(X, fs=16000, nperseg=512, noverlap=None, window="hann"):
    """
    Inverse STFT, channel by channel.
    X : (n_mics, n_freq, n_frames)  OR  (n_freq, n_frames) for a single channel
    returns time-domain signal(s).
    """
    if noverlap is None:
        noverlap = nperseg // 2

    if X.ndim == 2:  # single channel (e.g. beamformer output)
        _, x = istft(X, fs=fs, window=window, nperseg=nperseg, noverlap=noverlap)
        return x
    
    n_mics = X.shape[0]

    xs = []

    for m in range(n_mics):
        _, xm = istft(X[m], fs=fs, window=window, nperseg=nperseg, noverlap=noverlap)
        xs.append(xm)
        
    return np.stack(xs)


# Function for computing spatial covariance per frequency bin
def spatial_covariance(X, frame_mask=None):
    """
    Per-frequency spatial covariance from a multichannel STFT.
    X : (n_mics, n_freq, n_frames) complex
    frame_mask : (n_frames,) bool, which frames to include (None = all)
    returns
      R : (n_freq, n_mics, n_mics) complex Hermitian, one matrix per bin
    """
    # Select frames
    if frame_mask is None:
        frame_mask = np.ones(X.shape[2], dtype=bool)
    
    Xs = X[:, :, frame_mask] # (M, F, Nsel)
    nsel = Xs.shape[2]
    # for each bin f: sum_n x x^H  ->  einsum over mics
    # result[f, i, j] = sum_n Xs[i,f,n] * conj(Xs[j,f,n])
    R = np.einsum('ifn,jfn->fij', Xs, np.conj(Xs)) / max(nsel, 1)
    return R


# Function for speech presence detection
def speech_presence_detector(X_mix, X_tgt_truth, ref_mic=0, pfa=0.05, active_db=-25):
    """
    Per-(bin,frame) speech-presence mask.
    Calibrates a per-bin threshold on truth-defined noise-only cells,
    then tests the mixture cell-by-cell.

    Returns
      present : (n_freq, n_frames) bool,   True = target present in this cell
      truth   : (n_freq, n_frames) bool,   True = active cells 
    """
    _, n_freq, n_frames = X_mix.shape

    # Per-cell magnitude-squared at the reference mic
    P_mix = np.abs(X_mix[ref_mic])**2 
    P_tgt = np.abs(X_tgt_truth[ref_mic])**2 

    # Threshold for active cells: 10**(active_db/10) below the peak in each bin
    peak_per_bin = P_tgt.max(axis=1, keepdims=True) # (F,1)
    thr_active = peak_per_bin * 10**(active_db/10)
    truth = P_tgt > thr_active # (F, N)

    # Per-bin threshold gamma from H0 
    present = np.zeros((n_freq, n_frames), dtype=bool)
    for f in range(n_freq):
        h0 = P_mix[f, ~truth[f]] # mixture power in this bin's noise cells
        if h0.size > 0:
            gamma = np.quantile(h0, 1 - pfa)
        else:
            gamma = np.inf 
        present[f] = P_mix[f] > gamma

    return present, truth


# Function for estimating covariances per bin
def estimate_covariances_perbin(X_mix, present, lam_x=0.95, lam_n=0.95):
    """
    Per-bin detector-gated covariance estimation.
    present : (n_freq, n_frames) bool, per-cell target-present mask
    Returns R_x, R_n : (n_freq, n_mics, n_mics)
    """
    n_mics, n_freq, n_frames = X_mix.shape

    R_x = np.zeros((n_freq, n_mics, n_mics), dtype=complex)
    R_n = np.zeros((n_freq, n_mics, n_mics), dtype=complex)

    # Track whether each bins R_n has been initialized
    n_init = np.zeros(n_freq, dtype=bool)
    x_init = np.zeros(n_freq, dtype=bool)

    for n in range(n_frames):
        x = X_mix[:, :, n]  # (M, F)
        Rk = np.einsum('if,jf->fij', x, np.conj(x)) # (F, M, M) 

        # Sigma_x: update every frame, all bins
        new_x = ~x_init
        R_x[new_x] = Rk[new_x]
        R_x[x_init] = lam_x * R_x[x_init] + (1 - lam_x) * Rk[x_init]
        x_init[:] = True

        # Sigma_u: update only in bins where target is absent 
        noise_bins = ~present[:, n] # (F,) bool

        # Bins to initialize: noise this frame AND not yet initialized
        init = noise_bins & ~n_init
        upd = noise_bins &  n_init
        R_n[init] = Rk[init]
        R_n[upd] = lam_n * R_n[upd] + (1 - lam_n) * Rk[upd]
        n_init[init] = True

    # Bins that never saw a noise cell fall back to R_x
    if np.any(~n_init):
        R_n[~n_init] = R_x[~n_init]
        print(f"{np.sum(~n_init)} bins had no noise cell; fell back to R_x")

    return R_x, R_n


# Function for computing RTF from covariance
def rtf_from_covariance(R_s, ref_mic=0):
    """
    RTF as the principal eigenvector of the target spatial covariance.
    R_s : (n_freq, n_mics, n_mics) 
    returns
      rtf : (n_freq, n_mics) complex, normalized so rtf[:,ref_mic] == 1
    """
    n_freq, n_mics, _ = R_s.shape
    rtf = np.zeros((n_freq, n_mics), dtype=complex)
    for f in range(n_freq):
        _, V = np.linalg.eigh(R_s[f])
        a = V[:, -1] # principal eigenvector since they are in ascending order
        rtf[f] = a / a[ref_mic]  # normalize to reference mic
    return rtf

# Function for computing RTF from covariance whitening like in the overview paper
def rtf_covariance_whitening(R_x, R_n, ref_mic=0):
    """
    RTF via covariance whitening
    R_x, R_n : (n_freq, n_mics, n_mics) 
    returns
      rtf : (n_freq, n_mics) complex, normalized so rtf[:,ref_mic] == 1
    """
    n_freq, n_mics, _ = R_x.shape
    rtf = np.zeros((n_freq, n_mics), dtype=complex)
    e_ref = np.zeros(n_mics)
    e_ref[ref_mic] = 1.0

    for f in range(n_freq):
        # Cholesky decomposition of the noise covariance
        L = np.linalg.cholesky(R_n[f]) 
        Linv = np.linalg.inv(L)

        # Whiten the mixture covariance like: Rw = L^-1 R_x L^-H
        Rw = Linv @ R_x[f] @ Linv.conj().T
        Rw = 0.5 * (Rw + Rw.conj().T)      

        # Principal eigenvector in whitened space
        _, V = np.linalg.eigh(Rw)
        V_hat = V[:, -1]    

        # De-whiten: a = L phi
        a = L @ V_hat

        # normalize to reference mic
        rtf[f] = a / a[ref_mic]

    return rtf


# Function for checking the match between estimated and true RTFs using the cosine of the Hermitian angle per bin: |<a,b>| / (||a|| ||b||)
def rtf_match(a_est, a_true):
    num = np.abs(np.sum(np.conj(a_est) * a_true, axis=1))
    den = np.linalg.norm(a_est, axis=1) * np.linalg.norm(a_true, axis=1)
    return num / np.maximum(den, 1e-12)


# Delay and sum beamformer
def bf_delay_and_sum(rtf):
    """
    Delay-and-sum: align to the target steering vector
    Uses only the RTF : w = a / (a^H a)
    rtf : (n_freq, n_mics) -> w : (n_freq, n_mics)
    """
    # normalize so that w^H a = 1 (distortionless toward target)
    denom = np.sum(np.conj(rtf) * rtf, axis=1, keepdims=True).real  # a^H a
    w = rtf / denom
    return w

# MVDR beamformer
def bf_mvdr(rtf, R_n):
    """
    MVDR : w = R_n^-1 a / (a^H R_n^-1 a)
    Uses solve(R_n, a) instead of forming the inverse.
    """
    n_freq, n_mics = rtf.shape
    w = np.zeros((n_freq, n_mics), dtype=complex)
    for f in range(n_freq):
        a = rtf[f]
        Rinv_a = np.linalg.solve(R_n[f], a) # R_n^-1 a
        w[f] = Rinv_a / (np.conj(a) @ Rinv_a) # normalize: a^H R_n^-1 a
    return w

# Signal distortion weighted beamformer
def bf_sdw_mwf(rtf, R_n, R_s_scalar=None, mu=1.0):
    """
    SDW-MWF via the MVDR + single-channel postfilter decomposition
    w_SDW-MWF = w_MVDR *  sigma_ds^2 / (sigma_ds^2 + mu * sigma_du^2) 
    sigma_ds^2 : target power at the MVDR output
    sigma_du^2 : noise  power at the MVDR output
    """
    n_freq, _ = rtf.shape
    w_mvdr = bf_mvdr(rtf, R_n)
    w = np.zeros_like(w_mvdr)
    for f in range(n_freq):
        wm = w_mvdr[f]

        # Noise power at MVDR output: w^H R_n w
        sigma_du2 = (np.conj(wm) @ R_n[f] @ wm).real

        # Target power at MVDR output: w^H R_s w  (R_s = sigma_s^2 a a^H)
        sigma_ds2 = (np.conj(wm) @ R_s_scalar[f] @ wm).real
        gain = sigma_ds2 / (sigma_ds2 + mu * sigma_du2 + 1e-12)
        w[f] = wm * gain

    return w


# Function to apply frequency dependant beamformers to the multichannel STFT
def apply_beamformer(w, X_mix):
    """
    Apply per-bin weights to the multichannel STFT.
    w : (n_freq, n_mics)
    X_mix : (n_mics, n_freq, n_frames)
    returns Y : (n_freq, n_frames)  single-channel output STFT
    """
    # Y[f,n] = w[f]^H x[:,f,n] = sum_m conj(w[f,m]) X[m,f,n]
    Y = np.einsum('fm,mfn->fn', np.conj(w), X_mix)
    return Y

# Function to get the time signal back after applying the beamformer
def beamform_to_time(w, X_mix, nperseg=512):
    Y = apply_beamformer(w, X_mix)
    return multichannel_istft(Y, nperseg=nperseg)   


# Estimate R_s = R_x - R_n
def make_R_s_from_subtraction(R_x, R_n):
    R_s = R_x - R_n

    # project to PSD
    n_freq, _, _ = R_s.shape
    out = np.zeros_like(R_s)
    for f in range(n_freq):
        H = 0.5*(R_s[f] + R_s[f].conj().T)
        w_, V = np.linalg.eigh(H)
        w_ = np.clip(w_, 0, None)
        out[f] = (V * w_) @ V.conj().T

    return out


# Function to get SNR
def seg_snr(est, target, eps=1e-12):
    """Global SNR of an estimate against a clean target, both 1-D, aligned."""
    L = min(len(est), len(target))
    est, target = est[:L], target[:L]

    # scale-invariant: project est onto target to remove gain ambiguity
    alpha = np.dot(est, target) / (np.dot(target, target) + eps)
    s = alpha * target
    noise = est - s

    return 10*np.log10((np.dot(s,s)+eps)/(np.dot(noise,noise)+eps))


# Function to get all the metrics
def evaluate(est, target, fs=16000):
    L = min(len(est), len(target))
    est, target = est[:L], target[:L]
    return {
        "STOI": stoi(target, est, fs, extended=False),
        "SIIB": SIIB(target, est, fs, window="hann"),
        "SNR":  seg_snr(est, target),
    }