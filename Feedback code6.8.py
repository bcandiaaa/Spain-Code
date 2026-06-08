# ============================================================
# Correct M-PSK Constellation + OFDM I/Q Signal Simulation
# Spyder / Anaconda friendly version
# Requires only: numpy and matplotlib
# ============================================================

import numpy as np
import matplotlib.pyplot as plt


# ============================================================
# 1. BASIC PARAMETERS
# ============================================================

FS = 4_000_000              # Sampling frequency = 4 MHz
NUM_SUBCARRIERS = 64        # OFDM subcarriers
CYCLIC_PREFIX = 16          # Cyclic prefix length
SAMPLES_PER_SIGNAL = 1024   # One OFDM I/Q capture length

SNR_DB = 10                 # Signal-to-noise ratio in dB

MODULATION_CLASSES = [
    ("BPSK", "BPSK"),
    ("BPSK", "QPSK"),
    ("BPSK", "8PSK"),
    ("QPSK", "BPSK"),
    ("QPSK", "QPSK"),
    ("QPSK", "8PSK")
]


# ============================================================
# 2. IDEAL M-PSK SYMBOL GENERATION
# ============================================================

def get_m_value(mod_type):
    """
    Converts modulation name into M value.
    BPSK  -> M = 2
    QPSK  -> M = 4
    8PSK  -> M = 8
    16PSK -> M = 16
    """
    if mod_type == "BPSK":
        return 2
    elif mod_type == "QPSK":
        return 4
    elif mod_type == "8PSK":
        return 8
    elif mod_type == "16PSK":
        return 16
    else:
        raise ValueError("Unknown modulation type: " + mod_type)


def generate_mpsk_symbols(mod_type, num_symbols):
    """
    Generates ideal M-PSK symbols using:

        s_m = cos(2*pi*m/M) + j*sin(2*pi*m/M)

    This matches the formula from the notes/book.

    For BPSK:
        M = 2
        m = 0 -> point at +1 on I-axis
        m = 1 -> point at -1 on I-axis
    """
    M = get_m_value(mod_type)

    symbol_values = np.random.randint(0, M, num_symbols)

    symbols = np.exp(1j * 2 * np.pi * symbol_values / M)

    return symbols, symbol_values


def add_noise_to_symbols(symbols, noise_std=0.05):
    """
    Adds small noise to ideal constellation points.
    This makes the clean constellation appear as small clouds.
    """
    noise = noise_std * (
        np.random.randn(len(symbols)) + 1j * np.random.randn(len(symbols))
    )

    return symbols + noise


# ============================================================
# 3. PLOT IDEAL M-PSK CONSTELLATIONS
# ============================================================

def plot_ideal_mpsk_constellations():
    """
    Plots the correct ideal reference constellations for:
    BPSK, QPSK, 8PSK, and 16PSK.

    These are NOT OFDM signals yet.
    These are the raw modulation symbols.
    """
    modulation_list = ["BPSK", "QPSK", "8PSK", "16PSK"]

    plt.figure(figsize=(12, 10))

    for index, mod_type in enumerate(modulation_list):
        symbols, symbol_values = generate_mpsk_symbols(mod_type, 1000)

        received_symbols = add_noise_to_symbols(symbols, noise_std=0.04)

        plt.subplot(2, 2, index + 1)
        plt.scatter(received_symbols.real, received_symbols.imag, s=10)

        plt.title(mod_type + " Ideal M-PSK Constellation")
        plt.xlabel("In-phase I")
        plt.ylabel("Quadrature Q")

        plt.axhline(0, color="gray", linewidth=0.7)
        plt.axvline(0, color="gray", linewidth=0.7)

        plt.grid(True)
        plt.axis("equal")
        plt.xlim(-1.5, 1.5)
        plt.ylim(-1.5, 1.5)

    plt.tight_layout()
    plt.savefig("ideal_mpsk_constellations.png", dpi=150)
    plt.show()

    print("Saved: ideal_mpsk_constellations.png")


# ============================================================
# 4. AWGN NOISE FOR OFDM SIGNALS
# ============================================================

def add_awgn(signal, snr_db):
    """
    Adds Additive White Gaussian Noise to a complex signal.
    """
    signal_power = np.mean(np.abs(signal) ** 2)

    snr_linear = 10 ** (snr_db / 10)

    noise_power = signal_power / snr_linear

    noise = np.sqrt(noise_power / 2) * (
        np.random.randn(len(signal)) + 1j * np.random.randn(len(signal))
    )

    noisy_signal = signal + noise

    return noisy_signal


# ============================================================
# 5. CREATE ONE OFDM SYMBOL
# ============================================================

def create_ofdm_symbol(mod_type):
    """
    Creates one OFDM symbol.

    Steps:
    1. Generate ideal PSK symbols on 64 subcarriers
    2. Use IFFT to convert frequency-domain data to time-domain OFDM
    3. Add cyclic prefix
    """
    freq_symbols, symbol_values = generate_mpsk_symbols(
        mod_type,
        NUM_SUBCARRIERS
    )

    time_domain_signal = np.fft.ifft(freq_symbols, NUM_SUBCARRIERS)

    cyclic_prefix = time_domain_signal[-CYCLIC_PREFIX:]

    ofdm_symbol = np.concatenate((cyclic_prefix, time_domain_signal))

    return ofdm_symbol


# ============================================================
# 6. CREATE ONE 1024-SAMPLE OFDM I/Q SIGNAL
# ============================================================

def create_one_ofdm_iq_signal(header_mod, payload_mod, snr_db):
    """
    Creates one 1024-sample OFDM signal.

    First 2 OFDM symbols are treated as header.
    Remaining OFDM symbols are treated as payload.
    """
    frame = []

    ofdm_symbol_length = NUM_SUBCARRIERS + CYCLIC_PREFIX

    while len(frame) < SAMPLES_PER_SIGNAL:

        if len(frame) < 2 * ofdm_symbol_length:
            symbol = create_ofdm_symbol(header_mod)
        else:
            symbol = create_ofdm_symbol(payload_mod)

        frame.extend(symbol)

    frame = np.array(frame[:SAMPLES_PER_SIGNAL])

    noisy_frame = add_awgn(frame, snr_db)

    iq_signal = np.vstack((noisy_frame.real, noisy_frame.imag))

    return iq_signal


# ============================================================
# 7. PLOT ONE OFDM I/Q SIGNAL
# ============================================================

def plot_single_ofdm_signal(header_mod="BPSK", payload_mod="QPSK", snr_db=10):
    """
    Plots one OFDM I/Q signal.

    Important:
    This is NOT an ideal constellation plot.
    This is the time-domain OFDM signal after IFFT and noise.
    """
    iq_signal = create_one_ofdm_iq_signal(header_mod, payload_mod, snr_db)

    I = iq_signal[0, :]
    Q = iq_signal[1, :]

    time_us = np.arange(SAMPLES_PER_SIGNAL) / FS * 1e6

    class_name = header_mod + "+" + payload_mod

    print("Single OFDM I/Q signal created.")
    print("Class:", class_name)
    print("SNR:", snr_db, "dB")
    print("Shape:", iq_signal.shape)
    print("First 10 I values:")
    print(I[:10])
    print("First 10 Q values:")
    print(Q[:10])

    np.save("single_ofdm_iq_signal.npy", iq_signal)
    print("Saved: single_ofdm_iq_signal.npy")

    plt.figure(figsize=(10, 4))
    plt.plot(time_us, I, label="I Channel")
    plt.plot(time_us, Q, label="Q Channel")
    plt.title("OFDM Time-Domain I/Q Signal: " + class_name)
    plt.xlabel("Time (microseconds)")
    plt.ylabel("Amplitude")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("single_ofdm_iq_time_signal.png", dpi=150)
    plt.show()

    print("Saved: single_ofdm_iq_time_signal.png")

    plt.figure(figsize=(6, 5))
    plt.scatter(I, Q, s=5)
    plt.title("OFDM Time-Domain I/Q Scatter: " + class_name)
    plt.xlabel("In-phase I")
    plt.ylabel("Quadrature Q")
    plt.grid(True)
    plt.axis("equal")
    plt.tight_layout()
    plt.savefig("single_ofdm_iq_scatter.png", dpi=150)
    plt.show()

    print("Saved: single_ofdm_iq_scatter.png")


# ============================================================
# 8. PLOT FIRST 5 MICROSECONDS
# ============================================================

def plot_first_5_microseconds(header_mod="BPSK", payload_mod="QPSK", snr_db=10):
    """
    Plots only the first 5 microseconds of the OFDM I/Q signal.
    This gives a zoomed-in view of the waveform.
    """
    iq_signal = create_one_ofdm_iq_signal(header_mod, payload_mod, snr_db)

    I = iq_signal[0, :]
    Q = iq_signal[1, :]

    time_us = np.arange(SAMPLES_PER_SIGNAL) / FS * 1e6

    samples_5us = int(5e-6 * FS)

    class_name = header_mod + "+" + payload_mod

    plt.figure(figsize=(8, 4))
    plt.plot(time_us[:samples_5us], I[:samples_5us], marker="o", label="I Channel")
    plt.plot(time_us[:samples_5us], Q[:samples_5us], marker="o", label="Q Channel")

    plt.title("OFDM I/Q Signal, First 5 Microseconds: " + class_name)
    plt.xlabel("Time (microseconds)")
    plt.ylabel("Amplitude")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("ofdm_first_5_microseconds.png", dpi=150)
    plt.show()

    print("Saved: ofdm_first_5_microseconds.png")


# ============================================================
# 9. PLOT ALL SIX OFDM CLASSES
# ============================================================

def plot_all_ofdm_classes(snr_db=10):
    """
    Plots OFDM time-domain I/Q scatter plots for all 6 classes.

    These plots are expected to look like noisy clouds because they are
    OFDM time-domain samples, not ideal PSK constellation points.
    """
    plt.figure(figsize=(14, 9))

    for index, (header_mod, payload_mod) in enumerate(MODULATION_CLASSES):
        iq_signal = create_one_ofdm_iq_signal(header_mod, payload_mod, snr_db)

        I = iq_signal[0, :]
        Q = iq_signal[1, :]

        class_name = header_mod + "+" + payload_mod

        plt.subplot(2, 3, index + 1)
        plt.scatter(I, Q, s=4)

        plt.title("OFDM I/Q Scatter: " + class_name)
        plt.xlabel("In-phase I")
        plt.ylabel("Quadrature Q")
        plt.grid(True)
        plt.axis("equal")

    plt.tight_layout()
    plt.savefig("all_ofdm_iq_scatter_plots.png", dpi=150)
    plt.show()

    print("Saved: all_ofdm_iq_scatter_plots.png")


# ============================================================
# 10. MAIN PROGRAM
# ============================================================

def main():
    np.random.seed(42)

    print("Starting corrected constellation and OFDM simulation...")

    print("\n1. Plotting ideal M-PSK constellations...")
    plot_ideal_mpsk_constellations()

    print("\n2. Plotting one OFDM I/Q signal...")
    plot_single_ofdm_signal(
        header_mod="BPSK",
        payload_mod="QPSK",
        snr_db=SNR_DB
    )

    print("\n3. Plotting first 5 microseconds...")
    plot_first_5_microseconds(
        header_mod="BPSK",
        payload_mod="QPSK",
        snr_db=SNR_DB
    )

    print("\n4. Plotting all six OFDM I/Q scatter plots...")
    plot_all_ofdm_classes(
        snr_db=SNR_DB
    )

    print("\nFinished.")


if __name__ == "__main__":
    main()