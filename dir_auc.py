import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('TkAgg')  # Or 'MacOSX' if Tk isn't installed on macOS
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.neighbors import KernelDensity
from scipy.stats import kruskal
from scipy import stats as st
from sklearn.metrics import auc
import os
import helpers as h

#GLOBAL SETTINGS
nb_it = 500
bandwidth = 0.01
# With 100 points in x from 0..1, 44:56 ~ 0.44..0.56
AUC_XMIN = 44
AUC_XMAX = 56

#set intervaks as global variables
interval13_left = int(1/4.25 * 100)
interval13_right = int(1/3.75 * 100)
DIR_13 = int(0.25*100)

interval12_left = int(1/3.25 * 100)
interval12_right = int(1/2.75 * 100)
DIR_12 = int(0.33*100)

interval11_left = int(0.44 * 100)
interval11_right = int(0.56 * 100)
DIR_11 = int(0.5 * 100)

interval21_left = int((1-1/2.75)*100)
interval21_right = int((1-1/3.25)*100)
DIR_21 = int(0.66 * 100)

interval31_left = int((1-1/3.75)*100)
interval31_right = int((1-1/4.25)*100)
DIR_31 = int(0.75 * 100)



def plot_clade_curves(x, real_curves, shuffled_curves, clade_name, save_path):
    """
    Plot per-clade real, shuffled, and diff curves, plus mean curves.
    Adds black dashed vertical lines at AUC bounds.
    """
    print(f'Plotting average curve for clade: {clade_name}')
    fig, axs = plt.subplots(3, 1, figsize=(5, 8))

    # Use what we receive (already in consistent domain by caller)
    real_curves_norm = list(real_curves)
    shuffled_curves_norm = list(shuffled_curves)
    diff_curves = [r - s for r, s in zip(real_curves_norm, shuffled_curves_norm)]

    # Plot individual curves
    for curve in real_curves_norm:
        axs[0].plot(x, curve, alpha=0.2)
    for curve in shuffled_curves_norm:
        axs[1].plot(x, curve, alpha=0.2)
    for curve in diff_curves:
        axs[2].plot(x, curve, alpha=0.2)

    # Plot means only if available
    if len(real_curves_norm):
        mean_real = np.mean(np.vstack(real_curves_norm), axis=0)
        axs[0].plot(x, mean_real, linewidth=2, color='k', label='mean')
    if len(shuffled_curves_norm):
        mean_shuffled = np.mean(np.vstack(shuffled_curves_norm), axis=0)
        axs[1].plot(x, mean_shuffled, linewidth=2, color='k', label='mean')
    if len(diff_curves):
        mean_diff = np.mean(np.vstack(diff_curves), axis=0)
        axs[2].plot(x, mean_diff, linewidth=2, color='k', label='mean')

    # Add vertical lines at exact DIRs, add tick marks
    x_vals = x.squeeze()
    #left_idx = max(0, min(auc_xmin, len(x_vals) - 1))
    #right_idx = max(0, min(auc_xmax, len(x_vals) - 1))
    for ax in axs:
        #ax.axvline(x=float(x_vals[DIR_13]), color="black", linestyle="--", linewidth=1, label="1:3")
        #ax.axvline(x=float(x_vals[DIR_12]), color="black", linestyle="--", linewidth=1, label="1:2")
        #ax.axvline(x=float(x_vals[DIR_11]), color="black", linestyle="--", linewidth=1, label="1:1")
        #ax.axvline(x=float(x_vals[DIR_21]), color="black", linestyle="--", linewidth=1, label="2:1")
        #ax.axvline(x=float(x_vals[DIR_31]), color="black", linestyle="--", linewidth=1, label="3:1")
        # ax.axvline(x=float(x_vals[interval13_left]), color="black", linestyle="--", linewidth=1)
        # ax.axvline(x=float(x_vals[interval13_right]), color="black", linestyle="--", linewidth=1)
        # ax.axvline(x=float(x_vals[interval12_left]), color="black", linestyle="--", linewidth=1)
        # ax.axvline(x=float(x_vals[interval12_right]), color="black", linestyle="--", linewidth=1)
        # ax.axvline(x=float(x_vals[interval11_left]), color="black", linestyle="--", linewidth=1)
        # ax.axvline(x=float(x_vals[interval11_right]), color="black", linestyle="--", linewidth=1)
        # ax.axvline(x=float(x_vals[interval21_left]), color="black", linestyle="--", linewidth=1)
        # ax.axvline(x=float(x_vals[interval21_right]), color="black", linestyle="--", linewidth=1)
        # ax.axvline(x=float(x_vals[interval31_left]), color="black", linestyle="--", linewidth=1)
        # ax.axvline(x=float(x_vals[interval31_right]), color="black", linestyle="--", linewidth=1)
        ax.axvspan(x_vals[interval13_left], x_vals[interval13_right],
                   facecolor="0.2", alpha=0.08, linewidth=0, zorder=0)
        ax.axvspan(x_vals[interval12_left], x_vals[interval12_right],
                   facecolor="0.2", alpha=0.08, linewidth=0, zorder=0)
        ax.axvspan(x_vals[interval11_left], x_vals[interval11_right],
                   facecolor="0.2", alpha=0.08, linewidth=0, zorder=0)
        ax.axvspan(x_vals[interval21_left], x_vals[interval21_right],
                   facecolor="0.2", alpha=0.08, linewidth=0, zorder=0)
        ax.axvspan(x_vals[interval31_left], x_vals[interval31_right],
                   facecolor="0.2", alpha=0.08, linewidth=0, zorder=0)

    axs[0].set_title(f"{clade_name} - Real curves")
    axs[1].set_title(f"{clade_name} - Shuffled curves")
    axs[2].set_title(f"{clade_name} - Difference")

    for ax in axs:
        ax.set_ylabel("Density")
        ax.set_xlabel("Diadic interval ratio (DIR)")
        ax.set_ylim((0, 40))
    axs[2].set_ylim((-20, 20))

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


def _canonical_group(name):
    """Make group labels robust to capitalization/typos."""
    s = str(name).strip().lower()
    if 'oscines' == s:
        return 'Oscines'
    if 'humm' in s:
        return 'Hummingbirds'
    if 'nightjar' in s:
        return 'Nightjars'
    if 'suboscin' in s:  # handles 'Suboscines' / 'Subosciness'
        return 'Suboscines'
    return 'Unknown'


def plot_bird_auc_curves(syllable_csv):
    print('Plotting 4 AUC graphs, one per clade!')
    aucs = []

    # read file (xlsx expected)
    df = pd.read_excel(syllable_csv)

    # storage
    oscine_real_curves, oscine_shuffled_curves = [], []
    suboscine_real_curves, suboscine_shuffled_curves = [], []
    hummingbird_real_curves, hummingbird_shuffled_curves = [], []
    nightjar_real_curves, nightjar_shuffled_curves = [], []

    all_birds_real_curves, all_birds_shuffled_curves = [], []

    # loop through species
    for species_id, species in df.groupby('species_birdtree'):
        print(f"Species: {species_id}")
        iois = []

        # collect contiguous IOIs
        for i in range(len(species) - 1):
            cur_idx = species.iloc[i]['i_syll_in_song']
            next_idx = species.iloc[i + 1]['i_syll_in_song']
            if next_idx == cur_idx + 1:
                iois.append(species.iloc[i + 1]['onset_msec'] - species.iloc[i]['onset_msec'])
        if len(iois) < 11:
            print(f"  Skipping (insufficient IOIs): n={len(iois)}")
            continue

        # real DIRs
        real_dirs = [iois[i] / (iois[i] + iois[i + 1]) for i in range(len(iois) - 1)]
        x = np.linspace(0, 1, 100).reshape(-1, 1)
        vals = np.array(real_dirs, dtype=float).reshape(-1, 1)
        kde = KernelDensity(kernel='gaussian', bandwidth=bandwidth).fit(vals)
        real_curve_log = kde.score_samples(x)  # log-density (length 100)

        # shuffled curves (per-song shuffling)
        shuffled_curves_log = []
        song_groups = species.groupby("seg_id")

        for _ in range(nb_it):
            all_shuffled_dirs = []
            for _, song_df in song_groups:
                # per-song IOIs
                song_iois = []
                for j in range(len(song_df) - 1):
                    cur_idx = song_df.iloc[j]['i_syll_in_song']
                    next_idx = song_df.iloc[j + 1]['i_syll_in_song']
                    if next_idx == cur_idx + 1:
                        song_iois.append(song_df.iloc[j + 1]['onset_msec'] - song_df.iloc[j]['onset_msec'])
                if len(song_iois) < 2:
                    continue
                shuffled = np.random.permutation(song_iois)
                dirs = [shuffled[k] / (shuffled[k] + shuffled[k + 1]) for k in range(len(shuffled) - 1)]
                all_shuffled_dirs.extend(dirs)

            if len(all_shuffled_dirs) < 10:
                # not enough material this iteration
                continue

            kde_shuf = KernelDensity(kernel="gaussian", bandwidth=bandwidth).fit(
                np.array(all_shuffled_dirs, dtype=float).reshape(-1, 1)
            )
            shuffled_curves_log.append(kde_shuf.score_samples(x))  # log-density

        if not shuffled_curves_log:
            print("  Skipping (no shuffled curves could be generated).")
            continue

        # mean shuffled curve in the same domain as real_curve_log according to h.normalize_kde
        mean_shuffled_curve = np.mean(h.normalize_kde(shuffled_curves_log, x), axis=0)
        # Compute AUC using densities (exp of log-like arrays if normalize_kde keeps log domain)
        auc_xmin, auc_xmax = AUC_XMIN, AUC_XMAX
        portion_real = np.exp(real_curve_log[auc_xmin:auc_xmax])
        portion_shuf = np.exp(mean_shuffled_curve[auc_xmin:auc_xmax])
        auc_real = np.sum(portion_real) / 100.0
        auc_shuffled = np.sum(portion_shuf) / 100.0


        # Normalize curves for plotting & diff
        real_curve = h.normalize_kde_one(real_curve_log, x)           # length 100
        mean_shuf_norm = mean_shuffled_curve                           # already normalized by helper
        diff_curve = real_curve - mean_shuf_norm

        all_birds_real_curves.append(real_curve)
        all_birds_shuffled_curves.append(mean_shuf_norm)

        # Grouping
        group_label = _canonical_group(species.iloc[0].get('our_grouping', 'Unknown'))

        # Accumulate by clade
        if group_label == 'Oscines':
            oscine_real_curves.append(real_curve)
            oscine_shuffled_curves.append(mean_shuf_norm)
        elif group_label == 'Hummingbirds':
            hummingbird_real_curves.append(real_curve)
            hummingbird_shuffled_curves.append(mean_shuf_norm)
        elif group_label == 'Nightjars':
            nightjar_real_curves.append(real_curve)
            nightjar_shuffled_curves.append(mean_shuf_norm)
        elif group_label == 'Suboscines':
            suboscine_real_curves.append(real_curve)
            suboscine_shuffled_curves.append(mean_shuf_norm)

        aucs.append((species_id, auc_real, auc_shuffled, group_label))

    # Output plots per clade
    dirpath, _ = os.path.split(syllable_csv)
    x_flat = np.linspace(0, 1, 100).reshape(-1, 1)

    plot_clade_curves(
        x_flat,
        all_birds_real_curves,
        all_birds_shuffled_curves,
        "All birds",
        os.path.join(dirpath, 'dir_auc_plot_all_birds.png')
    )

    plot_clade_curves(x_flat, oscine_real_curves,      oscine_shuffled_curves,      "Oscines",
                      os.path.join(dirpath, 'dir_auc_plot_oscines.png'))
    plot_clade_curves(x_flat, suboscine_real_curves,   suboscine_shuffled_curves,   "Suboscines",
                      os.path.join(dirpath, 'dir_auc_plot_suboscines.png'))
    plot_clade_curves(x_flat, hummingbird_real_curves, hummingbird_shuffled_curves, "Hummingbirds",
                      os.path.join(dirpath, 'dir_auc_plot_hummingbirds.png'))
    plot_clade_curves(x_flat, nightjar_real_curves,    nightjar_shuffled_curves,    "Nightjars",
                      os.path.join(dirpath, 'dir_auc_plot_nightjars.png'))

    print('Counts — Oscines:', len(oscine_real_curves),
          'Suboscines:', len(suboscine_real_curves),
          'Nightjars:', len(nightjar_real_curves),
          'Hummingbirds:', len(hummingbird_real_curves))

    # Save AUC table
    if aucs:
        aucs_df = pd.DataFrame(aucs, columns=["species_birdtree", "AUC_real", "AUC_shuffled", "clade"])
        aucs_df.to_csv(os.path.join(dirpath, 'auc_values_birds.csv'), index=False)
    else:
        print("No AUCs to save for birds.")


def plot_music_auc_curves(syllable_csv, n_iter):
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    from sklearn.neighbors import KernelDensity
    import helpers as h

    bandwidth = 0.01
    print('Plotting music KDE curves')
    aucs = []
    fig, axs = plt.subplots(3, 1, figsize=(5, 8))
    df = pd.read_csv(syllable_csv)
    all_real_curves, all_shuffled_curves, all_difference_curves = [], [], []

    for title, track in df.groupby('title'):
        print(f"Processing {title}")

        is_note = track['is_note'].values
        duration = track['duration'].values

        iois = [duration[i] for i in range(len(duration) - 1)
                if is_note[i] and is_note[i + 1]]

        if len(iois) < 11:
            print(f"  Skipping {title} (insufficient IOIs): n={len(iois)}")
            continue

        real_dirs = np.array([iois[i] / (iois[i] + iois[i + 1]) for i in range(len(iois) - 1)], dtype=float)
        x = np.linspace(0, 1, 100).reshape(-1, 1)
        real_dirs = real_dirs[~np.isnan(real_dirs)]
        if len(real_dirs) < 2:
            print(f"  Skipping {title} (too few valid DIRs after NaN filter).")
            continue

        kde = KernelDensity(kernel='gaussian', bandwidth=bandwidth).fit(real_dirs.reshape(-1, 1))
        real_curve_log = kde.score_samples(x)
        # We'll append later after confirming shuffled curves exist

        shuffled_curves_log = []
        grouped = list(track.groupby('phrase_id'))
        for _ in range(n_iter):
            all_shuffled_dirs = []
            for _, phrase_df in grouped:
                is_note_phrase = phrase_df['is_note'].values
                duration_phrase = phrase_df['duration'].values
                iois_phrase = [duration_phrase[i] for i in range(len(duration_phrase) - 1)
                               if is_note_phrase[i] and is_note_phrase[i + 1]]
                if len(iois_phrase) < 2:
                    continue
                shuffled = np.random.permutation(iois_phrase)
                dirs = [shuffled[i] / (shuffled[i] + shuffled[i + 1])
                        for i in range(len(shuffled) - 1)]
                dirs = [d for d in dirs if not np.isnan(d)]
                all_shuffled_dirs.extend(dirs)

            if len(all_shuffled_dirs) < 10:
                continue

            kde_shuf = KernelDensity(kernel="gaussian", bandwidth=bandwidth).fit(
                np.array(all_shuffled_dirs, dtype=float).reshape(-1, 1))
            shuffled_curves_log.append(kde_shuf.score_samples(x))

        if not shuffled_curves_log:
            print(f"  Skipping {title} (no shuffled curves could be generated).")
            continue

        mean_shuffled_curve = np.mean(h.normalize_kde(shuffled_curves_log, x), axis=0)
        real_curve = h.normalize_kde_one(real_curve_log, x)
        difference_curve = real_curve - mean_shuffled_curve

        # AUC within window (exp to density as above)
        portion_real = np.exp(real_curve_log[AUC_XMIN:AUC_XMAX])
        portion_shuffled = np.exp(mean_shuffled_curve[AUC_XMIN:AUC_XMAX])
        auc_real = float(np.sum(portion_real) / 100.0)
        auc_shuffled = float(np.sum(portion_shuffled) / 100.0)

        aucs.append((title, auc_real, auc_shuffled))
        all_real_curves.append(real_curve)
        all_shuffled_curves.append(mean_shuffled_curve)
        all_difference_curves.append(difference_curve)

        axs[0].plot(x, real_curve, alpha=0.2)
        axs[1].plot(x, mean_shuffled_curve, alpha=0.2)
        axs[2].plot(x, difference_curve, alpha=0.2)

    # Plot means (if any)
    x = np.linspace(0, 1, 100).reshape(-1, 1)
    if len(all_real_curves):
        axs[0].plot(x, np.mean(np.vstack(all_real_curves), axis=0), linewidth=2, color='k')
    if len(all_shuffled_curves):
        axs[1].plot(x, np.mean(np.vstack(all_shuffled_curves), axis=0), linewidth=2, color='k')
    if len(all_difference_curves):
        axs[2].plot(x, np.mean(np.vstack(all_difference_curves), axis=0), linewidth=2, color='k')

    # --- Add vertical lines at AUC boundaries (black, dashed)
    x_vals = x.squeeze()
    left_idx = max(0, min(AUC_XMIN, len(x_vals) - 1))
    right_idx = max(0, min(AUC_XMAX, len(x_vals) - 1))
    for ax in axs:
        print(DIR_13, DIR_12, DIR_11, DIR_21, DIR_31)
        print(interval13_left, interval13_right, interval12_left, interval12_right, interval11_left, interval11_right,
              interval21_left, interval21_right, interval31_left, interval31_right)
        #ax.axvline(x=float(x_vals[DIR_13]), color="black", linewidth=1, label="1:3")
        #ax.axvline(x=float(x_vals[DIR_12]), color="black", linewidth=1, label="1:2")
        #ax.axvline(x=float(x_vals[DIR_11]), color="black", linewidth=1, label="1:1")
        #ax.axvline(x=float(x_vals[DIR_21]), color="black", linewidth=1, label="2:1")
        # #ax.axvline(x=float(x_vals[DIR_31]), color="black", linewidth=1, label="3:1")
        # ax.axvline(x=float(x_vals[interval13_left]), color="black", linestyle="--", linewidth=1)
        # ax.axvline(x=float(x_vals[interval13_right]), color="black", linestyle="--", linewidth=1)
        # ax.axvline(x=float(x_vals[interval12_left]), color="black", linestyle="--", linewidth=1)
        # ax.axvline(x=float(x_vals[interval12_right]), color="black", linestyle="--", linewidth=1)
        # ax.axvline(x=float(x_vals[interval11_left]), color="black", linestyle="--", linewidth=1)
        # ax.axvline(x=float(x_vals[interval11_right]), color="black", linestyle="--", linewidth=1)
        # ax.axvline(x=float(x_vals[interval21_left]), color="black", linestyle="--", linewidth=1)
        # ax.axvline(x=float(x_vals[interval21_right]), color="black", linestyle="--", linewidth=1)
        # ax.axvline(x=float(x_vals[interval31_left]), color="black", linestyle="--", linewidth=1)
        # ax.axvline(x=float(x_vals[interval31_right]), color="black", linestyle="--", linewidth=1)
        ax.axvspan(x_vals[interval13_left], x_vals[interval13_right],
                   facecolor = "0.2", alpha = 0.08, linewidth=0, zorder=0)
        ax.axvspan(x_vals[interval12_left], x_vals[interval12_right],
                   facecolor="0.2", alpha=0.08, linewidth=0,zorder=0)
        ax.axvspan(x_vals[interval11_left], x_vals[interval11_right],
                   facecolor="0.2", alpha=0.08, linewidth=0, zorder=0)
        ax.axvspan(x_vals[interval21_left], x_vals[interval21_right],
                   facecolor="0.2", alpha=0.08, linewidth=0, zorder=0)
        ax.axvspan(x_vals[interval31_left], x_vals[interval31_right],
                   facecolor="0.2", alpha=0.08, linewidth=0, zorder=0)

    axs[0].set_title("Music Real curves")
    axs[1].set_title("Music Mean shuffled curves")
    axs[2].set_title("Difference between real and shuffled")

    for ax in axs:
        ax.set_ylabel("Density")
        ax.set_xlabel("Diadic interval ratio (DIR)")
        ax.set_ylim((0, 40))
    axs[2].set_ylim((-20, 20))
    plt.tight_layout()

    # Save outputs
    dirpath, _ = os.path.split(syllable_csv)
    if aucs:
        aucs_df = pd.DataFrame(aucs, columns=["title", "AUC_real", "AUC_shuffled"])
        aucs_df.to_csv(os.path.join(dirpath, 'auc_values_music.csv'), index=False)
    else:
        print("No AUCs to save for music.")
    plt.savefig(os.path.join(dirpath, 'music_dir_kde.png'), dpi=300)

