import pandas as pd
import numpy as np
import os
import helpers as h

#set intervaks as global variables
interval13_left = 1/4.25
interval13_right = 1/3.75

interval12_left = 1/3.25
interval12_right = 1/2.75

interval11_left = 0.444
interval11_right = 0.555

interval21_left = 1-interval12_right
interval21_right = 1-interval12_left

interval31_left = 1-interval13_right
interval31_right = 1-interval13_left



def percent_iso_music(file, n_iter=500):
    import pandas as pd
    import numpy as np
    import os
    import helpers as h
    print("Intervals \n", "1:3", interval13_left, interval13_right, "\n",
          "1:2", interval12_left, interval12_right, "\n",
          "1:1", interval11_left, interval11_right, "\n",
          "2:1", interval21_left, interval21_right, "\n",
          "3:1", interval31_left, interval31_right, "\n")
    print('Calculating percentage of music DIRs within the isochrony zone.')
    df = pd.read_csv(file)
    obs_rand_mean = []
    rand_matrix_13 = []
    rand_matrix_12 = []
    rand_matrix_11 = []
    rand_matrix_21 = []
    rand_matrix_31 = []
    file_labels = []

    for song_id, song_df in df.groupby("title"):
        print(f"Processing {song_id}")

        # Extract relevant arrays once to avoid repeated DataFrame access
        is_note = song_df['is_note'].values
        duration = song_df['duration'].values
        phrase_ids = song_df['phrase_id'].values

        # Compute real IOIs within phrases
        iois = [duration[i] for i in range(len(duration) - 1)
                if is_note[i] and is_note[i + 1] and phrase_ids[i] == phrase_ids[i + 1]]

        if len(iois) < 11:
            print(f"Skipping {song_id} due to insufficient IOIs")
            continue

        real_dirs = np.array([iois[i] / (iois[i] + iois[i + 1]) for i in range(len(iois) - 1)])

        #calculate percentage of DIRs within each rhythm window (1:3, 1:2, 1:1, 2:1, 3:1)
        prop_real_13 = 100 * np.sum((real_dirs > interval13_left) & (real_dirs < interval13_right)) / len(real_dirs)
        prop_real_12 = 100 * np.sum((real_dirs > interval12_left) & (real_dirs < interval12_right)) / len(real_dirs)
        prop_real_11 = 100 * np.sum((real_dirs > interval11_left) & (real_dirs < interval11_right)) / len(real_dirs)
        prop_real_21 = 100 * np.sum((real_dirs > interval21_left) & (real_dirs < interval21_right)) / len(real_dirs)
        prop_real_31 = 100 * np.sum((real_dirs > interval31_left) & (real_dirs < interval31_right)) / len(real_dirs)

        #initialize empty arrays for randomizations
        prop_rand_13 = np.full(n_iter, np.nan)
        prop_rand_12 = np.full(n_iter, np.nan)
        prop_rand_11 = np.full(n_iter, np.nan)
        prop_rand_21 = np.full(n_iter, np.nan)
        prop_rand_31 = np.full(n_iter, np.nan)

        grouped = list(song_df.groupby("phrase_id"))

        for iter in range(n_iter):
            all_shuffled_dirs = []

            for _, phrase_df in grouped:
                is_note_phrase = phrase_df['is_note'].values
                duration_phrase = phrase_df['duration'].values
                iois_phrase = [duration_phrase[i] for i in range(len(duration_phrase) - 1)
                               if is_note_phrase[i] and is_note_phrase[i + 1]]
                if len(iois_phrase) < 2:
                    continue

                shuffled_iois = np.random.permutation(iois_phrase)
                shuffled_dirs = [shuffled_iois[i] / (shuffled_iois[i] + shuffled_iois[i + 1])
                                 for i in range(len(shuffled_iois) - 1)]
                all_shuffled_dirs.extend(shuffled_dirs)

            if len(all_shuffled_dirs) >= 10:
                all_shuffled_dirs = np.array(all_shuffled_dirs)

                #calculate random percentages for this iteration
                prop_rand_13[iter] = 100 * np.sum(
                    (all_shuffled_dirs > interval13_left) & (all_shuffled_dirs < interval13_right)) / len(
                    all_shuffled_dirs)
                prop_rand_12[iter] = 100 * np.sum(
                    (all_shuffled_dirs > interval12_left) & (all_shuffled_dirs < interval12_right)) / len(
                    all_shuffled_dirs)
                prop_rand_11[iter] = 100 * np.sum(
                    (all_shuffled_dirs > interval11_left) & (all_shuffled_dirs < interval11_right)) / len(
                    all_shuffled_dirs)
                prop_rand_21[iter] = 100 * np.sum(
                    (all_shuffled_dirs > interval21_left) & (all_shuffled_dirs < interval21_right)) / len(
                    all_shuffled_dirs)
                prop_rand_31[iter] = 100 * np.sum(
                    (all_shuffled_dirs > interval31_left) & (all_shuffled_dirs < interval31_right)) / len(
                    all_shuffled_dirs)


        #get average and std for randomizations for each interval
        mean_rand_13 = np.nanmean(prop_rand_13)
        mean_rand_12 = np.nanmean(prop_rand_12)
        mean_rand_11 = np.nanmean(prop_rand_11)
        mean_rand_21 = np.nanmean(prop_rand_21)
        mean_rand_31 = np.nanmean(prop_rand_31)

        std_rand_13 = np.nanstd(prop_rand_13)
        std_rand_12 = np.nanstd(prop_rand_12)
        std_rand_11 = np.nanstd(prop_rand_11)
        std_rand_21 = np.nanstd(prop_rand_21)
        std_rand_31 = np.nanstd(prop_rand_31)

        #append real percentaces, mean and std for random
        obs_rand_mean.append([song_id, prop_real_13, prop_real_12,
                                prop_real_11, prop_real_21, prop_real_31,
                                mean_rand_13, mean_rand_12,
                                mean_rand_11, mean_rand_21, mean_rand_31,
                                std_rand_13, std_rand_12,
                                std_rand_11, std_rand_21, std_rand_31])

        #save all randomizations
        rand_matrix_13.append(prop_rand_13)
        rand_matrix_12.append(prop_rand_12)
        rand_matrix_11.append(prop_rand_11)
        rand_matrix_21.append(prop_rand_21)
        rand_matrix_31.append(prop_rand_31)

        file_labels.append(song_id)

    obs_mean_random_df = pd.DataFrame(obs_rand_mean,
                                      columns=["title", "observed_1:3", "observed_1:2",
                                               "observed_1:1", "observed_2:1", "observed_3:1",
                                               "shuffled_1:3", "shuffled_1:2", "shuffled_1:1",
                                               "shuffled_2:1", "shuffled_3:1", "std_shuffled_1:3",
                                               "std_shuffled_1:2", "std_shuffled_1:1", "std_shuffled_2:1",
                                               "std_shuffled_3:1"])

    rand_matrix_df_13 = pd.DataFrame(np.array(rand_matrix_13).T,
                                  columns=h.make_valid_column_names(file_labels))
    rand_matrix_df_12 = pd.DataFrame(np.array(rand_matrix_12).T,
                                     columns=h.make_valid_column_names(file_labels))
    rand_matrix_df_11 = pd.DataFrame(np.array(rand_matrix_11).T,
                                     columns=h.make_valid_column_names(file_labels))
    rand_matrix_df_21 = pd.DataFrame(np.array(rand_matrix_21).T,
                                     columns=h.make_valid_column_names(file_labels))
    rand_matrix_df_31 = pd.DataFrame(np.array(rand_matrix_31).T,
                                     columns=h.make_valid_column_names(file_labels))

    folder, base = os.path.split(file)
    base_name, _ = os.path.splitext(base)
    output_path = os.path.join(folder, f"{base_name}_percent_iso.xlsx")

    with pd.ExcelWriter(output_path) as writer:
        obs_mean_random_df.to_excel(writer, sheet_name="obs+shuffled_mean", index=False)
        rand_matrix_df_13.to_excel(writer, sheet_name="500_shuffled_all_13", index=False)
        rand_matrix_df_12.to_excel(writer, sheet_name="500_shuffled_all_12", index=False)
        rand_matrix_df_11.to_excel(writer, sheet_name="500_shuffled_all_11", index=False)
        rand_matrix_df_21.to_excel(writer, sheet_name="500_shuffled_all_21", index=False)
        rand_matrix_df_31.to_excel(writer, sheet_name="500_shuffled_all_31", index=False)


    print(f"Saved output Excel to {output_path}")


def compute_percent_iso(file, n_iter=500):
    # Load input
    df = pd.read_excel(file)
    print(
        'Calculating percentage of music DIRs within the isochorony zone.',
        'Shuffling is done within song, real and random percentages are within species.')

    # Add IOI and DIR columns
    df = h.add_IOI_DIR_from_onsets(df)

    # Drop rows with NaNs in IOI or DIR
    df = df.dropna(subset=["IOI", "DIR"])

    # Convert species to string
    df["species_birdtree"] = df["species_birdtree"].astype(str)
    unique_species = df["species_birdtree"].unique()

    obs_rand_mean = []
    species_labels = []
    rand_matrix_13 = []
    rand_matrix_12 = []
    rand_matrix_11 = []
    rand_matrix_21 = []
    rand_matrix_31 = []

    for species in unique_species:
        print(species)
        df_species = df[df["species_birdtree"] == species].copy()
        if df_species.empty:
            continue

        observed_dir = df_species["DIR"].dropna().values
        if len(observed_dir) < 10:
            continue

        # Observed % in isochrony and small integer ratio ranges
        #prop_obs = 100 * np.sum((observed_dir > 0.444) & (observed_dir < 0.555)) / len(observed_dir)
        prop_real_13 = 100 * np.sum((observed_dir > interval13_left) & (observed_dir < interval13_right)) / len(observed_dir)
        prop_real_12 = 100 * np.sum((observed_dir > interval12_left) & (observed_dir < interval12_right)) / len(observed_dir)
        prop_real_11 = 100 * np.sum((observed_dir > interval11_left) & (observed_dir < interval11_right)) / len(observed_dir)
        prop_real_21 = 100 * np.sum((observed_dir > interval21_left) & (observed_dir < interval21_right)) / len(observed_dir)
        prop_real_31 = 100 * np.sum((observed_dir > interval31_left) & (observed_dir < interval31_right)) / len(observed_dir)

        # Shuffle IOIs within each song for the species

        # initialize empty arrays for randomizations
        prop_rand_13 = np.full(n_iter, np.nan)
        prop_rand_12 = np.full(n_iter, np.nan)
        prop_rand_11 = np.full(n_iter, np.nan)
        prop_rand_21 = np.full(n_iter, np.nan)
        prop_rand_31 = np.full(n_iter, np.nan)

        song_groups = df_species.groupby("seg_id")

        for j in range(n_iter):
            all_random_dirs = []

            for _, song_df in song_groups:
                song_iois = song_df["IOI"].dropna().values
                if len(song_iois) < 2:
                    continue
                shuffled_ioi = np.random.permutation(song_iois)
                random_dir = [shuffled_ioi[k] / (shuffled_ioi[k] + shuffled_ioi[k + 1])
                              for k in range(len(shuffled_ioi) - 1)]
                all_random_dirs.extend(random_dir)

            if len(all_random_dirs) == 0:
                prop_rand_13[j] = np.nan
                prop_rand_12[j] = np.nan
                prop_rand_11[j] = np.nan
                prop_rand_21[j] = np.nan
                prop_rand_31[j] = np.nan
            else:
                all_random_dirs = np.array(all_random_dirs)
                prop_rand_13[j] = 100 * np.sum(
                    (all_random_dirs > interval13_left) & (all_random_dirs < interval13_right)) / len(
                    all_random_dirs)
                prop_rand_12[j] = 100 * np.sum(
                    (all_random_dirs > interval12_left) & (all_random_dirs < interval12_right)) / len(
                    all_random_dirs)
                prop_rand_11[j] = 100 * np.sum(
                    (all_random_dirs > interval11_left) & (all_random_dirs < interval11_right)) / len(
                    all_random_dirs)
                prop_rand_21[j] = 100 * np.sum(
                    (all_random_dirs > interval21_left) & (all_random_dirs < interval21_right)) / len(
                    all_random_dirs)
                prop_rand_31[j] = 100 * np.sum(
                    (all_random_dirs > interval31_left) & (all_random_dirs < interval31_right)) / len(
                    all_random_dirs)

        if np.all(np.isnan(prop_rand_13)):
            continue

        # get average and std for randomizations for each interval
        mean_rand_13 = np.nanmean(prop_rand_13)
        mean_rand_12 = np.nanmean(prop_rand_12)
        mean_rand_11 = np.nanmean(prop_rand_11)
        mean_rand_21 = np.nanmean(prop_rand_21)
        mean_rand_31 = np.nanmean(prop_rand_31)

        std_rand_13 = np.nanstd(prop_rand_13)
        std_rand_12 = np.nanstd(prop_rand_12)
        std_rand_11 = np.nanstd(prop_rand_11)
        std_rand_21 = np.nanstd(prop_rand_21)
        std_rand_31 = np.nanstd(prop_rand_31)

        # append real percentaces, mean and std for random
        obs_rand_mean.append([species, prop_real_13, prop_real_12,
                                  prop_real_11, prop_real_21, prop_real_31,
                                  mean_rand_13, mean_rand_12,
                                  mean_rand_11, mean_rand_21, mean_rand_31,
                                  std_rand_13, std_rand_12,
                                  std_rand_11, std_rand_21, std_rand_31])

        # save all randomizations
        rand_matrix_13.append(prop_rand_13)
        rand_matrix_12.append(prop_rand_12)
        rand_matrix_11.append(prop_rand_11)
        rand_matrix_21.append(prop_rand_21)
        rand_matrix_31.append(prop_rand_31)

        species_labels.append(species)

    # Build output tables
    #obs_rand_df = pd.DataFrame(obs_rand_mean, columns=["species_birdtree", "observed_%", "shuffled_%"])
    #rand_matrix_df = pd.DataFrame(np.array(rand_matrix).T,
                                  #columns=h.make_valid_column_names(species_labels))

    obs_rand_df = pd.DataFrame(obs_rand_mean, columns=["species_birdtree", "observed_1:3", "observed_1:2",
                                               "observed_1:1", "observed_2:1", "observed_3:1",
                                               "shuffled_1:3", "shuffled_1:2", "shuffled_1:1",
                                               "shuffled_2:1", "shuffled_3:1", "std_shuffled_1:3",
                                               "std_shuffled_1:2", "std_shuffled_1:1", "std_shuffled_2:1",
                                               "std_shuffled_3:1"])

    rand_matrix_df_13 = pd.DataFrame(np.array(rand_matrix_13).T,
                                     columns=h.make_valid_column_names(species_labels))
    rand_matrix_df_12 = pd.DataFrame(np.array(rand_matrix_12).T,
                                     columns=h.make_valid_column_names(species_labels))
    rand_matrix_df_11 = pd.DataFrame(np.array(rand_matrix_11).T,
                                     columns=h.make_valid_column_names(species_labels))
    rand_matrix_df_21 = pd.DataFrame(np.array(rand_matrix_21).T,
                                     columns=h.make_valid_column_names(species_labels))
    rand_matrix_df_31 = pd.DataFrame(np.array(rand_matrix_31).T,
                                     columns=h.make_valid_column_names(species_labels))

    # Save output
    folder, base = os.path.split(file)
    base_name, _ = os.path.splitext(base)
    output_path = os.path.join(folder, f"{base_name}_percent_iso.xlsx")

    with pd.ExcelWriter(output_path) as writer:
        obs_rand_df.to_excel(writer, sheet_name="obs+shuffled_mean", index=False)
        rand_matrix_df_13.to_excel(writer, sheet_name="500_shuffled_all_13", index=False)
        rand_matrix_df_12.to_excel(writer, sheet_name="500_shuffled_all_12", index=False)
        rand_matrix_df_11.to_excel(writer, sheet_name="500_shuffled_all_11", index=False)
        rand_matrix_df_21.to_excel(writer, sheet_name="500_shuffled_all_21", index=False)
        rand_matrix_df_31.to_excel(writer, sheet_name="500_shuffled_all_31", index=False)


    print(f"Saved output Excel to {output_path}")


    with pd.ExcelWriter(output_path) as writer:
        obs_rand_df.to_excel(writer, sheet_name="obs+shuffled_mean", index=False)
        rand_matrix_df.to_excel(writer, sheet_name="500_shuffled_all", index=False)

    print(f"Saved output Excel to {output_path}")
