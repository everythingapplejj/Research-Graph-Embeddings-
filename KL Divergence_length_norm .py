import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import entropy
from itertools import combinations
from tqdm import tqdm


# File path (update this if necessary)
data_path = "../GSM3347525NR_FDR_0.1_pseudoGEM_10000_enrichTest_master.txt"  # Replace with your file path

# Load the dataset
df = pd.read_csv(data_path, sep="\t")
df = df[df['decis1'] == 'PASS'].copy()

# chromosome lengths in Mb
chromosome_lengths = {
    'chr2L': 23.5,
    'chr2R': 25.3,
    'chr3L': 28.1,
    'chr3R': 32.1,
    'chrX': 23.5,
    'chr4': 1.3,
    'chrY': 3.7
}

# Function to compute midpoint bin and pairwise differences within a row
def compute_midpoints_git_and_diffs(row):
    fragments = row['List_of_frag_coord'].split(';')
    midpoints = []
    chrom = row['GEM_coord'].split(':')[0]  # get chrom name
    chrom_len = chromosome_lengths[chrom]
    for fragment in fragments:
        start, end = map(int, fragment.split(':')[1].split('-'))
        start_bin = ((start / chrom_len) // 500) + 1
        end_bin = ((end / chrom_len) // 500) + 1
        midpoint = (start_bin + end_bin) // 2  # Calculate the midpoint
        midpoints.append(midpoint)

    # Calculate pairwise differences
    diffs = []
    n = len(midpoints)
    for i in range(n):
        for j in range(i + 1, n):
            diffs.append(abs(midpoints[j] - midpoints[i]))
    return diffs

# Process each chromosome separately
chromosome_diffs = {}
for chrom in df['GEM_coord'].str.split(':').str[0].unique():
    chrom_diffs = []
    for _, row in tqdm(df[df['GEM_coord'].str.contains(chrom)].iterrows(), total=df.shape[0],
                       desc=f"Processing {chrom}"):
        # Compute midpoints and pairwise differences
        pairwise_diffs = compute_midpoints_and_diffs(row)
        chrom_diffs.extend(pairwise_diffs)

    # Store the differences for the chromosome
    chromosome_diffs[chrom] = chrom_diffs

print(len(chromosome_diffs['chr2L']))
# Function to truncate and normalize PMF for each chromosome
def truncate_and_normalize(diffs, left_threshold=100):
    # Truncate the left side
    truncated_diffs = [d for d in diffs if d >= left_threshold]
    # data_range = max(truncated_diffs) - min(truncated_diffs)
    # bins = max(10, min(500, data_range))
    # Identify the right truncation point dynamically based on PMF
    counts, bin_edges = np.histogram(truncated_diffs, bins=500, density=True)
    cumulative_prob = np.cumsum(counts)
    total_prob = cumulative_prob[-1]

    # Find the point where the cumulative probability reaches a threshold (95%)
    cutoff_index = np.argmax(cumulative_prob >= 0.95 * total_prob)
    right_threshold = bin_edges[cutoff_index]

    # Filter on the right threshold
    final_diffs = [d for d in truncated_diffs if d <= right_threshold]

    # Renormalize PMF
    counts, _ = np.histogram(final_diffs, bins=500, density=True)
    pmf = counts / counts.sum()  # Normalize to get valid PMF
    return pmf, right_threshold


# Normalize and calculate PMF for each chromosome
chromosome_histograms = {}
for chrom, diffs in chromosome_diffs.items():
    if len(diffs) > 0:
        pmf, right_threshold = truncate_and_normalize(diffs)
        chromosome_histograms[chrom] = pmf
        print(f"{chrom}: PMF calculated with left threshold 100 and right threshold {right_threshold}")

# Calculate symmetric KL Divergence
kl_divergences = {}
chromosomes = list(chromosome_histograms.keys())

for chrom1, chrom2 in combinations(chromosomes, 2):
    if len(chromosome_histograms[chrom1]) == len(chromosome_histograms[chrom2]):
        hist1 = chromosome_histograms[chrom1]
        hist2 = chromosome_histograms[chrom2]

        # Avoid zero probabilities when computing KL divergence
        mask = (hist1 > 0) & (hist2 > 0)
        hist1 = hist1[mask]
        hist2 = hist2[mask]

        if len(hist1) > 0 and len(hist2) > 0:
            # Compute KL divergence symmetrically
            kl_div1 = entropy(hist1, hist2)
            kl_div2 = entropy(hist2, hist1)
            symmetric_kl = 0.5 * (kl_div1 + kl_div2)
            kl_divergences[(chrom1, chrom2)] = symmetric_kl

# Print the KL Divergences
print('KL Divergence without chrX and chr4:')
for (chrom1, chrom2), kl in kl_divergences.items():
    if chrom1 != 'chrX' and chrom1 != 'chr4' and chrom2 != 'chrX' and chrom2 != 'chr4':
        print(f"KL Divergence between {chrom1} and {chrom2}: {kl}")
print('KL Divergence involving chrX and chr4:')
for (chrom1, chrom2), kl in kl_divergences.items():
    if chrom1 == 'chrX' or chrom1 == 'chr4' or chrom2 == 'chrX' or chrom2 == 'chr4':
        print(f"KL Divergence between {chrom1} and {chrom2}: {kl}")


