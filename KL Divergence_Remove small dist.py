import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import ast
from itertools import combinations
from tqdm import tqdm
from scipy.stats import entropy

# read file
file_path = '../GSM3347525NR_FDR_0.1_pseudoGEM_10000_enrichTest_master.txt'
df = pd.read_csv(file_path, delimiter='\t')
# PASS tag only
df_pass = df[df['decis1'] == 'PASS'].copy()
# bin size
bin_size = 500

# filter out small pairwise distance (0-1500)
def filter_pairwise_diff(diffs):
    return [diff for diff in diffs if diff > 1500]

# pre-process data text and do binning
def extract_and_calculate_bins(row):
    frag_coord_string = row['List_of_frag_coord']
    fragments = frag_coord_string.split(';')    # several segments in form: chr2L:123-456
    representative_bins = []
    for fragment in fragments:
        chrom, coords = fragment.split(':')     # eg. chrom = chr2L, coords = 123-456
        start, end = map(int, coords.split('-'))    # start = 123, end = 456
        start_bin = (start // bin_size) + 1     # calculate bin
        end_bin = (end // bin_size) + 1

        # use midpoint as the representative bin (eg. 5000-5002 -> rep_bin = 5001)
        if start_bin == end_bin:
            representative_bin = start_bin
        else:
            representative_bin = (start_bin + end_bin) // 2
        representative_bins.append(representative_bin)
    return representative_bins

# calculate pairwise-diff within every piece of data (each row)
def calculate_pairwise_diff(representative_bins):
    pairwise_diffs = []
    for (bin1, bin2) in combinations(representative_bins, 2):
        pairwise_diffs.append(abs(bin1 - bin2))
    return filter_pairwise_diff(pairwise_diffs)

# Iterate every piece of data (row) to get all pairwise diff; store them by chromosome type (chr2L, chr2R ...)
chromosome_diffs = {}
for index, row in tqdm(df_pass.iterrows(), total=df_pass.shape[0], desc="Processing Rows"):
    representative_bins = extract_and_calculate_bins(row)
    pairwise_diffs = calculate_pairwise_diff(representative_bins)
    chromosome = row['GEM_coord'].split(':')[0]

    if chromosome not in chromosome_diffs:
        chromosome_diffs[chromosome] = []
    chromosome_diffs[chromosome].extend(pairwise_diffs)

# plotting histogram
chromosome_histograms = {}
for chromosome, diffs in chromosome_diffs.items():
    if len(diffs) > 0:
        counts, bin_edges = np.histogram(diffs, bins=30, density=True)
        chromosome_histograms[chromosome] = counts

        plt.figure(figsize=(8, 6))
        plt.hist(diffs, bins=30, density=True, edgecolor='black', alpha=0.75)
        plt.title(f'Normalized Pairwise Differences Histogram for {chromosome}')
        plt.xlabel('Pairwise Difference')
        plt.ylabel('Probability')
        plt.grid(True)
        plt.show()

# Calculate KL Divergence
kl_divergences = {}
chromosomes = list(chromosome_histograms.keys())
for chrom1, chrom2 in combinations(chromosomes, 2):
    hist1 = chromosome_histograms[chrom1]
    hist2 = chromosome_histograms[chrom2]
    kl_div = entropy(hist1, hist2)
    kl_divergences[(chrom1, chrom2)] = kl_div

# print KL Divergence
for (chrom1, chrom2), kl in kl_divergences.items():
    print(f"KL Divergence between {chrom1} and {chrom2}: {kl}")
