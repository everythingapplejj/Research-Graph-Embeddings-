# Implementation by Jiwoong Jung (JJ)
import numpy as np
from scipy.special import rel_entr
from scipy.spatial.distance import jensenshannon


def normalize_distribution(dist):
    
    total = sum(dist.values())
    if total == 0:
        raise ValueError("Distribution has a sum of 0, cannot normalize.")
    return {k: v / total for k, v in dist.items()}


def align_distributions(p_dict, q_dict):

    keys = set(p_dict.keys()).union(q_dict.keys())
    p = {key: p_dict.get(key, 0) for key in keys}
    q = {key: q_dict.get(key, 0) for key in keys}
    return p, q


def kl_divergence_dict_safe(p_dict, q_dict, smoothing=1e-10):


    p_dict = normalize_distribution(p_dict)
    q_dict = normalize_distribution(q_dict)

    p_dict, q_dict = align_distributions(p_dict, q_dict)


    keys = sorted(p_dict.keys())
    p = np.array([p_dict[key] for key in keys])
    q = np.array([q_dict[key] for key in keys])


    q += smoothing

    q = q / np.sum(q)

    return np.sum(rel_entr(p, q))


def total_variation_distance(p_dict, q_dict):
    
    p_dict, q_dict = align_distributions(p_dict, q_dict)
    

    p_dict = normalize_distribution(p_dict)
    q_dict = normalize_distribution(q_dict)
    

    tv = 0.5 * sum(abs(p_dict[key] - q_dict[key]) for key in p_dict)
    return tv


def verify_pinskers_inequality(p_dict, q_dict, smoothing=1e-10):

    kl = kl_divergence_dict_safe(p_dict, q_dict, smoothing)
    tv = total_variation_distance(p_dict, q_dict)
    bound = np.sqrt(0.5 * kl)
    holds = tv <= bound
    return tv, bound, holds


# Define your dictionaries here:
'''
Declare your dictionaries here. The function will normalize it for you. 
'''
freq_2l_final = ....
freq_2r_final = ....
freq_3l_final = ....
freq_3r_final = ....

freq_datasets = {
    "freq_2l_final": freq_2l_final,
    "freq_2r_final": freq_2r_final,
    "freq_3l_final": freq_3l_final,
    "freq_3r_final": freq_3r_final
}


kl_results = {}
pinsker_results = {}

for dataset1, data1 in freq_datasets.items():
    for dataset2, data2 in freq_datasets.items():
        if dataset1 != dataset2:
            kl_val = kl_divergence_dict_safe(data1, data2)
            kl_results[f"{dataset1} -> {dataset2}"] = kl_val

            tv, bound, holds = verify_pinskers_inequality(data1, data2)
            pinsker_results[f"{dataset1} -> {dataset2}"] = {
                "TV": tv,
                "Bound": bound,
                "Holds": holds
            }


print("=== KL Divergence Results ===")
for pair, kl_value in kl_results.items():
    print(f"{pair}: {kl_value}")


print("\n=== Pinsker's Inequality Verification ===")
for pair, result in pinsker_results.items():
    tv = result["TV"]
    bound = result["Bound"]
    holds = result["Holds"]
    print(f"{pair}: TV = {tv:.6f}, Bound = {bound:.6f}, Holds: {holds}")
