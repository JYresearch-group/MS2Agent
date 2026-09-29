

2016 neg:
Claude + Qwen3-Embedding-8B
Top-1 0.0370
Top-3 0.1111
Top-5 0.1728
Top-10 0.2840
Gemini + Qwen3-Embedding-8B
Top-1 0.0247
Top-3 0.0741
Top-5 0.0864
Top-10 0.1728
GPT-5.4 + Qwen3-Embedding-8B
Top-1 0.0247
Top-3 0.0617
Top-5 0.0617
Top-10 0.1481



retrieval_results ... 
{'embedding_only': {1: 0.08641975308641975, 3: 0.19753086419753085, 5: 0.2962962962962963, 10: 0.4691358024691358}, 
 'mass_filtered': {1: 0.5308641975308642, 3: 0.9012345679012346, 5: 0.9259259259259259, 10: 0.9259259259259259}, 
 'rank_stats': {'raw_mean_rank': 21.666666666666668, 'raw_median_rank': 12.0, 'mass_filtered_mean_rank': 6.530864197530864, 
                'mass_filtered_median_rank': 1.0, 'mean_candidate_count_after_mass_filter': 2.308641975308642, 
                'median_candidate_count_after_mass_filter': 2.0}}


