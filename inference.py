# Mini inference code here, add sampling settings and inference improvements

# Greedy vs Multinomial option
# stop tokens option 
# Max_new_tokens, is an option, also should be forced to be below ctx len 
# add TopK and TopP
# add a streaming option 
# Get KV cache working
# prefix suffix options 
# add repetition penalty options
# expand this to more tokenizers

# Tail free sampling, - Tail free sampling (TFS) is a text generation technique that aims to reduce the impact of less likely tokens, which may be less relevant, less coherent, or nonsensical, on the output. Similar to Top-P it tries to determine the bulk of the most likely tokens dynamically. But TFS filters out logits based on the second derivative of their probabilities. Adding tokens is stopped after the sum of the second derivatives reaches the parameter z. In short: TFS looks how quickly the probabilities of the tokens decrease and cuts off the tail of unlikely tokens using the parameter z. Typical values for z are in the range of 0.9 to 0.95. A value of 1.0 would include all tokens, and thus disables the effect of TFS.

# Locally Typical Sampling - Locally typical sampling promotes the generation of contextually coherent and diverse text by sampling tokens that are typical or expected based on the surrounding context. By setting the parameter p between 0 and 1, you can control the balance between producing text that is locally coherent and diverse. A value closer to 1 will promote more contextually coherent tokens, while a value closer to 0 will promote more diverse tokens. A value equal to 1 disables locally typical sampling.

# Smooth Sampling / Quadratic Sampling
#    - This sampling method differs from the truncation samplers (Top K, Top P, Min P) in that it is doing something that is fundamentally different to the raw token scores.
#    - We are tweaking the logits using a quadratic transformation, based on each token score's distance from the top token (the transformation centers on the top logit.) The coefficient is decided by the "smoothing factor" value.
#    - This is hard to explain without looking at the visualization, but the idea is that we make the topmost tokens more evenly probable while reducing the probability of extremely unlikely tokens.
#    - Higher values will be more deterministic, but it doesn't work quite like lower temperature would, as the scores of extremely closely competing top tokens will barely change. So if the original probabilities were 50/50 on the top two tokens, they will likely remain that way with higher smoothing factor values.
#    - The idea is that this can be used as an "all in one" sampler by itself, or in tandem with other methods if desired.

# The muse https://github.com/the-crypt-keeper/the-muse
# add beam search 
# Drugs https://github.com/EGjoni/DRUGS 
# minimum bayes risk decoding [https://github.com/ZurichNLP/mbr](https://github.com/ZurichNLP/mbr?scrlybrkr=4c9c022b)

# grammars
# - https://github.com/ggerganov/llama.cpp/blob/master/grammars/README.md
# - https://github.com/ggerganov/llama.cpp#constrained-output-with-grammars

# Mirostat
# - https://arxiv.org/abs/2007.14966

# EAGLE
# - https://arxiv.org/abs/2401.15077
# - https://github.com/SafeAILab/EAGLE

# Dynamic Temp
# - https://github.com/ggerganov/llama.cpp/issues/3483
