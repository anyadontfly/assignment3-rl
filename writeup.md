## Problem 5
1. Generation stage takes longer time than policy update stage. This is because during generation stage, the model have to run forward path for every generated token in autoregressive generation fashion. The update stage includs policy log prob calculation, backward computation, and optimizer step. However, log prob calculation can be batched and only run one forward to get log prob for every token, which is much more efficient than generation stage. 
2. Prompt: Write a story that includes the word: sound  
Response: Her dad smiled and said: "I didn't know, the birds were talking! How does that sound?"The old man and his dadre at the same time in the tree when the sky was full of colorful stars.<|endoftext|>Once upon a time, there was a big fish named Bob  
Prompt: Write a story that includes the word: town  
Response: Once upon a time, there was a big, small town. In this town, there was a lot of traffic. People were very happy and excited. They wanted to go to different places.One day, a little boy named Tim went to the left. He saw many things to


## Problem 6
![kl div](kl_divergence_over_time.png)
In my results, KL divergence grows exponentially at early stage and then remains for later steps. However, high KL divergence does not relates to higher validation rewards.

## Problem 7
![normalized time](timing_by_k.png)
There are intotal 512 samples processed (32 steps, group size 4, rollout batch size 4). 
| k value | Throughput (samples/sec) |
|---------|--------------------------|
|    1     |            25.37              |
|    2     |             35.53             |
|    4     |             41.83             |
|    8     |             41.96             |

| k value | Speedup |
|---------|---------|
|    1    |   1.00x |
|    2    |   1.40x |
|    4    |   1.65x |
|    8    |   1.65x |

As k increases, throughput improves due to better batching efficiency, with speedup plateauing around k=4.

## Problem 8



## Problem 9
The average time of weight transfer without RDT is around 353.6781 ms, and average time of weight transfer with RDT is around 29.3782 ms. The RDT brings around 12x speed up with NCCL collective group. 