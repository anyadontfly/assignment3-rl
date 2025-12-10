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
According to the normalized time breakd own plot, there is speed up in normalized time for k=2 comparing to k=1. However, the normalized time increases as k increases to larger numbers. The reason for this is because trajectory generation can be batched for larger k but advantage and log prob calculation are repeated k times which can not be batched. This is why normalized generation time is decreasing while learning time increases with proportion of k.


## Problem 9
The average time of weight transfer without RDT is around 353.6781 ms, and average time of weight transfer with RDT is around 29.3782 ms. The RDT brings around 12x speed up with NCCL collective group. 