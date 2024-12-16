# Neural Turing Machines

Replicating the paper "Neural Turing Machines".

## TODO - PLAN
- [ ] Figure out the training setup (http://karpathy.github.io/2015/05/21/rnn-effectiveness/)
    - Entering zeros?
    - Simply using padding??
- [ ] Implement the first experiment with a simple LSTM, check that it learns properly.
- [ ] Implement the NTM with an MLP controller, make it work. Then do the same with the LSTM controller.
- [ ] Perform the rest of the experiments.
- [ ] Tinker with the models, ask questions.

## Notes

- Basic Idea -> Augment a neural network (controller) with **memory**.


**Reading:**
- The memory matrix has shape [N, M], i.e. N memory vectors/locations of size M.
- A read head emits a vector of weightings w_t.
- The read vector is just the weighted combination of all the vectors in memory.

**Writing:**
- The writing operation is decomposed into an *erase* followed by an *add*.
    - Erase:
        - We have a weighting w_t emitted by a write head at time t
        - We have an erase vector e_t whose M elements lie in (0, 1)
        - The memory vectors M_(t-1) are modified as follows: M_t = M_(t-1)[1 - w_t*e_t] (elementwise)
        - Intuition, when both the weighting and the erase vector are 1s, then the vector at a given location is set to zero.
    - Add:
        - We still have the same weighting
        - The head also emits an add vector a_t, which is added to the memory.
        - The vector is added to memory as M_t = M_t + w_t*a_t (also elementwise)

That's cool but how do we actually emit the weighting/erase and add vectors?

- Two addressing mechanisms are combined: 
    - **content-based addressing** focuses attention on locations based on the similarity between their current values and values emitted by the controller.
        - In other words, the controller outputs an approximate output and then the exact value ir retrieved from memory. My intuition tells me that this is not very useful.
        - In the paper, they mention that this addressing method fails in tasks where the content of a variable is arbitrary, such as in arithmetic. For example, in x*y, both variables can take whatever value but the procedure should be clearly defined. In this case, we should store the values of x and y, store them in different adresses and then retrieve them and perform the multiplication algorithm.
    - In the latter example, variables are addressed by location, not content, which is the second addressing mechanism: **location-based addressing**. 
    - However, content-based is more general as we could store location information in memory, but the authors say that explicitly providing location-based addressing was essential for some forms of generalization.

- **Focusing by content:** Each head produces a key vector k_t of length M. In this case, the weighting is obtained by (i) computing the (cosine) similarity of k with each vector in memory and (ii) softmaxing the resulting vector multiplied by a scalar b_t that controls the "sharpness" of the final vector. This is really similar to what we do in attention (i.e. the keys and queries are used to obtain a weighted vector to aggregate the value vectors). We call the resulting vector wc_t (c for content)

- **Focusing by location:** The idea here is to make it easier for the controller to move across diferent locations. This is done by implementing a rotational shift of a weighting. Each head emits a scalar g_t in the range (0, 1), which is used to obtain the **gated weighting** wg_t = g_t * wc_t + (1 - g_t) * w_(t-1).
    - If g_t = 0, then the content weighting is ignored and the weighting from the previous step is used. 
- After interpolation, each heads emits a shift weighting s_t vector. What I understood is that, for example, if only shifts of -1, 0 and 1 are allowed, then s_t will be a vector of size 3 representing the "intensities" in which these shifts are performed. Thus, w_hat will be obtained by convolving this shift weighting vector s_t with the previous gated weighting wg_t (the conv is computed modulo N, i.e. the end and the start is connected).
- Finally, as the previous operation can "blur" the weighting vector, another sharpening peration is performed via another emitted gamma_t scalar, i.e. we raise w_t^gamma_t and normalise.  

Which controller to use?
- The authors experiment with a simple MLP and a LSTM.
- An LSTM has internal memory. This is analogous to the registers in a CPU.
- The memory access pattern of an MLP should be less obscure, but we are also limited because the number of concurrent read and write heads imposes a bottleneck on the type of computation the NTM can perform. 
    - One head -> unary transform on a single memory vector at each timestep.
    - Two heads -> Binary vector transforms, etc.


### First experiment: copy

- Tests whether the NTM can store and recall a long sequence of arbitrary information.
- Sequences of 8-bit random vectors of size 1-20.
- Train sequence-by-sequence, i.e. no batch
- Format: abcd0abcd

---
**LSTM**

Assume we have a sequence 10x8. The LSTM cell will process each item of the sequence iteratively. Assuming we don't use batches, the input to the LSTM will be x_0 of shape (1, 8) and will output h_1 and c_1, of shapes (1, hidden_size). Then, we will enter the next item of the sequence x_1, together with h_1 and c_1. 

- At the start, h_0 and c_0 are simply vectors of zeroes.
- Once that the complete sequence is feed, we keep iterating but we enter a vector of zeros for x_i. In other words, once that we have fed the input to the LSTM, we stop providing information. At each step, then the LSTM processes its internal states h_i and c_i.
- We will put a Linear layer to project the hidden activations into a vector of size 8. This will be the predicted sequence.

```
        a b c d
| | | | | | | |
a b c d 0 0 0 0
```

---