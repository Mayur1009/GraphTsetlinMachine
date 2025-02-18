#include <curand_kernel.h>

extern "C" {

__global__ void get_ta_states(unsigned int *ta_state, unsigned int chunks, unsigned int literals, unsigned int *out) {
    // :param: global_ta_state
    // Array of TAs for each literal.
    //
    // :param: chunks
    // Number of chunks.
    //
    // :param: literals
    // Number of literals.
    //
    // Shape:
    // For Clauses:
    //        (         CLAUSES,        LA_CHUNKS,           STATE_BITS)
    //        (number of clauses, number of chunks, number of state bits)
    // For Messages:
    //        (         CLAUSES,            MESSAGE_CHUNKS,           STATE_BITS)
    //        (number of clauses, number of message chunks, number of state bits)
    //
    //
    // :param: out
    // Output array to store the state values of each TA.
    // Shape: (CLAUSES, literals)
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (int clause = index; clause < CLAUSES; clause += stride) {
        unsigned int *local_ta_state = &ta_state[clause * chunks * STATE_BITS];

        for (int literal = 0; literal < literals; ++literal) {
            unsigned int state = 0;
            int chunk_nr = literal / INT_SIZE;
            int chunk_pos = literal % INT_SIZE;

            for (int bit = 0; bit < STATE_BITS; ++bit)
                if (local_ta_state[chunk_nr * STATE_BITS + bit] & (1 << chunk_pos)) state |= (1 << bit);

            out[clause * literals + literal] = state;
        }
    }
}
}
