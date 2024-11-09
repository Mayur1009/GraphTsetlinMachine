extern "C" {

// Increment the states of each of those 32 Tsetlin Automata flagged in the active bit vector.
__device__ inline void inc(unsigned int *ta_state, int chunk, unsigned int active) {
    unsigned int carry, carry_next;
    int id = chunk * STATE_BITS;
    carry = active;
    for (int b = 0; b < STATE_BITS; ++b) {
        if (carry == 0) break;

        carry_next = ta_state[id + b] & carry;        // Sets carry bits (overflow) passing on to next bit
        ta_state[id + b] = ta_state[id + b] ^ carry;  // Performs increments with XOR
        carry = carry_next;
    }

    if (carry > 0) {
        for (int b = 0; b < STATE_BITS; ++b) {
            ta_state[id + b] |= carry;
        }
    }
}

// Decrement the states of each of those 32 Tsetlin Automata flagged in the active bit vector.
__device__ inline void dec(unsigned int *ta_state, int chunk, unsigned int active) {
    unsigned int carry, carry_next;
    int id = chunk * STATE_BITS;
    carry = active;
    for (int b = 0; b < STATE_BITS; ++b) {
        if (carry == 0) break;

        carry_next = (~ta_state[id + b]) & carry;     // Sets carry bits (overflow) passing on to next bit
        ta_state[id + b] = ta_state[id + b] ^ carry;  // Performs increments with XOR
        carry = carry_next;
    }

    if (carry > 0) {
        for (int b = 0; b < STATE_BITS; ++b) {
            ta_state[id + b] &= ~carry;
        }
    }
}

__device__ inline void update_clause_message(curandState *localState, float s, int target_sign, unsigned int *ta_state,
                                             int clause_output, int clause_node, int number_of_include_actions,
                                             int *X) {
    if (target_sign > 0) {
        // Type I Feedback
        for (int la_chunk = 0; la_chunk < MESSAGE_CHUNKS; ++la_chunk) {
            // Generate random bit values
            unsigned int la_feedback = 0;
            for (int b = 0; b < INT_SIZE; ++b) {
                if (curand_uniform(localState) <= 1.0 / s) {
                    la_feedback |= (1 << b);
                }
            }

            if (clause_output && number_of_include_actions <= MAX_INCLUDED_LITERALS) {
#if BOOST_TRUE_POSITIVE_FEEDBACK == 1
                inc(ta_state, la_chunk, X[clause_node * MESSAGE_CHUNKS + la_chunk]);
#else
                inc(ta_state, la_chunk, X[clause_node * MESSAGE_CHUNKS + la_chunk] & (~la_feedback));
#endif

                dec(ta_state, la_chunk, (~X[clause_node * MESSAGE_CHUNKS + la_chunk]) & la_feedback);
            } else {
                dec(ta_state, la_chunk, la_feedback);
            }
        }
    } else if (target_sign < 0 && clause_output) {
        // Type II Feedback

        for (int la_chunk = 0; la_chunk < MESSAGE_CHUNKS; ++la_chunk) {
            inc(ta_state, la_chunk,
                (~X[clause_node * MESSAGE_CHUNKS + la_chunk]) & (~ta_state[la_chunk * STATE_BITS + STATE_BITS - 1]));
        }
    }
}

__device__ inline void update_clause(curandState *localState, float s, int target_sign, unsigned int *ta_state,
                                     int clause_output, int clause_node, int number_of_include_actions, int *X) {
    if (target_sign > 0) {
        // Type I Feedback
        for (int la_chunk = 0; la_chunk < LA_CHUNKS; ++la_chunk) {
            // Generate random bit values
            unsigned int la_feedback = 0;
            for (int b = 0; b < INT_SIZE; ++b) {
                if (curand_uniform(localState) <= 1.0 / s) {
                    la_feedback |= (1 << b);
                }
            }

            if (clause_output && number_of_include_actions <= MAX_INCLUDED_LITERALS) {
#if BOOST_TRUE_POSITIVE_FEEDBACK == 1
                inc(ta_state, la_chunk, X[clause_node * LA_CHUNKS + la_chunk]);
#else
                inc(ta_state, la_chunk, X[clause_node * LA_CHUNKS + la_chunk] & (~la_feedback));
#endif

                dec(ta_state, la_chunk, (~X[clause_node * LA_CHUNKS + la_chunk]) & la_feedback);
            } else {
                dec(ta_state, la_chunk, la_feedback);
            }
        }
    } else if (target_sign < 0 && clause_output) {
        // Type II Feedback

        for (int la_chunk = 0; la_chunk < LA_CHUNKS; ++la_chunk) {
            inc(ta_state, la_chunk,
                (~X[clause_node * LA_CHUNKS + la_chunk]) & (~ta_state[la_chunk * STATE_BITS + STATE_BITS - 1]));
        }
    }
}

__global__ void update_message(curandState *state, float s, unsigned int *global_ta_state, int number_of_nodes,
                               int *clause_node, int *number_of_include_actions, int *X, int *class_clause_update) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    curandState localState = state[index];

    // Calculate clause output first
    for (unsigned long long clause = index; clause < CLAUSES; clause += stride) {
        unsigned int *ta_state = &global_ta_state[clause * MESSAGE_CHUNKS * STATE_BITS];

        for (unsigned long long class_id = 0; class_id < CLASSES; ++class_id) {
            update_clause_message(&localState, s, class_clause_update[class_id * CLAUSES + clause], ta_state,
                                  clause_node[clause] != -1, clause_node[clause], number_of_include_actions[clause], X);
        }
    }

    state[index] = localState;
}

__global__ void update(curandState *state, float s, unsigned int *global_ta_state, int number_of_nodes, int graph_index,
                       int *clause_node, int *number_of_include_actions, int *X, int *class_clause_update) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    curandState localState = state[index];

    X = &X[graph_index * LA_CHUNKS];

    // Calculate clause output first
    for (unsigned long long clause = index; clause < CLAUSES; clause += stride) {
        unsigned int *ta_state = &global_ta_state[clause * LA_CHUNKS * STATE_BITS];

        for (unsigned long long class_id = 0; class_id < CLASSES; ++class_id) {
            update_clause(&localState, s, class_clause_update[class_id * CLAUSES + clause], ta_state,
                          clause_node[clause] != -1, clause_node[clause], number_of_include_actions[clause], X);
        }
    }

    state[index] = localState;
}
}
