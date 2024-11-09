extern "C" {
__global__ void prepare_message_ta_state(unsigned int *global_ta_state) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (unsigned long long clause = index; clause < CLAUSES; clause += stride) {
        unsigned int *ta_state = &global_ta_state[clause * MESSAGE_CHUNKS * STATE_BITS];
        for (int message_ta_chunk = 0; message_ta_chunk < MESSAGE_CHUNKS; ++message_ta_chunk) {
            for (int b = 0; b < STATE_BITS - 1; ++b) {
                ta_state[message_ta_chunk * STATE_BITS + b] = ~0;
            }
            ta_state[message_ta_chunk * STATE_BITS + STATE_BITS - 1] = 0;
        }
    }
}

__global__ void prepare(curandState *state, unsigned int *global_ta_state, int *clause_weights, int *class_sum) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    curandState localState = state[index];

    for (unsigned long long clause = index; clause < CLAUSES; clause += stride) {
        for (unsigned long long class_id = 0; class_id < CLASSES; ++class_id) {
#if NEGATIVE_CLAUSES == 1
            clause_weights[class_id * CLAUSES + clause] =
                1 - 2 * (curand(&localState) % 2);  // 1 - 2*(clause % CLASSES != class_id);
#else
            clause_weights[class_id * CLAUSES + clause] = 1;
#endif
        }

        unsigned int *ta_state = &global_ta_state[clause * LA_CHUNKS * STATE_BITS];
        for (int la_chunk = 0; la_chunk < LA_CHUNKS; ++la_chunk) {
            for (int b = 0; b < STATE_BITS - 1; ++b) {
                ta_state[la_chunk * STATE_BITS + b] = ~0;
            }
            ta_state[la_chunk * STATE_BITS + STATE_BITS - 1] = 0;
        }
    }

    state[index] = localState;
}
}
