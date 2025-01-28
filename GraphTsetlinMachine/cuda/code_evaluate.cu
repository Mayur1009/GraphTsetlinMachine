extern "C" {

// Counts number of include actions for a given clause
__device__ inline int count_number_of_include_actions_message(unsigned int *ta_state) {
    int number_of_include_actions = 0;
    for (int k = 0; k < MESSAGE_CHUNKS - 1; ++k) {
        unsigned int ta_pos = k * STATE_BITS + STATE_BITS - 1;
        number_of_include_actions += __popc(ta_state[ta_pos]);
    }
    unsigned int ta_pos = (MESSAGE_CHUNKS - 1) * STATE_BITS + STATE_BITS - 1;
    number_of_include_actions += __popc(ta_state[ta_pos] & MESSAGE_FILTER);

    return (number_of_include_actions);
}

// Counts number of include actions for a given clause
__device__ inline int count_number_of_include_actions(unsigned int *ta_state) {
    int number_of_include_actions = 0;
    for (int k = 0; k < LA_CHUNKS - 1; ++k) {
        unsigned int ta_pos = k * STATE_BITS + STATE_BITS - 1;
        number_of_include_actions += __popc(ta_state[ta_pos]);
    }
    unsigned int ta_pos = (LA_CHUNKS - 1) * STATE_BITS + STATE_BITS - 1;
    number_of_include_actions += __popc(ta_state[ta_pos] & FILTER);

    return (number_of_include_actions);
}

__global__ void evaluate(int *global_clause_node_output, int *clause_weights, int number_of_nodes, int *class_sum) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    int number_of_node_chunks = (number_of_nodes - 1) / INT_SIZE + 1;
    unsigned int node_filter;
    if ((number_of_nodes % INT_SIZE) != 0) {
        node_filter = (~(0xffffffff << (number_of_nodes % INT_SIZE)));
    } else {
        node_filter = 0xffffffff;
    }

    for (int clause = index; clause < CLAUSES; clause += stride) {
        int clause_output = 0;
        for (int k = 0; k < number_of_node_chunks - 1; ++k) {
            if (global_clause_node_output[clause * NODE_CHUNKS + k]) {
                clause_output = 1;
                break;
            }
        }

        if (global_clause_node_output[clause * NODE_CHUNKS + number_of_node_chunks - 1] & node_filter) {
            clause_output = 1;
        }

        if (clause_output) {
            for (int class_id = 0; class_id < CLASSES; ++class_id) {
                int clause_weight = clause_weights[class_id * CLAUSES + clause];
                atomicAdd(&class_sum[class_id], clause_weight);
            }
        }
    }
}

__global__ void select_clause_node(curandState *state, int *global_clause_node_output, int number_of_nodes,
                                   int *clause_node) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    curandState localState = state[index];

    int clause_true_node[MAX_NODES];
    int clause_true_node_len;

    for (int clause = index; clause < CLAUSES; clause += stride) {
        clause_true_node_len = 0;
        for (int node = 0; node < number_of_nodes; ++node) {
            int node_chunk = node / INT_SIZE;
            int node_pos = node % INT_SIZE;

            if (global_clause_node_output[clause * NODE_CHUNKS + node_chunk] & (1 << node_pos)) {
                clause_true_node[clause_true_node_len] = node;
                clause_true_node_len++;
            }
        }

        if (clause_true_node_len > 0) {
            clause_node[clause] = clause_true_node[curand(&localState) % (clause_true_node_len)];
        } else {
            clause_node[clause] = -1;
        }
    }

    state[index] = localState;
}

__global__ void select_clause_updates(curandState *state, int *clause_weights, int *class_sum, int *y, int example,
                                      int *clause_node, int *class_clause_update) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    curandState localState = state[index];

    for (int clause = index; clause < CLAUSES; clause += stride) {
        for (int class_id = 0; class_id < CLASSES; ++class_id) {
            int local_class_sum = class_sum[class_id];
            if (local_class_sum > THRESHOLD) {
                local_class_sum = THRESHOLD;
            } else if (local_class_sum < -THRESHOLD) {
                local_class_sum = -THRESHOLD;
            }

            int target = 1 - 2 * (local_class_sum > y[example * CLASSES + class_id]);
            int sign =
                (clause_weights[class_id * CLAUSES + clause] >= 0) - (clause_weights[class_id * CLAUSES + clause] < 0);
            int absolute_prediction_error = abs(y[example * CLASSES + class_id] - local_class_sum);

            if ((target == -1 && curand_uniform(&localState) > 1.0 * Q / max(1, CLASSES - 1)) ||
                (curand_uniform(&localState) > 1.0 * absolute_prediction_error / (2 * THRESHOLD))) {
                class_clause_update[class_id * CLAUSES + clause] = 0;
            } else {
                class_clause_update[class_id * CLAUSES + clause] = target * sign;

                if (target * sign > 0 && clause_node[clause] != -1 &&
                    abs(clause_weights[class_id * CLAUSES + clause]) < INT_MAX) {
                    clause_weights[class_id * CLAUSES + clause] += sign;
                } else if (target * sign < 0 && clause_node[clause] != -1) {
                    clause_weights[class_id * CLAUSES + clause] -= sign;

#if NEGATIVE_CLAUSES == 0
                    if (clause_weights[class_id * CLAUSES + clause] < 1) {
                        clause_weights[class_id * CLAUSES + clause] = 1;
                    }
#endif
                }
            }
        }
    }

    state[index] = localState;
}

__global__ void calculate_messages(unsigned int *global_ta_state, int *node_type, int number_of_node_types,
                                   int number_of_nodes, int graph_index, int *global_clause_node_output,
                                   int *number_of_include_actions, unsigned int *global_X) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    unsigned int clause_node_output;

    int number_of_node_chunks = (number_of_nodes - 1) / INT_SIZE + 1;
    unsigned int node_filter;
    if ((number_of_nodes % INT_SIZE) != 0) {
        node_filter = (~(0xffffffff << (number_of_nodes % INT_SIZE)));
    } else {
        node_filter = 0xffffffff;
    }

    unsigned int *X = &global_X[graph_index * LA_CHUNKS];

    for (int clause_node_chunk = index; clause_node_chunk < (CLAUSES) * (NODE_CHUNKS); clause_node_chunk += stride) {
        int clause = clause_node_chunk % CLAUSES;
        int node_chunk = clause_node_chunk / CLAUSES;

        unsigned int *ta_state = &global_ta_state[clause * LA_CHUNKS * STATE_BITS];

        if (node_chunk == 0) {
            number_of_include_actions[clause] = count_number_of_include_actions(ta_state);
        }

        clause_node_output = ~0;
        for (int node_pos = 0; (node_pos < INT_SIZE) && ((node_chunk * INT_SIZE + node_pos) < number_of_nodes);
             ++node_pos) {
            int node = node_chunk * INT_SIZE + node_pos;

            if (node_type[graph_index + node] == (clause % number_of_node_types)) {
                for (int la_chunk = 0; la_chunk < LA_CHUNKS - 1; ++la_chunk) {
                    if ((ta_state[la_chunk * STATE_BITS + STATE_BITS - 1] & X[node * LA_CHUNKS + la_chunk]) !=
                        ta_state[la_chunk * STATE_BITS + STATE_BITS - 1]) {
                        clause_node_output &= ~(1 << node_pos);
                    }
                }

                if ((ta_state[(LA_CHUNKS - 1) * STATE_BITS + STATE_BITS - 1] & X[node * LA_CHUNKS + LA_CHUNKS - 1] &
                     FILTER) != (ta_state[(LA_CHUNKS - 1) * STATE_BITS + STATE_BITS - 1] & FILTER)) {
                    clause_node_output &= ~(1 << node_pos);
                }
            } else {
                // printf("Normal: Wrong node type %d for clause %d \n", node_type[graph_index + node], clause);
                clause_node_output &= ~(1 << node_pos);
            }
        }

        if (node_chunk == number_of_node_chunks - 1) {
            global_clause_node_output[clause * NODE_CHUNKS + node_chunk] = clause_node_output & node_filter;
        } else {
            global_clause_node_output[clause * NODE_CHUNKS + node_chunk] = clause_node_output;
        }
    }
}

__global__ void calculate_messages_conditional(unsigned int *global_ta_state, int *node_type, int number_of_node_types,
                                               int number_of_nodes, int graph_index,
                                               int *global_clause_node_output_condition, int *global_clause_node_output,
                                               int *number_of_include_actions, unsigned int *X) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    unsigned int clause_node_output;

    int number_of_node_chunks = (number_of_nodes - 1) / INT_SIZE + 1;
    unsigned int node_filter;
    if ((number_of_nodes % INT_SIZE) != 0) {
        node_filter = (~(0xffffffff << (number_of_nodes % INT_SIZE)));
    } else {
        node_filter = 0xffffffff;
    }

    for (int clause_node_chunk = index; clause_node_chunk < (CLAUSES) * (NODE_CHUNKS); clause_node_chunk += stride) {
        int clause = clause_node_chunk / NODE_CHUNKS;
        int node_chunk = clause_node_chunk % NODE_CHUNKS;

        unsigned int *ta_state = &global_ta_state[clause * MESSAGE_CHUNKS * STATE_BITS];

        if (node_chunk == 0) {
            number_of_include_actions[clause] += count_number_of_include_actions_message(ta_state);
        }

        clause_node_output = ~0;
        for (int node_pos = 0; (node_pos < INT_SIZE) && ((node_chunk * INT_SIZE + node_pos) < number_of_nodes);
             ++node_pos) {
            int node = node_chunk * INT_SIZE + node_pos;

            if (node_type[graph_index + node] == (clause % number_of_node_types)) {
                for (int la_chunk = 0; la_chunk < MESSAGE_CHUNKS - 1; ++la_chunk) {
                    if ((ta_state[la_chunk * STATE_BITS + STATE_BITS - 1] & X[node * MESSAGE_CHUNKS + la_chunk]) !=
                        ta_state[la_chunk * STATE_BITS + STATE_BITS - 1]) {
                        clause_node_output &= ~(1 << node_pos);
                    }
                }

                if ((ta_state[(MESSAGE_CHUNKS - 1) * STATE_BITS + STATE_BITS - 1] &
                     X[node * MESSAGE_CHUNKS + MESSAGE_CHUNKS - 1] & MESSAGE_FILTER) !=
                    (ta_state[(MESSAGE_CHUNKS - 1) * STATE_BITS + STATE_BITS - 1] & MESSAGE_FILTER)) {
                    clause_node_output &= ~(1 << node_pos);
                }
            } else {
                // printf("Conditional: Wrong node type %d for clause %d\n", node_type[graph_index + node], clause);
                clause_node_output &= ~(1 << node_pos);
            }
        }

        if (node_chunk == number_of_node_chunks - 1) {
            global_clause_node_output[clause * NODE_CHUNKS + node_chunk] =
                global_clause_node_output_condition[clause * NODE_CHUNKS + node_chunk] & clause_node_output &
                node_filter;
        } else {
            global_clause_node_output[clause * NODE_CHUNKS + node_chunk] =
                global_clause_node_output_condition[clause * NODE_CHUNKS + node_chunk] & clause_node_output;
        }
    }
}

__global__ void prepare_messages(int number_of_nodes, unsigned int *clause_X_int) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (int node_message_bit = index; node_message_bit < number_of_nodes * MESSAGE_SIZE; node_message_bit += stride) {
        int node = node_message_bit / MESSAGE_SIZE;
        int message_bit = node_message_bit % MESSAGE_SIZE;

        clause_X_int[node * MESSAGE_LITERALS + message_bit] = 0;
        clause_X_int[node * MESSAGE_LITERALS + MESSAGE_SIZE + message_bit] = 1;
    }
}

__global__ void exchange_messages(int number_of_nodes, int *hypervectors, int *global_clause_node_output,
                                  int node_index, int global_edge_index, int *number_of_graph_node_edges, int *edge,
                                  unsigned int *clause_X_int) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    int bit[MESSAGE_BITS];

    for (int clause = index; clause < CLAUSES; clause += stride) {
        for (int bit_index = 0; bit_index < MESSAGE_BITS; ++bit_index) {
            bit[bit_index] = hypervectors[clause * MESSAGE_BITS + bit_index];
        }

        int edge_index = global_edge_index;
        for (int source_node = 0; source_node < number_of_nodes; ++source_node) {
            int source_node_chunk = source_node / INT_SIZE;
            int source_node_pos = source_node % INT_SIZE;

            if ((global_clause_node_output[clause * NODE_CHUNKS + source_node_chunk] & (1 << source_node_pos)) > 0) {
                for (int i = 0; i < number_of_graph_node_edges[node_index + source_node]; ++i) {
                    int destination_node = edge[(edge_index + i) * 2];
                    int edge_type = edge[(edge_index + i) * 2 + 1];

                    for (int bit_index = 0; bit_index < MESSAGE_BITS; ++bit_index) {
                        int shifted_bit = (bit[bit_index] + edge_type) % MESSAGE_SIZE;
                        clause_X_int[destination_node * MESSAGE_LITERALS + shifted_bit] = 1;
                        clause_X_int[destination_node * MESSAGE_LITERALS + MESSAGE_SIZE + shifted_bit] = 0;
                    }
                }
            }
            edge_index += number_of_graph_node_edges[node_index + source_node];
        }
    }
}

__global__ void encode_messages(int number_of_nodes, unsigned int *clause_X_int, unsigned int *clause_X) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (int node_message_chunk = index; node_message_chunk < number_of_nodes * MESSAGE_CHUNKS;
         node_message_chunk += stride) {
        int node = node_message_chunk / MESSAGE_CHUNKS;
        int message_chunk = node_message_chunk % MESSAGE_CHUNKS;
        int X_int_base = node * MESSAGE_LITERALS + message_chunk * INT_SIZE;

        int message = 0;
        for (int bit_pos = 0; (bit_pos < INT_SIZE) && (message_chunk * INT_SIZE + bit_pos < MESSAGE_LITERALS);
             ++bit_pos) {
            if (clause_X_int[X_int_base + bit_pos]) {
                message |= (1 << bit_pos);
            }
        }

        clause_X[node * MESSAGE_CHUNKS + message_chunk] = message;
    }
}
}
