extern "C" {

__global__ void transform(int *global_clause_node_output, int number_of_nodes, int *transformed_X) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    // int number_of_node_chunks = (number_of_nodes - 1) / INT_SIZE + 1;

    for (int clause = index; clause < CLAUSES; clause += stride) {
        int clause_output = 0;
        for (int n = 0; n < number_of_nodes; n++) {
            int chunk_nr = n / INT_SIZE;
            int chunk_pos = n % INT_SIZE;

            if (global_clause_node_output[clause * NODE_CHUNKS + chunk_nr] & (1 << chunk_pos)) {
                clause_output = 1;
                break;
            }
        }
        if (clause_output)
            transformed_X[clause] = 1;
        else
            transformed_X[clause] = 0;
    }
}

__global__ void transform_nodewise(int *global_clause_node_output, int number_of_nodes, int *transformed_X) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    // int number_of_node_chunks = (number_of_nodes - 1) / INT_SIZE + 1;
    for (int clause = index; clause < CLAUSES; clause += stride) {
        for (int n = 0; n < number_of_nodes; n++) {
            int chunk_nr = n / INT_SIZE;
            int chunk_pos = n % INT_SIZE;

            transformed_X[clause * number_of_nodes + n] =
                (global_clause_node_output[clause * NODE_CHUNKS + chunk_nr] & (1 << chunk_pos)) > 0;
        }
    }
}
}
