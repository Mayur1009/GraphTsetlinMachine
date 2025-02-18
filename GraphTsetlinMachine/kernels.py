# Copyright (c) 2024 Ole-Christoffer Granmo

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

# This code implements the Convolutional Tsetlin Machine from paper arXiv:1905.09688
# https://arxiv.org/abs/1905.09688

import pathlib

current_dir = pathlib.Path(__file__).parent


def get_kernel(file):
	path = current_dir.joinpath(file)
	with path.open("r") as f:
		ker = f.read()
	return ker


code_header = """
    #include <curand_kernel.h>
    
    #define INT_SIZE 32

    #define LA_CHUNKS (((LITERALS-1)/INT_SIZE + 1))
    #define CLAUSE_CHUNKS ((CLAUSES-1)/INT_SIZE + 1)

    #define MESSAGE_LITERALS (MESSAGE_SIZE*2)
    #define MESSAGE_CHUNKS (((MESSAGE_LITERALS-1)/INT_SIZE + 1))

    #define NODE_CHUNKS ((MAX_NODES-1)/INT_SIZE + 1)

    #if (LITERALS % 32 != 0)
    #define FILTER (~(0xffffffff << (LITERALS % INT_SIZE)))
    #else
    #define FILTER 0xffffffff
    #endif

    #if (MESSAGE_LITERALS % 32 != 0)
    #define MESSAGE_FILTER (~(0xffffffff << (MESSAGE_LITERALS % INT_SIZE)))
    #else
    #define MESSAGE_FILTER 0xffffffff
    #endif
"""

code_update = get_kernel("cuda/code_update.cu")
code_evaluate = get_kernel("cuda/code_evaluate.cu")
code_prepare = get_kernel("cuda/code_prepare.cu")
code_transform = get_kernel("cuda/code_transform.cu")
code_clauses = get_kernel("cuda/code_clauses.cu")
