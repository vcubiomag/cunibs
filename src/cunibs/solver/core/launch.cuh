#pragma once
#include "core/common.hpp"

#include <cstdint>
#include <utility>

// `kernel` over `threads` threads in blocks of `Block`. An empty range launches nothing, since a
// zero grid is a launch error, and grid dimensions are the only launch failure this code can cause
// on its own. Anything else surfaces at the next sync_stream, inside a graph capture through
// GraphCache, or when the native call returns.
template <int Block = kBlock, typename... Params, typename... Args>
void launch_threads(std::int64_t threads, void (*kernel)(Params...), cudaStream_t stream,
                    Args&&... args) {
    if (const unsigned blocks = grid_for(threads, Block)) {
        kernel<<<blocks, Block, 0, stream>>>(std::forward<Args>(args)...);
    }
}
