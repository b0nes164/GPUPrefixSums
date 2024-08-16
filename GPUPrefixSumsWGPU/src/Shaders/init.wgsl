//****************************************************************************
// GPUPrefixSums
//
// SPDX-License-Identifier: MIT
// Copyright Thomas Smith 8/4/2024
// https://github.com/b0nes164/GPUPrefixSums
//
//****************************************************************************
@group(0) @binding(0)
var<storage, read_write> scan_in: array<u32>;

@group(0) @binding(1)
var<storage, read_write> lazy_padding_0: array<u32>;

@group(0) @binding(2)
var<storage, read_write> reduction: array<u32>;

@group(0) @binding(3)
var<storage, read_write> index: array<u32>;

@group(0) @binding(4)
var<storage, read> info: array<u32>;

const BLOCK_DIM: u32 = 256;
@compute @workgroup_size(BLOCK_DIM, 1, 1)
fn main(
    @builtin(global_invocation_id) id: vec3<u32>,
    @builtin(num_workgroups) griddim: vec3<u32>) {
    
    let size: u32 = info[0];
    let thread_blocks: u32 = info[1];
    let stride = griddim.x * BLOCK_DIM;
    for(var i: u32 = id.x; i < size; i += stride){
        scan_in[i] = 1u;
        //scan_in[i] = i + 1u;
    }

    for(var i: u32 = id.x; i < thread_blocks; i += stride){
        reduction[i] = 0u;
    }

    if(id.x < 1){
        index[0] = 0u;
    }
}