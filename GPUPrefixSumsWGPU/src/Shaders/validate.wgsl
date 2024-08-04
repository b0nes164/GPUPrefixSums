//****************************************************************************
// GPUPrefixSums
//
// SPDX-License-Identifier: MIT
// Copyright Thomas Smith 8/4/2024
// https://github.com/b0nes164/GPUPrefixSums
//
//****************************************************************************
@group(0) @binding(0)
var<storage, read_write> scan: array<u32>;

@group(0) @binding(1)
var<storage, read> info: array<u32>;

@group(0) @binding(2)
var<storage, read_write> error: array<atomic<u32>>;

const BLOCK_DIM: u32 = 256;
@compute @workgroup_size(BLOCK_DIM, 1, 1)
fn main(
    @builtin(global_invocation_id) id: vec3<u32>,
    @builtin(num_workgroups) griddim: vec3<u32>) {
    
    let size: u32 = info[0];
    let stride = griddim.x * BLOCK_DIM;
    for(var i: u32 = id.x; i < size; i += stride){
        if(scan[i] != i + 1u){
            atomicAdd(&error[0], 1u);
        }
    }
}