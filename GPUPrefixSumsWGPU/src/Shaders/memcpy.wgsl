//****************************************************************************
// GPUPrefixSums
//
// SPDX-License-Identifier: MIT
// Copyright Thomas Smith 8/4/2024
// https://github.com/b0nes164/GPUPrefixSums
//
//****************************************************************************
@group(0) @binding(0)
var<storage, read_write> copy: array<u32>;

@group(0) @binding(1)
var<storage, read_write> lazy_padding_0: array<u32>;

@group(0) @binding(2)
var<storage, read_write> lazy_padding_1: array<u32>;

@group(0) @binding(3)
var<storage, read> lazy_padding_2: array<u32>;

const BLOCK_DIM: u32 = 256;
const SPT: u32 = 16;
const PART_SIZE: u32 = BLOCK_DIM * SPT;
@compute @workgroup_size(BLOCK_DIM, 1, 1)
fn main(
    @builtin(local_invocation_id) threadid: vec3<u32>,
    @builtin(workgroup_id) blockid: vec3<u32>,
    @builtin(num_workgroups) griddim: vec3<u32>) {

    for(var i: u32 = threadid.x + blockid.x * PART_SIZE; i < (blockid.x + 1) * PART_SIZE; i += BLOCK_DIM){
        var t: u32 = copy[i];
        t++;
        copy[i] = t;
    }
}