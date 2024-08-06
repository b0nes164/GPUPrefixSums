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
var<storage, read_write> reduction: array<atomic<u32>>;

@group(0) @binding(2)
var<storage, read_write> index: array<atomic<u32>>;

@group(0) @binding(3)
var<storage, read> info: array<u32>;

const BLOCK_DIM: u32 = 256;
const SPT: u32 = 13;
const PART_SIZE: u32 = BLOCK_DIM * SPT;

var<workgroup> s_scratch: array<u32, PART_SIZE>;
var<workgroup> s_reduce: array<u32, BLOCK_DIM>;

@compute @workgroup_size(BLOCK_DIM, 1, 1)
fn reduce(
    @builtin(local_invocation_id) threadid: vec3<u32>,
    @builtin(workgroup_id) blockid: vec3<u32>,
    @builtin(num_workgroups) griddim: vec3<u32>) {

    //No push constant 
    let size = info[0u];    
    let dev_offset = blockid.x * PART_SIZE;

    //Full
    if(blockid.x < griddim.x - 1u){
        for(var i: u32 = threadid.x; i < PART_SIZE; i += BLOCK_DIM){
            s_scratch[i] = scan[i + dev_offset];
        }
    }

    //Partial
    if(blockid.x == griddim.x - 1u){
        let final_part_size = size - dev_offset;
        for(var i: u32 = threadid.x; i < final_part_size; i += BLOCK_DIM){
            s_scratch[i] = scan[i + dev_offset];
        }
    }
    workgroupBarrier();

    var t_reduce: u32 = 0u;
    {
        let s_offset = threadid.x * SPT;
        for(var i: u32 = 0u; i < SPT; i += 1u){
            t_reduce += s_scratch[i + s_offset];
        }
    }
    s_reduce[threadid.x] = t_reduce;
    workgroupBarrier();

    //upsweep
    if(threadid.x < (BLOCK_DIM >> 1u)){
        s_reduce[(threadid.x << 1u) + 1u] += s_reduce[threadid.x << 1u];
    }

    var offset: u32 = 1;
    for(var j: u32 = (BLOCK_DIM >> 2u); j > 0u; j >>= 1u){
        workgroupBarrier();
        if(threadid.x < j){
            s_reduce[(((threadid.x << 1u) + 2u) << offset) - 1u] +=
            s_reduce[(((threadid.x << 1u) + 1u) << offset) - 1u];
        }
        offset += 1u;
    }
    workgroupBarrier();

    if(threadid.x == 0u){
        reduction[blockid.x] = s_reduce[BLOCK_DIM - 1];
    }
}

@compute @workgroup_size(BLOCK_DIM, 1, 1)
fn device_scan(
    @builtin(local_invocation_id) threadid: vec3<u32>) {
    
    //No push constant
    let size = info[1];
    let aligned_size = (size + PART_SIZE - 1) / PART_SIZE * PART_SIZE;
    var prev_reduction: u32 = 0u;
    for(var dev_offset: u32 = 0; dev_offset < aligned_size; dev_offset += PART_SIZE){
        for(var i: u32 = threadid.x; i < PART_SIZE; i += BLOCK_DIM){
            let dev_index = i + dev_offset;
            if(dev_index < size){
                s_scratch[i] = reduction[dev_index];
            }
        }
        workgroupBarrier();

        var t_scan = array<u32, SPT>();
        {
            let s_offset = threadid.x * SPT;
            for(var i: u32 = 0; i < SPT; i += 1u){
                t_scan[i] = s_scratch[i + s_offset];
                if(i != 0u){
                    t_scan[i] += t_scan[i - 1u];
                }
            }
        }
        s_reduce[threadid.x] = t_scan[SPT - 1u];
        workgroupBarrier();

        //upsweep
        if(threadid.x < (BLOCK_DIM >> 1u)){
            s_reduce[(threadid.x << 1u) + 1u] += s_reduce[threadid.x << 1u];
        }

        var offset: u32 = 1;
        for(var j: u32 = (BLOCK_DIM>> 2u); j > 0u; j >>= 1u){
            workgroupBarrier();
            if(threadid.x < j){
                s_reduce[(((threadid.x << 1u) + 2u) << offset) - 1u] +=
                s_reduce[(((threadid.x << 1u) + 1u) << offset) - 1u];
            }
            offset += 1u;
        }
        workgroupBarrier();

        //downsweep
        for(var j: u32 = 1u; j < BLOCK_DIM; j <<= 1u){
            offset -= 1u;
            workgroupBarrier();
            if(threadid.x < j - 1){
                s_reduce[(((threadid.x << 1u) + 3u) << offset) - 1u] +=
                s_reduce[(((threadid.x << 1u) + 2u) << offset) - 1u];
            }
        }
        workgroupBarrier();

        {
            let s_offset = threadid.x * SPT;
            for(var i: u32 = 0; i < SPT; i += 1u){
                s_scratch[i + s_offset] = t_scan[i] + select(0u, s_reduce[threadid.x - 1], threadid.x != 0u) + prev_reduction;
            }
        }
        workgroupBarrier();

        for(var i: u32 = threadid.x; i < PART_SIZE; i += BLOCK_DIM){
            let dev_index = i + dev_offset;
            if(dev_index < size){
                reduction[dev_index] = s_scratch[i];
            }
        }

        prev_reduction += s_reduce[BLOCK_DIM - 1];
        workgroupBarrier();
    }
}

@compute @workgroup_size(BLOCK_DIM, 1, 1)
fn downsweep(
    @builtin(local_invocation_id) threadid: vec3<u32>,
    @builtin(workgroup_id) blockid: vec3<u32>,
    @builtin(num_workgroups) griddim: vec3<u32>) {
    
    let size = info[0u];
    let prev_reduction = select(0u, reduction[blockid.x - 1u], blockid.x != 0u);

    //Load
    {
        //Full
        let dev_offset = blockid.x * PART_SIZE;
        if(blockid.x < griddim.x - 1u){
            for(var i: u32 = threadid.x; i < PART_SIZE; i += BLOCK_DIM){
                s_scratch[i] = scan[i + dev_offset];
            }
        }

        //Partial
        if(blockid.x == griddim.x - 1u){
            let final_part_size = size - dev_offset;
            for(var i: u32 = threadid.x; i < final_part_size; i += BLOCK_DIM){
                s_scratch[i] = scan[i + dev_offset];
            }
        }
    }
    workgroupBarrier();

    var t_scan = array<u32, SPT>();
    {
        let s_offset = threadid.x * SPT;
        for(var i: u32 = 0; i < SPT; i += 1u){
            t_scan[i] = s_scratch[i + s_offset];
            if(i != 0u){
                t_scan[i] += t_scan[i - 1u];
            }
        }
    }
    s_reduce[threadid.x] = t_scan[SPT - 1u];
    workgroupBarrier();

    //upsweep
    if(threadid.x < (BLOCK_DIM >> 1u)){
        s_reduce[(threadid.x << 1u) + 1u] += s_reduce[threadid.x << 1u];
    }

    var offset: u32 = 1;
    for(var j: u32 = (BLOCK_DIM>> 2u); j > 0u; j >>= 1u){
        workgroupBarrier();
        if(threadid.x < j){
            s_reduce[(((threadid.x << 1u) + 2u) << offset) - 1u] +=
             s_reduce[(((threadid.x << 1u) + 1u) << offset) - 1u];
        }
        offset += 1u;
    }
    workgroupBarrier();

    //downsweep
    for(var j: u32 = 1u; j < BLOCK_DIM; j <<= 1u){
        offset -= 1u;
        workgroupBarrier();
        if(threadid.x < j - 1){
            s_reduce[(((threadid.x << 1u) + 3u) << offset) - 1u] +=
             s_reduce[(((threadid.x << 1u) + 2u) << offset) - 1u];
        }
    }
    workgroupBarrier();

    {
        let s_offset = threadid.x * SPT;
        for(var i: u32 = 0; i < SPT; i += 1u){
            s_scratch[i + s_offset] = t_scan[i] + select(0u, s_reduce[threadid.x - 1], threadid.x != 0u) + prev_reduction;
        }
    }
    workgroupBarrier();

    //Write
    {
        //Full
        let dev_offset = blockid.x * PART_SIZE;
        if(blockid.x < griddim.x - 1u){
            for(var i: u32 = threadid.x; i < PART_SIZE; i += BLOCK_DIM){
                scan[i + dev_offset] = s_scratch[i];
            }
        }

        //Partial
        if(blockid.x == griddim.x - 1u){
            let final_part_size = size - dev_offset;
            for(var i: u32 = threadid.x; i < final_part_size; i += BLOCK_DIM){
                scan[i + dev_offset] = s_scratch[i];
            }
        }
    }
}