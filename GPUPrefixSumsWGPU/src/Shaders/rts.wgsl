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
var<storage, read_write> lazy_padding_0: array<atomic<u32>>;

@group(0) @binding(3)
var<storage, read> info: array<u32>;

const BLOCK_DIM: u32 = 256;
const MIN_SUBGROUP_SIZE: u32 = 8;
const MAX_REDUCE_SIZE: u32 = BLOCK_DIM / MIN_SUBGROUP_SIZE;

const SPT: u32 = 16;
const PART_SIZE: u32 = BLOCK_DIM * SPT;

var<workgroup> s_reduce: array<u32, MAX_REDUCE_SIZE>;

@compute @workgroup_size(BLOCK_DIM, 1, 1)
fn reduce(
    @builtin(subgroup_invocation_id) laneid: u32,
    @builtin(subgroup_id) sid: u32,
    @builtin(subgroup_size) lane_count: u32,
    @builtin(workgroup_id) blockid: vec3<u32>,
    @builtin(num_workgroups) griddim: vec3<u32>) {

    //No push constant 
    let size = info[0u];
    let s_offset = laneid + sid * lane_count * SPT;
    let dev_offset = blockid.x * PART_SIZE;
    var i: u32 = s_offset + dev_offset;

    var t_red = array<u32, SPT>();
    if(blockid.x < griddim.x - 1){
        for(var k: u32 = 0u; k < SPT; k += 1u){
            t_red[k] = scan[i];
            i += lane_count;
        }
    }

    if(blockid.x == griddim.x - 1){
        for(var k: u32 = 0u; k < SPT; k += 1u){
            t_red[k] = select(0u, scan[i], i < size);
            i += lane_count;
        }
    }
    
    var sub_red = 0u;
    for(var k: u32 = 0u; k < SPT; k += 1u){
        sub_red += subgroupAdd(t_red[k]);
    }

    if(laneid == lane_count - 1u){
        s_reduce[sid] = sub_red;
    }
    workgroupBarrier();

    if(sid == 0u){
        let pred = laneid < BLOCK_DIM / lane_count;
        let t = subgroupAdd(select(0u, s_reduce[laneid], pred));
        if(laneid == 0u){
            reduction[blockid.x] = t;
        }
    }
}

@compute @workgroup_size(BLOCK_DIM, 1, 1)
fn device_scan(
    @builtin(subgroup_invocation_id) laneid: u32,
    @builtin(subgroup_id) sid: u32,
    @builtin(subgroup_size) lane_count: u32) {
    
    //No push constant
    let size = info[1];
    let s_offset = laneid + sid * lane_count * SPT;
    let aligned_size = (size + PART_SIZE - 1) / PART_SIZE * PART_SIZE;
    var prev_reduction: u32 = 0u;
    var t_scan = array<u32, SPT>();

    for(var dev_offset: u32 = 0; dev_offset < aligned_size; dev_offset += PART_SIZE){
        {
            var i: u32 = s_offset + dev_offset;
            for(var k: u32 = 0u; k < SPT; k += 1u){
                if(i < size){
                    t_scan[k] = reduction[i];
                }
                i += lane_count;
            }

            var prev: u32 = 0u;
            for(var k: u32 = 0u; k < SPT; k += 1u){
                t_scan[k] = subgroupInclusiveAdd(t_scan[k]) + prev;
                prev = subgroupBroadcast(t_scan[k], lane_count - 1);
            }

            if(laneid == lane_count - 1u){
                s_reduce[sid] = t_scan[SPT - 1u];
            }
        }
        workgroupBarrier();

        if(sid == 0u){
            let pred = laneid < BLOCK_DIM / lane_count;
            let t = subgroupInclusiveAdd(select(0u, s_reduce[laneid], pred));
            if(pred){
                s_reduce[laneid] = t;
            }
        }
        workgroupBarrier();

        {
            let prev = select(0u, s_reduce[sid - 1u], sid != 0u) + prev_reduction;
            var i: u32 = s_offset + dev_offset;
            for(var k: u32 = 0u; k < SPT; k += 1u){
                if(i < size){
                    reduction[i] = t_scan[k] + prev;
                }
                i += lane_count;
            }
        }

        prev_reduction += subgroupBroadcast(s_reduce[BLOCK_DIM / lane_count - 1], 0u);
        workgroupBarrier();
    }
}

@compute @workgroup_size(BLOCK_DIM, 1, 1)
fn downsweep(
    @builtin(subgroup_invocation_id) laneid: u32,
    @builtin(subgroup_id) sid: u32,
    @builtin(subgroup_size) lane_count: u32,
    @builtin(workgroup_id) blockid: vec3<u32>,
    @builtin(num_workgroups) griddim: vec3<u32>) {

    //No push constant 
    let size = info[0u];
    let s_offset = laneid + sid * lane_count * SPT;
    let dev_offset = blockid.x * PART_SIZE;
    var i: u32 = s_offset + dev_offset;

    var t_scan = array<u32, SPT>();
    {
        if(blockid.x < griddim.x - 1){
            for(var k: u32 = 0u; k < SPT; k += 1u){
                t_scan[k] = scan[i];
                i += lane_count;
            }
        }

        if(blockid.x == griddim.x - 1){
            for(var k: u32 = 0u; k < SPT; k += 1u){
                if(i < size){
                    t_scan[k] = scan[i];
                }
                i += lane_count;
            }
        }
        
        var prev: u32 = 0u;
        for(var k: u32 = 0u; k < SPT; k += 1u){
            t_scan[k] = subgroupInclusiveAdd(t_scan[k]) + prev;
            prev = subgroupBroadcast(t_scan[k], lane_count - 1);
        }

        if(laneid == lane_count - 1u){
            s_reduce[sid] = t_scan[SPT - 1u];
        }
    }
    workgroupBarrier();

    if(sid == 0u){
        let pred = laneid < BLOCK_DIM / lane_count;
        let t = subgroupInclusiveAdd(select(0u, s_reduce[laneid], pred));
        if(pred){
            s_reduce[laneid] = t;
        }
    }
    workgroupBarrier();

    let prev = select(0u, s_reduce[sid - 1u], sid != 0u) + select(0u, reduction[blockid.x - 1u], blockid.x != 0u); 
    {
        let s_offset = laneid + sid * lane_count * SPT;
        let dev_offset =  blockid.x * PART_SIZE;
        var i: u32 = s_offset + dev_offset;

        if(blockid.x < griddim.x - 1){
            for(var k: u32 = 0u; k < SPT; k += 1u){
                scan[i] = t_scan[k] + prev;
                i += lane_count;
            }
        }

        if(blockid.x == griddim.x - 1){
            for(var k: u32 = 0u; k < SPT; k += 1u){
                if(i < size){
                    scan[i] = t_scan[k] + prev;
                }
                i += lane_count;
            }
        }
    }
}