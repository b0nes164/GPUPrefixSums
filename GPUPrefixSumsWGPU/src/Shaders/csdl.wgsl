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
const MIN_SUBGROUP_SIZE: u32 = 8;
const MAX_REDUCE_SIZE: u32 = BLOCK_DIM / MIN_SUBGROUP_SIZE;

const SPT: u32 = 16;
const PART_SIZE: u32 = BLOCK_DIM * SPT;

const FLAG_NOT_READY: u32 = 0;
const FLAG_REDUCTION: u32 = 1;
const FLAG_INCLUSIVE: u32 = 2;
const FLAG_MASK: u32 = 3;

var<workgroup> s_broadcast: u32;
var<workgroup> s_reduce: array<u32, MAX_REDUCE_SIZE>;

@compute @workgroup_size(BLOCK_DIM, 1, 1)
fn main(
    @builtin(local_invocation_id) threadid: vec3<u32>,
    @builtin(subgroup_invocation_id) laneid: u32,
    @builtin(subgroup_id) sid: u32,
    @builtin(subgroup_size) lane_count: u32,
    @builtin(num_workgroups) griddim: vec3<u32>) {

    //No push constant, so we use device memory instead...
    let size = info[0u];

    //acquire partition index, set the lock
    if(threadid.x == 0u){
        s_broadcast = atomicAdd(&index[0u], 1u);
    }
    workgroupBarrier();
    let part_id = s_broadcast;

    var t_scan = array<u32, SPT>();
    {
        let s_offset = laneid + sid * lane_count * SPT;
        let dev_offset =  part_id * PART_SIZE;
        var i: u32 = s_offset + dev_offset;

        if(part_id < griddim.x - 1u){
            for(var k: u32 = 0u; k < SPT; k += 1u){
                t_scan[k] = scan[i];
                i += lane_count;
            }
        }

        if(part_id == griddim.x - 1u){
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

    //Device broadcast
    if(threadid.x == 0u){
        atomicStore(&reduction[part_id], s_reduce[BLOCK_DIM / lane_count - 1u] << 2u |
         select(FLAG_INCLUSIVE, FLAG_REDUCTION, part_id != 0u));
    }

    //Lookback, single thread
    if(part_id != 0u){
        if(threadid.x == 0u){
            var lookback_id: u32 = part_id - 1u;
            var prev_reduction: u32 = 0u;
            loop{
                let flag_payload = atomicLoad(&reduction[lookback_id]);
                if((flag_payload & FLAG_MASK) == FLAG_INCLUSIVE){
                    prev_reduction += flag_payload >> 2u;
                    atomicAdd(&reduction[part_id], prev_reduction << 2u | 1u);
                    s_broadcast = prev_reduction;
                    break;
                }

                if((flag_payload & FLAG_MASK) == FLAG_REDUCTION){
                    prev_reduction += flag_payload >> 2u;
                    lookback_id -= 1u;
                }
            }
        }
        workgroupBarrier();
    }

    let prev = s_broadcast + select(0u, s_reduce[sid - 1u], sid != 0u); //s_broadcast convienently 0 for part_id 0
    {
        let s_offset = laneid + sid * lane_count * SPT;
        let dev_offset =  part_id * PART_SIZE;
        var i: u32 = s_offset + dev_offset;

        if(part_id < griddim.x - 1){
            for(var k: u32 = 0u; k < SPT; k += 1u){
                scan[i] = t_scan[k] + prev;
                i += lane_count;
            }
        }

        if(part_id == griddim.x - 1){
            for(var k: u32 = 0u; k < SPT; k += 1u){
                if(i < size){
                    scan[i] = t_scan[k] + prev;
                }
                i += lane_count;
            }
        }
    }
}