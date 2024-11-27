//****************************************************************************
// GPUPrefixSums
// Chained Scan with Decoupled Lookback: 
// aka Single Pass Scan, this will crash without forward thread progress guarantees.
//
// SPDX-License-Identifier: MIT
// Copyright Thomas Smith 10/23/2024
// https://github.com/b0nes164/GPUPrefixSums
//
//****************************************************************************

struct InfoStruct
{
    size: u32,
    vec_size: u32,
    thread_blocks: u32,
};

@group(0) @binding(0)
var<uniform> info : InfoStruct; 

@group(0) @binding(1)
var<storage, read_write> scan_in: array<vec4<u32>>;

@group(0) @binding(2)
var<storage, read_write> scan_out: array<vec4<u32>>;

@group(0) @binding(3)
var<storage, read_write> scan_bump: atomic<u32>;

@group(0) @binding(4)
var<storage, read_write> reduction: array<atomic<u32>>;

@group(0) @binding(5)
var<storage, read_write> misc: array<u32>;

const BLOCK_DIM = 256u;
const MIN_SUBGROUP_SIZE = 4u;
const MAX_REDUCE_SIZE = BLOCK_DIM / MIN_SUBGROUP_SIZE;

const VEC4_SPT = 4u;
const VEC_PART_SIZE = BLOCK_DIM * VEC4_SPT;

const FLAG_NOT_READY = 0u;
const FLAG_REDUCTION = 1u;
const FLAG_INCLUSIVE = 2u;
const FLAG_MASK = 3u;

var<workgroup> wg_broadcast: u32;
var<workgroup> wg_reduce: array<u32, MAX_REDUCE_SIZE>;

@compute @workgroup_size(BLOCK_DIM, 1, 1)
fn main(
    @builtin(local_invocation_id) threadid: vec3<u32>,
    @builtin(subgroup_invocation_id) laneid: u32,
    @builtin(subgroup_size) lane_count: u32) {
    
    let sid = threadid.x / lane_count;  //Caution 1D workgoup ONLY! Ok, but technically not in HLSL spec
    
    //acquire partition index
    if(threadid.x == 0u){
        wg_broadcast = atomicAdd(&scan_bump, 1u);
    }
    let part_id = workgroupUniformLoad(&wg_broadcast);

    var t_scan = array<vec4<u32>, VEC4_SPT>();
    {
        let s_offset = laneid + sid * lane_count * VEC4_SPT;
        let dev_offset =  part_id * VEC_PART_SIZE;
        var i = s_offset + dev_offset;

        if(part_id < info.thread_blocks- 1u){
            for(var k = 0u; k < VEC4_SPT; k += 1u){
                t_scan[k] = scan_in[i];
                t_scan[k].y += t_scan[k].x;
                t_scan[k].z += t_scan[k].y;
                t_scan[k].w += t_scan[k].z;
                i += lane_count;
            }
        }

        if(part_id == info.thread_blocks - 1u){
            for(var k = 0u; k < VEC4_SPT; k += 1u){
                if(i < info.vec_size){
                    t_scan[k] = scan_in[i];
                    t_scan[k].y += t_scan[k].x;
                    t_scan[k].z += t_scan[k].y;
                    t_scan[k].w += t_scan[k].z;
                }
                i += lane_count;
            }
        }

        var prev = 0u;
        let lane_mask = lane_count - 1u;
        let circular_shift = (laneid + lane_mask) & lane_mask;
        for(var k = 0u; k < VEC4_SPT; k += 1u){
            let t = subgroupShuffle(subgroupInclusiveAdd(t_scan[k].w), circular_shift);
            t_scan[k] += select(0u, t, laneid != 0u) + prev;
            prev += subgroupBroadcast(t, 0u);
        }

        if(laneid == 0u){
            wg_reduce[sid] = prev;
        }
    }
    workgroupBarrier();

    //Subgroup agnostic inclusive scan across subgroup reductions
    //At no point is a subgroup function called here in a state of subgroup divergence
    let lane_log = u32(countTrailingZeros(lane_count));
    let spine_size = BLOCK_DIM >> lane_log;
    {   
        var offset = 0u;
        let aligned_size = 1u << ((u32(countTrailingZeros(spine_size)) + lane_log - 1) / lane_log * lane_log);
        for(var j = lane_count; j <= aligned_size; j <<= lane_log){
            let i0 = ((threadid.x) << offset) - 1u;
            let pred0 = i0 < spine_size;
            let t0 = subgroupInclusiveAdd(select(0u, wg_reduce[i0], pred0));
            if(pred0){
                wg_reduce[i0] = t0;
            }
            workgroupBarrier();

            if(j != lane_count){
                let rshift = j >> lane_log;
                let i1 = threadid.x + rshift;
                if ((i1 & (j - 1u)) >= rshift){
                    let t1 = subgroupBroadcast(wg_reduce[((i1 >> offset) << offset) - 1u], 0u);
                    if(((i1 + 1) & (rshift - 1u)) != 0u){
                        wg_reduce[i1] += t1;
                    }
                }
            }
            offset += lane_log;
        }
    }   
    workgroupBarrier();

    //Device broadcast
    if(threadid.x == 0u){
        atomicStore(&reduction[part_id], (wg_reduce[spine_size - 1u] << 2u) |
            select(FLAG_INCLUSIVE, FLAG_REDUCTION, part_id != 0u));
    }

    //Lookback, single thread
    if(part_id != 0u){
        if(threadid.x == 0u){
            var lookback_id = part_id - 1u;
            var prev_reduction = 0u;
            loop{
                let flag_payload = atomicLoad(&reduction[lookback_id]);
                if((flag_payload & FLAG_MASK) == FLAG_INCLUSIVE){
                    prev_reduction += flag_payload >> 2u;
                    atomicStore(&reduction[part_id],
                        ((prev_reduction + wg_reduce[spine_size - 1u]) << 2u) | FLAG_INCLUSIVE);
                    wg_broadcast = prev_reduction;
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

    {
        let prev = wg_broadcast + select(0u, wg_reduce[sid - 1u], sid != 0u); //wg_broadcast is 0 for part_id 0
        let s_offset = laneid + sid * lane_count * VEC4_SPT;
        let dev_offset =  part_id * VEC_PART_SIZE;
        var i = s_offset + dev_offset;

        if(part_id < info.thread_blocks - 1u){
            for(var k = 0u; k < VEC4_SPT; k += 1u){
                scan_out[i] = t_scan[k] + prev;
                i += lane_count;
            }
        }

        if(part_id == info.thread_blocks - 1u){
            for(var k = 0u; k < VEC4_SPT; k += 1u){
                if(i < info.vec_size){
                    scan_out[i] = t_scan[k] + prev;
                }
                i += lane_count;
            }
        }
    }
}
