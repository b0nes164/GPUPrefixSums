//****************************************************************************
// GPUPrefixSums
// Chained Scan with Decoupled Lookback Decoupled Fallback: 
// CSDL but with an additional fallback routine, allowing the scan to work
// on hardware without forward thread progress guarantees
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
const MAX_REDUCE_SIZE = BLOCK_DIM / MIN_SUBGROUP_SIZE * 2;

const VEC4_SPT = 4u;
const VEC_PART_SIZE = BLOCK_DIM * VEC4_SPT;

const FLAG_NOT_READY = 0u;
const FLAG_REDUCTION = 1u;
const FLAG_INCLUSIVE = 2u;
const FLAG_MASK = 3u;

const MAX_SPIN_COUNT = 4u;
const LOCKED = 1u;
const UNLOCKED = 0u;

var<workgroup> wg_lock: u32;
var<workgroup> wg_broadcast: u32;
var<workgroup> wg_reduce: array<u32, MAX_REDUCE_SIZE>;
var<workgroup> wg_fallback: array<u32, MAX_REDUCE_SIZE>;

@compute @workgroup_size(BLOCK_DIM, 1, 1)
fn main(
    @builtin(local_invocation_id) threadid: vec3<u32>,
    @builtin(subgroup_invocation_id) laneid: u32,
    @builtin(subgroup_size) lane_count: u32) {
    
    let sid = threadid.x / lane_count;  //Caution 1D workgoup ONLY! Ok, but technically not in HLSL spec
    
    //acquire partition index, set the lock
    if(threadid.x == 0u){
        wg_broadcast = atomicAdd(&scan_bump, 1u);
        wg_lock = LOCKED;
    }
    let part_id = workgroupUniformLoad(&wg_broadcast);
    let s_offset = laneid + sid * lane_count * VEC4_SPT;

    var t_scan = array<vec4<u32>, VEC4_SPT>();
    {
        var i = s_offset + part_id * VEC_PART_SIZE;

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
            let t = subgroupShuffle(subgroupInclusiveAdd(select(prev, 0u, laneid != 0u) + t_scan[k].w), circular_shift);
            t_scan[k] += select(prev, t, laneid != 0u);
            prev = t;
        }

        if(laneid == 0u){
            wg_reduce[sid] = prev;
        }
    }
    workgroupBarrier();

    //Non-divergent subgroup agnostic inclusive scan across subgroup reductions
    let lane_log = u32(countTrailingZeros(lane_count));
    let spine_size = BLOCK_DIM >> lane_log;
    let aligned_size = 1u << ((u32(countTrailingZeros(spine_size)) + lane_log - 1u) / lane_log * lane_log);
    {   
        var top_offset = 0u;
        var offset = 0u;
        let lane_pred = laneid == lane_count - 1u;
        for(var j = lane_count; j <= aligned_size; j <<= lane_log){
            let step = spine_size >> offset;
            let pred1 = threadid.x < step;
            let t = subgroupInclusiveAdd(select(0u, wg_reduce[threadid.x], pred1));
            if(pred1) {
                wg_reduce[threadid.x + top_offset] = t;
                if(lane_pred){
                    wg_reduce[(threadid.x >> offset) + step + top_offset] = t;
                }
            }
            workgroupBarrier();

            let rshift = j >> lane_log;
            let index = threadid.x + rshift;
            if(index < spine_size && (index & (j - 1u)) >= rshift){
                wg_reduce[index] += wg_reduce[sid + top_offset - 1u];
            }
            top_offset += step;
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
        var prev_red = 0u;
        var lookback_id = part_id - 1u;

        var lock = workgroupUniformLoad(&wg_lock);
        while(lock == LOCKED){
            if(threadid.x == 0u){
                var spin_count = 0u;
                for(; spin_count < MAX_SPIN_COUNT; ){
                    let flag_payload = atomicLoad(&reduction[lookback_id]);
                    if((flag_payload & FLAG_MASK) > FLAG_NOT_READY){
                        prev_red += flag_payload >> 2u;
                        spin_count = 0u;
                        if((flag_payload & FLAG_MASK) == FLAG_INCLUSIVE){
                            atomicAdd(&reduction[part_id], (prev_red << 2u) | 1u);
                            wg_broadcast = prev_red;
                            wg_lock = UNLOCKED;
                            break;
                        } else {
                            lookback_id -= 1u;
                        }
                    } else {
                        spin_count += 1u;
                    }
                }

                //If we did not complete the lookback within the alotted spins,
                //broadcast the lookback id in shared memory to prepare for the fallback
                if(spin_count == MAX_SPIN_COUNT){
                    wg_broadcast = lookback_id;
                }
            }

            //Fallback if still locked
            lock = workgroupUniformLoad(&wg_lock);
            if(lock == LOCKED){
                let fallback_id = wg_broadcast;
                {
                    var i = s_offset + fallback_id * VEC_PART_SIZE;
                    var t_red = 0u;

                    for(var k = 0u; k < VEC4_SPT; k += 1u){
                        let t = scan_in[i];
                        t_red += dot(t, vec4<u32>(1u, 1u, 1u, 1u));
                        i += lane_count;
                    }

                    let s_red = subgroupAdd(t_red);
                    if(laneid == 0u){
                        wg_fallback[sid] = s_red;
                    }
                }
                workgroupBarrier();

                //Non-divergent subgroup agnostic reduction across subgroup reductions
                {
                    var offset = 0u;
                    for(var j = lane_count; j <= aligned_size; j <<= lane_log){
                        let i = ((threadid.x + 1u) << offset) - 1u;
                        let pred0 = i < spine_size;
                        let t = subgroupAdd(select(0u, wg_fallback[i], pred0));
                        if(pred0){
                            wg_fallback[i] = t;
                        }
                        workgroupBarrier();
                        offset += lane_log;
                    }
                }

                if(threadid.x == 0u){
                    //Max will store when no insertion has been made, but will not overwrite a tile
                    //which has already inserted, or been updated to FLAG_INCLUSIVE
                    let f_red = wg_fallback[spine_size - 1u];
                    let f_payload = atomicMax(&reduction[fallback_id],
                        (f_red << 2u) | select(FLAG_INCLUSIVE, FLAG_REDUCTION, fallback_id != 0u));
                    if(f_payload == 0u){
                        prev_red += f_red;
                    } else {
                        prev_red += f_payload >> 2u;
                    }

                    if(fallback_id == 0u || (f_payload & FLAG_MASK) == FLAG_INCLUSIVE){
                        atomicAdd(&reduction[part_id], (prev_red << 2u) | 1u);
                        wg_broadcast = prev_red;
                        wg_lock = UNLOCKED;
                    } else {
                        lookback_id -= 1u;
                    }
                }
                lock = workgroupUniformLoad(&wg_lock);
            }   
        }
    }

    {
        let prev = wg_broadcast + select(0u, wg_reduce[sid - 1u], sid != 0u); //wg_broadcast is 0 for part_id 0
        var i = s_offset + part_id * VEC_PART_SIZE;

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
