//****************************************************************************
// GPUPrefixSums
// CSDLDF Struct Occupancy Estimator Variant
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

const STRUCT_MEMBERS = 4u;

@group(0) @binding(0)
var<uniform> info : InfoStruct; 

@group(0) @binding(1)
var<storage, read_write> scan_in: array<vec4<u32>>;

@group(0) @binding(2)
var<storage, read_write> scan_out: array<vec4<u32>>;

@group(0) @binding(3)
var<storage, read_write> scan_bump: atomic<u32>;

@group(0) @binding(4)
var<storage, read_write> reduction: array<array<atomic<u32>, STRUCT_MEMBERS>>;

@group(0) @binding(5)
var<storage, read_write> misc: array<atomic<u32>>;

const BLOCK_DIM = 256u;
const MIN_SUBGROUP_SIZE = 4u;
const MAX_REDUCE_SIZE = BLOCK_DIM / MIN_SUBGROUP_SIZE;

const VEC4_SPT = 4u;
const VEC_PART_SIZE = BLOCK_DIM * VEC4_SPT;

const FLAG_NOT_READY = 0u;
const FLAG_REDUCTION = 1u;
const FLAG_INCLUSIVE = 2u;
const FLAG_MASK = 3u;

const MAX_SPIN_COUNT = 4u;
const LOCKED = 1u;
const UNLOCKED = 0u;

const OCC_COUNTER_INDEX = 1u;

var<workgroup> wg_lock: u32;
var<workgroup> wg_broadcast: u32;
var<workgroup> wg_lookback_broadcast: vec4<u32>;
var<workgroup> wg_reduce: array<vec4<u32>, MAX_REDUCE_SIZE>;
var<workgroup> wg_fallback: array<vec4<u32>, MAX_REDUCE_SIZE>;

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
    var part_id = workgroupUniformLoad(&wg_broadcast);

    //Only the initial threadblocks will ever increment this counter.
    //The remaining code is kept as identical as possible to the original.
    //The compiler is free to change register counts, so this number
    //should only be treated as an estimate
    if(threadid.x == 0u && part_id < info.thread_blocks){
        atomicAdd(&misc[OCC_COUNTER_INDEX], 1u);
    }

    while(part_id < info.thread_blocks){
        var t_scan = array<vec4<u32>, VEC4_SPT>();
        {
            let s_offset = laneid + sid * lane_count * VEC4_SPT;
            let dev_offset =  part_id * VEC_PART_SIZE;
            var i = s_offset + dev_offset;

            if(part_id < info.thread_blocks- 1u){
                for(var k = 0u; k < VEC4_SPT; k += 1u){
                    t_scan[k] = scan_in[i];
                    i += lane_count;
                }
            }

            if(part_id == info.thread_blocks - 1u){
                for(var k = 0u; k < VEC4_SPT; k += 1u){
                    if(i < info.vec_size){
                        t_scan[k] = scan_in[i];
                    }
                    i += lane_count;
                }
            }

            var prev = vec4(0u, 0u, 0u, 0u);
            let greatestLane = lane_count - 1u;
            for(var k = 0u; k < VEC4_SPT; k += 1u){
                t_scan[k] = subgroupInclusiveAdd(t_scan[k]) + prev;
                prev = subgroupShuffle(t_scan[k], greatestLane);
            }

            if(laneid == lane_count - 1u){
                wg_reduce[sid] = prev;
            }
        }
        workgroupBarrier();

        //Subgroup agnostic inclusive scan across subgroup reductions
        let wgSpineSize = BLOCK_DIM / lane_count;
        {
            let pred0 = threadid.x < wgSpineSize;
            let t0 = subgroupInclusiveAdd(select(vec4(0u, 0u, 0u, 0u), wg_reduce[threadid.x], pred0));
            if(pred0){
                wg_reduce[threadid.x] = t0;
            }
            workgroupBarrier();

            let laneLog = countTrailingZeros(lane_count) + 1u;
            var offset = laneLog;
            var j = laneLog;
            for(; j < (wgSpineSize >> 1u); j <<= laneLog){
                let index1 = ((threadid.x + 1u) << offset) - 1u;
                let pred1 = index1 < wgSpineSize;
                let t1 = subgroupInclusiveAdd(select(vec4(0u, 0u, 0u, 0u), wg_reduce[index1], pred1));
                if(pred1){
                    wg_reduce[index1] = t1;
                }
                workgroupBarrier();

                if((threadid.x & ((j << laneLog) - 1u)) >= j){  //Guaranteed lane aligned
                    let pred2 = ((threadid.x + 1u) & (j - 1u))!= 0u;
                    let t2 = subgroupBroadcast(wg_reduce[((threadid.x >> offset) << offset) - 1u], 0u); //index guaranteed gt 0
                    if(pred2){
                        wg_reduce[threadid.x] += t2;
                    }
                }
                offset += laneLog;
            }
            workgroupBarrier();

            let finalIndex = threadid.x + j;    //Guaranteed lane aligned
            if(finalIndex < wgSpineSize){    
                wg_reduce[finalIndex] += subgroupBroadcast(wg_reduce[((finalIndex >> offset) << offset) - 1u], 0u); //index guaranteed gt 0
            }
        }
        workgroupBarrier();

        //Device broadcast
        if(threadid.x == 0u){
            for(var k = 0u; k < STRUCT_MEMBERS; k += 1u){
                atomicStore(&reduction[part_id][k], (wg_reduce[wgSpineSize - 1u][k] << 2u) |
                    select(FLAG_INCLUSIVE, FLAG_REDUCTION, part_id != 0u));
            }
        }

        //Lookback, single thread
        if(part_id != 0u){
            var lookback_id = part_id - 1u;
            var prev_red = vec4(0u, 0u, 0u, 0u);
            var inc_complete = vec4(false, false, false, false);
            
            var lock = workgroupUniformLoad(&wg_lock);
            while(lock == LOCKED){
                var red_complete = vec4(false, false, false, false);
                if(threadid.x == 0u){
                    var can_advance: bool;
                    for(var spin_count = 0u; spin_count < MAX_SPIN_COUNT; ){
                        //Attempt Lookback
                        can_advance = true;
                        for(var k = 0u; k < STRUCT_MEMBERS; k += 1u){
                            if(!inc_complete[k] && !red_complete[k]){
                                let flag_payload = atomicLoad(&reduction[lookback_id][k]);
                                if((flag_payload & FLAG_MASK) != FLAG_NOT_READY){
                                    spin_count = 0u;
                                    prev_red[k] += flag_payload >> 2u;
                                    if((flag_payload & FLAG_MASK) == FLAG_INCLUSIVE){
                                        inc_complete[k] = true;
                                        wg_lookback_broadcast[k] = prev_red[k];
                                        atomicStore(&reduction[part_id][k],
                                            ((prev_red[k] + wg_reduce[wgSpineSize - 1u][k]) << 2u) | FLAG_INCLUSIVE);
                                    } else {
                                        red_complete[k] = true;
                                    }
                                } else {
                                    can_advance = false;
                                }
                            }
                        }

                        //Have we completed the current reduction or inclusive sum for all struct members?
                        if(can_advance){
                            //Are all lookbacks complete?
                            var all_complete = inc_complete[0u];
                            for(var k = 1u; k < STRUCT_MEMBERS; k += 1u){
                                all_complete &= inc_complete[k];
                            }
                            if(all_complete){
                                wg_lock = UNLOCKED;
                                break;
                            } else {
                                lookback_id -= 1u;
                                red_complete = vec4(false, false, false, false);
                            }
                        } else {
                            spin_count += 1u;
                        }
                    }

                    //If we did not complete the lookback within the alotted spins,
                    //broadcast the lookback id in shared memory to prepare for the fallback
                    if(!can_advance){
                        wg_broadcast = lookback_id;
                    }
                }

                //Fallback if still locked
                lock = workgroupUniformLoad(&wg_lock);
                if(lock == LOCKED){
                    let fallback_id = wg_broadcast;
                    {
                        let s_offset = laneid + sid * lane_count * VEC4_SPT;
                        let dev_offset =  fallback_id * VEC_PART_SIZE;
                        var i = s_offset + dev_offset;
                        var t_red = vec4(0u, 0u, 0u, 0u);

                        for(var k = 0u; k < VEC4_SPT; k += 1u){
                            t_red += scan_in[i];
                            i += lane_count;
                        }

                        let s_red = subgroupAdd(t_red);
                        if(laneid == 0u){
                            wg_fallback[sid] = s_red;
                        }
                    }
                    workgroupBarrier();

                    //Subgroup agnostic reduction across fallback subgroup reductions
                    {
                        let pred0 = threadid.x < wgSpineSize;
                        let t0 = subgroupAdd(select(vec4(0u, 0u, 0u, 0u), wg_fallback[threadid.x], pred0));
                        if(pred0){
                            wg_fallback[threadid.x] = t0;
                        }
                        workgroupBarrier();

                        let laneLog = countTrailingZeros(lane_count) + 1u;
                        var offset = laneLog;
                        for(var j = lane_count; j < (wgSpineSize >> 1u); j <<= laneLog){

                            let index1 = ((threadid.x + 1u) << offset) - 1u;
                            let pred1 = index1 < wgSpineSize;
                            let t1 = subgroupAdd(select(vec4(0u, 0u, 0u, 0u), wg_fallback[index1], pred1));
                            if(pred1){
                                wg_fallback[index1] = t1;
                            }
                            workgroupBarrier();
                            offset += laneLog;
                        }
                    }

                    //Fallback and attempt insertion of status flag
                    if(threadid.x == 0u){
                        var all_complete = true;
                        for(var k = 0u; k < STRUCT_MEMBERS; k += 1u){
                            if(!red_complete[k]){
                                if(!inc_complete[k]){
                                    //Max will store when no insertion has been made, but will not overwrite a tile
                                    //which has already inserted, or been updated to FLAG_INCLUSIVE
                                    let f_red = wg_fallback[wgSpineSize - 1u][k];
                                    let f_payload = atomicMax(&reduction[fallback_id][k],
                                        (f_red << 2u) | select(FLAG_INCLUSIVE, FLAG_REDUCTION, fallback_id != 0u));
                                    if(f_payload == 0u){
                                        prev_red[k] += f_red;
                                    } else {
                                        prev_red[k] += f_payload >> 2u;
                                    }

                                    if(fallback_id == 0u || (f_payload & FLAG_MASK) == FLAG_INCLUSIVE){
                                        atomicStore(&reduction[part_id][k],
                                            ((prev_red[k] + wg_reduce[wgSpineSize - 1u][k]) << 2u) | FLAG_INCLUSIVE);
                                        wg_lookback_broadcast[k] = prev_red[k];
                                        inc_complete[k] = true; 
                                    } else {
                                        all_complete = false;
                                    }
                                }
                            } else {
                                all_complete = false;
                            }
                        }

                        //At this point, the reductions are guaranteed to be complete,
                        //so try unlocking, else, keep looking back
                        if(all_complete){
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
            let prev = select(vec4(0u, 0u, 0u, 0u), wg_lookback_broadcast, part_id != 0u)
                + select(vec4(0u, 0u, 0u, 0u), wg_reduce[sid - 1u], sid != 0u);
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

        workgroupBarrier();

        if(threadid.x == 0u){
            wg_broadcast = atomicAdd(&scan_bump, 1u);
            wg_lock = LOCKED;
        }
        part_id = workgroupUniformLoad(&wg_broadcast);
    }
}
