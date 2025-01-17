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
var<storage, read_write> reduction: array<array<atomic<u32>, 2>>;

@group(0) @binding(5)
var<storage, read_write> misc: array<u32>;

const BLOCK_DIM = 256u;
const SPLIT_MEMBERS = 2u;
const MIN_SUBGROUP_SIZE = 4u;
const MAX_REDUCE_SIZE = BLOCK_DIM / MIN_SUBGROUP_SIZE * 2u; //Double for conflict avoidance

const VEC4_SPT = 4u;
const VEC_PART_SIZE = BLOCK_DIM * VEC4_SPT;

const FLAG_NOT_READY = 0u;
const FLAG_READY = 0x40000000u;
const FLAG_INCLUSIVE = 0x80000000u;
const FLAG_MASK = 0xC0000000u;
const VALUE_MASK = 0xffffu;
const ALL_READY = 3u;

const MAX_SPIN_COUNT = 4u;
const LOCKED = 1u;
const UNLOCKED = 0u;

var<workgroup> wg_lock: u32;
var<workgroup> wg_broadcast: u32;
var<workgroup> wg_reduce: array<u32, MAX_REDUCE_SIZE>;
var<workgroup> wg_fallback: array<u32, MAX_REDUCE_SIZE>;

//Wrap all values
fn join(mine: u32, tid: u32) -> u32 {
    let xor = tid ^ 1;
    let theirs = subgroupShuffle(mine, xor);
    return (mine << (16u * tid)) | (theirs << (16u * xor));
}

fn split(x: u32, tid: u32) -> u32 {
    return (x >> (tid * 16u)) & VALUE_MASK; //bitcast as needed
}

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
        var offset = 0u;
        var top_offset = 0u;
        let lane_pred = laneid == lane_count - 1u;
        for(var j = lane_count; j <= aligned_size; j <<= lane_log){
            let step = spine_size >> offset;
            let pred = threadid.x < step;
            let t = subgroupInclusiveAdd(select(0u, wg_reduce[threadid.x + top_offset], pred));
            if(pred){
                wg_reduce[threadid.x + top_offset] = t;
                if(lane_pred){
                    wg_reduce[sid + step + top_offset] = t;
                }
            }
            workgroupBarrier();

            if(j != lane_count){
                let rshift = j >> lane_log;
                let index = threadid.x + rshift;
                if(index < spine_size && (index & (j - 1u)) >= rshift){
                    wg_reduce[index] += wg_reduce[(index >> offset) + top_offset - 1u];
                }
            }
            top_offset += step;
            offset += lane_log;
        }
    }   
    workgroupBarrier();

    //Device broadcast
    if(threadid.x < SPLIT_MEMBERS){
        let t = split(wg_reduce[spine_size - 1u], threadid.x) | select(FLAG_READY, FLAG_INCLUSIVE, part_id == 0u);
        atomicStore(&reduction[part_id][threadid.x], t);
    }

    //lookback, single subgroup
    if(part_id != 0u){
        var prev_red = 0u;
        var lookback_id = part_id - 1u;
        var lock = workgroupUniformLoad(&wg_lock);
        while(lock == LOCKED){
            if(threadid.x < lane_count){
                var spin_count = 0u;
                while(spin_count < MAX_SPIN_COUNT){
                    var flag_payload = select(0u, atomicLoad(&reduction[lookback_id][threadid.x]), threadid.x < SPLIT_MEMBERS);
                    if(subgroupBallot((flag_payload & FLAG_MASK) > FLAG_NOT_READY).x == ALL_READY) {
                        var incl_bal = subgroupBallot((flag_payload & FLAG_MASK) == FLAG_INCLUSIVE).x;
                        if(incl_bal != 0u) {
                            //Did we find any inclusive? Alright, the rest are guaranteed to be on their way, lets just wait. 
                            while(incl_bal != ALL_READY){
                                flag_payload = select(0u, atomicLoad(&reduction[lookback_id][threadid.x]), threadid.x < SPLIT_MEMBERS);
                                incl_bal = subgroupBallot((flag_payload & FLAG_MASK) == FLAG_INCLUSIVE).x;
                            }
                            prev_red += join(flag_payload & VALUE_MASK, threadid.x);
                            if(threadid.x < SPLIT_MEMBERS){
                                let t = split(prev_red + wg_reduce[spine_size - 1u], threadid.x) | FLAG_INCLUSIVE;
                                atomicStore(&reduction[part_id][threadid.x], t);
                            }
                            if(threadid.x == 0u){
                                wg_lock = UNLOCKED;
                                wg_broadcast = prev_red;
                            }
                            break;
                        } else {
                            prev_red += join(flag_payload & VALUE_MASK, threadid.x);
                            spin_count = 0u;
                            lookback_id -= 1u;
                        }
                    } else {
                        spin_count += 1u;
                    }

                    if(threadid.x == 0 && spin_count == MAX_SPIN_COUNT) {
                        wg_broadcast = lookback_id;
                    }
                }
            }

            //Fallback if still locked
            lock = workgroupUniformLoad(&wg_lock);
            if(lock == LOCKED){
                let fallback_id = wg_broadcast;
                {
                    var t_red = 0u;
                    var i = s_offset + fallback_id * VEC_PART_SIZE;
                    for(var k = 0u; k < VEC4_SPT; k += 1u){
                        t_red += dot(scan_in[i], vec4<u32>(1u, 1u, 1u, 1u));
                        i += lane_count;
                    }

                    let s_red = subgroupAdd(t_red);
                    if(laneid == 0u){
                        wg_fallback[sid] = s_red;
                    }
                }
                workgroupBarrier();

                //Non-divergent subgroup agnostic reduction across subgroup reductions
                var f_red = 0u;
                {
                    var offset = 0u;
                    var top_offset = 0u;
                    let lane_pred = laneid == lane_count - 1u;
                    for(var j = lane_count; j <= aligned_size; j <<= lane_log){
                        let step = spine_size >> offset;
                        let pred = threadid.x < step;
                        f_red = subgroupAdd(select(0u, wg_fallback[threadid.x + top_offset], pred));
                        if(pred && lane_pred){
                            wg_fallback[sid + step + top_offset] = f_red;
                        }
                        workgroupBarrier();
                        top_offset += step;
                        offset += lane_log;
                    }
                }

                if(threadid.x < lane_count){
                    let f_split = split(f_red, threadid.x) | select(FLAG_READY, FLAG_INCLUSIVE, fallback_id == 0u);
                    //This ternary does not play well for some reason?
                    //let f_payload = select(0u, atomicMax(&reduction[fallback_id][threadid.x], f_split), threadid.x < SPLIT_MEMBERS);
                    var f_payload = 0u;
                    if(threadid.x < SPLIT_MEMBERS) {
                        f_payload = atomicMax(&reduction[fallback_id][threadid.x], f_split);
                    }
                    let incl_found = subgroupBallot((f_payload & FLAG_MASK) == FLAG_INCLUSIVE).x == ALL_READY;
                    if(incl_found){
                        prev_red += join(f_payload & VALUE_MASK, threadid.x); 
                    } else {
                        prev_red += f_red;
                    }

                    if(fallback_id == 0u || incl_found){
                        if(threadid.x < SPLIT_MEMBERS){
                            let t = split(prev_red + wg_reduce[spine_size - 1u], threadid.x) | FLAG_INCLUSIVE;
                            atomicStore(&reduction[part_id][threadid.x], t);
                        }
                        if(threadid.x == 0u){
                            wg_lock = UNLOCKED;
                            wg_broadcast = prev_red;
                        }
                    } else {
                        lookback_id -= 1u;
                    }
                }
                lock = workgroupUniformLoad(&wg_lock);
            }
        }
    }

    {
        var i = s_offset + part_id * VEC_PART_SIZE;
        let prev = wg_broadcast + select(0u, wg_reduce[sid - 1u], sid != 0u); //wg_broadcast is 0 for part_id 0
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
