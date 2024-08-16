//****************************************************************************
// GPUPrefixSums
//
// SPDX-License-Identifier: MIT
// Copyright Thomas Smith 8/4/2024
// https://github.com/b0nes164/GPUPrefixSums
//
//****************************************************************************
@group(0) @binding(0)
var<storage, read_write> scan_in: array<vec4<u32>>;

@group(0) @binding(1)
var<storage, read_write> scan_out: array<vec4<u32>>;

@group(0) @binding(2)
var<storage, read_write> reduction: array<atomic<u32>>;

@group(0) @binding(3)
var<storage, read_write> index: array<atomic<u32>>;

@group(0) @binding(4)
var<storage, read> info: array<u32>;

const BLOCK_DIM: u32 = 256;
const MIN_SUBGROUP_SIZE: u32 = 8;
const MAX_REDUCE_SIZE: u32 = BLOCK_DIM / MIN_SUBGROUP_SIZE;

const VEC4_SPT: u32 = 4;
const VEC_PART_SIZE: u32 = BLOCK_DIM * VEC4_SPT;

const FLAG_NOT_READY: u32 = 0;
const FLAG_REDUCTION: u32 = 1;
const FLAG_INCLUSIVE: u32 = 2;
const FLAG_MASK: u32 = 3;

const MAX_SPIN_COUNT: u32 = 16;
const LOCKED: u32 = 1;
const UNLOCKED: u32 = 0;

var<workgroup> s_broadcast: u32;
var<workgroup> s_lock: u32;
var<workgroup> s_reduce: array<u32, MAX_REDUCE_SIZE>;
var<workgroup> s_fallback: array<u32, MAX_REDUCE_SIZE>;

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
        s_lock = LOCKED;
    }
    workgroupBarrier();
    let part_id = s_broadcast;

    var t_scan = array<vec4<u32>, VEC4_SPT>();
    {
        let lane_mask = lane_count - 1;
        let circular_shift = laneid + lane_mask & lane_mask;
        let s_offset = laneid + sid * lane_count * VEC4_SPT;
        let dev_offset =  part_id * VEC_PART_SIZE;
        var i: u32 = s_offset + dev_offset;
        var prev: u32 = 0u;

        if(part_id < griddim.x - 1u){
            t_scan[0] = scan_in[i];
            t_scan[0].y += t_scan[0].x;
            t_scan[0].z += t_scan[0].y;
            t_scan[0].w += t_scan[0].z;
            i += lane_count;

            t_scan[1] = scan_in[i];
            t_scan[1].y += t_scan[1].x;
            t_scan[1].z += t_scan[1].y;
            t_scan[1].w += t_scan[1].z;
            i += lane_count;

            t_scan[2] = scan_in[i];
            t_scan[2].y += t_scan[2].x;
            t_scan[2].z += t_scan[2].y;
            t_scan[2].w += t_scan[2].z;
            i += lane_count;

            t_scan[3] = scan_in[i];
            t_scan[3].y += t_scan[3].x;
            t_scan[3].z += t_scan[3].y;
            t_scan[3].w += t_scan[3].z;
        }

        if(part_id == griddim.x - 1u){
            if(i < size){
                t_scan[0] = scan_in[i];
                t_scan[0].y += t_scan[0].x;
                t_scan[0].z += t_scan[0].y;
                t_scan[0].w += t_scan[0].z;
                i += lane_count;
            }

            if(i < size){
                t_scan[1] = scan_in[i];
                t_scan[1].y += t_scan[1].x;
                t_scan[1].z += t_scan[1].y;
                t_scan[1].w += t_scan[1].z;
                i += lane_count;
            }   

            if(i < size){
                t_scan[2] = scan_in[i];
                t_scan[2].y += t_scan[2].x;
                t_scan[2].z += t_scan[2].y;
                t_scan[2].w += t_scan[2].z;
                i += lane_count;
            }
            
            if(i < size){
                t_scan[3] = scan_in[i];
                t_scan[3].y += t_scan[3].x;
                t_scan[3].z += t_scan[3].y;
                t_scan[3].w += t_scan[3].z;
            }
        }

        {
            let t = subgroupShuffle(subgroupInclusiveAdd(t_scan[0].w), circular_shift);
            t_scan[0] += select(0u, t, laneid != 0u);
            prev = subgroupBroadcast(t, 0u);
        }

        {
            let t = subgroupShuffle(subgroupInclusiveAdd(t_scan[1].w), circular_shift);
            t_scan[1] += select(0u, t, laneid != 0u) + prev;
            prev += subgroupBroadcast(t, 0u);
        }

        {
            let t = subgroupShuffle(subgroupInclusiveAdd(t_scan[2].w), circular_shift);
            t_scan[2] += select(0u, t, laneid != 0u) + prev;
            prev += subgroupBroadcast(t, 0u);
        }

        {
            let t = subgroupShuffle(subgroupInclusiveAdd(t_scan[3].w), circular_shift);
            t_scan[3] += select(0u, t, laneid != 0u) + prev;
            prev += subgroupBroadcast(t, 0u);
        }

        if(laneid == 0u){
            s_reduce[sid] = prev;
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
        //It doesnt matter if a fallback already inserted,
        //this is guarunteed to be the same exact value
        atomicStore(&reduction[part_id], s_reduce[BLOCK_DIM / lane_count - 1u] << 2u |
         select(FLAG_INCLUSIVE, FLAG_REDUCTION, part_id != 0u));
    }

    //Lookback, with decoupled fallback
    if(part_id != 0u){
        var prev_reduction: u32 = 0u;
        var lookback_id: u32 = part_id - 1u;

        while(s_lock == LOCKED)
        {
            workgroupBarrier();

            if(threadid.x == 0){
                for(var spin_count: u32 = 0; spin_count < MAX_SPIN_COUNT; ){
                    let flag_payload = atomicLoad(&reduction[lookback_id]);
                    if((flag_payload & FLAG_MASK) > FLAG_NOT_READY){
                        prev_reduction += flag_payload >> 2u;
                        if((flag_payload & FLAG_MASK) == FLAG_INCLUSIVE){
                            //atomicStore(&reduction[part_id], prev_reduction + 
                            // s_reduce[BLOCK_DIM / lane_count - 1u] << 2u| FLAG_INCLUSIVE);
                            atomicAdd(&reduction[part_id], prev_reduction << 2u | 1u);
                            s_broadcast = prev_reduction;
                            s_lock = UNLOCKED;
                            break;
                        }
                        if((flag_payload & FLAG_MASK) == FLAG_REDUCTION){
                            lookback_id -= 1u;
                        }
                    } else {
                        spin_count += 1u;
                    }
                }

                if(s_lock == LOCKED){
                    s_broadcast = lookback_id;
                }
            }
            workgroupBarrier();

            //Fallback
            if(s_lock == LOCKED){
                let fallback_id = s_broadcast;
                let s_offset = laneid + sid * lane_count * VEC4_SPT;
                let dev_offset =  fallback_id * VEC_PART_SIZE;
                var i: u32 = s_offset + dev_offset;
                var red: u32 = 0u;

                red += dot(scan_in[i], vec4<u32>(1u, 1u, 1u, 1u));
                i += lane_count;

                red += dot(scan_in[i], vec4<u32>(1u, 1u, 1u, 1u));
                i += lane_count;

                red += dot(scan_in[i], vec4<u32>(1u, 1u, 1u, 1u));
                i += lane_count;

                red += dot(scan_in[i], vec4<u32>(1u, 1u, 1u, 1u));

                let sub_red = subgroupAdd(red);

                if(laneid == lane_count - 1u){
                    s_fallback[sid] = sub_red;
                }
                workgroupBarrier();

                if(sid == 0u){
                    let pred = laneid < BLOCK_DIM / lane_count;
                    let t = subgroupAdd(select(0u, s_fallback[laneid], pred));
                    if(pred){
                        s_fallback[laneid] = t;
                    }
                }
                workgroupBarrier();
                
                if(threadid.x == 0u){
                    //Check if another tile inserted before attempting our own insertion
                    let prev_payload = atomicLoad(&reduction[fallback_id]);
                    if(prev_payload == 0u){
                        //Max will store when no insertion has been made, but will not overwrite
                        //a tile which has updated to FLAG_INCLUSIVE
                        let f_red = s_fallback[BLOCK_DIM / lane_count - 1u];
                        let payload = f_red << 2u | select(FLAG_INCLUSIVE, FLAG_REDUCTION, fallback_id != 0u);
                        atomicMax(&reduction[fallback_id], payload);
                        prev_reduction += f_red;
                    } else {
                        prev_reduction += prev_payload >> 2u;
                    }

                    if(fallback_id == 0u || (prev_payload & FLAG_MASK) == FLAG_INCLUSIVE){
                        atomicAdd(&reduction[part_id], prev_reduction << 2u | 1u);
                        s_broadcast = prev_reduction;
                        s_lock = UNLOCKED;
                    } else {
                        lookback_id -= 1u;
                    }
                }
                workgroupBarrier();
            }
        }
    }

    let prev = select(0u, s_reduce[sid - 1u], sid != 0u) + s_broadcast;
    {
        let s_offset = laneid + sid * lane_count * VEC4_SPT;
        let dev_offset = part_id * VEC_PART_SIZE;
        var i: u32 = s_offset + dev_offset;

        if(part_id < griddim.x - 1u){
            scan_out[i] = t_scan[0] + prev;
            i += lane_count;

            scan_out[i] = t_scan[1] + prev;
            i += lane_count;

            scan_out[i] = t_scan[2] + prev;
            i += lane_count;

            scan_out[i] = t_scan[3] + prev;
        }

        if(part_id == griddim.x - 1u){
            if(i < size){
                scan_out[i] = t_scan[0] + prev;
                i += lane_count;
            }
            
            if(i < size){
                scan_out[i] = t_scan[1] + prev;
                i += lane_count;
            }
            
            if(i < size){
                scan_out[i] = t_scan[2] + prev;
                i += lane_count;
            }
            
            if(i < size){
                scan_out[i] = t_scan[3] + prev;
            }
        }
    }
}