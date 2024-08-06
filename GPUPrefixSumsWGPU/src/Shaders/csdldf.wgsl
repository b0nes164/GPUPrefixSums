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

const FLAG_NOT_READY: u32 = 0;
const FLAG_REDUCTION: u32 = 1;
const FLAG_INCLUSIVE: u32 = 2;
const FLAG_MASK: u32 = 3;

const MAX_SPIN_COUNT: u32 =  4;
const LOCKED: u32 = 1;
const UNLOCKED: u32 = 0;

var<workgroup> s_broadcast: u32;
var<workgroup> s_lock: u32;
var<workgroup> s_scan: array<u32, PART_SIZE>;
var<workgroup> s_reduce: array<u32, BLOCK_DIM>;
var<workgroup> s_fallback: array<u32, BLOCK_DIM>;

@compute @workgroup_size(BLOCK_DIM, 1, 1)
fn main(
    @builtin(local_invocation_id) threadid: vec3<u32>,
    @builtin(num_workgroups) griddim: vec3<u32>) {

    //No push constant, so we use device memory instead...
    let size = info[0u];

    //acquire partition index, set the lock
    if(threadid.x == 0u){
        s_broadcast = atomicAdd(&index[0u], 1u);
        s_lock = 1u;
    }
    workgroupBarrier();
    let partition_index = s_broadcast;

    //Load
    {
        //Full
        let dev_offset = partition_index * PART_SIZE;
        if(partition_index < griddim.x - 1u){
            for(var i: u32 = threadid.x; i < PART_SIZE; i += BLOCK_DIM){
                s_scan[i] = scan[i + dev_offset];
            }
        }

        //Partial
        if(partition_index == griddim.x - 1u){
            let final_part_size = size - dev_offset;
            for(var i: u32 = threadid.x; i < final_part_size; i += BLOCK_DIM){
                s_scan[i] = scan[i + dev_offset];
            }
        }
    }
    workgroupBarrier();

    var t_scan = array<u32, SPT>();
    {
        let s_offset = threadid.x * SPT;
        for(var i: u32 = 0; i < SPT; i += 1u){
            t_scan[i] = s_scan[i + s_offset];
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

    //Device broadcast
    if(threadid.x == 0u){
        atomicCompareExchangeWeak(&reduction[partition_index], 0u, s_reduce[BLOCK_DIM - 1u] << 2u |
         select(FLAG_INCLUSIVE, FLAG_REDUCTION, partition_index != 0u));
    }

    //Lookback, with decoupled fallback
    if(partition_index != 0u){
        var prev_reduction: u32 = 0u;
        var lookback_index: u32 = partition_index - 1u;

        while(s_lock == LOCKED)
        {
            workgroupBarrier();
            
            if(threadid.x == 0){
                for(var spin_count: u32 = 0; spin_count < MAX_SPIN_COUNT; ){
                    let flag_payload = atomicLoad(&reduction[lookback_index]);
                    if((flag_payload & FLAG_MASK) > FLAG_NOT_READY){
                        prev_reduction += flag_payload >> 2u;
                        if((flag_payload & FLAG_MASK) == FLAG_INCLUSIVE){
                            atomicAdd(&reduction[partition_index], prev_reduction << 2u | 1u);
                            s_broadcast = prev_reduction;
                            s_lock = UNLOCKED;
                            break;
                        }
                        if((flag_payload & FLAG_MASK) == FLAG_REDUCTION){
                            lookback_index -= 1u;
                        }
                    } else {
                        spin_count += 1u;
                    }
                }

                if(s_lock == LOCKED){
                    s_broadcast = lookback_index;
                }
            }
            workgroupBarrier();

            //Fallback
            if(s_lock == LOCKED){
                //Full
                let fallback_index = s_broadcast;
                let dev_offset = partition_index * PART_SIZE;
                if(partition_index < griddim.x - 1u){
                    for(var i: u32 = threadid.x; i < PART_SIZE; i += BLOCK_DIM){
                        s_scan[i] = scan[i + dev_offset];
                    }
                }

                //Partial
                if(partition_index == griddim.x - 1u){
                    let final_part_size = size - dev_offset;
                    for(var i: u32 = threadid.x; i < final_part_size; i += BLOCK_DIM){
                        s_scan[i] = scan[i + dev_offset];
                    }
                }
                workgroupBarrier();

                var t_reduce: u32 = 0u;
                {
                    let s_offset = threadid.x * SPT;
                    for(var i: u32 = 0u; i < SPT; i += 1u){
                        t_reduce += s_scan[i + s_offset];
                    }
                }
                s_fallback[threadid.x] = t_reduce;
                workgroupBarrier();

                //upsweep
                if(threadid.x < (BLOCK_DIM >> 1u)){
                    s_fallback[(threadid.x << 1u) + 1u] += s_fallback[threadid.x << 1u];
                }

                var offset: u32 = 1;
                for(var j: u32 = (BLOCK_DIM >> 2u); j > 0u; j >>= 1u){
                    workgroupBarrier();
                    if(threadid.x < j){
                        s_fallback[(((threadid.x << 1u) + 2u) << offset) - 1u] +=
                        s_fallback[(((threadid.x << 1u) + 1u) << offset) - 1u];
                    }
                    offset += 1u;
                }
                workgroupBarrier();

                if(threadid.x == 0u){
                    let prev = atomicCompareExchangeWeak(&reduction[fallback_index], 0u, s_fallback[BLOCK_DIM - 1u] << 2u |
                        select(FLAG_INCLUSIVE, FLAG_REDUCTION, fallback_index != 0u));

                    if(prev.exchanged){
                        prev_reduction += s_fallback[BLOCK_DIM - 1u];
                    } else {
                        prev_reduction += prev.old_value >> 2u;
                    }

                    if(fallback_index == 0u || (prev.old_value & FLAG_MASK) == FLAG_INCLUSIVE){
                        atomicAdd(&reduction[partition_index], prev_reduction << 2u | 1u);
                        s_broadcast = prev_reduction;
                        s_lock = UNLOCKED;
                    } else {
                        lookback_index -= 1u;
                    }
                }
            }
            workgroupBarrier();
        }
    }

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
        let prev = s_broadcast + select(0u, s_reduce[threadid.x - 1u], threadid.x != 0u);
        for(var i: u32 = 0; i < SPT; i += 1u){
            s_scan[i + s_offset] = t_scan[i] + prev;
        }
    }
    workgroupBarrier();

    //Write
    {
        //Full
        let prev = s_broadcast;
        let dev_offset = partition_index * PART_SIZE;
        if(partition_index < griddim.x - 1u){
            for(var i: u32 = threadid.x; i < PART_SIZE; i += BLOCK_DIM){
                scan[i + dev_offset] = s_scan[i];
            }
        }

        //Partial
        if(partition_index == griddim.x - 1u){
            let final_part_size = size - dev_offset;
            for(var i: u32 = threadid.x; i < final_part_size; i += BLOCK_DIM){
                scan[i + dev_offset] = s_scan[i];
            }
        }
    }
}