//****************************************************************************
// GPUPrefixSums
// Reduce then Scan: 
// aka a "Tree Scan" or "Two Level Scan" Reduces values to intermediates,
// which are scanned over then passed into a final downsweep pass.
//
// SPDX-License-Identifier: MIT
// Copyright Thomas Smith 10/23/2024
// https://github.com/b0nes164/GPUPrefixSums
//
//****************************************************************************

struct InfoStruct
{
    size: u32,
    thread_blocks: u32,
};

@group(0) @binding(0)
var<uniform> info : InfoStruct; 

@group(0) @binding(1)
var<storage, read_write> scan_in: array<vec4<u32>>;

@group(0) @binding(2)
var<storage, read_write> scan_out: array<vec4<u32>>;

@group(0) @binding(3)
var<storage, read_write> scan_bump: u32;

@group(0) @binding(4)
var<storage, read_write> reduction: array<u32>;

@group(0) @binding(5)
var<storage, read_write> misc: array<u32>;

const BLOCK_DIM = 256u;
const MIN_SUBGROUP_SIZE = 4u;
const MAX_REDUCE_SIZE = BLOCK_DIM / MIN_SUBGROUP_SIZE;

const VEC4_SPT = 4u;
const VEC_PART_SIZE = BLOCK_DIM * VEC4_SPT;

const SPINE_SPT = 16u;
const SPINE_PART_SIZE = BLOCK_DIM * SPINE_SPT;

var<workgroup> wg_reduce: array<u32, MAX_REDUCE_SIZE>;

@compute @workgroup_size(BLOCK_DIM, 1, 1)
fn reduce(
    @builtin(local_invocation_id) threadid: vec3<u32>,
    @builtin(subgroup_invocation_id) laneid: u32,
    @builtin(subgroup_size) lane_count: u32,
    @builtin(workgroup_id) wgid: vec3<u32>) {
    
    let sid = threadid.x / lane_count;  //Caution 1D workgoup ONLY! Ok, but technically not in HLSL spec
    let s_offset = laneid + sid * lane_count * VEC4_SPT;
    let dev_offset = wgid.x * VEC_PART_SIZE;
    var i: u32 = s_offset + dev_offset;

    var s_red = 0u;
    if(wgid.x < info.thread_blocks - 1){
        for(var k = 0u; k < VEC4_SPT; k += 1u){
            let t = scan_in[i];
            s_red += dot(t, vec4(1u, 1u, 1u, 1u));
            i += lane_count;
        }
    }

    if(wgid.x == info.thread_blocks - 1){
        for(var k = 0u; k < VEC4_SPT; k += 1u){
            let t = select(vec4<u32>(0u, 0u, 0u, 0u), scan_in[i], i < info.size);
            s_red += dot(t, vec4(1u, 1u, 1u, 1u));
            i += lane_count;
        }
    }

    s_red = subgroupAdd(s_red);
    if(laneid == 0u){
        wg_reduce[sid] = s_red;
    }
    workgroupBarrier();

    //Subgroup agnostic reduction across subgroup reductions
    let redSize = BLOCK_DIM / lane_count;
    let pred0 = threadid.x < redSize;
    let t0 = subgroupAdd(select(0u, wg_reduce[threadid.x], pred0));
    if(pred0){
        wg_reduce[threadid.x] = t0;
    }
    workgroupBarrier();

    let laneLog = countTrailingZeros(lane_count) + 1u;
    var offset = laneLog;
    for(var j = lane_count; j < (redSize >> 1u); j <<= laneLog){

        let index1 = ((threadid.x + 1u) << offset) - 1u;
        let pred1 = index1 < redSize;
        let t1 = subgroupAdd(select(0u, wg_reduce[index1], pred1));
        if(pred1){
            wg_reduce[index1] = t1;
        }
        workgroupBarrier();
        offset += laneLog;
    }

    if(threadid.x == 0u){
        reduction[wgid.x] = wg_reduce[redSize - 1u];
    }
}

//Spine unvectorized
@compute @workgroup_size(BLOCK_DIM, 1, 1)
fn spine_scan(
    @builtin(local_invocation_id) threadid: vec3<u32>,
    @builtin(subgroup_invocation_id) laneid: u32,
    @builtin(subgroup_size) lane_count: u32) {
    
    let sid = threadid.x / lane_count;  //Caution 1D workgoup ONLY! Ok, but technically not in HLSL spec
    let s_offset = laneid + sid * lane_count * SPINE_SPT;
    let s_scanSize = BLOCK_DIM / lane_count;
    let aligned_size = (info.thread_blocks + SPINE_PART_SIZE - 1u) / SPINE_PART_SIZE * SPINE_PART_SIZE;
    var t_scan = array<u32, SPINE_SPT>();
    
    var prev_red = 0u;
    for(var dev_offset = 0u; dev_offset < aligned_size; dev_offset += SPINE_PART_SIZE){
        {
            var i = s_offset + dev_offset;
            for(var k = 0u; k < SPINE_SPT; k += 1u){
                if(i < info.thread_blocks){
                    t_scan[k] = reduction[i];
                }
                i += lane_count;
            }
        }

        var prev = 0u;
        for(var k = 0u; k < SPINE_SPT; k += 1u){
            t_scan[k] = subgroupInclusiveAdd(t_scan[k]) + prev;
            prev = subgroupShuffle(t_scan[k], lane_count - 1);
        }

        if(laneid == lane_count - 1u){
            wg_reduce[sid] = t_scan[SPINE_SPT - 1u];
        }
        workgroupBarrier();

        //Subgroup agnostic inclusive scan across subgroup reductions
        {
            let pred0 = threadid.x < s_scanSize;
            let t0 = subgroupInclusiveAdd(select(0u, wg_reduce[threadid.x], pred0));
            if(pred0){
                wg_reduce[threadid.x] = t0;
            }
            workgroupBarrier();

            let laneLog = countTrailingZeros(lane_count) + 1u;
            var offset = laneLog;
            var j = laneLog;
            for(; j < (s_scanSize >> 1u); j <<= laneLog){
                let index1 = ((threadid.x + 1u) << offset) - 1u;
                let pred1 = index1 < s_scanSize;
                let t1 = subgroupInclusiveAdd(select(0u, wg_reduce[index1], pred1));
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
            if(finalIndex < s_scanSize){    
                wg_reduce[finalIndex] += subgroupBroadcast(wg_reduce[((finalIndex >> offset) << offset) - 1u], 0u); //index guaranteed gt 0 
            }
        }
        workgroupBarrier();

        {
            let prev = select(0u, wg_reduce[sid - 1u], sid != 0u) + prev_red;
            var i: u32 = s_offset + dev_offset;
            for(var k = 0u; k < SPINE_SPT; k += 1u){
                if(i < info.thread_blocks){
                    reduction[i] = t_scan[k] + prev;
                }
                i += lane_count;
            }
        }

        prev_red += subgroupBroadcast(wg_reduce[s_scanSize - 1u], 0u);
        workgroupBarrier();
    }
}    

@compute @workgroup_size(BLOCK_DIM, 1, 1)
fn downsweep(
    @builtin(local_invocation_id) threadid: vec3<u32>,
    @builtin(subgroup_invocation_id) laneid: u32,
    @builtin(subgroup_size) lane_count: u32,
    @builtin(workgroup_id) wgid: vec3<u32>) {
    
    let sid = threadid.x / lane_count;  //Caution 1D workgoup ONLY! Ok, but technically not in HLSL spec
    var t_scan = array<vec4<u32>, VEC4_SPT>();

    {
        let s_offset = laneid + sid * lane_count * VEC4_SPT;
        let dev_offset = wgid.x * VEC_PART_SIZE;
        var i: u32 = s_offset + dev_offset;

        if(wgid.x < info.thread_blocks- 1u){
            for(var k = 0u; k < VEC4_SPT; k += 1u){
                t_scan[k] = scan_in[i];
                t_scan[k].y += t_scan[k].x;
                t_scan[k].z += t_scan[k].y;
                t_scan[k].w += t_scan[k].z;
                i += lane_count;
            }
        }

        if(wgid.x == info.thread_blocks - 1u){
            for(var k = 0u; k < VEC4_SPT; k += 1u){
                if(i < info.size){
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
    {
        let scanSize = BLOCK_DIM / lane_count;
        let pred0 = threadid.x < scanSize;
        let t0 = subgroupInclusiveAdd(select(0u, wg_reduce[threadid.x], pred0));
        if(pred0){
            wg_reduce[threadid.x] = t0;
        }
        workgroupBarrier();

        let laneLog = countTrailingZeros(lane_count) + 1u;
        var offset = laneLog;
        var j = laneLog;
        for(; j < (scanSize >> 1u); j <<= laneLog){
            let index1 = ((threadid.x + 1u) << offset) - 1u;
            let pred1 = index1 < scanSize;
            let t1 = subgroupInclusiveAdd(select(0u, wg_reduce[index1], pred1));
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
        if(finalIndex < scanSize){    
            wg_reduce[finalIndex] += subgroupBroadcast(wg_reduce[((finalIndex >> offset) << offset) - 1u], 0u); //index guaranteed gt 0 
        }
    }
    workgroupBarrier();
    
    {
        let prev = select(0u, reduction[wgid.x - 1u], wgid.x != 0u) + select(0u, wg_reduce[sid - 1u], sid != 0u);
        let s_offset = laneid + sid * lane_count * VEC4_SPT;
        let dev_offset =  wgid.x * VEC_PART_SIZE;
        var i = s_offset + dev_offset;

        if(wgid.x < info.thread_blocks - 1u){
            for(var k = 0u; k < VEC4_SPT; k += 1u){
                scan_out[i] = t_scan[k] + prev;
                i += lane_count;
            }
        }

        if(wgid.x == info.thread_blocks - 1u){
            for(var k = 0u; k < VEC4_SPT; k += 1u){
                if(i < info.size){
                    scan_out[i] = t_scan[k] + prev;
                }
                i += lane_count;
            }
        }
    }
}
