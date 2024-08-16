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
var<storage, read_write> reduction: array<u32>;

@group(0) @binding(3)
var<storage, read_write> lazy_padding_0: array<u32>;

@group(0) @binding(4)
var<storage, read> info: array<u32>;

const BLOCK_DIM: u32 = 256;
const MIN_SUBGROUP_SIZE: u32 = 8;
const MAX_REDUCE_SIZE: u32 = BLOCK_DIM / MIN_SUBGROUP_SIZE;

const VEC4_SPT: u32 = 4;
const VEC_PART_SIZE: u32 = BLOCK_DIM * VEC4_SPT;

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
    let s_offset = laneid + sid * lane_count * VEC4_SPT;
    let dev_offset = blockid.x * VEC_PART_SIZE;
    var i: u32 = s_offset + dev_offset;

    var t_red = array<vec4<u32>, VEC4_SPT>();
    if(blockid.x < griddim.x - 1){
        t_red[0] = scan_in[i];
        i += lane_count;

        t_red[1] = scan_in[i];
        i += lane_count;

        t_red[2] = scan_in[i];
        i += lane_count;

        t_red[3] = scan_in[i];
        i += lane_count;
    }

    if(blockid.x == griddim.x - 1){
        t_red[0] = select(vec4<u32>(0u, 0u, 0u, 0u), scan_in[i], i < size);
        i += lane_count;

        t_red[1] = select(vec4<u32>(0u, 0u, 0u, 0u), scan_in[i], i < size);
        i += lane_count;

        t_red[2] = select(vec4<u32>(0u, 0u, 0u, 0u), scan_in[i], i < size);
        i += lane_count;

        t_red[3] = select(vec4<u32>(0u, 0u, 0u, 0u), scan_in[i], i < size);
        i += lane_count;
    }

    var sub_red: u32 = dot(t_red[0], vec4(1u, 1u, 1u, 1u));
    sub_red += dot(t_red[1], vec4(1u, 1u, 1u, 1u));
    sub_red += dot(t_red[2], vec4(1u, 1u, 1u, 1u));
    sub_red += dot(t_red[3], vec4(1u, 1u, 1u, 1u));
    sub_red = subgroupAdd(sub_red);

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
fn downsweep(
    @builtin(subgroup_invocation_id) laneid: u32,
    @builtin(subgroup_id) sid: u32,
    @builtin(subgroup_size) lane_count: u32,
    @builtin(workgroup_id) blockid: vec3<u32>,
    @builtin(num_workgroups) griddim: vec3<u32>) {

    //No push constant 
    let size = info[0u];
    var t_scan = array<vec4<u32>, VEC4_SPT>();
    {
        let lane_mask = lane_count - 1;
        let circular_shift = laneid + lane_mask & lane_mask;
        let s_offset = laneid + sid * lane_count * VEC4_SPT;
        let dev_offset =  blockid.x * VEC_PART_SIZE;
        var i: u32 = s_offset + dev_offset;
        var prev: u32 = 0u;

        if(blockid.x < griddim.x - 1u){
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

        if(blockid.x == griddim.x - 1u){
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

    let prev = select(0u, s_reduce[sid - 1u], sid != 0u) + select(0u, reduction[blockid.x - 1u], blockid.x != 0u); 
    {
        let s_offset = laneid + sid * lane_count * VEC4_SPT;
        let dev_offset =  blockid.x * VEC_PART_SIZE;
        var i: u32 = s_offset + dev_offset;

        if(blockid.x < griddim.x - 1u){
            scan_out[i] = t_scan[0] + prev;
            i += lane_count;

            scan_out[i] = t_scan[1] + prev;
            i += lane_count;

            scan_out[i] = t_scan[2] + prev;
            i += lane_count;

            scan_out[i] = t_scan[3] + prev;
        }

        if(blockid.x == griddim.x - 1u){
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