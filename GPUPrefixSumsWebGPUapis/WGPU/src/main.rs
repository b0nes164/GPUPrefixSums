/******************************************************************************
 * GPUPrefixSums
 *
 * SPDX-License-Identifier: MIT
 * Copyright Thomas Smith 10/23/2024
 * https://github.com/b0nes164/GPUPrefixSums
 *
 ******************************************************************************/

use std::{env, vec};
use wgpu::util::DeviceExt;

fn div_round_up(x: u32, y: u32) -> u32 {
    (x + y - 1) / y
}

enum ScanType {
    Rts,
    Csdl,
    Csdldf,
    CsdldfStats,
    CsdldfOcc,
    CsdldfStruct,
    CsdldfStructStats,
    CsdldfStructOcc,
}

struct GPUContext {
    device: wgpu::Device,
    queue: wgpu::Queue,
    query_set: wgpu::QuerySet,
    timestamp_freq: f32,
}

impl GPUContext {
    async fn init() -> Self {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            backends: wgpu::Backends::all(),
            flags: wgpu::InstanceFlags::empty(),
            dx12_shader_compiler: wgpu::Dx12Compiler::default(),
            gles_minor_version: wgpu::Gles3MinorVersion::default(),
        });

        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                compatible_surface: None,
                force_fallback_adapter: false,
            })
            .await
            .expect("Failed to find an appropriate adapter");

        let (device, queue) = adapter
            .request_device(
                &wgpu::DeviceDescriptor {
                    label: None,
                    required_features: wgpu::Features::TIMESTAMP_QUERY | wgpu::Features::SUBGROUP,
                    required_limits: wgpu::Limits::default(),
                    memory_hints: wgpu::MemoryHints::Performance,
                },
                None,
            )
            .await
            .expect("Failed to create device");

        let query_set = device.create_query_set(&wgpu::QuerySetDescriptor {
            label: Some("Timestamp Query Set"),
            count: 6,
            ty: wgpu::QueryType::Timestamp,
        });

        let timestamp_freq = queue.get_timestamp_period();

        GPUContext {
            device,
            queue,
            query_set,
            timestamp_freq,
        }
    }
}

struct GPUBuffers {
    info: wgpu::Buffer,
    scan_in: wgpu::Buffer,
    scan_out: wgpu::Buffer,
    scan_bump: wgpu::Buffer,
    reduction: wgpu::Buffer,
    timestamp: wgpu::Buffer,
    timestamp_readback: wgpu::Buffer,
    readback: wgpu::Buffer,
    misc: wgpu::Buffer,
}

impl GPUBuffers {
    fn init(
        gpu: &GPUContext,
        size: usize,
        thread_blocks: usize,
        max_pass_count: usize,
        max_readback_size: usize,
        misc_size: usize,
    ) -> Self {
        let buffer_size = (size * std::mem::size_of::<u32>()) as u64;

        //Vectorize the size here
        let info_info: Vec<u32> = vec![
            size as u32,
            div_round_up(size as u32, 4),
            thread_blocks as u32,
        ];
        let info = gpu
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Info"),
                contents: bytemuck::cast_slice(&info_info),
                usage: wgpu::BufferUsages::UNIFORM,
            });

        let scan_in: wgpu::Buffer = gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Scan In"),
            size: buffer_size,
            usage: wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        });

        let scan_out: wgpu::Buffer = gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Scan Out"),
            size: buffer_size,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let scan_bump: wgpu::Buffer = gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Scan Bump"),
            size: std::mem::size_of::<u32>() as u64,
            usage: wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        });

        //Oversize this buffer to accomodate struct version when necessary
        let reduction: wgpu::Buffer = gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Intermediate Reduction"),
            size: ((thread_blocks * 4usize) * std::mem::size_of::<u32>()) as u64,
            usage: wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        });

        let timestamp_size = (max_pass_count * 2usize * std::mem::size_of::<u64>()) as u64;
        let timestamp = gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Timestamp"),
            size: timestamp_size,
            usage: wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::QUERY_RESOLVE,
            mapped_at_creation: false,
        });

        let timestamp_readback = gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Timestamp Readback"),
            size: timestamp_size,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let readback = gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Readback"),
            size: ((max_readback_size) * std::mem::size_of::<u32>()) as u64,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let misc = gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Misc"),
            size: (misc_size * std::mem::size_of::<u32>()) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        GPUBuffers {
            info,
            scan_in,
            scan_out,
            scan_bump,
            reduction,
            timestamp,
            timestamp_readback,
            readback,
            misc,
        }
    }
}

//For simplicity we are going to use the bind group and layout
//for all of the kernels except the validation
struct ComputeShader {
    bind_group: wgpu::BindGroup,
    compute_pipeline: wgpu::ComputePipeline,
    label: String,
}

impl ComputeShader {
    fn init(
        gpu: &GPUContext,
        gpu_buffers: &GPUBuffers,
        entry_point: &str,
        module: &wgpu::ShaderModule,
        cs_label: &str,
    ) -> Self {
        let bind_group_layout =
            gpu.device
                .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    label: Some(&format!("Bind Group Layout {}", cs_label)),
                    entries: &[
                        wgpu::BindGroupLayoutEntry {
                            binding: 0,
                            visibility: wgpu::ShaderStages::COMPUTE,
                            ty: wgpu::BindingType::Buffer {
                                ty: wgpu::BufferBindingType::Uniform,
                                has_dynamic_offset: false,
                                min_binding_size: None,
                            },
                            count: None,
                        },
                        wgpu::BindGroupLayoutEntry {
                            binding: 1,
                            visibility: wgpu::ShaderStages::COMPUTE,
                            ty: wgpu::BindingType::Buffer {
                                ty: wgpu::BufferBindingType::Storage { read_only: false },
                                has_dynamic_offset: false,
                                min_binding_size: None,
                            },
                            count: None,
                        },
                        wgpu::BindGroupLayoutEntry {
                            binding: 2,
                            visibility: wgpu::ShaderStages::COMPUTE,
                            ty: wgpu::BindingType::Buffer {
                                ty: wgpu::BufferBindingType::Storage { read_only: false },
                                has_dynamic_offset: false,
                                min_binding_size: None,
                            },
                            count: None,
                        },
                        wgpu::BindGroupLayoutEntry {
                            binding: 3,
                            visibility: wgpu::ShaderStages::COMPUTE,
                            ty: wgpu::BindingType::Buffer {
                                ty: wgpu::BufferBindingType::Storage { read_only: false },
                                has_dynamic_offset: false,
                                min_binding_size: None,
                            },
                            count: None,
                        },
                        wgpu::BindGroupLayoutEntry {
                            binding: 4,
                            visibility: wgpu::ShaderStages::COMPUTE,
                            ty: wgpu::BindingType::Buffer {
                                ty: wgpu::BufferBindingType::Storage { read_only: false },
                                has_dynamic_offset: false,
                                min_binding_size: None,
                            },
                            count: None,
                        },
                        wgpu::BindGroupLayoutEntry {
                            binding: 5,
                            visibility: wgpu::ShaderStages::COMPUTE,
                            ty: wgpu::BindingType::Buffer {
                                ty: wgpu::BufferBindingType::Storage { read_only: false },
                                has_dynamic_offset: false,
                                min_binding_size: None,
                            },
                            count: None,
                        },
                    ],
                });

        let bind_group = gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some(&format!("Bind Group {}", cs_label)),
            layout: &bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: gpu_buffers.info.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: gpu_buffers.scan_in.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: gpu_buffers.scan_out.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: gpu_buffers.scan_bump.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: gpu_buffers.reduction.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: gpu_buffers.misc.as_entire_binding(),
                },
            ],
        });

        let pipeline_layout_init =
            gpu.device
                .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                    label: Some(&format!("Pipeline Layout {}", cs_label)),
                    bind_group_layouts: &[&bind_group_layout],
                    push_constant_ranges: &[],
                });

        let compute_pipeline =
            gpu.device
                .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                    label: Some(&format!("Compute Pipeline {}", cs_label)),
                    layout: Some(&pipeline_layout_init),
                    module,
                    entry_point: Some(entry_point),
                    compilation_options: Default::default(),
                    cache: Default::default(),
                });

        ComputeShader {
            bind_group,
            compute_pipeline,
            label: cs_label.to_string(),
        }
    }
}

struct Shaders {
    init: ComputeShader,
    reduce: ComputeShader,
    spine_scan: ComputeShader,
    downsweep: ComputeShader,
    csdl: ComputeShader,
    csdldf: ComputeShader,
    csdldf_struct: ComputeShader,
    csdldf_stats: ComputeShader,
    csdldf_struct_stats: ComputeShader,
    csdldf_occ: ComputeShader,
    csdldf_struct_occ: ComputeShader,
    validate: ComputeShader,
    validate_struct: ComputeShader,
}

impl Shaders {
    fn init(gpu: &GPUContext, gpu_buffers: &GPUBuffers) -> Self {
        let init_mod = gpu
            .device
            .create_shader_module(wgpu::include_wgsl!("../../SharedShaders/init.wgsl"));
        let valid_mod = gpu
            .device
            .create_shader_module(wgpu::include_wgsl!("../../SharedShaders/validate.wgsl"));
        let rts_mod = gpu
            .device
            .create_shader_module(wgpu::include_wgsl!("../../SharedShaders/rts.wgsl"));
        let csdl_mod = gpu
            .device
            .create_shader_module(wgpu::include_wgsl!("../../SharedShaders/csdl.wgsl"));
        let csdldf_mod = gpu
            .device
            .create_shader_module(wgpu::include_wgsl!("../../SharedShaders/csdldf.wgsl"));
        let csdldf_stats_mod = gpu.device.create_shader_module(wgpu::include_wgsl!(
            "../../SharedShaders/TestVariants/csdldf_stats.wgsl"
        ));
        let csdldf_occ_mod = gpu.device.create_shader_module(wgpu::include_wgsl!(
            "../../SharedShaders/TestVariants/csdldf_occ.wgsl"
        ));
        let csdldf_struct_mod = gpu.device.create_shader_module(wgpu::include_wgsl!(
            "../../SharedShaders/csdldf_struct.wgsl"
        ));
        let csdldf_struct_stats_mod = gpu.device.create_shader_module(wgpu::include_wgsl!(
            "../../SharedShaders/TestVariants/csdldf_struct_stats.wgsl"
        ));
        let csdldf_struct_occ_mod = gpu.device.create_shader_module(wgpu::include_wgsl!(
            "../../SharedShaders/TestVariants/csdldf_struct_occ.wgsl"
        ));
        
        let init = ComputeShader::init(gpu, gpu_buffers, "main", &init_mod, "Init");
        let reduce = ComputeShader::init(gpu, gpu_buffers, "reduce", &rts_mod, "Reduce");
        let spine_scan =
            ComputeShader::init(gpu, gpu_buffers, "spine_scan", &rts_mod, "Spine Scan");
        let downsweep = ComputeShader::init(gpu, gpu_buffers, "downsweep", &rts_mod, "Downsweep");
        let csdl = ComputeShader::init(gpu, gpu_buffers, "main", &csdl_mod, "CSDL");
        let csdldf = ComputeShader::init(gpu, gpu_buffers, "main", &csdldf_mod, "CSDLDF");
        let csdldf_stats =
            ComputeShader::init(gpu, gpu_buffers, "main", &csdldf_stats_mod, "CSDLDF Stats");
        let csdldf_occ = ComputeShader::init(
            gpu,
            gpu_buffers,
            "main",
            &csdldf_occ_mod,
            "CSDLDF Occupancy",
        );
        let csdldf_struct = ComputeShader::init(
            gpu,
            gpu_buffers,
            "main",
            &csdldf_struct_mod,
            "CSDLDF Struct",
        );
        let csdldf_struct_stats = ComputeShader::init(
            gpu,
            gpu_buffers,
            "main",
            &csdldf_struct_stats_mod,
            "CSDLDF Struct Stats",
        );
        let csdldf_struct_occ = ComputeShader::init(
            gpu,
            gpu_buffers,
            "main",
            &csdldf_struct_occ_mod,
            "CSDLDF Struct Occupancy",
        );
        let validate = ComputeShader::init(gpu, gpu_buffers, "main", &valid_mod, "Validate");
        let validate_struct = ComputeShader::init(
            gpu,
            gpu_buffers,
            "validate_struct",
            &valid_mod,
            "Validate Struct",
        );

        Shaders {
            init,
            reduce,
            spine_scan,
            downsweep,
            csdl,
            csdldf,
            csdldf_stats,
            csdldf_occ,
            csdldf_struct,
            csdldf_struct_stats,
            csdldf_struct_occ,
            validate,
            validate_struct,
        }
    }
}

fn set_compute_pass(
    query: &wgpu::QuerySet,
    cs: &ComputeShader,
    com_encoder: &mut wgpu::CommandEncoder,
    thread_blocks: u32,
    timestamp_offset: u32,
) {
    let mut pass = com_encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
        label: Some(&format!("{} Pass", cs.label)),
        timestamp_writes: Some(wgpu::ComputePassTimestampWrites {
            query_set: query,
            beginning_of_pass_write_index: Some(timestamp_offset),
            end_of_pass_write_index: Some(timestamp_offset + 1u32),
        }),
    });
    pass.set_pipeline(&cs.compute_pipeline);
    pass.set_bind_group(0, &cs.bind_group, &[]);
    pass.dispatch_workgroups(thread_blocks, 1, 1);
}

fn readback_back(tester: &Tester, data_out: &mut Vec<u32>, readback_size: u64) {
    let readback_slice = &tester.gpu_buffers.readback.slice(0..readback_size);
    readback_slice.map_async(wgpu::MapMode::Read, |result| {
        result.unwrap();
    });
    tester.gpu_context.device.poll(wgpu::Maintain::wait());
    let data = readback_slice.get_mapped_range();
    data_out.extend_from_slice(bytemuck::cast_slice(&data));
}

fn validate_base(tester: &Tester, cs: &ComputeShader) -> bool {
    let mut valid_command =
        tester
            .gpu_context
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Valid Command Encoder"),
            });
    {
        let mut valid_pass = valid_command.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("Validate Pass"),
            timestamp_writes: None,
        });
        valid_pass.set_pipeline(&cs.compute_pipeline);
        valid_pass.set_bind_group(0, &cs.bind_group, &[]);
        valid_pass.dispatch_workgroups(256, 1, 1);
    }
    valid_command.copy_buffer_to_buffer(
        &tester.gpu_buffers.misc,
        0u64,
        &tester.gpu_buffers.readback,
        0u64,
        std::mem::size_of::<u32>() as u64,
    );
    tester
        .gpu_context
        .queue
        .submit(Some(valid_command.finish()));

    let mut data_out: Vec<u32> = vec![];
    readback_back(tester, &mut data_out, std::mem::size_of::<u32>() as u64);
    tester.gpu_buffers.readback.unmap();

    if data_out[0] != 0 {
        println!("Err count {}", data_out[0]);
    }
    data_out[0] == 0
}

fn validate_generic(tester: &Tester) -> bool {
    validate_base(tester, &tester.gpu_shaders.validate)
}

fn validate_struct(tester: &Tester) -> bool {
    validate_base(tester, &tester.gpu_shaders.validate_struct)
}

fn get_stats(tester: &Tester, stats_out: &mut Vec<u32>) {
    let mut stats_command =
        tester
            .gpu_context
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Stats Command Encoder"),
            });
    stats_command.copy_buffer_to_buffer(
        &tester.gpu_buffers.misc,
        std::mem::size_of::<u32>() as u64,
        &tester.gpu_buffers.readback,
        0u64,
        3u64 * std::mem::size_of::<u32>() as u64,
    );
    tester
        .gpu_context
        .queue
        .submit(Some(stats_command.finish()));
    readback_back(tester, stats_out, 3u64 * std::mem::size_of::<u32>() as u64);
    tester.gpu_buffers.readback.unmap();
}

fn get_occupancy(tester: &Tester, occ_out: &mut Vec<u32>) {
    let mut stats_command =
        tester
            .gpu_context
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Occupancy Command Encoder"),
            });
    stats_command.copy_buffer_to_buffer(
        &tester.gpu_buffers.misc,
        std::mem::size_of::<u32>() as u64,
        &tester.gpu_buffers.readback,
        0u64,
        std::mem::size_of::<u32>() as u64,
    );
    tester
        .gpu_context
        .queue
        .submit(Some(stats_command.finish()));
    readback_back(tester, occ_out, std::mem::size_of::<u32>() as u64);
    tester.gpu_buffers.readback.unmap();
}

trait PassLogic {
    fn main_pass(&self, tester: &Tester, com_encoder: &mut wgpu::CommandEncoder);
    fn validation_pass(&self, tester: &Tester) -> bool;
    fn stats(&self, _tester: &Tester, _stats_out: &mut Vec<u32>);
    fn occupancy(&self, _tester: &Tester, _occ_out: &mut Vec<u32>);
    fn pass_count(&self) -> u32;
}

struct RtsPass;
impl PassLogic for RtsPass {
    fn main_pass(&self, tester: &Tester, com_encoder: &mut wgpu::CommandEncoder) {
        set_compute_pass(
            &tester.gpu_context.query_set,
            &tester.gpu_shaders.reduce,
            com_encoder,
            tester.thread_blocks,
            0u32,
        );
        set_compute_pass(
            &tester.gpu_context.query_set,
            &tester.gpu_shaders.spine_scan,
            com_encoder,
            1u32,
            2u32,
        );
        set_compute_pass(
            &tester.gpu_context.query_set,
            &tester.gpu_shaders.downsweep,
            com_encoder,
            tester.thread_blocks,
            4u32,
        );
    }

    fn validation_pass(&self, tester: &Tester) -> bool {
        validate_generic(tester)
    }

    fn stats(&self, _tester: &Tester, _stats_out: &mut Vec<u32>) {}

    fn occupancy(&self, _tester: &Tester, _occ_out: &mut Vec<u32>) {}

    fn pass_count(&self) -> u32 {
        3u32
    }
}

struct CsdlPass;
impl PassLogic for CsdlPass {
    fn main_pass(&self, tester: &Tester, com_encoder: &mut wgpu::CommandEncoder) {
        set_compute_pass(
            &tester.gpu_context.query_set,
            &tester.gpu_shaders.csdl,
            com_encoder,
            tester.thread_blocks,
            0u32,
        );
    }

    fn validation_pass(&self, tester: &Tester) -> bool {
        validate_generic(tester)
    }

    fn stats(&self, _tester: &Tester, _stats_out: &mut Vec<u32>) {}

    fn occupancy(&self, _tester: &Tester, _occ_out: &mut Vec<u32>) {}

    fn pass_count(&self) -> u32 {
        1u32
    }
}

struct CsdldfPass;
impl PassLogic for CsdldfPass {
    fn main_pass(&self, tester: &Tester, com_encoder: &mut wgpu::CommandEncoder) {
        set_compute_pass(
            &tester.gpu_context.query_set,
            &tester.gpu_shaders.csdldf,
            com_encoder,
            tester.thread_blocks,
            0u32,
        );
    }

    fn validation_pass(&self, tester: &Tester) -> bool {
        validate_generic(tester)
    }

    fn stats(&self, _tester: &Tester, _stats_out: &mut Vec<u32>) {}

    fn occupancy(&self, _tester: &Tester, _occ_out: &mut Vec<u32>) {}

    fn pass_count(&self) -> u32 {
        1u32
    }
}

struct CsdldfStatsPass;
impl PassLogic for CsdldfStatsPass {
    fn main_pass(&self, tester: &Tester, com_encoder: &mut wgpu::CommandEncoder) {
        set_compute_pass(
            &tester.gpu_context.query_set,
            &tester.gpu_shaders.csdldf_stats,
            com_encoder,
            tester.thread_blocks,
            0u32,
        );
    }

    fn validation_pass(&self, tester: &Tester) -> bool {
        validate_generic(tester)
    }

    fn stats(&self, _tester: &Tester, _stats_out: &mut Vec<u32>) {
        get_stats(_tester, _stats_out);
    }

    fn occupancy(&self, _tester: &Tester, _occ_out: &mut Vec<u32>) {}

    fn pass_count(&self) -> u32 {
        1u32
    }
}

struct CsdldfOccPass;
impl PassLogic for CsdldfOccPass {
    fn main_pass(&self, tester: &Tester, com_encoder: &mut wgpu::CommandEncoder) {
        set_compute_pass(
            &tester.gpu_context.query_set,
            &tester.gpu_shaders.csdldf_occ,
            com_encoder,
            tester.thread_blocks,
            0u32,
        );
    }

    fn validation_pass(&self, tester: &Tester) -> bool {
        validate_generic(tester)
    }

    fn stats(&self, _tester: &Tester, _stats_out: &mut Vec<u32>) {}

    fn occupancy(&self, _tester: &Tester, _occ_out: &mut Vec<u32>) {
        get_occupancy(_tester, _occ_out);
    }

    fn pass_count(&self) -> u32 {
        1u32
    }
}

struct CsdldfStructPass;
impl PassLogic for CsdldfStructPass {
    fn main_pass(&self, tester: &Tester, com_encoder: &mut wgpu::CommandEncoder) {
        set_compute_pass(
            &tester.gpu_context.query_set,
            &tester.gpu_shaders.csdldf_struct,
            com_encoder,
            tester.thread_blocks,
            0u32,
        );
    }

    fn validation_pass(&self, tester: &Tester) -> bool {
        validate_struct(tester)
    }

    fn stats(&self, _tester: &Tester, _stats_out: &mut Vec<u32>) {}

    fn occupancy(&self, _tester: &Tester, _occ_out: &mut Vec<u32>) {}

    fn pass_count(&self) -> u32 {
        1u32
    }
}

struct CsdldfStructStatsPass;
impl PassLogic for CsdldfStructStatsPass {
    fn main_pass(&self, tester: &Tester, com_encoder: &mut wgpu::CommandEncoder) {
        set_compute_pass(
            &tester.gpu_context.query_set,
            &tester.gpu_shaders.csdldf_struct_stats,
            com_encoder,
            tester.thread_blocks,
            0u32,
        );
    }

    fn validation_pass(&self, tester: &Tester) -> bool {
        validate_struct(tester)
    }

    fn stats(&self, _tester: &Tester, _stats_out: &mut Vec<u32>) {
        get_stats(_tester, _stats_out);
    }

    fn occupancy(&self, _tester: &Tester, _occ_out: &mut Vec<u32>) {}

    fn pass_count(&self) -> u32 {
        1u32
    }
}

struct CsdldfStructOccPass;
impl PassLogic for CsdldfStructOccPass {
    fn main_pass(&self, tester: &Tester, com_encoder: &mut wgpu::CommandEncoder) {
        set_compute_pass(
            &tester.gpu_context.query_set,
            &tester.gpu_shaders.csdldf_struct_occ,
            com_encoder,
            tester.thread_blocks,
            0u32,
        );
    }

    fn validation_pass(&self, tester: &Tester) -> bool {
        validate_struct(tester)
    }

    fn stats(&self, _tester: &Tester, _stats_out: &mut Vec<u32>) {}

    fn occupancy(&self, _tester: &Tester, _occ_out: &mut Vec<u32>) {
        get_occupancy(_tester, _occ_out);
    }

    fn pass_count(&self) -> u32 {
        1u32
    }
}

struct Tester {
    gpu_context: GPUContext,
    gpu_buffers: GPUBuffers,
    gpu_shaders: Shaders,
    size: u32,
    thread_blocks: u32,
}

impl Tester {
    async fn init(
        size: u32,
        thread_blocks: u32,
        max_pass_count: usize,
        max_readback_size: usize,
        misc_size: usize,
    ) -> Self {
        let gpu_context = GPUContext::init().await;
        let gpu_buffers = GPUBuffers::init(
            &gpu_context,
            size as usize,
            thread_blocks as usize,
            max_pass_count,
            max_readback_size,
            misc_size,
        );
        let gpu_shaders = Shaders::init(&gpu_context, &gpu_buffers);
        Tester {
            gpu_context,
            gpu_buffers,
            gpu_shaders,
            size,
            thread_blocks,
        }
    }

    fn init_pass(&self, com_encoder: &mut wgpu::CommandEncoder) {
        let mut init_pass = com_encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("Init Pass"),
            timestamp_writes: None,
        });
        init_pass.set_pipeline(&self.gpu_shaders.init.compute_pipeline);
        init_pass.set_bind_group(0, &self.gpu_shaders.init.bind_group, &[]);
        init_pass.dispatch_workgroups(256, 1, 1);
    }

    fn resolve_time_query(&self, com_encoder: &mut wgpu::CommandEncoder, pass_count: u32) {
        let entries_to_resolve = pass_count * 2;
        com_encoder.resolve_query_set(
            &self.gpu_context.query_set,
            0..entries_to_resolve,
            &self.gpu_buffers.timestamp,
            0u64,
        );
        com_encoder.copy_buffer_to_buffer(
            &self.gpu_buffers.timestamp,
            0u64,
            &self.gpu_buffers.timestamp_readback,
            0u64,
            entries_to_resolve as u64 * std::mem::size_of::<u64>() as u64,
        );
    }

    fn time(&self, pass_count: usize) -> u64 {
        let query_slice = self.gpu_buffers.timestamp_readback.slice(..);
        query_slice.map_async(wgpu::MapMode::Read, |result| {
            result.unwrap();
        });
        self.gpu_context.device.poll(wgpu::Maintain::wait());
        let query_out = query_slice.get_mapped_range();
        let timestamp: Vec<u64> = bytemuck::cast_slice(&query_out).to_vec();
        let mut total_time = 0u64;
        for i in 0..pass_count {
            total_time += u64::wrapping_sub(timestamp[i * 2 + 1], timestamp[i * 2]);
        }
        total_time
    }

    fn readback_results(&self, readback_size: u32) {
        let mut copy_command =
            self.gpu_context
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("Copy Command Encoder"),
                });
        copy_command.copy_buffer_to_buffer(
            &self.gpu_buffers.scan_out,
            0u64,
            &self.gpu_buffers.readback,
            0u64,
            readback_size as u64 * std::mem::size_of::<u32>() as u64,
        );
        self.gpu_context.queue.submit(Some(copy_command.finish()));
        let readback_slice = self
            .gpu_buffers
            .readback
            .slice(0..((readback_size as usize * std::mem::size_of::<u32>()) as u64));
        readback_slice.map_async(wgpu::MapMode::Read, |result| {
            result.unwrap();
        });
        self.gpu_context.device.poll(wgpu::Maintain::wait());
        let data = readback_slice.get_mapped_range();
        let data_out: Vec<u32> = bytemuck::cast_slice(&data).to_vec();
        println!("{:?}", data_out);
        // for i in 0..readback_size{
        //     println!("{} {}", i, data_out[i as usize]);
        // }
    }

    async fn run(
        &self,
        should_readback: bool,
        should_time: bool,
        should_validate: bool,
        readback_size: u32,
        batch_size: u32,
        pass: Box<dyn PassLogic>,
    ) {
        let mut total_spins: u32 = 0;
        let mut fallbacks_initiated: u32 = 0;
        let mut successful_insertions: u32 = 0;

        let mut tests_passed: u32 = 0;
        let mut total_time: u64 = 0;
        for i in 0..batch_size {
            let mut command =
                self.gpu_context
                    .device
                    .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                        label: Some("Command Encoder"),
                    });

            self.init_pass(&mut command);
            pass.main_pass(self, &mut command);
            if should_time {
                self.resolve_time_query(&mut command, pass.pass_count())
            }
            self.gpu_context.queue.submit(Some(command.finish()));

            //The first test is always discarded to prep caches and TLB
            if should_time && i != 0u32 {
                total_time += self.time(pass.pass_count() as usize);
                self.gpu_buffers.timestamp_readback.unmap();
            }

            if should_validate {
                let test_passed = pass.validation_pass(self);
                if test_passed {
                    tests_passed += 1u32;
                }
            }

            let mut stats_out: Vec<u32> = vec![];
            pass.stats(self, &mut stats_out);
            if !stats_out.is_empty() {
                total_spins += stats_out[0];
                fallbacks_initiated += stats_out[1];
                successful_insertions += stats_out[2];
            }
        }

        if should_readback {
            self.readback_results(readback_size);
            self.gpu_buffers.readback.unmap();
        }

        if should_time {
            let mut f_time = total_time as f64;
            f_time /= 1000000000.0f64;
            println!("\nTotal time elapsed: {}", f_time);
            let speed = ((self.size as u64) * ((batch_size - 1) as u64)) as f64
                / (f_time * self.gpu_context.timestamp_freq as f64);
            println!("Estimated speed {:e} ele/s", speed);
        }

        if should_validate {
            if tests_passed == batch_size {
                println!("ALL TESTS PASSED: {} / {}", tests_passed, batch_size);
            } else {
                println!("TESTS FAILED: {} / {}", tests_passed, batch_size);
            }
        }

        let mut occ_out: Vec<u32> = vec![];
        pass.occupancy(self, &mut occ_out);
        if !occ_out.is_empty() {
            println!("\nEstimated Occupancy across GPU: {}", occ_out[0]);
        }

        //dumb hack
        if total_spins != 0u32 {
            let avg_total_spins = total_spins as f64 / batch_size as f64;
            let avg_fallback_init = fallbacks_initiated as f64 / batch_size as f64;
            let avg_success_insert = successful_insertions as f64 / batch_size as f64;
            println!("\nThread Blocks Launched: {}", self.thread_blocks);
            println!("Average Total Spins per Pass: {}", avg_total_spins);
            println!(
                "Average Fallbacks Initiated per Pass: {}",
                avg_fallback_init
            );
            println!(
                "Average Successful Fallback Insertions per Pass: {}",
                avg_success_insert
            );
        }
    }

    pub async fn run_test(
        &self,
        should_readback: bool,
        should_time: bool,
        should_validate: bool,
        readback_size: u32,
        args: Vec<String>,
    ) {
        let scan_type = match args[1].as_str() {
            "rts" => Some(ScanType::Rts),
            "csdl" => Some(ScanType::Csdl),
            "csdldf" => Some(ScanType::Csdldf),
            "csdldf_stats" => Some(ScanType::CsdldfStats),
            "csdldf_occ" => Some(ScanType::CsdldfOcc),
            "csdldf_struct" => Some(ScanType::CsdldfStruct),
            "csdldf_struct_stats" => Some(ScanType::CsdldfStructStats),
            "csdldf_struct_occ" => Some(ScanType::CsdldfStructOcc),
            _ => None,
        };

        let scan_type = match scan_type {
            Some(scan_type) => scan_type,
            None => {
                eprintln!("Error: Unknown scan type {}", &args[1]);
                return;
            }
        };

        let pass: Box<dyn PassLogic> = match scan_type {
            ScanType::Rts => Box::new(RtsPass),
            ScanType::Csdl => Box::new(CsdlPass),
            ScanType::Csdldf => Box::new(CsdldfPass),
            ScanType::CsdldfStats => Box::new(CsdldfStatsPass),
            ScanType::CsdldfOcc => Box::new(CsdldfOccPass),
            ScanType::CsdldfStruct => Box::new(CsdldfStructPass),
            ScanType::CsdldfStructStats => Box::new(CsdldfStructStatsPass),
            ScanType::CsdldfStructOcc => Box::new(CsdldfStructOccPass),
        };

        let batch_size: u32 = match args[3].parse() {
            Ok(num) => num,
            Err(_) => {
                eprintln!("Error: Batch Size must be a positive integer");
                std::process::exit(1);
            }
        };

        self.run(
            should_readback,
            should_time,
            should_validate,
            readback_size,
            batch_size,
            pass,
        )
        .await;
    }
}

//warning, absolutely no guard rails
pub async fn run_the_runner(args: Vec<String>) {
    let pow_of_two: u32 = match args[2].parse() {
        Ok(num) if (num < 26) => num,
        Ok(_) => {
            eprintln!("Error: input size power must be a value between 0 and 25");
            std::process::exit(1);
        }
        Err(_) => {
            eprintln!("Error: input size power must be a positive integer");
            std::process::exit(1);
        }
    };

    let size: u32 = 1 << pow_of_two; //Input size to test, must be a multiple of 4
    let part_size: u32 = 4096; //MUST match partition size described in shaders
    let thread_blocks =                //Thread Blocks to launch based on input
        div_round_up(size, part_size);
    let max_pass_count: usize = 3; //Max number of passes to track with our query set
    let max_readback_size: usize = 8192; //Max size of our readback buffer
    let misc_size: usize = 4; //Max scratch memory we use to track various stats
    let tester = Tester::init(
        size,
        thread_blocks,
        max_pass_count,
        max_readback_size,
        misc_size,
    )
    .await;

    let should_validate = true; //Perform validation?
    let should_readback = false; //Use readback to sanity check results
    let should_time = true; //Time results?
    let readback_size = 256; //How many elements to readback, must be less than max
    tester
        .run_test(
            should_readback,
            should_time,
            should_validate,
            readback_size,
            args,
        )
        .await;
}

fn main() {
    let args: Vec<String> = env::args().collect();
    if args.len() != 4 {
        eprintln!(
            "Usage: <Scan Type: String>
            <Input Size as Power of Two: u32> <Test Batch Size: u32>"
        );
        std::process::exit(1);
    }
    pollster::block_on(run_the_runner(args));
}
