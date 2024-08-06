/******************************************************************************
 * GPUPrefixSums
 *
 * SPDX-License-Identifier: MIT
 * Copyright Thomas Smith 8/4/2024
 * https://github.com/b0nes164/GPUPrefixSums
 *
 ******************************************************************************/
use wgpu::util::DeviceExt;

const PART_SIZE: u32 = 3328;    //256 * 13

fn div_round_up(x: u32, y: u32) -> u32 {
    return (x + y - 1) / y;
}

struct GPUContext{
    device: wgpu::Device,
    queue: wgpu::Queue,
    query_set: wgpu::QuerySet,
}

impl GPUContext{
    async fn init() -> Self{
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor{
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
                    required_features: wgpu::Features::TIMESTAMP_QUERY | wgpu::Features::TIMESTAMP_QUERY_INSIDE_PASSES,
                    required_limits: wgpu::Limits::default(),
                    memory_hints: wgpu::MemoryHints::Performance,
                },
                None,
            )
            .await
            .expect("Failed to create device");

        let query_set = device.create_query_set(&wgpu::QuerySetDescriptor {
            label: Some("Timestamp Query Set"),
            count: 2,
            ty: wgpu::QueryType::Timestamp,
        });

        GPUContext {
            device,
            queue,
            query_set,
        }
    }
}

struct GPUBuffers{
    info: wgpu::Buffer,
    readback: wgpu::Buffer,
    scan: wgpu::Buffer,
    reduction: wgpu::Buffer,
    index: wgpu::Buffer,
    timestamp: wgpu::Buffer,
    readback_timestamp: wgpu::Buffer,
    error: wgpu::Buffer,
}

impl GPUBuffers{
    fn init(gpu: &GPUContext, size: usize) -> Self{

        //no push constants...
        let thread_blocks = div_round_up(size as u32, PART_SIZE);
        let info_info: Vec<u32> = vec![size as u32, thread_blocks as u32]; 
        let info = gpu.device.create_buffer_init(&wgpu::util::BufferInitDescriptor{
            label: Some("Info"),
            contents: bytemuck::cast_slice(&info_info),
            usage: wgpu::BufferUsages::STORAGE,
        });

        let buffer_size = (size * std::mem::size_of::<u32>()) as wgpu::BufferAddress;
        let readback = gpu.device.create_buffer(&wgpu::BufferDescriptor{
            label: Some("Readback"),
            size: buffer_size,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let scan = gpu.device.create_buffer(&wgpu::BufferDescriptor{
            label: Some("Scan"),
            size: buffer_size,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let reduction = gpu.device.create_buffer(&wgpu::BufferDescriptor{
            label: Some("Reduction"),
            size: (thread_blocks as usize * std::mem::size_of::<u32>()) as wgpu::BufferAddress,
            usage: wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        });

        let index = gpu.device.create_buffer(&wgpu::BufferDescriptor{
            label: Some("Index"),
            size: std::mem::size_of::<u32>() as wgpu::BufferAddress,
            usage: wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        });

        let timestamp_size = std::mem::size_of::<u64>() as wgpu::BufferAddress * 2;
        let timestamp = gpu.device.create_buffer(&wgpu::BufferDescriptor{
            label: Some("Timestamp"),
            size: timestamp_size,
            usage: wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::QUERY_RESOLVE,
            mapped_at_creation: false,
        });

        let readback_timestamp = gpu.device.create_buffer(&wgpu::BufferDescriptor{
            label: Some("Readback Timestamp"),
            size: timestamp_size,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let error = gpu.device.create_buffer(&wgpu::BufferDescriptor{
            label: Some("Error"),
            size: std::mem::size_of::<u32>() as wgpu::BufferAddress,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        GPUBuffers{
            info,
            readback,
            scan,
            reduction,
            index,
            timestamp,
            readback_timestamp,
            error,
        }
    }
}

//For simplicity we are going to use the bind group and layout 
//for all of the kernels except the validation
struct ComputeShader{
    bind_group: wgpu::BindGroup,
    compute_pipeline: wgpu::ComputePipeline,
}

impl  ComputeShader{
    fn init_main(entry_point: &str, gpu: &GPUContext, gpu_buffers: &GPUBuffers, module: &wgpu::ShaderModule) -> Self{
        let bind_group_layout = gpu.device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Bind Group Layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
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
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

        let bind_group =  gpu.device.create_bind_group(&wgpu::BindGroupDescriptor{
            label: Some("Bind Group"),
            layout: &bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry{
                    binding: 0,
                    resource: gpu_buffers.scan.as_entire_binding(),
                },
                wgpu::BindGroupEntry{
                    binding: 1,
                    resource: gpu_buffers.reduction.as_entire_binding(),
                },
                wgpu::BindGroupEntry{
                    binding: 2,
                    resource: gpu_buffers.index.as_entire_binding(),
                },
                wgpu::BindGroupEntry{
                    binding: 3,
                    resource: gpu_buffers.info.as_entire_binding(),
                },
            ],
        });

        let pipeline_layout_init = gpu.device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor{
            label: Some("Pipeline Layout"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });

        let compute_pipeline = gpu.device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor{
            label: Some("Compute Pipeline"),
            layout: Some(&pipeline_layout_init),
            module: &module,
            entry_point: entry_point,
            compilation_options: Default::default(),
            cache: Default::default(),
        });

        ComputeShader{
            bind_group,
            compute_pipeline,
        }
    }

    fn init_valid(gpu: &GPUContext, gpu_buffers: &GPUBuffers, module: &wgpu::ShaderModule) -> Self{
        let bind_group_layout = gpu.device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Bind Group Layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
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
            ],
        });

        let bind_group =  gpu.device.create_bind_group(&wgpu::BindGroupDescriptor{
            label: Some("Bind Group"),
            layout: &bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry{
                    binding: 0,
                    resource: gpu_buffers.scan.as_entire_binding(),
                },
                wgpu::BindGroupEntry{
                    binding: 1,
                    resource: gpu_buffers.info.as_entire_binding(),
                },
                wgpu::BindGroupEntry{
                    binding: 2,
                    resource: gpu_buffers.error.as_entire_binding(),
                },
            ],
        });

        let pipeline_layout_init = gpu.device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor{
            label: Some("Pipeline Layout"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });

        let compute_pipeline = gpu.device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor{
            label: Some("Compute Pipeline"),
            layout: Some(&pipeline_layout_init),
            module: &module,
            entry_point: "main",
            compilation_options: Default::default(),
            cache: Default::default(),
        });

        ComputeShader{
            bind_group,
            compute_pipeline,
        }
    }
}

struct Shaders{
    init: ComputeShader,
    _reduce: ComputeShader,
    _dev_scan: ComputeShader,
    _downsweep: ComputeShader,
    _csdl: ComputeShader,
    _csdldf: ComputeShader,
    validate: ComputeShader,
}

impl Shaders{
    fn init(gpu: &GPUContext, gpu_buffers: &GPUBuffers) -> Self{
        
        let init_module = gpu.device.create_shader_module(wgpu::include_wgsl!("Shaders/init.wgsl"));
        let rts_module = gpu.device.create_shader_module(wgpu::include_wgsl!("Shaders/rts.wgsl"));
        let csdl_module = gpu.device.create_shader_module(wgpu::include_wgsl!("Shaders/csdl.wgsl"));
        let csdldf_module = gpu.device.create_shader_module(wgpu::include_wgsl!("Shaders/csdldf.wgsl"));
        let valid_module = gpu.device.create_shader_module(wgpu::include_wgsl!("Shaders/validate.wgsl"));

        let init = ComputeShader::init_main("main", gpu, gpu_buffers, &init_module);
        let _reduce = ComputeShader::init_main("reduce", gpu, gpu_buffers, &rts_module);
        let _dev_scan = ComputeShader::init_main("device_scan", gpu, gpu_buffers, &rts_module);
        let _downsweep = ComputeShader::init_main("downsweep", gpu, gpu_buffers, &rts_module);
        let _csdl = ComputeShader::init_main("main", gpu, gpu_buffers, &csdl_module);
        let _csdldf = ComputeShader::init_main("main", gpu, gpu_buffers, &csdldf_module);
        let validate = ComputeShader::init_valid(gpu, gpu_buffers, &valid_module);

        Shaders{
            init,
            _reduce,
            _dev_scan,
            _downsweep,
            _csdl,
            _csdldf,
            validate,
        }
    }
}

//TODO rename
fn init_buffers(com_encoder: &mut wgpu::CommandEncoder, gpu_shaders: &Shaders){
    let mut init_pass = com_encoder.begin_compute_pass(&wgpu::ComputePassDescriptor{
        label: Some("Init Pass"),
        timestamp_writes: None,
    });
    init_pass.set_pipeline(&gpu_shaders.init.compute_pipeline);
    init_pass.set_bind_group(0, &gpu_shaders.init.bind_group, &[]);
    init_pass.dispatch_workgroups(256, 1, 1);
}

fn set_validate_pass(com_encoder: &mut wgpu::CommandEncoder, gpu_shaders: &Shaders){
    let mut valid_pass = com_encoder.begin_compute_pass(&wgpu::ComputePassDescriptor{
        label: Some("Validate Pass"),
        timestamp_writes: None,
    });
    valid_pass.set_pipeline(&gpu_shaders.validate.compute_pipeline);
    valid_pass.set_bind_group(0, &gpu_shaders.validate.bind_group, &[]);
    valid_pass.dispatch_workgroups(256, 1, 1);
}

fn set_rts_passes(com_encoder: &mut wgpu::CommandEncoder, gpu: &GPUContext, gpu_shaders: &Shaders, thread_blocks: u32){
    {
        let mut red_pass = com_encoder.begin_compute_pass(&wgpu::ComputePassDescriptor{
            label: Some("Reduce Pass"),
            timestamp_writes: Some(wgpu::ComputePassTimestampWrites {
                query_set: &gpu.query_set,
                beginning_of_pass_write_index: Some(0u32),
                end_of_pass_write_index: Some(1u32) }),
        });
        red_pass.set_pipeline(&gpu_shaders._reduce.compute_pipeline);
        red_pass.set_bind_group(0, &gpu_shaders._reduce.bind_group, &[]);
        red_pass.dispatch_workgroups(thread_blocks, 1, 1);
    }


    {
        let mut dev_scan_pass = com_encoder.begin_compute_pass(&wgpu::ComputePassDescriptor{
            label: Some("Device Scan Pass"),
            timestamp_writes: Some(wgpu::ComputePassTimestampWrites {
                query_set: &gpu.query_set,
                beginning_of_pass_write_index: Some(0u32),
                end_of_pass_write_index: Some(1u32) }),
        });
        dev_scan_pass.set_pipeline(&gpu_shaders._dev_scan.compute_pipeline);
        dev_scan_pass.set_bind_group(0, &gpu_shaders._dev_scan.bind_group, &[]);
        dev_scan_pass.dispatch_workgroups(1, 1, 1);
    }

    {
        let mut downsweep_pass = com_encoder.begin_compute_pass(&wgpu::ComputePassDescriptor{
            label: Some("Downsweep Pass"),
            timestamp_writes: Some(wgpu::ComputePassTimestampWrites {
                query_set: &gpu.query_set,
                beginning_of_pass_write_index: Some(0u32),
                end_of_pass_write_index: Some(1u32) }),
        });
        downsweep_pass.set_pipeline(&gpu_shaders._downsweep.compute_pipeline);
        downsweep_pass.set_bind_group(0, &gpu_shaders._downsweep.bind_group, &[]);
        downsweep_pass.dispatch_workgroups(thread_blocks, 1, 1);
    }
}

fn _set_csdl_pass(com_encoder: &mut wgpu::CommandEncoder, gpu: &GPUContext, gpu_shaders: &Shaders, thread_blocks: u32){
    let mut csdl_pass = com_encoder.begin_compute_pass(&wgpu::ComputePassDescriptor{
        label: Some("CSDL Pass"),
        timestamp_writes: Some(wgpu::ComputePassTimestampWrites {
            query_set: &gpu.query_set,
            beginning_of_pass_write_index: Some(0u32),
            end_of_pass_write_index: Some(1u32) }),
    });
    csdl_pass.set_pipeline(&gpu_shaders._csdl.compute_pipeline);
    csdl_pass.set_bind_group(0, &gpu_shaders._csdl.bind_group, &[]);
    csdl_pass.dispatch_workgroups(thread_blocks, 1, 1);
}

fn _set_csdldf_pass(com_encoder: &mut wgpu::CommandEncoder, gpu: &GPUContext, gpu_shaders: &Shaders, thread_blocks: u32){
    let mut csdldf_pass = com_encoder.begin_compute_pass(&wgpu::ComputePassDescriptor{
        label: Some("CSDLDF Pass"),
        timestamp_writes: Some(wgpu::ComputePassTimestampWrites {
            query_set: &gpu.query_set,
            beginning_of_pass_write_index: Some(0u32),
            end_of_pass_write_index: Some(1u32) }),
    });
    csdldf_pass.set_pipeline(&gpu_shaders._csdldf.compute_pipeline);
    csdldf_pass.set_bind_group(0, &gpu_shaders._csdldf.bind_group, &[]);
    csdldf_pass.dispatch_workgroups(thread_blocks, 1, 1);
}

async fn time(gpu: &GPUContext, gpu_buffers: &GPUBuffers) -> u64 {
    let query_slice = gpu_buffers.readback_timestamp.slice(..);
    query_slice.map_async(wgpu::MapMode::Read, |result| {
        result.unwrap();
    });
    gpu.device.poll(wgpu::Maintain::wait());
    let query_out = query_slice.get_mapped_range();
    let timestamp: Vec<u64> = bytemuck::cast_slice(&query_out).to_vec();
    return timestamp[1] - timestamp[0];
}

async fn validate(gpu: &GPUContext, gpu_buffers: &GPUBuffers, gpu_shaders: &Shaders) -> bool {
    let mut valid_command =  gpu.device.create_command_encoder(&wgpu::CommandEncoderDescriptor{
        label: Some("Valid Command Encoder"),
    });
    let zero = vec![0u8; 4 as usize];
    gpu.queue.write_buffer(&gpu_buffers.error, 0, &zero);
    set_validate_pass(&mut valid_command, &gpu_shaders);
    valid_command.copy_buffer_to_buffer(
        &gpu_buffers.error,
        0u64,
        &gpu_buffers.readback,
        0u64,
        std::mem::size_of::<u32>() as wgpu::BufferAddress);
    gpu.queue.submit(Some(valid_command.finish()));
    let readback_slice = gpu_buffers.readback.slice(0..4);
    readback_slice.map_async(wgpu::MapMode::Read, |result|{
        result.unwrap();
    });
    gpu.device.poll(wgpu::Maintain::wait());
    let data = readback_slice.get_mapped_range();
    let data_out: Vec<u32> = bytemuck::cast_slice(&data).to_vec();
    //println!("{}", data_out[0]);
    return data_out[0] == 0;
}

async fn readback_results(gpu: &GPUContext, gpu_buffers: &GPUBuffers, readback_size : u32){
    let mut copy_command =  gpu.device.create_command_encoder(&wgpu::CommandEncoderDescriptor{
        label: Some("Copy Command Encoder"),
    });
    
    copy_command.copy_buffer_to_buffer(
        &gpu_buffers.scan,
        0u64,
        &gpu_buffers.readback,
        0u64,
        readback_size as u64 * std::mem::size_of::<u32>() as wgpu::BufferAddress);
    gpu.queue.submit(Some(copy_command.finish()));
    let readback_slice = gpu_buffers.readback.slice(0..((readback_size as usize * std::mem::size_of::<u32>()) as u64));
    readback_slice.map_async(wgpu::MapMode::Read, |result|{
        result.unwrap();
    });
    gpu.device.poll(wgpu::Maintain::wait());
    let data = readback_slice.get_mapped_range();
    let data_out: Vec<u32> = bytemuck::cast_slice(&data).to_vec();
    println!("{:?}", data_out);
    // for i in 0..readback_size{
    //     println!("{} {}", i, data_out[i as usize]);
    // }
}

pub async fn run(should_readback : bool, should_time : bool, readback_size : u32, size : u32, batch_size : u32){
    let gpu_context = GPUContext::init().await;
    let gpu_buffers = GPUBuffers::init(&gpu_context, size as usize);
    let gpu_shaders = Shaders::init(&gpu_context, &gpu_buffers);
    
    let mut tests_passed: u32 = 0;
    let mut total_time: u64 = 0;
    for i in 0 .. batch_size{
        let mut command = gpu_context.device.create_command_encoder(&wgpu::CommandEncoderDescriptor{
            label: Some("Command Encoder"),
        });
    
        init_buffers(&mut command, &gpu_shaders);
        set_rts_passes(&mut command, &gpu_context, &gpu_shaders, div_round_up(size, PART_SIZE));
        //_set_csdl_pass(&mut command, &gpu_context, &gpu_shaders, div_round_up(size, PART_SIZE));
        //_set_csdldf_pass(&mut command, &gpu_context, &gpu_shaders, div_round_up(size, PART_SIZE));
        if should_time {
            command.resolve_query_set(&gpu_context.query_set, 0..2, &gpu_buffers.timestamp, 0u64);
            command.copy_buffer_to_buffer(
                &gpu_buffers.timestamp, 
                0u64, 
                &gpu_buffers.readback_timestamp, 
                0u64,
                2u64 * std::mem::size_of::<u64>() as wgpu::BufferAddress);
        }
        gpu_context.queue.submit(Some(command.finish()));
        
        if should_time {
            total_time += time(&gpu_context, &gpu_buffers).await;
            gpu_buffers.readback_timestamp.unmap();
        }

        let test_passed = validate(&gpu_context, &gpu_buffers, &gpu_shaders).await;
        gpu_buffers.readback.unmap();
        if test_passed{
            tests_passed += 1u32;
        }
    
        if should_readback {
            readback_results(&gpu_context, &gpu_buffers, readback_size).await;
            gpu_buffers.readback.unmap();
        }

        if (i & 15) == 0 {
            print!(".");
        }
    }

    if should_time{
        let mut f_time = total_time as f64;
        f_time /= 1000000000f64;
        println!("\nTotal time elapsed: {}", f_time);
        let speed = ((size as u64) * (batch_size as u64)) as f64 / f_time;
        println!("Estimated speed {:e} ele/s", speed);
    }

    if tests_passed == batch_size{
        println!("\nALL TESTS PASSED");
    } else {
        println!("TESTS FAILED: {} / {}", tests_passed, batch_size);
    }
}

fn main() {
    pollster::block_on(run(false, true, 1024, 1 << 25, 100));
    println!("OK!");
}
