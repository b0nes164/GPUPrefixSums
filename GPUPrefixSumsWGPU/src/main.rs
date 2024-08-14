/******************************************************************************
 * GPUPrefixSums
 *
 * SPDX-License-Identifier: MIT
 * Copyright Thomas Smith 8/4/2024
 * https://github.com/b0nes164/GPUPrefixSums
 *
 ******************************************************************************/
use wgpu::util::DeviceExt;
use std::env;

const PART_SIZE: u32 = 3328;    //256 * 13

fn div_round_up(x: u32, y: u32) -> u32 {
    return (x + y - 1) / y;
}

enum ScanType{
    Rts,
    Csdl,
    Csdldf,
    Memcpy,
}

struct GPUContext{
    device: wgpu::Device,
    queue: wgpu::Queue,
    query_set: wgpu::QuerySet,
    timestamp_freq: f32,
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
                    required_features: wgpu::Features::TIMESTAMP_QUERY,
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

        let timestamp_size = std::mem::size_of::<u64>() as wgpu::BufferAddress * 6;
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
    reduce: ComputeShader,
    dev_scan: ComputeShader,
    downsweep: ComputeShader,
    csdl: ComputeShader,
    csdldf: ComputeShader,
    memcpy: ComputeShader,
    validate: ComputeShader,
}

impl Shaders{
    fn init(gpu: &GPUContext, gpu_buffers: &GPUBuffers) -> Self{
        
        let init_module = gpu.device.create_shader_module(wgpu::include_wgsl!("Shaders/init.wgsl"));
        let rts_module: wgpu::ShaderModule;
        let csdl_module: wgpu::ShaderModule;
        let csdldf_module: wgpu::ShaderModule;
        let memcpy_module: wgpu::ShaderModule;
        unsafe{
            rts_module = gpu.device.create_shader_module_unchecked(wgpu::include_wgsl!("Shaders/rts.wgsl"));
            csdl_module = gpu.device.create_shader_module_unchecked(wgpu::include_wgsl!("Shaders/csdl.wgsl"));
            csdldf_module = gpu.device.create_shader_module_unchecked(wgpu::include_wgsl!("Shaders/csdldf.wgsl"));
            memcpy_module = gpu.device.create_shader_module_unchecked(wgpu::include_wgsl!("Shaders/memcpy.wgsl"));
        }
        let valid_module = gpu.device.create_shader_module(wgpu::include_wgsl!("Shaders/validate.wgsl"));

        let init = ComputeShader::init_main("main", gpu, gpu_buffers, &init_module);
        let reduce = ComputeShader::init_main("reduce", gpu, gpu_buffers, &rts_module);
        let dev_scan = ComputeShader::init_main("device_scan", gpu, gpu_buffers, &rts_module);
        let downsweep = ComputeShader::init_main("downsweep", gpu, gpu_buffers, &rts_module);
        let csdl = ComputeShader::init_main("main", gpu, gpu_buffers, &csdl_module);
        let csdldf = ComputeShader::init_main("main", gpu, gpu_buffers, &csdldf_module);
        let memcpy = ComputeShader::init_main("main", gpu, gpu_buffers, &memcpy_module);
        let validate = ComputeShader::init_valid(gpu, gpu_buffers, &valid_module);

        Shaders{
            init,
            reduce,
            dev_scan,
            downsweep,
            csdl,
            csdldf,
            memcpy,
            validate,
        }
    }
}

struct Tester{
    gpu_context: GPUContext,
    gpu_buffers: GPUBuffers,
    gpu_shaders: Shaders,
    size: u32,
    partitions: u32,
}

impl Tester{
    async fn init(size : u32) -> Self{
        let gpu_context = GPUContext::init().await;
        let gpu_buffers = GPUBuffers::init(&gpu_context, size as usize);
        let gpu_shaders = Shaders::init(&gpu_context, &gpu_buffers);
        let partitions = div_round_up(size, PART_SIZE);
        Tester{
            gpu_context,
            gpu_buffers,
            gpu_shaders,
            size,
            partitions,
        }
    }

    fn set_init_pass(&self, com_encoder: &mut wgpu::CommandEncoder){
        let mut init_pass = com_encoder.begin_compute_pass(&wgpu::ComputePassDescriptor{
            label: Some("Init Pass"),
            timestamp_writes: None,
        });
        init_pass.set_pipeline(&self.gpu_shaders.init.compute_pipeline);
        init_pass.set_bind_group(0, &self.gpu_shaders.init.bind_group, &[]);
        init_pass.dispatch_workgroups(256, 1, 1);
    }

    fn set_validate_pass(&self, com_encoder: &mut wgpu::CommandEncoder){
        let mut valid_pass = com_encoder.begin_compute_pass(&wgpu::ComputePassDescriptor{
            label: Some("Validate Pass"),
            timestamp_writes: None,
        });
        valid_pass.set_pipeline(&self.gpu_shaders.validate.compute_pipeline);
        valid_pass.set_bind_group(0, &self.gpu_shaders.validate.bind_group, &[]);
        valid_pass.dispatch_workgroups(256, 1, 1);
    }

    fn set_rts_passes(&self, com_encoder: &mut wgpu::CommandEncoder){
        {
            let mut red_pass = com_encoder.begin_compute_pass(&wgpu::ComputePassDescriptor{
                label: Some("Reduce Pass"),
                timestamp_writes: Some(wgpu::ComputePassTimestampWrites {
                    query_set: &self.gpu_context.query_set,
                    beginning_of_pass_write_index: Some(0u32),
                    end_of_pass_write_index: Some(1u32) }),
            });
            red_pass.set_pipeline(&self.gpu_shaders.reduce.compute_pipeline);
            red_pass.set_bind_group(0, &self.gpu_shaders.reduce.bind_group, &[]);
            red_pass.dispatch_workgroups(self.partitions, 1, 1);
        }
    
    
        {
            let mut dev_scan_pass = com_encoder.begin_compute_pass(&wgpu::ComputePassDescriptor{
                label: Some("Device Scan Pass"),
                timestamp_writes: Some(wgpu::ComputePassTimestampWrites {
                    query_set: &self.gpu_context.query_set,
                    beginning_of_pass_write_index: Some(2u32),
                    end_of_pass_write_index: Some(3u32) }),
            });
            dev_scan_pass.set_pipeline(&self.gpu_shaders.dev_scan.compute_pipeline);
            dev_scan_pass.set_bind_group(0, &self.gpu_shaders.dev_scan.bind_group, &[]);
            dev_scan_pass.dispatch_workgroups(1, 1, 1);
        }
    
        {
            let mut downsweep_pass = com_encoder.begin_compute_pass(&wgpu::ComputePassDescriptor{
                label: Some("Downsweep Pass"),
                timestamp_writes: Some(wgpu::ComputePassTimestampWrites {
                    query_set: &self.gpu_context.query_set,
                    beginning_of_pass_write_index: Some(4u32),
                    end_of_pass_write_index: Some(5u32) }),
            });
            downsweep_pass.set_pipeline(&self.gpu_shaders.downsweep.compute_pipeline);
            downsweep_pass.set_bind_group(0, &self.gpu_shaders.downsweep.bind_group, &[]);
            downsweep_pass.dispatch_workgroups(self.partitions, 1, 1);
        }
    }

    fn set_csdl_pass(&self, com_encoder: &mut wgpu::CommandEncoder){
        let mut csdl_pass = com_encoder.begin_compute_pass(&wgpu::ComputePassDescriptor{
            label: Some("CSDL Pass"),
            timestamp_writes: Some(wgpu::ComputePassTimestampWrites {
                query_set: &self.gpu_context.query_set,
                beginning_of_pass_write_index: Some(0u32),
                end_of_pass_write_index: Some(1u32) }),
        });
        csdl_pass.set_pipeline(&self.gpu_shaders.csdl.compute_pipeline);
        csdl_pass.set_bind_group(0, &self.gpu_shaders.csdl.bind_group, &[]);
        csdl_pass.dispatch_workgroups(self.partitions, 1, 1);
    }

    fn set_csdldf_pass(&self, com_encoder: &mut wgpu::CommandEncoder){
        let mut csdldf_pass = com_encoder.begin_compute_pass(&wgpu::ComputePassDescriptor{
            label: Some("CSDLDF Pass"),
            timestamp_writes: Some(wgpu::ComputePassTimestampWrites {
                query_set: &self.gpu_context.query_set,
                beginning_of_pass_write_index: Some(0u32),
                end_of_pass_write_index: Some(1u32) }),
        });
        csdldf_pass.set_pipeline(&self.gpu_shaders.csdldf.compute_pipeline);
        csdldf_pass.set_bind_group(0, &self.gpu_shaders.csdldf.bind_group, &[]);
        csdldf_pass.dispatch_workgroups(self.partitions, 1, 1);
    }

    fn set_memcpy_pass(&self, com_encoder: &mut wgpu::CommandEncoder){
        let mut memcpy_pass = com_encoder.begin_compute_pass(&wgpu::ComputePassDescriptor{
            label: Some("Memcpy Pass"),
            timestamp_writes: Some(wgpu::ComputePassTimestampWrites {
                query_set: &self.gpu_context.query_set,
                beginning_of_pass_write_index: Some(0u32),
                end_of_pass_write_index: Some(1u32) }),
        });
        memcpy_pass.set_pipeline(&self.gpu_shaders.memcpy.compute_pipeline);
        memcpy_pass.set_bind_group(0, &self.gpu_shaders.memcpy.bind_group, &[]);
        memcpy_pass.dispatch_workgroups(self.size / PART_SIZE, 1, 1);
    }

    fn resolve_time_query(&self, com_encoder: &mut wgpu::CommandEncoder, pass_count: u32){
        let entries_to_resolve = pass_count * 2;
        com_encoder.resolve_query_set(
            &self.gpu_context.query_set,
            0..entries_to_resolve,
            &self.gpu_buffers.timestamp,
            0u64);
        com_encoder.copy_buffer_to_buffer(
            &self.gpu_buffers.timestamp, 
            0u64, 
            &self.gpu_buffers.readback_timestamp, 
            0u64,
            entries_to_resolve as u64 * std::mem::size_of::<u64>() as wgpu::BufferAddress);
    }

    async fn time(&self, pass_count: usize) -> u64 {
        let query_slice = self.gpu_buffers.readback_timestamp.slice(..);
        query_slice.map_async(wgpu::MapMode::Read, |result| {
            result.unwrap();
        });
        self.gpu_context.device.poll(wgpu::Maintain::wait());
        let query_out = query_slice.get_mapped_range();
        let timestamp: Vec<u64> = bytemuck::cast_slice(&query_out).to_vec();
        let mut total_time = 0u64;
        for i in 0..pass_count{
            total_time += u64::wrapping_sub(timestamp[i * 2 + 1], timestamp[i * 2]);
        }
        return total_time;
    }

    async fn validate(&self) -> bool {
        let mut valid_command =  self.gpu_context.device.create_command_encoder(&wgpu::CommandEncoderDescriptor{
            label: Some("Valid Command Encoder"),
        });
        let zero = vec![0u8; 4 as usize];
        self.gpu_context.queue.write_buffer(&self.gpu_buffers.error, 0, &zero);
        self.set_validate_pass(&mut valid_command);
        valid_command.copy_buffer_to_buffer(
            &self.gpu_buffers.error,
            0u64,
            &self.gpu_buffers.readback,
            0u64,
            std::mem::size_of::<u32>() as wgpu::BufferAddress);
        self.gpu_context.queue.submit(Some(valid_command.finish()));
        let readback_slice = self.gpu_buffers.readback.slice(0..4);
        readback_slice.map_async(wgpu::MapMode::Read, |result|{
            result.unwrap();
        });
        self.gpu_context.device.poll(wgpu::Maintain::wait());
        let data = readback_slice.get_mapped_range();
        let data_out: Vec<u32> = bytemuck::cast_slice(&data).to_vec();
        if data_out[0] != 0 {
            println!("Err count {}", data_out[0]);
        }
        return data_out[0] == 0;
    }

    async fn readback_results(&self, readback_size : u32){
        let mut copy_command =  self.gpu_context.device.create_command_encoder(&wgpu::CommandEncoderDescriptor{
            label: Some("Copy Command Encoder"),
        });
        copy_command.copy_buffer_to_buffer(
            &self.gpu_buffers.scan,
            0u64,
            &self.gpu_buffers.readback,
            0u64,
            readback_size as u64 * std::mem::size_of::<u32>() as wgpu::BufferAddress);
        self.gpu_context.queue.submit(Some(copy_command.finish()));
        let readback_slice = self.gpu_buffers.readback.slice(0..((readback_size as usize * std::mem::size_of::<u32>()) as u64));
        readback_slice.map_async(wgpu::MapMode::Read, |result|{
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
        should_readback : bool,
        should_time : bool,
        should_validate : bool,
        readback_size : u32,
        batch_size : u32,
        passes : u32,
        set_main_pass: impl Fn(&Tester, &mut wgpu::CommandEncoder))
    {
        let mut tests_passed: u32 = 0;
        let mut total_time: u64 = 0;
        for i in 0 .. batch_size{
            let mut command = self.gpu_context.device.create_command_encoder(&wgpu::CommandEncoderDescriptor{
                label: Some("Command Encoder"),
            });
        
            self.set_init_pass(&mut command);
            set_main_pass(self, &mut command);
            if should_time {
                self.resolve_time_query(&mut command, passes)
            }
            self.gpu_context.queue.submit(Some(command.finish()));

            if should_time {
                total_time += self.time(passes as usize).await;
                self.gpu_buffers.readback_timestamp.unmap();
            }

            if should_validate {
                let test_passed = self.validate().await;
                self.gpu_buffers.readback.unmap();
                if test_passed{
                    tests_passed += 1u32;
                }
            }
        
            if should_readback {
                self.readback_results(readback_size).await;
                self.gpu_buffers.readback.unmap();
            }

            if (i & 15) == 0 {
                print!(".");
            }
        }

        if should_time{
            let mut f_time = total_time as f64;
            f_time /= 1000000000.0f64;
            println!("\nTotal time elapsed: {}", f_time);
            let speed = ((self.size as u64) * (batch_size as u64)) as f64 / (f_time * self.gpu_context.timestamp_freq as f64);
            println!("Estimated speed {:e} ele/s", speed);
        }

        if should_validate {
            if tests_passed == batch_size{
                println!("\nALL TESTS PASSED");
            } else {
                println!("TESTS FAILED: {} / {}", tests_passed, batch_size);
            }
        }
    }

    fn parse(arg: &str) -> Option<ScanType> {
        match arg {
            "rts" => Some(ScanType::Rts),
            "csdl" => Some(ScanType::Csdl),
            "csdldf" => Some(ScanType::Csdldf),
            "memcpy" => Some(ScanType::Memcpy),
            _ => None,
        }
    }

    pub async fn run_test(
        &self,
        should_readback : bool,
        should_time : bool,
        readback_size : u32,
        batch_size : u32,
        args : Vec<String>)
    {
        let scan_type = Tester::parse(&args[1]);
        match scan_type{
            Some(ScanType::Rts) => self.run(should_readback, should_time, true, readback_size, batch_size, 3, Self::set_rts_passes).await,
            Some(ScanType::Csdl) => self.run(should_readback, should_time, true, readback_size, batch_size, 1, Self::set_csdl_pass).await,
            Some(ScanType::Csdldf) => self.run(should_readback, should_time, true, readback_size, batch_size, 1, Self::set_csdldf_pass).await,
            Some(ScanType::Memcpy) => self.run(false, should_time, false, 0u32, batch_size, 1, Self::set_memcpy_pass).await,
            None => println!("Err, arg not found"),
        };
    }
}

pub async fn run_the_runner(args : Vec<String>)
{
    let tester = Tester::init(1 << 25).await;
    let should_readback = false;
    let should_time = true;
    let readback_size = 1024;
    let batch_size = 1000;
    tester.run_test(should_readback, should_time, readback_size, batch_size, args).await;
}

fn main() {
    let args: Vec<String> = env::args().collect();
    if args.len() != 2 {
        println!("Err, invalid number of args.");
        return;
    }
    pollster::block_on(run_the_runner(args));
    println!("OK!");
}