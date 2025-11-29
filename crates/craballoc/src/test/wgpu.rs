//! `wgpu` linkage for the test module.

use crate::{
    runtime::WgpuRuntime,
    test::{BackendUpdate, GpuUpdateTest},
};

pub const ENTRY_POINT: &str = "apply_data_changes";

fn shader() -> wgpu::ShaderModuleDescriptor<'static> {
    wgpu::include_spirv!("../test/shaders/apply_data_changes.spv")
}

pub struct TestBackendWgpu {
    bindgroup_layout: wgpu::BindGroupLayout,
    pipeline: wgpu::ComputePipeline,
}

impl TestBackendWgpu {
    pub fn new(runtime: WgpuRuntime) -> Self {
        let bindgroup_layout =
            runtime
                .device
                .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    label: Some("test"),
                    entries: &[
                        // data_slab
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
                        // changes slab
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
                    ],
                });
        let pipeline_layout =
            runtime
                .device
                .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                    label: Some("test"),
                    bind_group_layouts: &[&bindgroup_layout],
                    push_constant_ranges: &[],
                });
        let module = runtime.device.create_shader_module(shader());
        let pipeline = runtime
            .device
            .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("test"),
                layout: Some(&pipeline_layout),
                module: &module,
                entry_point: Some(ENTRY_POINT),
                compilation_options: Default::default(),
                cache: None,
            });

        Self {
            bindgroup_layout,
            pipeline,
        }
    }
}

impl BackendUpdate for GpuUpdateTest<WgpuRuntime, TestBackendWgpu> {
    fn apply_backend_changes(&mut self) {
        let runtime = self.arena.runtime();
        let bindgroup = runtime
            .device
            .create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("test"),
                layout: &self.backend_updater.bindgroup_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: self.arena.get_buffer().unwrap().as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: self.changes_arena.get_buffer().unwrap().as_entire_binding(),
                    },
                ],
            });

        let mut encoder = runtime
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("test"),
            });
        {
            let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("test"),
                timestamp_writes: None,
            });
            compute_pass.set_pipeline(&self.backend_updater.pipeline);
            compute_pass.set_bind_group(0, &bindgroup, &[]);
            let wg = self.invocation.get().workgroup_dimensions();
            log::info!("dispatching {wg:?} workgroups");
            compute_pass.dispatch_workgroups(wg.x, wg.y, wg.z);
        }
        let submission = runtime.queue.submit(Some(encoder.finish()));
        runtime
            .device
            .poll(wgpu::PollType::WaitForSubmissionIndex(submission))
            .unwrap();
    }
}
