//! `wgpu` linkage for the test module.

use crate::{
    runtime::{IsRuntime, WgpuRuntime},
    test::{BackendUpdate, GpuUpdateTest},
};
use craballoc_test_shaders::apply_data_changes_shader;

pub const ENTRY_POINT: &str = "apply_data_changes";

/// The counters buffer slots: successful invocations, skipped ones.
const COUNTERS_LEN: u32 = 2;

fn shader_source() -> String {
    apply_data_changes_shader::WGSL_SOURCE
        .wgsl_source()
        .expect("failed to assemble the apply_data_changes shader source")
}

pub struct TestBackendWgpu {
    bindgroup_layout: wgpu::BindGroupLayout,
    pipeline: wgpu::ComputePipeline,
    /// The shader counts invocations in this `array<atomic<u32>>`
    /// buffer (WGSL forbids atomics on the plain-`u32` data slab).
    counters: wgpu::Buffer,
    invocations_ran: u32,
    invocations_skipped: u32,
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
                        // invocation counters
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
        let pipeline_layout =
            runtime
                .device
                .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                    label: Some("test"),
                    bind_group_layouts: &[Some(&bindgroup_layout)],
                    immediate_size: 0,
                });
        let module = runtime
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("apply_data_changes"),
                source: wgpu::ShaderSource::Wgsl(shader_source().into()),
            });
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
        let counters = runtime.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("test counters"),
            size: (COUNTERS_LEN * 4) as u64,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        Self {
            bindgroup_layout,
            pipeline,
            counters,
            invocations_ran: 0,
            invocations_skipped: 0,
        }
    }
}

impl BackendUpdate for GpuUpdateTest<WgpuRuntime, TestBackendWgpu> {
    fn apply_backend_changes(&mut self) {
        let runtime = self.arena.runtime();
        // Zero the counters before the dispatch.
        runtime
            .queue
            .write_buffer(&self.backend_updater.counters, 0, &[0; 8]);

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
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: self.backend_updater.counters.as_entire_binding(),
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
            // The shader reads its index from `global_id.x` alone, so a
            // flat dispatch over the invocation count suffices; extra
            // invocations land in the skipped counter.
            let total = self.invocation.get().total_invocations_required();
            let workgroups = total.div_ceil(16);
            log::info!("dispatching {workgroups} workgroups for {total} invocations");
            compute_pass.dispatch_workgroups(workgroups, 1, 1);
        }
        let _submission = runtime.queue.submit(Some(encoder.finish()));
        runtime
            .device
            .poll(wgpu::PollType::wait_indefinitely())
            .unwrap();

        // Read back the counters so `invocations_ran` can verify the
        // dispatch.
        let counters = futures_lite::future::block_on(runtime.buffer_read(
            &self.backend_updater.counters,
            COUNTERS_LEN as usize,
            0..COUNTERS_LEN as usize,
        ))
        .unwrap();
        self.backend_updater.invocations_ran = counters[0];
        self.backend_updater.invocations_skipped = counters[1];
    }

    fn invocations_ran(&mut self) -> u32 {
        self.backend_updater.invocations_ran
    }

    fn invocations_skipped(&mut self) -> u32 {
        self.backend_updater.invocations_skipped
    }
}
