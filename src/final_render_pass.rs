use std::sync::Arc;

use bytemuck::{Pod, Zeroable};
use egui_winit_vulkano::Gui;
use vulkano::{
    buffer::{BufferUsage, CpuAccessibleBuffer, CpuBufferPool, TypedBufferAccess},
    command_buffer::{
        AutoCommandBufferBuilder, CommandBufferInheritanceInfo, CommandBufferUsage,
        RenderPassBeginInfo, SubpassContents,
    },
    descriptor_set::{PersistentDescriptorSet, WriteDescriptorSet},
    device::{Device, Queue},
    format::Format,
    image::{ImageAccess, ImageViewAbstract},
    impl_vertex,
    pipeline::{
        graphics::{
            input_assembly::InputAssemblyState,
            vertex_input::BuffersDefinition,
            viewport::{Viewport, ViewportState},
        },
        GraphicsPipeline, Pipeline, PipelineBindPoint,
    },
    render_pass::{Framebuffer, FramebufferCreateInfo, RenderPass, Subpass},
    sampler::{Filter, Sampler, SamplerAddressMode, SamplerCreateInfo},
    sync::GpuFuture,
};
use vulkano_util::{
    context::VulkanoContext,
    renderer::{DeviceImageView, SwapchainImageView},
};

/// Simple read-only buffer type
type Buffer<T> = Arc<CpuAccessibleBuffer<[T]>>;

#[repr(C)]
#[derive(Clone, Copy, Default, Pod, Zeroable)]
struct QuadVertex {
    position: [f32; 2],
    tex_coords: [f32; 2],
}
impl_vertex!(QuadVertex, position, tex_coords);

pub type ViewportTransform = vs::ty::Uniforms;

pub struct FinalRenderPass {
    device: Arc<Device>,
    graphics_queue: Arc<Queue>,
    render_pass: Arc<RenderPass>,
    pipeline: Arc<GraphicsPipeline>,

    // Viewport quad
    vertex_buffer: Buffer<QuadVertex>,
    index_buffer: Buffer<u32>,
    uniform_buffer: CpuBufferPool<ViewportTransform>,
}

impl FinalRenderPass {
    pub fn new(context: &VulkanoContext, format: Format) -> Self {
        let render_pass = Self::create_render_pass(context.device(), format);
        let pipeline = Self::create_pipeline(context.device(), render_pass.clone());
        let (vertex_buffer, index_buffer) = Self::create_viewport_quad(context.device());
        let uniform_buffer =
            CpuBufferPool::<ViewportTransform>::new(context.device(), BufferUsage::all());

        Self {
            device: context.device(),
            graphics_queue: context.graphics_queue(),
            render_pass,
            pipeline,
            vertex_buffer,
            index_buffer,
            uniform_buffer,
        }
    }

    pub fn render<F>(
        &self,
        before_future: F,
        target: SwapchainImageView,
        gui: &mut Gui,
        viewport_view: DeviceImageView,
        viewport_bounds: Viewport,
        viewport_transform: ViewportTransform,
    ) -> Box<dyn GpuFuture>
    where
        F: GpuFuture + 'static,
    {
        // Get dimensions
        let image_dimensions = target.image().dimensions();

        // Create framebuffer (must be in same order as render pass description in `new`
        let framebuffer = Framebuffer::new(
            self.render_pass.clone(),
            FramebufferCreateInfo {
                attachments: vec![target],
                ..Default::default()
            },
        )
        .unwrap();

        // Create primary command buffer builder
        let mut primary_builder = AutoCommandBufferBuilder::primary(
            self.device.clone(),
            self.graphics_queue.family(),
            CommandBufferUsage::OneTimeSubmit,
        )
        .unwrap();

        // Begin render pass
        primary_builder
            .begin_render_pass(
                RenderPassBeginInfo {
                    clear_values: vec![Some([0.0, 0.0, 0.0, 1.0].into())],
                    ..RenderPassBeginInfo::framebuffer(framebuffer)
                },
                SubpassContents::SecondaryCommandBuffers,
            )
            .unwrap();

        // Create secondary command buffer from texture pipeline & send draw commands
        let mut secondary_builder = AutoCommandBufferBuilder::secondary(
            self.device.clone(),
            self.graphics_queue.family(),
            CommandBufferUsage::MultipleSubmit,
            CommandBufferInheritanceInfo {
                render_pass: Some(self.viewport_subpass().into()),
                ..Default::default()
            },
        )
        .unwrap();

        let descriptor_set = self.create_descriptor_set(viewport_view, viewport_transform);
        secondary_builder
            .bind_pipeline_graphics(self.pipeline.clone())
            .set_viewport(0, vec![viewport_bounds])
            .bind_vertex_buffers(0, self.vertex_buffer.clone())
            .bind_index_buffer(self.index_buffer.clone())
            .bind_descriptor_sets(
                PipelineBindPoint::Graphics,
                self.pipeline.layout().clone(),
                0,
                descriptor_set,
            )
            .draw_indexed(self.index_buffer.len() as u32, 1, 0, 0, 0)
            .unwrap();

        // Render viewport
        let cb = secondary_builder.build().unwrap();
        primary_builder.execute_commands(cb).unwrap();

        // Render gui
        primary_builder
            .next_subpass(SubpassContents::SecondaryCommandBuffers)
            .unwrap();

        let cb = gui.draw_on_subpass_image(image_dimensions.width_height());
        primary_builder.execute_commands(cb).unwrap();

        // End render pass
        primary_builder.end_render_pass().unwrap();

        // Build command buffer
        let command_buffer = primary_builder.build().unwrap();

        // Execute primary command buffer
        let after_future = before_future
            .then_execute(self.graphics_queue.clone(), command_buffer)
            .unwrap();

        after_future.boxed()
    }

    fn create_descriptor_set(
        &self,
        viewport_view: Arc<dyn ImageViewAbstract>,
        viewport_transform: ViewportTransform,
    ) -> Arc<PersistentDescriptorSet> {
        let sampler = Sampler::new(
            self.graphics_queue.device().clone(),
            SamplerCreateInfo {
                mag_filter: Filter::Linear,
                min_filter: Filter::Linear,
                address_mode: [SamplerAddressMode::ClampToBorder; 3],
                ..Default::default()
            },
        )
        .unwrap();

        let uniform_buffer_subbuffer = self.uniform_buffer.next(viewport_transform).unwrap();

        let layout = self.pipeline.layout().set_layouts().get(0).unwrap();
        PersistentDescriptorSet::new(
            layout.clone(),
            [
                WriteDescriptorSet::buffer(0, uniform_buffer_subbuffer),
                WriteDescriptorSet::image_view_sampler(1, viewport_view.clone(), sampler),
            ],
        )
        .unwrap()
    }

    fn create_render_pass(device: Arc<Device>, format: Format) -> Arc<RenderPass> {
        vulkano::ordered_passes_renderpass!(
            device,
            attachments: {
                color: {
                    load: Clear,
                    store: Store,
                    format: format,
                    samples: 1,
                }
            },
            passes: [
                { color: [color], depth_stencil: {}, input: [] }, // Draw viewport result
                { color: [color], depth_stencil: {}, input: [] }  // Gui render pass
            ]
        )
        .expect("error creating final render pass")
    }

    fn create_pipeline(device: Arc<Device>, render_pass: Arc<RenderPass>) -> Arc<GraphicsPipeline> {
        let vs = vs::load(device.clone()).expect("failed to create shader module");
        let fs = fs::load(device.clone()).expect("failed to create shader module");

        GraphicsPipeline::start()
            .vertex_input_state(BuffersDefinition::new().vertex::<QuadVertex>())
            .vertex_shader(vs.entry_point("main").unwrap(), ())
            .input_assembly_state(InputAssemblyState::new())
            .fragment_shader(fs.entry_point("main").unwrap(), ())
            .viewport_state(ViewportState::viewport_dynamic_scissor_irrelevant())
            .render_pass(Subpass::from(render_pass, 0).unwrap())
            .build(device)
            .expect("error creating pipeline")
    }

    fn create_viewport_quad(device: Arc<Device>) -> (Buffer<QuadVertex>, Buffer<u32>) {
        let verticies = vec![
            QuadVertex {
                position: [-1.0, -1.0],
                tex_coords: [0.0, 0.0],
            },
            QuadVertex {
                position: [-1.0, 1.0],
                tex_coords: [0.0, 1.0],
            },
            QuadVertex {
                position: [1.0, 1.0],
                tex_coords: [1.0, 1.0],
            },
            QuadVertex {
                position: [1.0, -1.0],
                tex_coords: [1.0, 0.0],
            },
        ];
        let vertex_buffer = CpuAccessibleBuffer::<[QuadVertex]>::from_iter(
            device.clone(),
            BufferUsage::vertex_buffer(),
            false,
            verticies,
        )
        .expect("failed to create quad vertex buffer");

        let indicies = vec![0, 2, 1, 0, 3, 2];
        let index_buffer = CpuAccessibleBuffer::<[u32]>::from_iter(
            device,
            BufferUsage::index_buffer(),
            false,
            indicies,
        )
        .expect("failed to create quad index buffer");

        (vertex_buffer, index_buffer)
    }

    pub fn viewport_subpass(&self) -> Subpass {
        Subpass::from(self.render_pass.clone(), 0).expect("failed to create subpass")
    }

    pub fn gui_subpass(&self) -> Subpass {
        Subpass::from(self.render_pass.clone(), 1).expect("failed to create subpass")
    }
}

mod vs {
    vulkano_shaders::shader! {
        ty: "vertex",
        src: "
#version 450
layout(location=0) in vec2 position;
layout(location=1) in vec2 tex_coords;

layout(location = 0) out vec2 f_tex_coords;

layout(set = 0, binding = 0) uniform Uniforms {
    vec2 offset;
    float scale;
    float aspect_ratio;
} uniforms;

void main() {
    gl_Position =  vec4(uniforms.scale * (position * vec2(1.0, uniforms.aspect_ratio)) + uniforms.offset, 0.0, 1.0);
    f_tex_coords = tex_coords;
}
        ",
        types_meta: {
            use bytemuck::{Pod, Zeroable};

            #[derive(Clone, Copy, Zeroable, Pod)]
        }
    }
}

impl Default for vs::ty::Uniforms {
    fn default() -> Self {
        Self {
            offset: [0.0, 0.0],
            scale: 1.0,
            aspect_ratio: 1.0,
        }
    }
}

mod fs {
    vulkano_shaders::shader! {
        ty: "fragment",
        src: "
#version 450
layout(location = 0) in vec2 v_tex_coords;

layout(location = 0) out vec4 f_color;

layout(set = 0, binding = 1) uniform sampler2D tex;

void main() {
    f_color = texture(tex, v_tex_coords);
}
"
    }
}
