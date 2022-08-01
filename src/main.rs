use egui::Context;
use egui_winit_vulkano::Gui;
use lazy_static::lazy_static;
use vulkano::{
    device::{DeviceExtensions, Features},
    format::Format,
    image::ImageUsage,
    instance::{InstanceCreateInfo, InstanceExtensions},
    pipeline::graphics::viewport::Viewport,
};
use vulkano_util::{
    context::{VulkanoConfig, VulkanoContext},
    window::{VulkanoWindows, WindowDescriptor},
};
use winit::{
    dpi::LogicalSize,
    event::{Event, WindowEvent},
    event_loop::{ControlFlow, EventLoop},
};

use crate::final_render_pass::{FinalRenderPass, ViewportTransform};

mod final_render_pass;

lazy_static! {
    static ref INSTANCE_EXTENSIONS: InstanceExtensions = InstanceExtensions {
        ..vulkano_win::required_extensions()
    };
}

const INSTANCE_LAYERS: Vec<String> = vec![];

const DEVICE_EXTENSIONS: DeviceExtensions = DeviceExtensions {
    khr_swapchain: true,
    ..DeviceExtensions::none()
};

const FEATURES: Features = Features { ..Features::none() };

// Required format by egui (otherwise egui colors look weird)
const SWAPCHAIN_FORMAT: Format = Format::B8G8R8A8_SRGB;

const WINDOW_SIZE: LogicalSize<f32> = LogicalSize::new(700.0, 500.0);
const WINDOW_TITLE: &str = "Conway's Game of Life";

fn main() {
    // Create vulkano context
    let vulkano_context = VulkanoContext::new(VulkanoConfig {
        instance_create_info: InstanceCreateInfo {
            enabled_extensions: *INSTANCE_EXTENSIONS,
            enabled_layers: INSTANCE_LAYERS,
            ..Default::default()
        },
        device_features: FEATURES,
        device_extensions: DEVICE_EXTENSIONS,
        ..Default::default()
    });

    println!("Using device: {}", vulkano_context.device_name());

    // Create window
    let mut windows = VulkanoWindows::default();
    let event_loop = EventLoop::new();
    let main_window_id = windows.create_window(
        &event_loop,
        &vulkano_context,
        &WindowDescriptor {
            width: WINDOW_SIZE.width,
            height: WINDOW_SIZE.height,
            title: WINDOW_TITLE.to_string(),
            ..WindowDescriptor::default()
        },
        |swapchain_create_info| {
            swapchain_create_info.image_format = Some(SWAPCHAIN_FORMAT);
        },
    );
    let window_renderer = windows.get_primary_renderer_mut().unwrap();

    let final_render_pass = FinalRenderPass::new(&vulkano_context, SWAPCHAIN_FORMAT);
    let viewport_view_id = 0;
    window_renderer.add_additional_image_view(
        viewport_view_id,
        SWAPCHAIN_FORMAT,
        ImageUsage { 
            sampled: true,
            storage: false,
            color_attachment: true,
            ..ImageUsage::none()
        },
    );

    // Create gui context
    let mut gui = Gui::new_with_subpass(
        window_renderer.surface(),
        vulkano_context.graphics_queue(),
        final_render_pass.gui_subpass(),
    );

    let mut viewport_transform = ViewportTransform::default();

    // Run the event loop to keep window open
    event_loop.run(move |event, _, control_flow| {
        let window_renderer = windows.get_primary_renderer_mut().unwrap();
        
        match event {
            Event::WindowEvent { event, window_id } => {
                if window_id == main_window_id {
                    let pass_events_to_app = !gui.update(&event);
                    if pass_events_to_app {
                    }

                    match event {
                        WindowEvent::Resized(_) => {
                            window_renderer.resize();
                        }
                        WindowEvent::ScaleFactorChanged { .. } => {
                            window_renderer.resize();
                        }
                        WindowEvent::CloseRequested => {
                            *control_flow = ControlFlow::Exit;
                        }
                        _ => (),
                    }
                }
            }
            Event::RedrawRequested(window_id) => {
                if window_id == main_window_id {
                    // Create immediate ui
                    let context = gui.context();
                    gui.immediate_ui(|_| {
                        egui::SidePanel::left("left_panel")
                            .min_width(200.0)
                            .show(&context, |ui| {
                            
                            ui.vertical_centered(|ui| {
                                ui.heading("Settings");
                            });
                            ui.separator();
                            ui.horizontal(|ui| {
                                ui.label("Offset X:");
                                ui.add(egui::Slider::new(&mut viewport_transform.offset[0], -1.0..=1.0));
                            });

                            ui.horizontal(|ui| {
                                ui.label("Offset Y:");
                                ui.add(egui::Slider::new(&mut viewport_transform.offset[1], -1.0..=1.0));
                            });

                            ui.horizontal(|ui| {
                                ui.label("Scale:");
                                ui.add(egui::Slider::new(&mut viewport_transform.scale, 0.1..=50.0));
                            });
                        });
                    });

                    // Calculate viwport so as not to render behind egui components
                    let viewport_bounds = calculate_viewport(
                        &context,
                        window_renderer.window().scale_factor() as f32,
                    );

                    // Update image aspect ratio
                    viewport_transform.aspect_ratio = viewport_bounds.dimensions[0] / viewport_bounds.dimensions[1];

                    let before_pipeline_future = window_renderer
                        .acquire()
                        .expect("failed to acquire window renderer future");
                    let viewport_view = window_renderer.get_additional_image_view(viewport_view_id);

                    // Render viewport

                    // Render gui
                    let after_gui_future = final_render_pass.render(
                        before_pipeline_future,
                        window_renderer.swapchain_image_view(),
                        &mut gui,
                        viewport_view,
                        viewport_bounds,
                        viewport_transform,
                    );

                    // Present to surface
                    window_renderer.present(after_gui_future, true);
                }
            }
            Event::MainEventsCleared => {
                window_renderer.surface().window().request_redraw();
            }
            _ => (),
        }
    });
}

fn calculate_viewport(context: &Context, scale_factor: f32) -> Viewport {
    let origin = context.available_rect().left_top();
    let dimensions = context.available_rect().right_bottom() - origin;

    Viewport {
        origin: [origin.x * scale_factor, origin.y * scale_factor],
        dimensions: [dimensions.x * scale_factor, dimensions.y * scale_factor],
        depth_range: 0.0..1.0,
    }
}
