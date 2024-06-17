mod objects;

use glam::Vec3A as Vec3;
use itertools::Itertools;
use objects::{Color, Hittable, Ray, Sphere, Triangle};
use pixels::{Error, Pixels, SurfaceTexture};
use rayon::iter::{IntoParallelIterator, ParallelIterator};
use std::fs::read_to_string;
use std::{process, time::Instant};
use winit::{
    dpi::LogicalSize,
    event::{Event, WindowEvent},
    event_loop::EventLoop,
    keyboard::KeyCode,
    window::WindowBuilder,
};
use winit_input_helper::WinitInputHelper;

const WIDTH: usize = 800;
const HEIGHT: usize = 600;

const SCALE: f32 = 1.0;

fn read_obj(filename: &str) -> Vec<Triangle> {
    let mut verts = Vec::new();
    let mut tris = Vec::new();
    for line in read_to_string(filename).unwrap().lines() {
        if let Some(stripped) = line.strip_prefix("v ") {
            let (a, b, c) = stripped
                .split(' ')
                .filter_map(|s| s.parse::<f32>().ok())
                .collect_tuple()
                .unwrap();
            verts.push(Vec3::new(a, b - 2.0, c + 4.0));
        } else if let Some(stripped) = line.strip_prefix("f ")  {
            let (a, b, c) = stripped
                .split(' ')
                .filter_map(|s| s.parse::<usize>().ok())
                .collect_tuple()
                .unwrap();
            tris.push(Triangle {
                a: verts[a - 1],
                b: verts[b - 1],
                c: verts[c - 1],
                color: Color::new(128, 128, 128),
            });
        }
    }
    tris
}

fn draw(frame: &mut [u8], t: f32) {
    let aspect_ratio = WIDTH / HEIGHT;
    let origin = Vec3::new(0.0, 0.0, 0.0);
    // Far far away
    let sun = Vec3::new(f32::sin(t/15.0)*100_000.0, 100_000.0, f32::cos(t/15.0)*100_000.0);
    let s = Sphere {
        center: Vec3::new(0.0, 0.0, 2.5),
        radius: 1.0,
        color: Color::new(0, 0, 255),
    };

    //let objects: Vec<Box<dyn Hittable>> = vec![Box::new(t), Box::new(t2), Box::new(s)];
    println!("hi");
    let objects: Vec<Triangle> = read_obj("teapot.obj");
    println!("len: {}", objects.len());

    let rows = (0..HEIGHT)
        .into_par_iter()
        .map(|y| {
            let mut row = Vec::with_capacity(WIDTH);
            for x in 0..WIDTH {
                let xx =
                    (2.0 * (x as f32 + 0.5) / (WIDTH as f32) - 1.0) * aspect_ratio as f32 * SCALE;
                let yy = (1.0 - 2.0 * (y as f32 + 0.5) / HEIGHT as f32) * SCALE;

                let mut hit_obj = None;
                let mut hit_data = None;
                let ray = Ray {
                    origin,
                    dir: Vec3::new(xx, yy, 1.0),
                    //inv_dir: 1 / Vec3::new(xx, yy, 1.0),
                };

                let mut closest = f32::INFINITY;

                for obj in objects.iter() {
                    if let Some(hit) = obj.ray_hits(&ray, closest) {
                        if closest > hit.t {
                            closest = hit.t;
                            hit_obj = Some(obj);
                            hit_data = Some(hit);
                        }
                    }
                }

                if let Some(hit) = hit_obj {
                    let hit_dat = hit_data.unwrap();
                    let sun_dir = sun - hit_dat.pos;
                    let mut can_see_sun = true;
                    let sun_ray = Ray {
                        dir: sun_dir,
                        origin: hit_dat.pos,
                    };
                    for obj in objects.iter() {
                        if obj.ray_hits(&sun_ray, closest).is_some() {
                            can_see_sun = false;
                            break;
                        }
                    }
                    if can_see_sun {
                        let color = hit.get_color();
                        row.push(color.clone());
                    } else {
                        row.push(Color { r: 0, g: 0, b: 0 });
                    }
                } else {
                    row.push(Color {
                        r: 0x87,
                        g: 0xce,
                        b: 0xeb,
                    });
                }
            }
            row
        })
        .collect::<Vec<Vec<Color>>>();

    for (y, row) in rows.iter().enumerate() {
        for (x, color) in row.iter().enumerate() {
            frame[y * 4 * WIDTH + (x * 4)] = color.r;
            frame[y * 4 * WIDTH + (x * 4) + 1] = color.g;
            frame[y * 4 * WIDTH + (x * 4) + 2] = color.b;
            frame[y * 4 * WIDTH + (x * 4) + 3] = 0xff;
        }
    }
}

fn main() -> Result<(), Error> {
    let event_loop = EventLoop::new().unwrap();
    let mut input = WinitInputHelper::new();
    let window = {
        let size = LogicalSize::new(WIDTH as f64, HEIGHT as f64);
        WindowBuilder::new()
            .with_title("Ray Tracer")
            .with_inner_size(size)
            .with_min_inner_size(size)
            .build(&event_loop)
            .unwrap()
    };

    let mut pixels = {
        let window_size = window.inner_size();
        let surface_texture = SurfaceTexture::new(window_size.width, window_size.height, &window);
        Pixels::new(WIDTH as u32, HEIGHT as u32, surface_texture)?
    };

    let mut frames = 0;
    let mut frame_timer = Instant::now();

    let start = Instant::now();

    let _ = event_loop.run(move |event, _| {
        match &event {
            Event::WindowEvent {
                window_id: _,
                event: inner_event,
            } => {
                if *inner_event != WindowEvent::RedrawRequested {
                    return;
                }

                draw(pixels.frame_mut(), start.elapsed().as_secs_f32());

                frames += 1;
                if frame_timer.elapsed().as_secs() >= 1 {
                    println!(
                        "fps: {}",
                        frames as f32 / frame_timer.elapsed().as_secs_f32()
                    );
                    frames = 0;
                    frame_timer = Instant::now();
                }
                if let Err(err) = pixels.render() {
                    println!("ERROR RENDER: {:?}", err);
                    process::exit(1);
                }
            }
            Event::AboutToWait => {
                window.request_redraw();
            }
            _ => (),
        }
        if input.update(&event) {
            // Close events
            if input.key_pressed(KeyCode::Escape) || input.close_requested() {
                process::exit(0);
            }

            // Resize the window
            if let Some(size) = input.window_resized() {
                if let Err(err) = pixels.resize_surface(size.width, size.height) {
                    println!("ERROR RESIZE: {:?}", err);
                    process::exit(1);
                }
            }
        }
    });
    Ok(())
}
