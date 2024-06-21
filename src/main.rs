mod objects;

use glam::Vec3A as Vec3;
use itertools::Itertools;
use objects::{
    box_intersection_check, BoxShape, Color, Hittable, Material, Object, Ray, Sphere, Triangle, World, BLACK
};
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

const WIDTH: usize = 500;
const HEIGHT: usize = 500;

const SCALE: f32 = 1.0;

fn calculate_normal(a: Vec3, b: Vec3, c: Vec3) -> Vec3 {
    (b - a).cross(c - a).normalize()
}

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
            verts.push(Vec3::new(a, b - 2.0, c + 6.0));
        } else if let Some(stripped) = line.strip_prefix("f ") {
            let (a, b, c) = stripped
                .split(' ')
                .filter_map(|s| s.parse::<usize>().ok())
                .collect_tuple()
                .unwrap();
            tris.push(Triangle {
                a: verts[a - 1],
                b: verts[b - 1],
                c: verts[c - 1],
                normal: calculate_normal(verts[a - 1], verts[b - 1], verts[c - 1]),
                color: Color::from_u8(0x50, 0xc8, 0x78),
            });
        }
    }
    tris
}

fn trace_ray(ray: &Ray, world: &World) -> Color {
    let mut hit_tri = None;
    let mut hit_data = None;
    let mut hit_obj = None;

    let mut closest = f32::INFINITY;

    for obj in world.objects.iter() {
        if box_intersection_check(&ray, &obj.bounding_box) {
            for tri in obj.tris.iter() {
                if let Some(hit) = tri.ray_hits(&ray, closest) {
                    if closest > hit.t {
                        closest = hit.t;
                        hit_tri = Some(tri);
                        hit_obj = Some(obj);
                        hit_data = Some(hit);
                    }
                }
            }
        }
    }

    if let Some(hit) = hit_tri {
        let hit_dat = hit_data.unwrap();
        let inside_obj = hit_obj.unwrap();
        let sun_dir = (world.sun - hit_dat.pos).normalize();
        let mut can_see_sun = true;
        let sun_ray = Ray {
            origin: hit_dat.pos,
            dir: sun_dir,
            inv_dir: sun_dir.recip(),
        };

        // We do a little cheating
        if hit.normal.dot(sun_dir) < 0.0 {
            return BLACK;
        }

        for tri in inside_obj.tris.iter() {
            if tri.ray_hits(&sun_ray, closest).is_some() {
                can_see_sun = false;
                break;
            }
        }
        if can_see_sun {
            for obj in world.objects.iter() {
                if !std::ptr::eq(obj, inside_obj) || box_intersection_check(&ray, &obj.bounding_box) {
                    for tri in obj.tris.iter() {
                        if tri.ray_hits(&sun_ray, closest).is_some() {
                            can_see_sun = false;
                            break;
                        }
                    }
                }
            }
        }
        if can_see_sun {
            let color = hit.get_color();
            let light_power = hit.normal.dot(sun_dir); // Assuming intensity 1
            return color.mul(light_power * inside_obj.material.albedo * 2.2); // 2.2 is gamma
        } else {
            return BLACK;
        }
    } else {
       return Color::from_u8(0x87, 0xce, 0xeb);
    }
}

fn draw(frame: &mut [u8], world: &World, t: f32) {
    let aspect_ratio = WIDTH / HEIGHT;
    let origin = Vec3::new(0.0, 0.0, 0.0);

    let current_material = Material { albedo: 1.0 };

    let rows = (0..HEIGHT)
        .into_par_iter()
        .map(|y| {
            let mut row = Vec::with_capacity(WIDTH);
            for x in 0..WIDTH {
                let xx =
                    (2.0 * (x as f32 + 0.5) / (WIDTH as f32) - 1.0) * aspect_ratio as f32 * SCALE;
                let yy = (1.0 - 2.0 * (y as f32 + 0.5) / HEIGHT as f32) * SCALE;

                let ray = Ray {
                    origin,
                    dir: Vec3::new(xx, yy, 1.0),
                    inv_dir: Vec3::new(xx, yy, 1.0).recip(),
                };
                
                row.push(trace_ray(&ray, world));
            }
            row
        })
        .collect::<Vec<Vec<Color>>>();

    for (y, row) in rows.iter().enumerate() {
        for (x, color) in row.iter().enumerate() {
            frame[y * 4 * WIDTH + (x * 4)] = (color.r * 255.0).min(255.0) as u8;
            frame[y * 4 * WIDTH + (x * 4) + 1] = (color.g * 255.0).min(255.0) as u8;
            frame[y * 4 * WIDTH + (x * 4) + 2] = (color.b * 255.0).min(255.0) as u8;
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

    println!("hi");
    let teapot: Vec<Triangle> = read_obj("teapot.obj");
    let mut objects: Vec<Object> = Vec::new();
    objects.push(Object::from(teapot, Material::new(1.0)));
    objects.push(Object::from(vec![
        Triangle {
            a: Vec3::new(-10000.0, -5.0, -10000.0),
            b: Vec3::new(-10000.0, -5.0, 10000.0),
            c: Vec3::new(10000.0, -5.0, 10000.0),
            normal: Vec3::new(0.0, 1.0, 0.0),
            color: Color::from_u8(128, 128, 128),
        },
        Triangle {
            a: Vec3::new(-10000.0, -5.0, 10000.0),
            b: Vec3::new(10000.0, -5.0, 10000.0),
            c: Vec3::new(-10000.0, -5.0, 10000.0),
            normal: Vec3::new(0.0, 1.0, 0.0),
            color: Color::from_u8(0, 128, 0),
        },
    ], Material::new(1.0)));
    println!(
        "len: {}",
        objects.iter().map(|obj| obj.tris.len()).sum::<usize>()
    );

    let _ = event_loop.run(move |event, _| {
        match &event {
            Event::WindowEvent {
                window_id: _,
                event: inner_event,
            } => {
                if *inner_event != WindowEvent::RedrawRequested {
                    return;
                }

                let t = start.elapsed().as_secs_f32();

                // Far far away
                let sun = Vec3::new(
                    f32::sin(t / 10.0) * 100_000.0,
                    25_000.0,
                    f32::cos(t / 10.0) * 100_000.0,
                );
                let o = objects.clone();
                let world = World {
                    objects: o,
                    sun
                };


                draw(pixels.frame_mut(), &world, t);

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
