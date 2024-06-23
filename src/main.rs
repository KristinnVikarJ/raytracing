mod objects;
mod simd_accel;

use glam::Vec3A as Vec3;
use itertools::Itertools;
use objects::{
    box_intersection_check, Color, Hit, Hittable, Material, Object, Ray, Triangle, World, BLACK,
};
use pixels::{Error, Pixels, SurfaceTexture};
use rayon::iter::{IntoParallelIterator, ParallelIterator};
use simd_accel::{extract_f32_from_m256, pack_triangles, ray_to_avx};
use std::arch::x86_64::_mm256_set1_ps;
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

const WIDTH: usize = 1000;
const HEIGHT: usize = 1000;

const SCALE: f32 = 1.0;

fn calculate_normal(a: Vec3, b: Vec3, c: Vec3) -> Vec3 {
    (b - a).cross(c - a).normalize()
}

fn read_obj(filename: &str, offset: Vec3, color: Color) -> Vec<Triangle> {
    let mut verts = Vec::new();
    let mut tris = Vec::new();
    for line in read_to_string(filename).unwrap().lines() {
        if let Some(stripped) = line.strip_prefix("v ") {
            let (a, b, c) = stripped
                .split(' ')
                .filter_map(|s| s.parse::<f32>().ok())
                .collect_tuple()
                .unwrap();
            verts.push(Vec3::new(a, b, c) + offset);
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
                color: color.clone(),
            });
        }
    }
    tris
}

fn trace_ray(ray: &Ray, world: &World, depth: u8) -> Color {
    let mut hit_tri = None;
    let mut hit_data = None;
    let mut hit_obj = None;

    let mut closest = f32::INFINITY;
    let mut closest_splat = unsafe { _mm256_set1_ps(closest) };

    let mut color = BLACK;

    let mut triangle_tmp = Vec::with_capacity(8);
    let mut c = 0;
    let (ray_origin, ray_direction) = ray_to_avx(&ray);

    for obj in world.objects.iter() {
        if box_intersection_check(ray, &obj.bounding_box) {
            for tri in obj.tris.iter() {
                triangle_tmp.push(tri);
                c += 1;
                if c == 8 {
                    let packed = pack_triangles(&triangle_tmp);
                    let (t_values, hit_mask) =
                        packed.intersect(ray_origin, ray_direction, closest_splat);
                    if hit_mask != 0xFF {
                        let t_arr = extract_f32_from_m256(t_values);
                        // At least 1 hit!
                        for i in 0..8 {
                            if t_arr[i] > 0.0 && closest > t_arr[i] {
                                closest = t_arr[i];
                                hit_tri = Some(triangle_tmp[i]);
                                hit_obj = Some(obj);
                                hit_data = Some(Hit {
                                    pos: ray.at(t_arr[i]) + (triangle_tmp[i].normal * 0.00001),
                                    t: t_arr[i],
                                });
                            }
                        }
                    }
                    closest_splat = unsafe { _mm256_set1_ps(closest) };
                    triangle_tmp.clear();
                    c = 0;
                }
            }

            if c > 0 {
                let packed = pack_triangles(&triangle_tmp);
                let (t_values, hit_mask) =
                    packed.intersect(ray_origin, ray_direction, closest_splat);
                if hit_mask != 0xFF {
                    let t_arr = extract_f32_from_m256(t_values);
                    // At least 1 hit!
                    for i in 0..c {
                        if t_arr[i] > 0.0 && closest > t_arr[i] {
                            hit_tri = Some(triangle_tmp[i]);
                            hit_obj = Some(obj);
                            hit_data = Some(Hit {
                                pos: ray.at(t_arr[i]) + (triangle_tmp[i].normal * 0.00001),
                                t: t_arr[i],
                            });
                        }
                    }
                }
                triangle_tmp.clear()
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
        let (ray_origin, ray_direction) = ray_to_avx(&sun_ray);

        // We do a little cheating
        if hit.normal.dot(sun_dir) > 0.0 {
            for tri in inside_obj.tris.iter() {
                triangle_tmp.push(tri);
                c += 1;
                if c == 8 {
                    let packed = pack_triangles(&triangle_tmp);
                    let (_, hit_mask) =
                        packed.intersect(ray_origin, ray_direction, closest_splat);
                    if hit_mask != 0xFF {
                        can_see_sun = false;
                        break;
                    }
                    triangle_tmp.clear();
                    c = 0;
                }
            }
            if c > 0 {
                let packed = pack_triangles(&triangle_tmp);
                let (_, hit_mask) =
                    packed.intersect(ray_origin, ray_direction, closest_splat);
                if hit_mask != 0xFF {
                    can_see_sun = false;
                }
                triangle_tmp.clear();
                c = 0;
            }
            if can_see_sun {
                for obj in world.objects.iter() {
                    if !std::ptr::eq(obj, inside_obj)
                        || box_intersection_check(ray, &obj.bounding_box)
                    {
                        for tri in obj.tris.iter() {
                            triangle_tmp.push(tri);
                            c += 1;
                            if c == 8 {
                                let packed = pack_triangles(&triangle_tmp);
                                let (_, hit_mask) =
                                    packed.intersect(ray_origin, ray_direction, closest_splat);
                                if hit_mask != 0xFF {
                                    can_see_sun = false;
                                    break;
                                }
                                triangle_tmp.clear();
                                c = 0;
                            }
                        }
                        if c > 0 {
                            let packed = pack_triangles(&triangle_tmp);
                            let (_, hit_mask) =
                                packed.intersect(ray_origin, ray_direction, closest_splat);
                            if hit_mask != 0xFF {
                                can_see_sun = false;
                            }
                            triangle_tmp.clear()
                        }
                        if !can_see_sun {
                            break;
                        }
                    }
                }
            }
            if can_see_sun {
                let hit_color = hit.get_color();
                let light_power = hit.normal.dot(sun_dir); // Assuming intensity 1
                color = hit_color.mul(light_power * inside_obj.material.albedo * 2.2)
                // 2.2 is gamma
            }
        }

        if depth < 5 {
            let reflection_dir = ray.dir - 2.0 * hit.normal * (ray.dir.dot(hit.normal));
            color = color.mul(1.0 - inside_obj.material.reflectivity);
            color = color.add(
                trace_ray(
                    &Ray {
                        origin: hit_dat.pos,
                        dir: reflection_dir,
                        inv_dir: reflection_dir.recip(),
                    },
                    world,
                    depth + 1,
                )
                .mul(inside_obj.material.reflectivity),
            );
        }
    } else {
        return Color::from_u8(0x87, 0xce, 0xeb);
    }

    color
}

fn draw(frame: &mut [u8], world: &World, t: f32) {
    let aspect_ratio = WIDTH / HEIGHT;
    let origin = Vec3::new(0.0, 0.0, 0.0);

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

                row.push(trace_ray(&ray, world, 1));
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

    let teapot: Vec<Triangle> = read_obj(
        "teapot.obj",
        Vec3::new(-3.0, -2.0, 7.0),
        Color::from_u8(0x50, 0xc8, 0x78),
    );
    let teapot2: Vec<Triangle> = read_obj(
        "teapot.obj",
        Vec3::new(3.0, -2.0, 6.0),
        Color::from_u8(0xFF, 0, 0),
    );
    let mut objects: Vec<Object> = Vec::new();
    objects.push(Object::from(teapot, Material::new(1.0, 1.0)));
    objects.push(Object::from(teapot2, Material::new(1.0, 0.1)));
    objects.push(Object::from(
        vec![
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
        ],
        Material::new(1.0, 0.0),
    ));
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
                let world = World { objects: o, sun };

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
