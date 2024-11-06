mod objects;
mod opt;
mod simd_accel;

use glam::Vec3;
use itertools::Itertools;
use objects::{
    box_intersection_check, new_triangle, Color, Hittable, Material, Object, PackedObject, Ray,
    ScreenColor, Triangle, TriangleData, World, BLACK, SCREEN_BLACK,
};
use opt::{optimize_model, pack_model};
use pixels::{Error, Pixels, SurfaceTexture};
use rand::random;
use rayon::iter::{IntoParallelIterator, ParallelIterator};
use simd_accel::{extract_f32_from_m256, ray_to_avx};
use std::arch::x86_64::_mm256_set1_ps;
use std::f32::consts::PI;
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
const HEIGHT: usize = 800;

const SCALE: f32 = 1.0;

const REFLECT_DEPTH: u8 = 4;
const DIFFUSE_DEPTH: u8 = 2;
const DIFFUSE_SCATT: u32 = 10; // How many new rays to scatter on ray hit

pub fn read_obj(
    filename: &str,
    offset: Vec3,
    color: Color,
) -> (Vec<Triangle>, Vec<TriangleData>, Vec<Vec3>) {
    let mut verts = Vec::new();
    let mut tris = Vec::new();
    let mut tris_data = Vec::new();
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
            let (tri, tri_data) = new_triangle(
                a as u32 - 1,
                b as u32 - 1,
                c as u32 - 1,
                &verts,
                color.clone(),
            );
            tris.push(tri);
            tris_data.push(tri_data);
        }
    }
    (tris, tris_data, verts)
}

fn get_tangent_vectors(norm: Vec3) -> (Vec3, Vec3) {
    let arbitrary = if norm.z.abs() < 1e-6 {
        Vec3::Z
    } else {
        Vec3::X
    };

    let tang1 = norm.cross(arbitrary).normalize();

    let tang2 = norm.cross(tang1).normalize();

    (tang1, tang2)
}

fn trace_ray(ray: &Ray, world: &World, depth: u8) -> (Color, f32) {
    let mut hit_tri = None;
    let mut hit_position = None;
    let mut hit_obj = None;

    let mut closest = f32::INFINITY;
    let mut closest_splat = unsafe { _mm256_set1_ps(closest) };

    let mut color = BLACK;

    let simd_ray = ray_to_avx(&ray);

    for obj in world.objects.iter() {
        if box_intersection_check(ray, &obj.obj.bounding_box) {
            // TODO: Simplify this mofo
            for (idx, packed_bounds) in obj.packed_tri_bounds.iter().enumerate() {
                let (t_values, hit_mask) =
                    packed_bounds.intersect(&simd_ray, closest_splat);
                if hit_mask != 0xFF {
                    // At least 1 hit!
                    let t_arr = extract_f32_from_m256(t_values);
                    for i in 0..8 {
                        if t_arr[i] > 0.0 {
                            let (t_values, hit_mask) =
                                obj.packed_tris[(idx*8)+i].intersect(&simd_ray, closest_splat);
                            if hit_mask != 0xFF {
                                let t_arr = extract_f32_from_m256(t_values);
                                // At least 1 hit!
                                for j in 0..8 {
                                    if t_arr[j] > 0.0 && closest > t_arr[j] {
                                        closest = t_arr[j];
                                        hit_tri = Some(obj.obj.tri_data[(((idx*8)+i)*8) + j].clone());
                                        hit_obj = Some(obj);
                                        hit_position = Some(
                                            ray.at(t_arr[j])
                                                + (obj.obj.tri_data[(((idx*8)+i)*8) + j].normal * 0.00001),
                                        );
                                    }
                                }
                                closest_splat = unsafe { _mm256_set1_ps(closest) };
                            }
                        }
                    }
                }
            }
            
            for (idx, bound) in obj.rest_bounds.iter().enumerate() {
                if box_intersection_check(&ray, bound) {
                    let (t_values, hit_mask) =
                        obj.packed_tris[(obj.packed_tri_bounds.len()*8)+idx].intersect(&simd_ray, closest_splat);
                    if hit_mask != 0xFF {
                        let t_arr = extract_f32_from_m256(t_values);
                        // At least 1 hit!
                        for i in 0..8 {
                            if t_arr[i] > 0.0 && closest > t_arr[i] {
                                closest = t_arr[i];
                                hit_tri = Some(obj.obj.tri_data[(((obj.packed_tri_bounds.len()*8)+idx)*8) + i].clone());
                                hit_obj = Some(obj);
                                hit_position = Some(
                                    ray.at(t_arr[i])
                                        + (obj.obj.tri_data[(((obj.packed_tri_bounds.len()*8)+idx)*8) + i].normal * 0.00001),
                                );
                            }
                        }
                        closest_splat = unsafe { _mm256_set1_ps(closest) };
                    }
                }
            }

            let c = obj.rest_tri.len();
            for (idx, tri) in obj.rest_tri.iter().enumerate() {
                if let Some(t) = tri.ray_hits(&ray, closest, &obj.obj.verts) {
                    if closest > t {
                        hit_tri = Some(obj.obj.tri_data[obj.obj.tris.len() - c + idx].clone());
                        hit_obj = Some(obj);
                        hit_position = Some(
                            ray.at(t)
                                + (obj.obj.tri_data[obj.obj.tris.len() - c + idx].normal * 0.00001),
                        );
                    }
                }
            }
        }
    }

    if let Some(hit_data) = hit_tri {
        let hit_pos = hit_position.unwrap();
        let inside_obj = hit_obj.unwrap();
        let sun_dir = (world.sun - hit_pos).normalize();
        let mut can_see_sun = true;
        let sun_ray = Ray {
            origin: hit_pos,
            dir: sun_dir,
            inv_dir: sun_dir.recip(),
        };
        let simd_ray = ray_to_avx(&sun_ray);

        let closest = f32::INFINITY;
        let closest_splat = unsafe { _mm256_set1_ps(closest) };

        // We do a little cheating
        if hit_data.normal.dot(sun_dir) > 0.0 {
            for obj in world.objects.iter() {
                if box_intersection_check(&sun_ray, &obj.obj.bounding_box) {
                    for (idx, packed_bounds) in obj.packed_tri_bounds.iter().enumerate() {
                        let (t_values, hit_mask) =
                            packed_bounds.intersect(&simd_ray, closest_splat);
                        if hit_mask != 0xFF {
                            let t_arr = extract_f32_from_m256(t_values);
                            for i in 0..8 {
                                if t_arr[i] > 0.0 {
                                    let (_, hit_mask) =
                                        obj.packed_tris[(idx*8)+i].intersect(&simd_ray, closest_splat);
                                    if hit_mask != 0xFF {
                                        can_see_sun = false;
                                        break;
                                    }
                                }
                            }
                        }
                        if !can_see_sun {
                            break; // dont check more packed
                        }
                    }

                    for (idx, bound) in obj.rest_bounds.iter().enumerate() {
                        if box_intersection_check(&sun_ray, bound) {
                            let (_, hit_mask) =
                                obj.packed_tris[(obj.packed_tri_bounds.len()*8)+idx].intersect(&simd_ray, closest_splat);
                            if hit_mask != 0xFF {
                                can_see_sun = false;
                                break;
                            }
                        }
                    }
                    if can_see_sun {
                        can_see_sun = !obj
                            .rest_tri
                            .iter()
                            .any(|tri| tri.ray_hits(&sun_ray, closest, &obj.obj.verts).is_some());
                    }
                    if !can_see_sun {
                        break;
                    }
                }
            }

            if can_see_sun {
                let light_power = hit_data.normal.dot(sun_dir); // Assuming light intensity 1
                color = hit_data
                    .color
                    .mul(light_power * inside_obj.obj.material.albedo)
            }
        }

        let mut reflect_color = Color::new(0.0, 0.0, 0.0);
        if depth < REFLECT_DEPTH && inside_obj.obj.material.reflectivity > 0.0 {
            let reflection_dir = ray.dir - 2.0 * hit_data.normal * (ray.dir.dot(hit_data.normal));
            let (col, dist) = trace_ray(
                &Ray {
                    origin: hit_pos,
                    dir: reflection_dir,
                    inv_dir: reflection_dir.recip(),
                },
                world,
                depth + 1,
            );
            reflect_color = col
                .mul(inside_obj.obj.material.reflectivity);
        }

        let mut diffuse_color = Color::new(0.0, 0.0, 0.0);
        if depth < DIFFUSE_DEPTH {
            for _ in 0..DIFFUSE_SCATT {
                let sin_theta = random::<f32>().sqrt();
                let cos_theta = (1.0f32 - sin_theta * sin_theta).sqrt();
                let psi = random::<f32>() * 2.0 * PI;

                // Calculate the components of the vector
                let a = sin_theta * psi.cos();
                let b = sin_theta * psi.sin();
                let c = cos_theta;

                // Compute the velocity vector as v1 + v2 + v3
                let (tang1, tang2) = get_tangent_vectors(hit_data.normal);
                let vel = Vec3::new(
                    a * tang1.x + b * tang2.x + c * hit_data.normal.x, 
                    a * tang1.y + b * tang2.y + c * hit_data.normal.y,
                    a * tang1.z + b * tang2.z + c * hit_data.normal.z
                ).normalize();

                let light_power = hit_data.normal.dot(vel);
                let il = 1.0; // Light intensity

                let (col, dist) = trace_ray(
                    &Ray {
                        origin: hit_pos,
                        dir: vel,
                        inv_dir: vel.recip(),
                    },
                    world,
                    depth + 1,
                );

                diffuse_color = diffuse_color.add(
                    col
                    .mul_col(&hit_data.color)
                    .mul(light_power * inside_obj.obj.material.roughness),
                );
            }
            diffuse_color = diffuse_color.div(DIFFUSE_SCATT as f32);
        }
        color = color.add(reflect_color);
        color = color.add(diffuse_color);
    } else {
        return (Color::from_u8(0x87, 0xce, 0xeb), f32::INFINITY);
    }

    if depth == 1 {
        (color.pow(1.0/2.2), closest)
    } else {
        (color, closest)
    }
}

fn draw(frame: &mut [u8], world: &World, _t: f32) {
    let aspect_ratio = WIDTH / HEIGHT;
    let origin = Vec3::new(0.0, 0.0, 0.0);

    let rows = (0..HEIGHT)
        .into_par_iter()
        .map(|y| {
            let mut row = [SCREEN_BLACK; WIDTH];
            for x in 0..WIDTH {
                let xx =
                    (2.0 * (x as f32 + 0.25) / (WIDTH as f32) - 1.0) * aspect_ratio as f32 * SCALE;
                let yy = (1.0 - 2.0 * (y as f32 + 0.25) / HEIGHT as f32) * SCALE;

                let ray = Ray {
                    origin,
                    dir: Vec3::new(xx, yy, 1.0),
                    inv_dir: Vec3::new(xx, yy, 1.0).recip(),
                };
                let (col, _) = trace_ray(&ray, world, 1);

                // Sample 2
                let xx =
                    (2.0 * (x as f32 + 0.75) / (WIDTH as f32) - 1.0) * aspect_ratio as f32 * SCALE;
                let yy = (1.0 - 2.0 * (y as f32 + 0.75) / HEIGHT as f32) * SCALE;
                let ray = Ray {
                    origin,
                    dir: Vec3::new(xx, yy, 1.0),
                    inv_dir: Vec3::new(xx, yy, 1.0).recip(),
                };
                let (col2, _) = trace_ray(&ray, world, 1);

                // Sample 3
                let xx =
                    (2.0 * (x as f32 + 0.25) / (WIDTH as f32) - 1.0) * aspect_ratio as f32 * SCALE;
                let yy = (1.0 - 2.0 * (y as f32 + 0.75) / HEIGHT as f32) * SCALE;
                let ray = Ray {
                    origin,
                    dir: Vec3::new(xx, yy, 1.0),
                    inv_dir: Vec3::new(xx, yy, 1.0).recip(),
                };
                let (col3, _) = trace_ray(&ray, world, 1);

                // Sample 4
                let xx =
                    (2.0 * (x as f32 + 0.75) / (WIDTH as f32) - 1.0) * aspect_ratio as f32 * SCALE;
                let yy = (1.0 - 2.0 * (y as f32 + 0.25) / HEIGHT as f32) * SCALE;
                let ray = Ray {
                    origin,
                    dir: Vec3::new(xx, yy, 1.0),
                    inv_dir: Vec3::new(xx, yy, 1.0).recip(),
                };
                let (col4, _) = trace_ray(&ray, world, 1);
                row[x] = ScreenColor::from(col.add(col2).add(col3).add(col4).div(4.0)); // AVG of 2 samples
            }
            row
        })
        .collect::<Vec<[ScreenColor; WIDTH]>>();

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
    // TODO: Use clap for CLI params
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

    let (teapot_tris, teapot_tri_data, teapot_verts) = read_obj(
        "teapot.obj",
        Vec3::new(-0.9, -2.0, 9.0),
        Color::from_u8(0x50, 0xc8, 0x78),
    );
    let (teapot2_tris, teapot2_tri_data, teapot2_verts) = read_obj(
        "teapot.obj",
        Vec3::new(3.0, -2.0, 6.0),
        Color::from_u8(0xFF, 0, 0),
    );
    let mut objects: Vec<Object> = Vec::new();
    objects.push(Object::from(
        teapot_tris,
        teapot_tri_data,
        teapot_verts,
        Material::new(1.0, 0.9, 0.1),
    ));
    objects.push(Object::from(
        teapot2_tris,
        teapot2_tri_data,
        teapot2_verts,
        Material::new(1.0, 0.025, 0.975),
    ));
    objects.push(Object::from(
        vec![Triangle { a: 0, b: 1, c: 2 }, Triangle { a: 0, b: 3, c: 2 }],
        vec![
            TriangleData {
                color: Color::from_u8(128, 128, 128),
                normal: Vec3::new(0.0, 1.0, 0.0),
            },
            TriangleData {
                color: Color::from_u8(128, 128, 128),
                normal: Vec3::new(0.0, 1.0, 0.0),
            },
        ],
        vec![
            Vec3::new(-10000.0, -3.0, -10000.0),
            Vec3::new(-10000.0, -3.0, 10000.0),
            Vec3::new(10000.0, -3.0, 10000.0),
            Vec3::new(10000.0, -3.0, -10000.0),
        ],
        Material::new(1.0, 0.0, 1.0),
    ));

    for obj in objects.iter_mut() {
        optimize_model(obj);
    }

    let objects: Vec<PackedObject> = objects.into_iter().map(|obj| pack_model(obj)).collect();

    println!(
        "len: {}",
        objects.iter().map(|obj| obj.obj.tris.len()).sum::<usize>()
    );
    let mut world = World {
        objects,
        lights: Vec::new(), // TODO
        sun: Vec3::new(0.0, 0.0, 0.0),
    };

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
                world.sun = sun;

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
