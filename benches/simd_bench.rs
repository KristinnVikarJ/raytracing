use std::{thread::sleep, time::Duration};

use criterion::{black_box, criterion_group, criterion_main, Criterion};
use glam::Vec3;
use rays::{pack_triangles, Triangle};

fn benchmark_pack_triangles(c: &mut Criterion) {
    let verts = vec![
        Vec3::new(1.0, 1.1, 1.2),
        Vec3::new(2.0, 2.1, 2.2),
        Vec3::new(3.0, 3.1, 3.2),
        Vec3::new(4.0, 4.1, 4.2),
        Vec3::new(5.0, 5.1, 5.2),
        Vec3::new(6.0, 6.1, 6.2),
        Vec3::new(7.0, 7.1, 7.2),
        Vec3::new(8.0, 8.1, 8.2),
        Vec3::new(9.0, 9.1, 9.2),
        Vec3::new(10.0, 10.1, 10.2),
        Vec3::new(11.0, 11.1, 11.2),
        Vec3::new(12.0, 12.1, 12.2),
        Vec3::new(13.0, 13.1, 13.2),
        Vec3::new(14.0, 14.1, 14.2),
        Vec3::new(15.0, 15.1, 15.2),
        Vec3::new(16.0, 16.1, 16.2),
        Vec3::new(17.0, 17.1, 17.2),
        Vec3::new(18.0, 18.1, 18.2),
        Vec3::new(19.0, 19.1, 19.2),
        Vec3::new(20.0, 20.1, 20.2),
        Vec3::new(21.0, 21.1, 21.2),
        Vec3::new(22.0, 22.1, 22.2),
        Vec3::new(23.0, 23.1, 23.2),
        Vec3::new(24.0, 24.1, 24.2),
    ];
    let triangles = vec![
        Triangle { a: 0, b: 1, c: 2 },
        Triangle { a: 3, b: 4, c: 5 },
        Triangle { a: 6, b: 7, c: 8 },
        Triangle { a: 9, b: 10, c: 11 },
        Triangle {
            a: 12,
            b: 13,
            c: 14,
        },
        Triangle {
            a: 15,
            b: 16,
            c: 17,
        },
        Triangle {
            a: 18,
            b: 19,
            c: 20,
        },
        Triangle {
            a: 21,
            b: 22,
            c: 23,
        },
    ];

    sleep(Duration::from_secs(30)); // let cpu cool down after compile
    c.bench_function("pack_triangles", |b| {
        b.iter(|| pack_triangles(black_box(&triangles), black_box(&verts)))
    });
}

criterion_group!(benches, benchmark_pack_triangles);
criterion_main!(benches);
