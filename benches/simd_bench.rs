use criterion::{black_box, criterion_group, criterion_main, Criterion};
use glam::Vec3A;
use rays::{pack_triangles, Color, Triangle};

fn benchmark_pack_triangles(c: &mut Criterion) {
    let triangles = vec![
        Triangle {
            a: Vec3A::new(1.0, 1.1, 1.2),
            b: Vec3A::new(2.0, 2.1, 2.2),
            c: Vec3A::new(3.0, 3.1, 3.2),
            normal: Vec3A::new(0.0, 0.0, 1.0),
            color: Color::from_u8(255, 0, 0),
        },
        Triangle {
            a: Vec3A::new(4.0, 4.1, 4.2),
            b: Vec3A::new(5.0, 5.1, 5.2),
            c: Vec3A::new(6.0, 6.1, 6.2),
            normal: Vec3A::new(0.0, 0.0, 1.0),
            color: Color::from_u8(0, 255, 0),
        },
        Triangle {
            a: Vec3A::new(7.0, 7.1, 7.2),
            b: Vec3A::new(8.0, 8.1, 8.2),
            c: Vec3A::new(9.0, 9.1, 9.2),
            normal: Vec3A::new(0.0, 0.0, 1.0),
            color: Color::from_u8(0, 0, 255),
        },
        Triangle {
            a: Vec3A::new(10.0, 10.1, 10.2),
            b: Vec3A::new(11.0, 11.1, 11.2),
            c: Vec3A::new(12.0, 12.1, 12.2),
            normal: Vec3A::new(0.0, 0.0, 1.0),
            color: Color::from_u8(255, 255, 0),
        },
        Triangle {
            a: Vec3A::new(13.0, 13.1, 13.2),
            b: Vec3A::new(14.0, 14.1, 14.2),
            c: Vec3A::new(15.0, 15.1, 15.2),
            normal: Vec3A::new(0.0, 0.0, 1.0),
            color: Color::from_u8(0, 255, 255),
        },
        Triangle {
            a: Vec3A::new(16.0, 16.1, 16.2),
            b: Vec3A::new(17.0, 17.1, 17.2),
            c: Vec3A::new(18.0, 18.1, 18.2),
            normal: Vec3A::new(0.0, 0.0, 1.0),
            color: Color::from_u8(255, 0, 255),
        },
        Triangle {
            a: Vec3A::new(19.0, 19.1, 19.2),
            b: Vec3A::new(20.0, 20.1, 20.2),
            c: Vec3A::new(21.0, 21.1, 21.2),
            normal: Vec3A::new(0.0, 0.0, 1.0),
            color: Color::from_u8(255, 255, 255),
        },
        Triangle {
            a: Vec3A::new(22.0, 22.1, 22.2),
            b: Vec3A::new(23.0, 23.1, 23.2),
            c: Vec3A::new(24.0, 24.1, 24.2),
            normal: Vec3A::new(0.0, 0.0, 1.0),
            color: Color::from_u8(0, 0, 0),
        },
    ];

    c.bench_function("pack_triangles", |b| {
        b.iter(|| pack_triangles(black_box(&triangles.iter().collect::<Vec<&Triangle>>())))
    });
}

criterion_group!(benches, benchmark_pack_triangles);
criterion_main!(benches);
