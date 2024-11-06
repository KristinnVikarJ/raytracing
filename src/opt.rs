use core::f32;

use glam::Vec3;

use crate::{
    objects::{BoxShape, PackedObject},
    simd_accel::{pack_boxes, pack_triangles},
    Object, Triangle,
};

pub fn optimize_model(obj: &mut Object) {
    let mut combined: Vec<(usize, Triangle)> = obj.tris.iter().cloned().enumerate().collect();

    combined.sort_by(|a, b| {
        let amin = a.1.a.min(a.1.b).min(a.1.c);
        let bmin = b.1.a.min(b.1.b).min(b.1.c);
        if amin == bmin {
            let amax = a.1.a.max(a.1.b).max(a.1.c);
            let bmax = b.1.a.max(b.1.b).max(b.1.c);
            bmax.cmp(&amax)
        } else {
            amin.cmp(&bmin)
        }
    });

    let mut tris = Vec::new();
    let mut tridata = Vec::new();

    for (idx, tri) in combined {
        tris.push(tri);
        tridata.push(obj.tri_data[idx].clone())
    }

    obj.tris = tris;
    obj.tri_data = tridata;
}

pub fn pack_model(obj: Object) -> PackedObject {
    let mut packed_tris = Vec::new();
    let mut tri_bounds = Vec::new();
    let mut packed_tri_bounds = Vec::new();

    for k in 0..obj.tris.len() / 8 {
        let packed = pack_triangles(&obj.tris[k * 8..(k + 1) * 8], &obj.verts);

        // Calculate AABB for this batch of packed triangles
        let mut min = Vec3::new(f32::INFINITY, f32::INFINITY, f32::INFINITY);
        let mut max = Vec3::new(f32::NEG_INFINITY, f32::NEG_INFINITY, f32::NEG_INFINITY);
        for tri in obj.tris[k * 8..(k + 1) * 8].iter() {
            let tri_min = obj.verts[tri.a as usize]
                .min(obj.verts[tri.b as usize])
                .min(obj.verts[tri.c as usize]);
            let tri_max = obj.verts[tri.a as usize]
                .max(obj.verts[tri.b as usize])
                .max(obj.verts[tri.c as usize]);
            min = min.min(tri_min);
            max = max.max(tri_max);
        }

        tri_bounds.push(BoxShape { min, max });
        packed_tris.push(packed);
    }

    for k in 0..tri_bounds.len() / 8 {
        let packed = pack_boxes(&tri_bounds[k * 8..(k + 1) * 8]);
        packed_tri_bounds.push(packed);
    }

    let mut rest_tri = Vec::new();
    let mut rest_bounds = Vec::new();
    rest_tri.extend_from_slice(&obj.tris[(obj.tris.len() / 8) * 8..obj.tris.len()]);
    rest_bounds.extend_from_slice(&tri_bounds[(tri_bounds.len() / 8) * 8..tri_bounds.len()]);
    PackedObject {
        obj,
        packed_tris,
        rest_tri,
        rest_bounds,
        packed_tri_bounds,
    }
}

#[cfg(test)]
mod tests {
    use glam::Vec3;

    use crate::{
        objects::{Material, BLACK},
        read_obj,
    };

    use super::*;

    #[test]
    fn validate() {
        let (teapot_tris, teapot_tri_data, teapot_verts) =
            read_obj("teapot.obj", Vec3::splat(0.0), BLACK);

        let mut obj = Object::from(
            teapot_tris,
            teapot_tri_data,
            teapot_verts,
            Material::new(1.0, 0.9, 0.1),
        );

        optimize_model(&mut obj);
    }
}
