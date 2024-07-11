use crate::{Object, Triangle};

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


#[cfg(test)]
mod tests {
    use glam::Vec3;

    use crate::{objects::{Material, BLACK}, read_obj};

    use super::*;

    #[test]
    fn validate() {
        let (teapot_tris, teapot_tri_data, teapot_verts) = read_obj("teapot.obj", Vec3::splat(0.0), BLACK);

        let mut obj = Object::from(
            teapot_tris,
            teapot_tri_data,
            teapot_verts,
            Material::new(1.0, 0.9),
        );

        optimize_model(&mut obj);
    }
}
