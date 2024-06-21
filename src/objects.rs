use glam::Vec3A as Vec3;

pub trait Hittable {
    fn ray_hits(&self, ray: &Ray, max_dist: f32) -> Option<Hit>;
    fn get_color(&self) -> &Color;
}

pub struct World {
    pub objects: Vec<Object>,
    pub sun: Vec3,
}

#[derive(Clone)]
pub struct Object {
    pub tris: Vec<Triangle>,
    pub bounding_box: BoxShape,
    pub material: Material,
}

impl Object {
    pub fn from(tris: Vec<Triangle>, mat: Material) -> Self {
        let mut min = tris[0].a;
        let mut max = tris[0].a;
        for t in tris.iter() {
            min = min.min(t.a).min(t.b).min(t.c);
            max = max.max(t.a).max(t.b).max(t.c);
        }
        Object {
            tris,
            bounding_box: BoxShape { min, max },
            material: mat,
        }
    }
}

pub struct Ray {
    pub origin: Vec3,
    pub dir: Vec3,
    pub inv_dir: Vec3,
}

pub struct Hit {
    pub pos: Vec3,
    pub t: f32,
}

impl Ray {
    pub fn at(&self, t: f32) -> Vec3 {
        self.origin + self.dir * t
    }
}

#[derive(Debug, Clone)]
pub struct Color {
    pub r: f32,
    pub g: f32,
    pub b: f32,
}

impl Color {
    pub fn new(r: f32, g: f32, b: f32) -> Self {
        Self { r, g, b }
    }

    #[inline(always)]
    pub fn from_u8(r: u8, g: u8, b: u8) -> Self {
        Self {
            r: (r as f32) / 255.0,
            g: (g as f32) / 255.0,
            b: (b as f32) / 255.0,
        }
    }

    #[inline(always)]
    pub fn mul(&self, val: f32) -> Self {
        Color {
            r: self.r * val,
            g: self.g * val,
            b: self.b * val,
        }
    }

    #[inline(always)]
    pub fn add(&self, rhs: Color) -> Self {
        Color {
            r: self.r + rhs.r,
            g: self.g + rhs.g,
            b: self.b + rhs.b,
        }
    }
}
pub const BLACK: Color = Color {
    r: 0.0,
    g: 0.0,
    b: 0.0,
};

#[derive(Clone)]
pub struct Material {
    pub albedo: f32,
    pub reflectivity: f32,
}

impl Material {
    pub fn new(albedo: f32, reflectivity: f32) -> Self {
        Material {
            albedo: albedo / std::f32::consts::PI,
            reflectivity,
        }
    }
}

#[derive(Debug, Clone)]
pub struct Triangle {
    pub a: Vec3,
    pub b: Vec3,
    pub c: Vec3,
    pub normal: Vec3,
    pub color: Color,
}

#[derive(Clone)]
pub struct Sphere {
    pub center: Vec3,
    pub radius: f32,
    pub color: Color,
}

#[derive(Clone)]
pub struct BoxShape {
    pub min: Vec3,
    pub max: Vec3,
}

impl Hittable for Sphere {
    fn ray_hits(&self, ray: &Ray, max_dist: f32) -> Option<Hit> {
        let oc = ray.origin - self.center;
        let a = ray.dir.length_squared();
        let half_b = oc.dot(ray.dir);
        let c = oc.length_squared() - self.radius * self.radius;
        let discriminant = (half_b * half_b) - (a * c);

        if discriminant >= 0.0 {
            let sqrtd = discriminant.sqrt();
            let root_a = ((-half_b) - sqrtd) / a;
            let root_b = ((-half_b) + sqrtd) / a;

            // Optimization: root_a is always closer? (not sure if 100% true)
            // If weird behavior revert to in [root_a, root_b]

            for root in [root_a].into_iter() {
                if root < max_dist {
                    return Some(Hit {
                        pos: ray.at(root),
                        t: root,
                    });
                }
            }
        }
        None
    }
    fn get_color(&self) -> &Color {
        &self.color
    }
}

impl Hittable for Triangle {
    fn ray_hits(&self, ray: &Ray, _: f32) -> Option<Hit> {
        let e1 = self.b - self.a;
        let e2 = self.c - self.a;

        let ray_cross_e2 = ray.dir.cross(e2);
        let det = e1.dot(ray_cross_e2);

        if det > -f32::EPSILON && det < f32::EPSILON {
            return None;
        }

        let inv_det = 1.0 / det;
        let s = ray.origin - self.a;
        let u = inv_det * s.dot(ray_cross_e2);
        if !(0.0..=1.0).contains(&u) {
            return None;
        }

        let s_cross_e1 = s.cross(e1);
        let v = inv_det * ray.dir.dot(s_cross_e1);
        if v < 0.0 || u + v > 1.0 {
            return None;
        }
        // At this stage we can compute t to find out where the intersection point is on the line.
        let t = inv_det * e2.dot(s_cross_e1);

        if t > f32::EPSILON {
            // ray intersection
            Some(Hit {
                pos: ray.at(t) + (self.normal * 0.00001),
                t,
            })
        } else {
            // This means that there is a line intersection but not a ray intersection.
            None
        }
    }
    fn get_color(&self) -> &Color {
        &self.color
    }
}

pub fn is_inside_box(ray: &Ray, cb: &BoxShape) -> bool {
    let o = ray.origin;

    o.x <= cb.max.x
        && o.x >= cb.min.x
        && o.x <= cb.max.y
        && o.y >= cb.min.y
        && o.x <= cb.max.z
        && o.z >= cb.min.z
}

pub fn box_intersection_check(ray: &Ray, check_box: &BoxShape) -> bool {
    let t1 = (check_box.min - ray.origin) * ray.inv_dir;
    let t2 = (check_box.max - ray.origin) * ray.inv_dir;

    let tmin = t1.min(t2);
    let tmax = t1.max(t2);

    let t_near = tmin.x.max(tmin.y).max(tmin.z);
    let t_far = tmax.x.min(tmax.y).min(tmax.z);

    t_near.min(0.0) <= t_far
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn it_works() {
        let res = box_intersection_check(
            &Ray {
                origin: Vec3::new(0.0, 0.0, -2.0),
                dir: Vec3::new(0.0, 0.0, 1.0),
                inv_dir: Vec3::new(0.0, 0.0, 1.0).recip(),
            },
            &BoxShape {
                min: Vec3::new(-1.0, -1.0, -1.0),
                max: Vec3::new(1.0, 1.0, 1.0),
            },
        );
        assert!(res == true);
    }
}
