use glam::Vec3A as Vec3;

pub trait Hittable {
    fn ray_hits(&self, ray: &Ray, max_dist: f32) -> Option<Hit>;
    fn get_color(&self) -> &Color;
}

pub struct Ray {
    pub origin: Vec3,
    pub dir: Vec3,
    //pub inv_dir: Vec3,
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
    pub r: u8,
    pub g: u8,
    pub b: u8,
}

impl Color {
    pub fn new(r: u8, g: u8, b: u8) -> Self {
        Self { r, g, b }
    }
}

#[derive(Debug)]
pub struct Triangle {
    pub a: Vec3,
    pub b: Vec3,
    pub c: Vec3,
    pub color: Color,
}

pub struct Sphere {
    pub center: Vec3,
    pub radius: f32,
    pub color: Color,
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
            Some(Hit { pos: ray.at(t - 0.01), t })
        } else {
            // This means that there is a line intersection but not a ray intersection.
            None
        }
    }
    fn get_color(&self) -> &Color {
        &self.color
    }
}
