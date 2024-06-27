use std::{arch::x86_64::*, hint};

use glam::Vec3;

use crate::objects::{PackedTriangles, Ray, Triangle};

#[inline(always)]
pub fn pack_triangles(triangles: &[Triangle], verts: &[Vec3]) -> PackedTriangles {
    // Initialize arrays to hold 8 values for each coordinate component
    let mut e1_x = [0.0; 8];
    let mut e1_y = [0.0; 8];
    let mut e1_z = [0.0; 8];
    let mut e2_x = [0.0; 8];
    let mut e2_y = [0.0; 8];
    let mut e2_z = [0.0; 8];
    let mut v0_x = [0.0; 8];
    let mut v0_y = [0.0; 8];
    let mut v0_z = [0.0; 8];

    if triangles.len() != 8 {
        unsafe { hint::unreachable_unchecked() }
    }

    // Load triangle vertices into arrays
    for i in 0..8 {
        let e1 = verts[triangles[i].b as usize];
        e1_x[i] = e1[0];
        e1_y[i] = e1[1];
        e1_z[i] = e1[2];

        let e2 = verts[triangles[i].c as usize];
        e2_x[i] = e2[0];
        e2_y[i] = e2[1];
        e2_z[i] = e2[2];

        let v0 = verts[triangles[i].a as usize];
        v0_x[i] = v0[0];
        v0_y[i] = v0[1];
        v0_z[i] = v0[2];
    }
    unsafe {
        // Create __m256 vectors from the arrays
        let a = [
            _mm256_set_ps(
                e1_x[7], e1_x[6], e1_x[5], e1_x[4], e1_x[3], e1_x[2], e1_x[1], e1_x[0],
            ),
            _mm256_set_ps(
                e1_y[7], e1_y[6], e1_y[5], e1_y[4], e1_y[3], e1_y[2], e1_y[1], e1_y[0],
            ),
            _mm256_set_ps(
                e1_z[7], e1_z[6], e1_z[5], e1_z[4], e1_z[3], e1_z[2], e1_z[1], e1_z[0],
            ),
        ];
        let b = [
            _mm256_set_ps(
                e2_x[7], e2_x[6], e2_x[5], e2_x[4], e2_x[3], e2_x[2], e2_x[1], e2_x[0],
            ),
            _mm256_set_ps(
                e2_y[7], e2_y[6], e2_y[5], e2_y[4], e2_y[3], e2_y[2], e2_y[1], e2_y[0],
            ),
            _mm256_set_ps(
                e2_z[7], e2_z[6], e2_z[5], e2_z[4], e2_z[3], e2_z[2], e2_z[1], e2_z[0],
            ),
        ];
        let v0 = [
            _mm256_set_ps(
                v0_x[7], v0_x[6], v0_x[5], v0_x[4], v0_x[3], v0_x[2], v0_x[1], v0_x[0],
            ),
            _mm256_set_ps(
                v0_y[7], v0_y[6], v0_y[5], v0_y[4], v0_y[3], v0_y[2], v0_y[1], v0_y[0],
            ),
            _mm256_set_ps(
                v0_z[7], v0_z[6], v0_z[5], v0_z[4], v0_z[3], v0_z[2], v0_z[1], v0_z[0],
            ),
        ];

        let mut e1 = [_mm256_undefined_ps(); 3];
        let mut e2 = [_mm256_undefined_ps(); 3];
        avx_multi_sub(&mut e1, a, v0);
        avx_multi_sub(&mut e2, b, v0);

        /*let mask = if count != 8 {
            // Create the mask, setting the lower bits according to the number of triangles
            u8_mask_to_m256(((1u16 << count) - 1) as u8 ^ 0xFF)
        } else {
            _mm256_castsi256_ps(_mm256_set1_epi8(0)) // 1's represent masked out
        };*/

        PackedTriangles { e1, e2, v0 }
    }
}

#[repr(C)]
union C {
    a: __m256i,
    c: [i8; 32],
}

const BITMASK: __m256i = unsafe {
    C {
        c: [
            1, 1, 1, 1, 2, 2, 2, 2, 4, 4, 4, 4, 8, 8, 8, 8, 16, 16, 16, 16, 32, 32, 32, 32, 64, 64,
            64, 64, -128, -128, -128, -128,
        ],
    }
    .a
};

#[inline(always)]
fn u8_mask_to_m256(value: u8) -> __m256 {
    unsafe {
        // Create a __m256i register with the value repeated across all elements
        let broadcast_value = _mm256_set1_epi8(value as i8);

        // Use a comparison to generate the mask
        let mask_i = _mm256_cmpeq_epi8(_mm256_and_si256(broadcast_value, BITMASK), BITMASK);

        // Cast the __m256i register to a __m256 register
        _mm256_castsi256_ps(mask_i)
    }
}

#[inline(always)]
fn one_m256() -> __m256 {
    unsafe { _mm256_set1_ps(1.0) }
}

#[inline(always)]
fn minus_one_m256() -> __m256 {
    unsafe { _mm256_set1_ps(-1.0) }
}

#[inline(always)]
fn positive_epsilon_m256() -> __m256 {
    unsafe { _mm256_set1_ps(1e-6) }
}

#[inline(always)]
fn negative_epsilon_m256() -> __m256 {
    unsafe { _mm256_set1_ps(-1e-6) }
}

#[inline(always)]
fn zero_m256() -> __m256 {
    unsafe { _mm256_set1_ps(0.0) }
}

#[inline(always)]
fn avx_multi_cross(result: &mut [__m256; 3], a: [__m256; 3], b: [__m256; 3]) {
    unsafe {
        result[0] = _mm256_fmsub_ps(a[1], b[2], _mm256_mul_ps(b[1], a[2]));
        result[1] = _mm256_fmsub_ps(a[2], b[0], _mm256_mul_ps(b[2], a[0]));
        result[2] = _mm256_fmsub_ps(a[0], b[1], _mm256_mul_ps(b[0], a[1]));
    }
}

#[inline(always)]
fn avx_multi_dot(a: [__m256; 3], b: [__m256; 3]) -> __m256 {
    unsafe {
        _mm256_fmadd_ps(
            a[2],
            b[2],
            _mm256_fmadd_ps(a[1], b[1], _mm256_mul_ps(a[0], b[0])),
        )
    }
}

#[inline(always)]
fn avx_multi_sub(result: &mut [__m256; 3], a: [__m256; 3], b: [__m256; 3]) {
    unsafe {
        result[0] = _mm256_sub_ps(a[0], b[0]);
        result[1] = _mm256_sub_ps(a[1], b[1]);
        result[2] = _mm256_sub_ps(a[2], b[2]);
    }
}

pub fn extract_f32_from_m256(m: __m256) -> [f32; 8] {
    let mut result: [f32; 8] = [0.0; 8];
    unsafe {
        _mm256_storeu_ps(result.as_mut_ptr(), m);
    }
    result
}

pub fn ray_to_avx(ray: &Ray) -> ([__m256; 3], [__m256; 3]) {
    unsafe {
        let origin_x = _mm256_set1_ps(ray.origin.x);
        let origin_y = _mm256_set1_ps(ray.origin.y);
        let origin_z = _mm256_set1_ps(ray.origin.z);

        let direction_x = _mm256_set1_ps(ray.dir.x);
        let direction_y = _mm256_set1_ps(ray.dir.y);
        let direction_z = _mm256_set1_ps(ray.dir.z);

        (
            [origin_x, origin_y, origin_z],
            [direction_x, direction_y, direction_z],
        )
    }
}

impl PackedTriangles {
    pub fn intersect(
        &self,
        ray_origin: [__m256; 3],
        ray_direction: [__m256; 3],
        ray_length: __m256,
    ) -> (__m256, i32) {
        unsafe {
            let mut q = [_mm256_undefined_ps(); 3];
            avx_multi_cross(&mut q, ray_direction, self.e2);

            let a = avx_multi_dot(self.e1, q);

            let f = _mm256_div_ps(one_m256(), a);

            let mut s = [_mm256_undefined_ps(); 3];
            avx_multi_sub(&mut s, ray_origin, self.v0);

            let u = _mm256_mul_ps(f, avx_multi_dot(s, q));

            let mut r = [_mm256_undefined_ps(); 3];
            avx_multi_cross(&mut r, s, self.e1);

            let v = _mm256_mul_ps(f, avx_multi_dot(ray_direction, r));

            let t = _mm256_mul_ps(f, avx_multi_dot(self.e2, r));

            // Failure conditions
            let mut failed = _mm256_and_ps(
                _mm256_cmp_ps(a, negative_epsilon_m256(), _CMP_GT_OQ),
                _mm256_cmp_ps(a, positive_epsilon_m256(), _CMP_LT_OQ),
            );

            failed = _mm256_or_ps(failed, _mm256_cmp_ps(u, zero_m256(), _CMP_LT_OQ));

            failed = _mm256_or_ps(failed, _mm256_cmp_ps(v, zero_m256(), _CMP_LT_OQ));

            let u_plus_v = _mm256_add_ps(u, v);
            failed = _mm256_or_ps(failed, _mm256_cmp_ps(u_plus_v, one_m256(), _CMP_GT_OQ));

            failed = _mm256_or_ps(failed, _mm256_cmp_ps(t, zero_m256(), _CMP_LT_OQ));

            failed = _mm256_or_ps(failed, _mm256_cmp_ps(t, ray_length, _CMP_GT_OQ));

            //failed = _mm256_or_ps(failed, self.mask);

            let t_results = _mm256_blendv_ps(t, minus_one_m256(), failed);

            (t_results, _mm256_movemask_ps(t_results))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn almost_equal(a: f32, b: f32, epsilon: f32) -> bool {
        (a - b).abs() < epsilon
    }

    fn extract_m256_element(m256: __m256, index: usize) -> f32 {
        unsafe {
            let arr: [f32; 8] = std::mem::transmute(m256);
            arr[index]
        }
    }

    #[test]
    fn test_pack_triangles() {
        let epsilon = 1e-6;
        let verts = vec![
            Vec3::new(1.0, 1.1, 1.2),
            Vec3::new(2.0, 2.1, 2.2),
            Vec3::new(3.0, 3.1, 3.2),
            Vec3::new(4.0, 4.1, 4.2),
            Vec3::new(5.0, 5.1, 5.2),
            Vec3::new(6.0, 6.1, 6.2),
        ];
        let triangles = vec![
            Triangle { a: 0, b: 1, c: 2 },
            Triangle { a: 0, b: 1, c: 2 },
            // Add more triangles if necessary
        ];

        let packet = pack_triangles(&triangles, &verts);

        // Check the mask
        /*unsafe {
            assert_eq!(
                _mm256_testz_ps(packet.mask, u8_mask_to_m256((1u8 << triangles.len()) - 1)),
                0xFFFF
            );
        }*/

        // Check the packed coordinates
        for i in 0..triangles.len() {
            let a = verts[triangles[i].a as usize].to_array();
            let b = verts[triangles[i].b as usize].to_array();
            let c = verts[triangles[i].c as usize].to_array();

            assert!(almost_equal(
                extract_m256_element(packet.e1[0], i),
                a[0],
                epsilon
            ));
            assert!(almost_equal(
                extract_m256_element(packet.e1[1], i),
                a[1],
                epsilon
            ));
            assert!(almost_equal(
                extract_m256_element(packet.e1[2], i),
                a[2],
                epsilon
            ));

            assert!(almost_equal(
                extract_m256_element(packet.e2[0], i),
                b[0],
                epsilon
            ));
            assert!(almost_equal(
                extract_m256_element(packet.e2[1], i),
                b[1],
                epsilon
            ));
            assert!(almost_equal(
                extract_m256_element(packet.e2[2], i),
                b[2],
                epsilon
            ));

            assert!(almost_equal(
                extract_m256_element(packet.v0[0], i),
                c[0],
                epsilon
            ));
            assert!(almost_equal(
                extract_m256_element(packet.v0[1], i),
                c[1],
                epsilon
            ));
            assert!(almost_equal(
                extract_m256_element(packet.v0[2], i),
                c[2],
                epsilon
            ));
        }
    }
}
