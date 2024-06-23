use std::arch::x86_64::*;

use crate::objects::{PackedTriangles, Ray, Triangle};

pub fn pack_triangles(triangles: &[&Triangle]) -> PackedTriangles {
    // Initialize arrays to hold 8 values for each coordinate component
    let mut a_x = [0.0; 8];
    let mut a_y = [0.0; 8];
    let mut a_z = [0.0; 8];
    let mut b_x = [0.0; 8];
    let mut b_y = [0.0; 8];
    let mut b_z = [0.0; 8];
    let mut c_x = [0.0; 8];
    let mut c_y = [0.0; 8];
    let mut c_z = [0.0; 8];

    let count = triangles.len().min(8);

    // Load triangle vertices into arrays
    for i in 0..count {
        /*
        let a = triangles[i].a.to_array();
        let b = triangles[i].b.to_array();
        let c = triangles[i].c.to_array();
        */
        let a = triangles[i].b - triangles[i].a;
        a_x[i] = a[0];
        a_y[i] = a[1];
        a_z[i] = a[2];

        let b = triangles[i].c - triangles[i].a;
        b_x[i] = b[0];
        b_y[i] = b[1];
        b_z[i] = b[2];

        let c = triangles[i].a;
        c_x[i] = c[0];
        c_y[i] = c[1];
        c_z[i] = c[2];
    }

    // Create __m256 vectors from the arrays
    let a = [
        unsafe {
            _mm256_set_ps(
                a_x[7], a_x[6], a_x[5], a_x[4], a_x[3], a_x[2], a_x[1], a_x[0],
            )
        },
        unsafe {
            _mm256_set_ps(
                a_y[7], a_y[6], a_y[5], a_y[4], a_y[3], a_y[2], a_y[1], a_y[0],
            )
        },
        unsafe {
            _mm256_set_ps(
                a_z[7], a_z[6], a_z[5], a_z[4], a_z[3], a_z[2], a_z[1], a_z[0],
            )
        },
    ];
    let b = [
        unsafe {
            _mm256_set_ps(
                b_x[7], b_x[6], b_x[5], b_x[4], b_x[3], b_x[2], b_x[1], b_x[0],
            )
        },
        unsafe {
            _mm256_set_ps(
                b_y[7], b_y[6], b_y[5], b_y[4], b_y[3], b_y[2], b_y[1], b_y[0],
            )
        },
        unsafe {
            _mm256_set_ps(
                b_z[7], b_z[6], b_z[5], b_z[4], b_z[3], b_z[2], b_z[1], b_z[0],
            )
        },
    ];
    let c = [
        unsafe {
            _mm256_set_ps(
                c_x[7], c_x[6], c_x[5], c_x[4], c_x[3], c_x[2], c_x[1], c_x[0],
            )
        },
        unsafe {
            _mm256_set_ps(
                c_y[7], c_y[6], c_y[5], c_y[4], c_y[3], c_y[2], c_y[1], c_y[0],
            )
        },
        unsafe {
            _mm256_set_ps(
                c_z[7], c_z[6], c_z[5], c_z[4], c_z[3], c_z[2], c_z[1], c_z[0],
            )
        },
    ];
    let mask = if count != 8 {
        // Create the mask, setting the lower bits according to the number of triangles
        u8_mask_to_m256(((1u16 << count) - 1) as u8 ^ 0xFF)
    } else {
        unsafe { _mm256_castsi256_ps(_mm256_set1_epi8(0)) } // 1's represent masked out
    };

    PackedTriangles {
        e1: a,
        e2: b,
        v0: c,
        mask,
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

            failed = _mm256_or_ps(failed, self.mask);

            let t_results = _mm256_blendv_ps(t, minus_one_m256(), failed);

            (t_results, _mm256_movemask_ps(t_results))
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::objects::Color;

    use super::*;
    use glam::Vec3A;

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
            // Add more triangles if necessary
        ];

        let packet = pack_triangles(&triangles.iter().collect::<Vec<&Triangle>>());

        unsafe {
            assert_eq!(
                _mm256_testz_ps(packet.mask, u8_mask_to_m256((1u8 << triangles.len()) - 1)),
                0xFFFF
            );
        }
        // Check the mask

        // Check the packed coordinates
        for i in 0..triangles.len() {
            let a = triangles[i].a.to_array();
            let b = triangles[i].b.to_array();
            let c = triangles[i].c.to_array();

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
