use std::{arch::x86_64::*, hint, io::{stdout, Write}};

use glam::Vec3;

use crate::objects::{BoxShape, PackedBoxes, PackedTriangles, Ray, Triangle};

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
        let mut a = [
            _mm256_loadu_ps(e1_x.as_ptr()),
            _mm256_loadu_ps(e1_y.as_ptr()),
            _mm256_loadu_ps(e1_z.as_ptr()),
        ];

        let mut b = [
            _mm256_loadu_ps(e2_x.as_ptr()),
            _mm256_loadu_ps(e2_y.as_ptr()),
            _mm256_loadu_ps(e2_z.as_ptr()),
        ];

        let v0 = [
            _mm256_loadu_ps(v0_x.as_ptr()),
            _mm256_loadu_ps(v0_y.as_ptr()),
            _mm256_loadu_ps(v0_z.as_ptr()),
        ];

        inplace_avx_multi_sub(&mut a, v0);
        inplace_avx_multi_sub(&mut b, v0);

        PackedTriangles { e1: a, e2: b, v0 }
    }
}

#[inline(always)]
pub fn pack_boxes(boxes: &[BoxShape]) -> PackedBoxes {
    // Initialize arrays to hold 8 values for each coordinate component
    let mut min_x = [0.0; 8];
    let mut min_y = [0.0; 8];
    let mut min_z = [0.0; 8];
    let mut max_x = [0.0; 8];
    let mut max_y = [0.0; 8];
    let mut max_z = [0.0; 8];

    if boxes.len() != 8 {
        unsafe { hint::unreachable_unchecked() }
    }

    for i in 0..8 {
        let min = boxes[i].min;
        min_x[i] = min[0];
        min_y[i] = min[1];
        min_z[i] = min[2];

        let max = boxes[i].max;
        max_x[i] = max[0];
        max_y[i] = max[1];
        max_z[i] = max[2];
    }
    unsafe {
        // Create __m256 vectors from the arrays
        let min = [
            _mm256_loadu_ps(min_x.as_ptr()),
            _mm256_loadu_ps(min_y.as_ptr()),
            _mm256_loadu_ps(min_z.as_ptr()),
        ];

        let max = [
            _mm256_loadu_ps(max_x.as_ptr()),
            _mm256_loadu_ps(max_y.as_ptr()),
            _mm256_loadu_ps(max_z.as_ptr()),
        ];
        PackedBoxes { min, max }
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
    unsafe { _mm256_set1_ps(f32::EPSILON) }
}

#[inline(always)]
fn negative_epsilon_m256() -> __m256 {
    unsafe { _mm256_set1_ps(-f32::EPSILON) }
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

#[inline(always)]
fn inplace_avx_multi_sub(a: &mut [__m256; 3], b: [__m256; 3]) {
    unsafe {
        a[0] = _mm256_sub_ps(a[0], b[0]);
        a[1] = _mm256_sub_ps(a[1], b[1]);
        a[2] = _mm256_sub_ps(a[2], b[2]);
    }
}

#[inline(always)]
fn avx_reduce_min(a: [__m256; 3]) -> __m256 {
    unsafe {
        _mm256_min_ps(a[0], _mm256_min_ps(a[1], a[2]))
    }
}

#[inline(always)]
fn avx_reduce_max(a: [__m256; 3]) -> __m256 {
    unsafe {
        _mm256_max_ps(a[0], _mm256_max_ps(a[1], a[2]))
    }
}

#[inline(always)]
fn inplace_avx_multi_max(a: &mut [__m256; 3], b: [__m256; 3]) {
    unsafe {
        a[0] = _mm256_max_ps(a[0], b[0]);
        a[1] = _mm256_max_ps(a[1], b[1]);
        a[2] = _mm256_max_ps(a[2], b[2]);
    }
}
#[inline(always)]
fn avx_multi_max(a: [__m256; 3], b: [__m256; 3]) -> [__m256; 3] {
    unsafe {
        [
            _mm256_max_ps(a[0], b[0]),
            _mm256_max_ps(a[1], b[1]),
            _mm256_max_ps(a[2], b[2])
        ]
    }
}

#[inline(always)]
fn avx_multi_min(a: [__m256; 3], b: [__m256; 3]) -> [__m256; 3] {
    unsafe {
        [
            _mm256_min_ps(a[0], b[0]),
            _mm256_min_ps(a[1], b[1]),
            _mm256_min_ps(a[2], b[2])
        ]
    }
}

/// AVX Component-wise multiplication
#[inline(always)]
#[allow(unused)]
fn avx_multi_mul(result: &mut [__m256; 3], a: [__m256; 3], b: [__m256; 3]) {
    unsafe {
        result[0] = _mm256_mul_ps(a[0], b[0]);
        result[1] = _mm256_mul_ps(a[1], b[1]);
        result[2] = _mm256_mul_ps(a[2], b[2]);
    }
}

/// AVX in-place component-wise multiplication
#[inline(always)]
fn inplace_avx_multi_mul(a: &mut [__m256; 3], b: [__m256; 3]) {
    unsafe {
        a[0] = _mm256_mul_ps(a[0], b[0]);
        a[1] = _mm256_mul_ps(a[1], b[1]);
        a[2] = _mm256_mul_ps(a[2], b[2]);
    }
}

#[inline(always)]
pub fn extract_f32_from_m256(m: __m256) -> [f32; 8] {
    let mut result: [f32; 8] = [0.0; 8];
    unsafe {
        _mm256_storeu_ps(result.as_mut_ptr(), m);
    }
    result
}

pub struct SimdRay {
    pub origin:  [__m256; 3],
    pub dir: [__m256; 3],
    pub inv_dir: [__m256; 3],
    pub inv_sign_mask: [bool; 3], // sign bit mask for inv_dir
}

pub fn ray_to_avx(ray: &Ray) -> SimdRay {
    unsafe {
        let origin_x = _mm256_set1_ps(ray.origin.x);
        let origin_y = _mm256_set1_ps(ray.origin.y);
        let origin_z = _mm256_set1_ps(ray.origin.z);

        let direction_x = _mm256_set1_ps(ray.dir.x);
        let direction_y = _mm256_set1_ps(ray.dir.y);
        let direction_z = _mm256_set1_ps(ray.dir.z);

        let inv_dir_x = _mm256_set1_ps(ray.inv_dir.x);
        let inv_dir_y = _mm256_set1_ps(ray.inv_dir.y);
        let inv_dir_z = _mm256_set1_ps(ray.inv_dir.z);

        let inv_sign_x = ray.inv_dir.x.signum() == 1.0;
        let inv_sign_y = ray.inv_dir.y.signum() == 1.0;
        let inv_sign_z = ray.inv_dir.z.signum() == 1.0;

        SimdRay {
            origin: [origin_x, origin_y, origin_z],
            dir: [direction_x, direction_y, direction_z],
            inv_dir: [inv_dir_x, inv_dir_y, inv_dir_z],
            inv_sign_mask: [inv_sign_x, inv_sign_y, inv_sign_z],
        }
    }
}

impl PackedTriangles {
    pub fn intersect(
        &self,
        simd_ray: &SimdRay,
        ray_length: __m256,
    ) -> (__m256, i32) {
        unsafe {
            let mut q = [_mm256_undefined_ps(); 3];
            avx_multi_cross(&mut q, simd_ray.dir, self.e2);

            let a = avx_multi_dot(self.e1, q);

            let f = _mm256_div_ps(one_m256(), a);

            let mut s = [_mm256_undefined_ps(); 3];
            avx_multi_sub(&mut s, simd_ray.origin, self.v0);

            let u = _mm256_mul_ps(f, avx_multi_dot(s, q));

            let mut r = [_mm256_undefined_ps(); 3];
            avx_multi_cross(&mut r, s, self.e1);

            let v = _mm256_mul_ps(f, avx_multi_dot(simd_ray.dir, r));

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

const SWAP_TABLE: [[usize; 2]; 2] = [[1, 0], [0, 1]];

unsafe fn avx_conditional_swap_in_place(ts: &mut [[__m256; 3]; 2], mask: [bool; 3]) {
    // Branchless swapping
    (ts[0][0], ts[1][0]) = (ts[SWAP_TABLE[mask[0] as usize][0]][0], ts[SWAP_TABLE[mask[0] as usize][1]][0]); // Swap
    (ts[0][1], ts[1][1]) = (ts[SWAP_TABLE[mask[1] as usize][0]][1], ts[SWAP_TABLE[mask[1] as usize][1]][1]); // Swap
    (ts[0][2], ts[1][2]) = (ts[SWAP_TABLE[mask[2] as usize][0]][2], ts[SWAP_TABLE[mask[2] as usize][1]][2]); // Swap
}

impl PackedBoxes {
    pub fn intersect(
        &self,
        simd_ray: &SimdRay,
        ray_length: __m256,
    ) -> (__m256, i32) {
        unsafe {
            // let t1 = (check_box.min - ray.origin) * ray.inv_dir;
            let mut t1 = [_mm256_undefined_ps(); 3];
            avx_multi_sub(&mut t1, self.min, simd_ray.origin);
            inplace_avx_multi_mul(&mut t1, simd_ray.inv_dir);

            // let t2 = (check_box.max - ray.origin) * ray.inv_dir;
            let mut t2 = [_mm256_undefined_ps(); 3];
            avx_multi_sub(&mut t2, self.max, simd_ray.origin);
            inplace_avx_multi_mul(&mut t2, simd_ray.inv_dir);
            
            let mut tmp = [t1, t2];
            avx_conditional_swap_in_place(&mut tmp, simd_ray.inv_sign_mask);
            t1 = tmp[0];
            t2 = tmp[1];

            // Calculate tmin and tmax
            let tmin = _mm256_max_ps(_mm256_max_ps(t1[0], t1[1]), t1[2]);
            let tmax = _mm256_min_ps(_mm256_min_ps(_mm256_min_ps(t2[0], t2[1]), t2[2]), ray_length);            

            // Ensure tmin <= tmax for a valid intersection
            let mask = _mm256_cmp_ps(tmin, tmax, _CMP_LE_OQ);

            // blend mask
            let t_results = _mm256_blendv_ps(minus_one_m256(), one_m256(), mask);
            
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
