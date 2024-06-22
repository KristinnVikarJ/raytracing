pub mod objects;
pub mod simd_accel;

pub use objects::{Color, PackedTriangles, Triangle};
pub use simd_accel::pack_triangles;
