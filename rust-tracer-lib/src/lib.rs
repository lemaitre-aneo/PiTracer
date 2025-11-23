mod camera;
mod color;
mod scene;
mod sphere;
mod vector;

pub use camera::*;
pub use color::*;
pub use scene::*;
pub use sphere::*;
pub use vector::*;

pub fn select<T>(cond: bool, then: T, otherwise: T) -> T {
    // unsafe {
    //     let mut array = [
    //         std::mem::ManuallyDrop::new(otherwise),
    //         std::mem::ManuallyDrop::new(then),
    //     ];
    //     std::mem::ManuallyDrop::drop(&mut array[(!cond) as usize]);
    //     std::mem::ManuallyDrop::take(&mut array[cond as usize])
    // }
    if cond { then } else { otherwise }
}
