use rand::{rngs::StdRng, RngExt, SeedableRng};

pub fn shuffle<T>(arr: &mut [T], rng: &mut StdRng) {
    for i in (1..arr.len()).rev() {
        let j = rng.random_range(0..=i);
        arr.swap(i, j);
    }
}