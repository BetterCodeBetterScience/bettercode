/// Sum of squares of random numbers.

use std::env;
use std::hint::black_box;
use std::time::Instant;

/// Simple RNG matching Python's random (Mersenne Twister would be complex, 
/// so we use an LCG that produces similar statistical properties)
struct Rng {
    state: u64,
}

impl Rng {
    fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    fn random(&mut self) -> f64 {
        // LCG parameters (same as glibc)
        // the wrapping operations ensure we don't overflow
        self.state = self.state.wrapping_mul(1103515245).wrapping_add(12345);
        // Convert to float in [0, 1)
        (self.state >> 16) as f64 / (1u64 << 48) as f64
    }
}

fn sum_of_squares(n: usize, seed: u64) -> f64 {
    let mut rng = Rng::new(seed);
    let mut total = 0.0;
    for _ in 0..n {
        let x = rng.random();
        total += x * x;
    }
    total
}

fn main() {
    let seed: u64 = env::args()
        .nth(1)
        .and_then(|s| s.parse().ok())
        .unwrap_or(42);
    let n = 1_000_000;
    let iterations = 100;

    let start = Instant::now();
    for _ in 0..iterations {
        // black_box to prevent compiler optimizations that could eliminate the computation
        black_box(sum_of_squares(n, seed));    
    }
    let elapsed = start.elapsed();
    
    let avg_time = elapsed.as_secs_f64() / iterations as f64;

    println!("Seed: {}, N: {}", seed, n);
    println!("Average time (Rust): {:.4} seconds", avg_time);
}