//! Production Slot::step, without JSON/actor history/PyTorch. Not training FPS.
use isaac_sim::ffi::Slot;
use rayon::prelude::*;
fn main() {
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(4)
        .build()
        .unwrap();
    for n in [1, 4, 16, 64, 128] {
        let mut slots: Vec<_> = (0..n).map(|i| Slot::new(100 + i)).collect();
        let start = std::time::Instant::now();
        let mut completed = 0;
        for tick in 0..2000 {
            pool.install(|| {
                slots
                    .par_iter_mut()
                    .for_each(|s| s.step(&[(tick / 30 % 9) * 5 + 1, (tick % 120 == 0) as i32, 0]))
            });
            for (i, s) in slots.iter_mut().enumerate() {
                if s.done {
                    completed += 1;
                    *s = Slot::new(10000 + tick as u32 * n + i as u32);
                }
            }
        }
        println!(
            "{}",
            serde_json::json!({"envs":n,"decisions":n*2000,"seconds":start.elapsed().as_secs_f64(),"decisions_per_s":(n*2000)as f64/start.elapsed().as_secs_f64(),"complete_benchmark_episodes":completed,"threads":4,"includes_observation_and_model":false})
        );
    }
}
