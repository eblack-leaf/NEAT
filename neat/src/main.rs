// mod neat;
mod new_neat;

use crate::new_neat::neat;
use foliage::Foliage;

const VIEW_AREA: (f32, f32) = (1700.0, 800.0);
fn main() {
    // let mut foliage = Foliage::new();
    // foliage.set_desktop_size(VIEW_AREA);
    // foliage.photosynthesize();
    let mut solved = 0;
    let mut runs = 100;
    for run in 0..runs {
        if let Some((solution, gen)) = neat() {
            solved += 1;
            println!(
                "solved {} / {} genome: {}, {}, {} @ gen: {}",
                solved,
                runs,
                solution.fitness,
                solution.nodes.len(),
                solution
                    .connections
                    .iter()
                    .filter(|c| c.enabled)
                    .collect::<Vec<_>>()
                    .len(),
                gen
            );
        }
    }
}
