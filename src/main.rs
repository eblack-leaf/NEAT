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
    let mut runs = 1000;
    let mut hidden_node_count = 0;
    let mut connection_count = 0;
    let mut gen_done = 0;
    for run in 0..runs {
        if let Some((solution, gen)) = neat() {
            solved += 1;
            hidden_node_count += solution.nodes.len() - 4;
            connection_count += solution.connections.iter().filter(|c| c.enabled).count();
            gen_done += gen;
            println!(
                "solved {} / {} genome: {}, {}, {} @ gen: {}",
                run + 1,
                runs,
                solution.fitness,
                solution.nodes.len(),
                solution.connections.iter().filter(|c| c.enabled).count(),
                gen
            );
        } else {
            println!("FAILED {} / {}", run + 1, runs);
        }
    }
    println!(
        "percent_solved: {} avg-hidden: {} avg-conns: {} gen-done: {}",
        solved as f32 / runs as f32,
        hidden_node_count as f32 / solved as f32,
        connection_count as f32 / solved as f32,
        gen_done as f32 / solved as f32,
    );
}
