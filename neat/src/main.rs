pub(crate) mod snake;
pub(crate) mod gene;
pub(crate) mod genome;
pub(crate) mod connection;
pub(crate) mod node;
pub(crate) mod food;
pub(crate) mod game;
pub(crate) mod input;
pub(crate) mod output;
pub(crate) mod mutation;
pub(crate) mod selection;
pub(crate) mod network;
pub(crate) mod crossover;
pub(crate) mod fitness;
pub(crate) mod activation;
pub(crate) mod bias;
pub(crate) mod speciation;
pub(crate) mod population;
pub(crate) mod distance;
mod neat;

use foliage::Foliage;
const VIEW_AREA: (f32, f32) = (1700.0, 800.0);
fn main() {
    let mut foliage = Foliage::new();
    foliage.set_desktop_size(VIEW_AREA);
    foliage.photosynthesize();
}
