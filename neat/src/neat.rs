#[test]
fn neat() {
    // population (uniform) networks @ 6x2
    // for g in generations
    //      next_gen = []
    //      for p in population.len()
    //          parent1 = population.random()
    //          parent2 = population.random() // compatible with other (in parent1.species)
    //          c = crossover(parent1, parent2)
    //          next_gen.push(c)
    //      population = next_gen
    // ans = population.max_fitness()
}
pub(crate) type NodeId = usize;
pub(crate) enum NodeType {
    Input,
    Hidden,
    Output,
}
pub(crate) struct Node {
    pub(crate) value: f32,
    pub(crate) ty: NodeType,
}
pub(crate) struct Connection {
    pub(crate) from: NodeId,
    pub(crate) to: NodeId,
    pub(crate) weight: f32,
    pub(crate) enabled: bool,
    pub(crate) innovation: Innovation
}
pub(crate) struct Innovation {
    pub(crate) a: NodeId,
    pub(crate) b: NodeId,
}
pub(crate) struct Genome {
    pub(crate) connections: Vec<Connection>,
}
pub(crate) struct Compatibility {
    pub(crate) c1: f32,
    pub(crate) c2: f32,
    pub(crate) c3: f32,
    pub(crate) n: f32,
    pub(crate) threshold: f32,
}
impl Compatibility {
    pub(crate) fn distance(&self, excess: f32, disjoint: f32, wd: f32) -> f32 {
        todo!()
    }
}
pub(crate) struct Organism {
    pub(crate) genome: Genome,
    pub(crate) explicit_fitness_sharing: f32,
}
pub(crate) struct Species {
    pub(crate) order: Vec<Organism>
}
