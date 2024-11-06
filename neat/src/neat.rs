use rand::Rng;

#[test]
fn neat() {
    let mut population = vec![Genome::new(2, 1, -30.0, 30.0); 150];
    let generations = 300;
    let fitness_threshold = 3.9; // if above this after fitness error calc => we end sim
    let compatibility = Compatibility::new(1.0, 1.0, 0.5, 3.0);
    let perfect_fitness = 1.0 * XOR_INPUT.len() as f32;
    for g in 0..generations {
        let mut next_gen = vec![];
        for genome in population.iter_mut() {
            genome.fitness = perfect_fitness;
            for (i, xi) in XOR_INPUT.iter().enumerate() {
                let predicted = genome.activate(xi.to_vec());
                let xo = XOR_OUTPUT[i];
                genome.fitness -= (predicted[0] - xo).powi(2);
            }
            // TODO normalize by species???
            // need distance of each other in population * share-fn 0/1 modifier
        }
        // let species_proportion = species-fitness-value (avg-of-genomes-in-species [after share-norm]) / population-total (all-species-totals);
        // let species_offspring_num = species_proportion * population.len();
        // selection of fittest (top 20% of total population)
        let mut selection = vec![];
        // TODO modify below to account for how many offspring in each species (eases the species compat part)
        for _ in population.len() {
            // parent1 = random
            // parent2 = random (compatible w/ species of first)
            // crossover
            // mutate
            // next_gen.push(mutated_crossover);
        }
        population = next_gen;
    }
    // ans = population.max_fitness() (iter to find best)
}
pub(crate) const XOR_INPUT: [[f32; 2]; 4] = [[1.0, 1.0], [1.0, 1.0], [1.0, 1.0], [1.0, 1.0]];
pub(crate) const XOR_OUTPUT: [f32; 4] = [0.0, 1.0, 1.0, 0.0];
pub(crate) type NodeId = usize;
pub(crate) enum NodeType {
    Input,
    Hidden,
    Output,
    Bias,
}
pub(crate) fn sigmoid(z: f32) -> f32 {
    1.0 / (1.0 + (-z).exp())
}
pub(crate) struct Node {
    pub(crate) id: NodeId,
    pub(crate) value: f32,
    pub(crate) ty: NodeType,
}
impl Node {
    pub(crate) fn new(id: NodeId, ty: NodeType) -> Self {
        Self { id, value: 0.0, ty }
    }
}
pub(crate) struct Connection {
    pub(crate) from: NodeId,
    pub(crate) to: NodeId,
    pub(crate) weight: f32,
    pub(crate) enabled: bool,
    pub(crate) innovation: Innovation,
}
impl Connection {
    pub(crate) fn new(from: NodeId, to: NodeId, weight: f32, enabled: bool) -> Self {
        Self {
            from,
            to,
            weight,
            enabled,
            innovation: Innovation::new(from, to),
        }
    }
}
#[derive(PartialOrd, PartialEq, Hash, Copy, Clone, Debug)]
pub(crate) struct Innovation {
    pub(crate) a: NodeId,
    pub(crate) b: NodeId,
}
impl Innovation {
    pub(crate) fn new(a: NodeId, b: NodeId) -> Self {
        Self { a, b }
    }
}
pub(crate) struct Genome {
    pub(crate) nodes: Vec<Node>,
    pub(crate) connections: Vec<Connection>,
    pub(crate) fitness: f32,
}
impl Genome {
    pub(crate) fn new(inputs: usize, outputs: usize, weight_min: f32, weight_max: f32) -> Self {
        let mut nodes = vec![];
        for i in 0..inputs {
            nodes.push(Node::new(i, NodeType::Input));
        }
        for o in 0..outputs {
            nodes.push(Node::new(inputs + o, NodeType::Output));
        }
        let mut connections = vec![];
        for i in 0..inputs {
            for o in 0..outputs {
                connections.push(Connection::new(
                    i,
                    o,
                    rand::thread_rng().gen_range(weight_min..weight_max),
                    true,
                ));
            }
        }
        nodes.push(Node::new(nodes.len(), NodeType::Bias).value(1.0));
        connections.push(Connection::new(
            nodes.len().checked_sub(1).unwrap_or_default(),
            inputs + 1,
            rand::thread_rng().gen_range(weight_min..weight_max),
            true,
        ));
        Self {
            nodes,
            connections,
            fitness: 0.0,
        }
    }
    pub(crate) fn activate(&self, inputs: Vec<f32>) -> Vec<f32> {
        todo!()
    }
}
pub(crate) struct Compatibility {
    pub(crate) c1: f32,
    pub(crate) c2: f32,
    pub(crate) c3: f32,
    pub(crate) threshold: f32,
}
impl Compatibility {
    pub(crate) fn distance(&self, n: f32, excess: f32, disjoint: f32, wd: f32) -> f32 {
        todo!()
    }
    pub(crate) fn new(c1: f32, c2: f32, c3: f32, threshold: f32) -> Self {
        Self {
            c1,
            c2,
            c3,
            threshold,
        }
    }
}
pub(crate) struct SpeciesDescriptor {
    pub(crate) genome: Genome,
    pub(crate) explicit_fitness_sharing: f32,
}
pub(crate) struct Species {
    pub(crate) order: Vec<SpeciesDescriptor>,
}
