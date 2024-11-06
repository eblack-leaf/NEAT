use rand::Rng;

#[test]
fn neat() {
    // population (uniform) networks @ 2x1
    let mut population = vec![Genome::new(2, 1); 150];
    let generations = 300;
    let fitness_threshold = 3.9;// if above this after fitness error calc => we end sim
    let compatibility = Compatibility::new(1.0, 1.0, 0.5, 3.0);
    let perfect_fitness = 1.0 * XOR_INPUT.len() as f32;
    // for g in generations
    for g in 0..generations {
        //      next_gen = []
        let mut next_gen = vec![];
        //      for p in population.len()
        for genome in population.iter() {
            let mut fitness = perfect_fitness;
            for (i, xi) in XOR_INPUT.iter().enumerate() {
                let predicted = genome.activate(xi.to_vec());
                let xo = XOR_OUTPUT[i];
                fitness -= (predicted[0] - xo).powi(2);
            }
        }
        // selection of fittest
        let mut selection = vec![];
        for _ in population.len() {
            // parent1 = random
            // parent2 = random (compatible w/ species of first)
            // crossover
            // mutate
            // store in next_gen
        }
        //      population = next_gen
    }
    // ans = population.max_fitness()
}
pub(crate) const XOR_INPUT: [[f32; 2]; 4] = [[1.0, 1.0], [1.0, 1.0], [1.0, 1.0], [1.0, 1.0]];
pub(crate) const XOR_OUTPUT: [f32; 4] = [0.0, 1.0, 1.0, 0.0];
pub(crate) type NodeId = usize;
pub(crate) enum NodeType {
    Input,
    Hidden,
    Output,
}
pub(crate) fn sigmoid(z: f32) -> f32 {
    1.0 / (1.0 + (-z).exp())
}
pub(crate) struct Node {
    pub(crate) value: f32,
    pub(crate) ty: NodeType,
}
impl Node {
    pub(crate) fn new(ty: NodeType) -> Self {
        Self { value: 0.0, ty }
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
    pub(crate) fn new(inputs: usize, outputs: usize) -> Self {
        let mut nodes = vec![];
        for i in 0..inputs {
            nodes.push(Node::new(NodeType::Input));
        }
        for o in 0..outputs {
            nodes.push(Node::new(NodeType::Output));
        }
        let mut connections = vec![];
        for i in 0..inputs {
            for o in 0..outputs {
                connections.push(Connection::new(
                    i,
                    o,
                    rand::thread_rng().gen_range(0.0..1.0),
                    true,
                ));
            }
        }
        Self { nodes, connections, fitness: 0.0 }
    }
    pub(crate) fn activate(&self, inputs: Vec<f32>) -> Vec<f32> {
        // add-bias term direct to output in evaluation
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
