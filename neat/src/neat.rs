use rand::Rng;
pub(crate) const XOR_INPUT: [[f32; 2]; 4] = [[1.0, 1.0], [1.0, 1.0], [1.0, 1.0], [1.0, 1.0]];
pub(crate) const XOR_OUTPUT: [f32; 4] = [0.0, 1.0, 1.0, 0.0];
#[test]
fn neat() {
    let mut population = Population::new(2, 1, 150);
    let generations = 300;
    let fitness_threshold = 3.9; // if above this after fitness error calc => we end sim
    let compatibility = Compatibility::new(1.0, 1.0, 0.4, 3.0);
    let perfect_fitness = 1.0 * XOR_INPUT.len() as f32;
    let environment = Environment::new();
    let mut species_tree = SpeciesTree::new();
    species_tree.speciate(&mut population.genomes, &compatibility);
    for g in 0..generations {
        let mut next_gen = vec![];
        for (i, genome) in population.genomes.iter_mut().enumerate() {
            genome.fitness = perfect_fitness;
            for (i, xi) in XOR_INPUT.iter().enumerate() {
                let predicted = genome.activate(xi.to_vec());
                let xo = XOR_OUTPUT[i];
                genome.fitness -= (predicted[0] - xo).powi(2);
            }
            let mut summation = 0.0;
            for (j, other) in population.genomes.iter().cloned().enumerate() {
                if i != j {
                    let distance = compatibility.distance(genome.compatibility_metrics(&other));
                    let share = distance < compatibility.threshold;
                    summation += distance * f32::from(share);
                }
            }
            genome.fitness /= summation;
            species_tree
                .order
                .get_mut(genome.species_id)
                .unwrap()
                .explicit_fitness_sharing += genome.fitness;
            species_tree.total_fitness += genome.fitness;
        }
        // selection of fittest (top 20% of total population)
        let mut selection = vec![];
        let mut total_remaining = population.count;
        for (i, species) in species_tree.order.iter().enumerate() {
            if species.count > 0 {
                let requested_offspring = if i + 1 == species_tree.num_active_species() {
                    total_remaining
                } else {
                    let requested_offspring =
                        (species.explicit_fitness_sharing / species_tree.total_fitness
                            * population.count as f32) as usize;
                    total_remaining -= requested_offspring.min(total_remaining);
                    requested_offspring
                };
                for offspring_request in 0..requested_offspring {
                    // parent1 = random (from species.nodes)
                    // parent2 = random (from species.nodes) or environment.interspecies [all]
                    // crossover
                    // mutate
                    // next_gen.push(mutated_crossover);
                }
            }
        }
        population.genomes = next_gen;
        species_tree.speciate(&mut population.genomes, &compatibility);
    }
    // ans = population.max_fitness() (iter to find best any species can win)
}
pub(crate) struct Evaluation {}
pub(crate) struct Population {
    pub(crate) genomes: Vec<Genome>,
    pub(crate) count: usize,
}
impl Population {
    pub(crate) fn new(inputs: usize, outputs: usize, count: usize) -> Self {
        let mut genomes = vec![];
        for i in 0..count {
            genomes.push(Genome::new(inputs, outputs, i));
        }
        Self { genomes, count }
    }
}
pub(crate) type NodeId = usize;
#[derive(Clone)]
pub(crate) enum NodeType {
    Input,
    Hidden,
    Output,
    Bias,
}
pub(crate) fn sigmoid(z: f32) -> f32 {
    1.0 / (1.0 + (-z).exp())
}
#[derive(Clone)]
pub(crate) struct Node {
    pub(crate) id: NodeId,
    pub(crate) value: f32,
    pub(crate) ty: NodeType,
    pub(crate) disabled: bool,
}
impl Node {
    pub(crate) fn new(id: NodeId, ty: NodeType) -> Self {
        Self {
            id,
            value: 0.0,
            ty,
            disabled: false,
        }
    }
    pub(crate) fn value(mut self, v: f32) -> Self {
        self.value = v;
        self
    }
}
#[derive(Clone)]
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
pub(crate) type SpeciesId = usize;
#[derive(Clone)]
pub(crate) struct Genome {
    pub(crate) nodes: Vec<Node>,
    pub(crate) connections: Vec<Connection>,
    pub(crate) fitness: f32,
    pub(crate) node_id_generator: NodeId,
    pub(crate) species_id: SpeciesId,
    pub(crate) id: GenomeId,
}
impl Genome {
    pub(crate) fn compatibility_metrics(&self, other: &Self) -> CompatibilityMetrics {
        todo!()
    }
    pub(crate) fn new(inputs: usize, outputs: usize, id: GenomeId) -> Self {
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
                    rand::thread_rng().gen_range(0.0..1.0),
                    true,
                ));
            }
        }
        nodes.push(Node::new(nodes.len(), NodeType::Bias).value(1.0));
        connections.push(Connection::new(
            nodes.len().checked_sub(1).unwrap_or_default(),
            inputs + 1,
            rand::thread_rng().gen_range(0.0..1.0),
            true,
        ));
        Self {
            nodes,
            connections,
            fitness: 0.0,
            node_id_generator: 0,
            species_id: 0,
            id,
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
pub(crate) struct CompatibilityMetrics {
    pub(crate) n: f32,
    pub(crate) excess: f32,
    pub(crate) disjoint: f32,
    pub(crate) weight_difference: f32,
}
impl Compatibility {
    pub(crate) fn distance(&self, metrics: CompatibilityMetrics) -> f32 {
        self.c1 * metrics.excess / metrics.n
            + self.c2 * metrics.disjoint / metrics.n
            + self.c3 * metrics.weight_difference
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
pub(crate) type GenomeId = usize;
pub(crate) struct Species {
    pub(crate) current_organisms: Vec<GenomeId>,
    pub(crate) representation: Genome,
    pub(crate) count: usize,
    pub(crate) explicit_fitness_sharing: f32,
}
impl Species {
    pub(crate) fn new(genome: Genome) -> Self {
        Self {
            current_organisms: vec![],
            representation: genome,
            count: 0,
            explicit_fitness_sharing: 0.0,
        }
    }
}
pub(crate) struct SpeciesTree {
    pub(crate) order: Vec<Species>,
    pub(crate) total_fitness: f32,
}

impl SpeciesTree {
    fn new() -> Self {
        Self {
            order: vec![],
            total_fitness: 0.0,
        }
    }
    pub(crate) fn num_active_species(&self) -> usize {
        todo!()
    }
    pub(crate) fn speciate(&mut self, population: &mut Vec<Genome>, compatibility: &Compatibility) {
        self.total_fitness = 0.0;
        for species in self.order.iter_mut() {
            species.count = 0;
            species.explicit_fitness_sharing = 0.0;
            species.current_organisms.clear();
        }
        for genome in population.iter_mut() {
            let mut found = None;
            for (i, species) in self.order.iter().enumerate() {
                let distance =
                    compatibility.distance(genome.compatibility_metrics(&species.representation));
                if distance < compatibility.threshold {
                    found = Some(i);
                    break;
                }
            }
            let s = if let Some(f) = found {
                self.order
                    .get_mut(f)
                    .unwrap()
                    .current_organisms
                    .push(genome.id);
                f
            } else {
                self.order.push(Species::new(genome.clone()));
                self.order.len().checked_sub(1).unwrap_or_default()
            };
            genome.species_id = s;
        }
        for genome in population.iter() {
            self.order.get_mut(genome.species_id).unwrap().count += 1;
        }
    }
}

pub(crate) struct Environment {
    pub(crate) connection_weight: (f32, f32, f32),
    pub(crate) disable_gene: f32,
    pub(crate) skip_crossover: f32,
    pub(crate) interspecies: f32,
    pub(crate) add_node: f32,
    pub(crate) add_connection: f32,
    pub(crate) stagnation_threshold: usize,
    pub(crate) champion_network_count: usize,
}
impl Environment {
    pub(crate) fn new() -> Self {
        Self {
            connection_weight: (0.8, 0.9, 0.1),
            disable_gene: 0.75,
            skip_crossover: 0.25,
            interspecies: 0.001,
            add_node: 0.03,
            add_connection: 0.05,
            stagnation_threshold: 15,
            champion_network_count: 5,
        }
    }
}
