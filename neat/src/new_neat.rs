use std::collections::HashMap;
pub(crate) const XOR_INPUT: [[f32; 2]; 4] = [[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]];
pub(crate) const XOR_OUTPUT: [f32; 4] = [0.0, 1.0, 1.0, 0.0];
#[test]
fn test() {
    let mut environment = Environment::new();
    environment.set_c1(1.0);
    environment.set_c2(1.0);
    environment.set_c3(0.4);
    environment.set_elitism(0.3);
    environment.set_stagnation_threshold(15);
    environment.set_add_connection(0.05);
    environment.set_add_node(0.03);
    environment.set_inherit_disable(0.75);
    environment.set_only_mutate(0.25);
    let evaluation = Evaluation::new(|genome: &Genome, input: Data, actual: Output| -> Fitness {
        let mut fitness = 4.0;
        for (i, xi) in input.data.iter().enumerate() {
            let output = genome.activate(xi);
            fitness -= (output.data[0] - actual.data[i]).powi(2);
        }
        fitness
    });
    let mut population = Population::new(150, 2, 1);
    let mut species_manager = SpeciesManager::new();
    let mut runner = Runner::new(150, 3.9);
    for gen in 0..runner.generations {
        for genome in population.genomes.iter_mut() {
            genome.fitness = (evaluation.func)(genome, XOR_INPUT.into(), XOR_OUTPUT.into());
            if genome.fitness >= runner.limit {
                runner.history.push(GenerationHistory::new(
                    gen,
                    genome.clone(),
                    species_manager.total(),
                ));
                // TODO save history / print-out
                return;
            }
        }
        runner.min_fitness = population
            .genomes
            .iter()
            .min_by(|a, b| a.fitness.partial_cmp(&b.fitness).unwrap())
            .unwrap()
            .fitness;
        runner.max_fitness = population
            .genomes
            .iter()
            .max_by(|a, b| a.fitness.partial_cmp(&b.fitness).unwrap())
            .unwrap()
            .fitness;
        runner.fitness_range = (runner.max_fitness - runner.min_fitness).max(1.0);

    }
}
pub(crate) struct Data {
    pub(crate) data: Vec<Vec<f32>>,
}
impl<V: Into<Vec<Vec<f32>>>> From<V> for Data {
    fn from(v: V) -> Self {
        Self { data: v.into() }
    }
}
pub(crate) type Generation = usize;
pub(crate) struct Input {
    pub(crate) data: Vec<f32>,
}
impl<V: Into<Vec<f32>>> From<V> for Input {
    fn from(value: V) -> Self {
        Input { data: value.into() }
    }
}
pub(crate) struct Output {
    pub(crate) data: Vec<f32>,
}
impl<V: Into<Vec<f32>>> From<V> for Output {
    fn from(value: V) -> Self {
        Self { data: value.into() }
    }
}
pub(crate) struct Population {
    pub(crate) genomes: Vec<Genome>,
    pub(crate) next_gen: Vec<Genome>,
}

impl Population {
    pub(crate) fn new(count: usize, input_dim: usize, output_dim: usize) -> Population {
        todo!()
    }
}

pub(crate) type GenomeId = usize;
pub(crate) struct Genome {
    pub(crate) id: GenomeId,
    pub(crate) connections: Vec<Connection>,
    pub(crate) nodes: Vec<Node>,
    pub(crate) node_id_gen: NodeId,
    pub(crate) species_id: SpeciesId,
    pub(crate) fitness: Fitness,
}
impl Genome {
    pub(crate) fn activate<I: Into<Input>>(&self, input: I) -> Output {
        todo!()
    }
}
pub(crate) type NodeId = usize;
pub(crate) struct Node {
    pub(crate) id: NodeId,
}
pub(crate) struct Connection {
    pub(crate) from: NodeId,
    pub(crate) to: NodeId,
    pub(crate) weight: f32,
    pub(crate) enabled: bool,
    pub(crate) innovation: Innovation,
}
pub(crate) type Innovation = usize;
pub(crate) struct ExistingInnovation {
    pub(crate) existing: HashMap<(NodeId, NodeId), Innovation>,
    pub(crate) generator: Innovation,
}
pub(crate) type Fitness = f32;
pub(crate) struct Evaluation<FFN: Fn(&Genome, Data, Output) -> Fitness> {
    pub(crate) func: FFN,
}
impl<FFN> Evaluation<FFN> {
    pub(crate) fn new(evaluation: FFN) -> Self {
        Self { func: evaluation }
    }
}
pub(crate) type SpeciesId = usize;
pub(crate) struct Species {
    pub(crate) id: SpeciesId,
    pub(crate) members: Vec<GenomeId>,
    pub(crate) representative: Genome,
    pub(crate) explicit_fitness_sharing: f32,
    pub(crate) max_fitness: f32,
    pub(crate) last_improvement: Generation,
}
pub(crate) struct Compatibility {
    pub(crate) excess: f32,
    pub(crate) disjoint: f32,
    pub(crate) weight_difference: f32,
    pub(crate) n: f32,
}
impl Compatibility {
    pub(crate) fn distance(&self, environment: Environment) -> f32 {
        todo!()
    }
}
pub(crate) struct SpeciesManager {
    pub(crate) total_fitness: f32,
    pub(crate) species: Vec<Species>,
    pub(crate) species_id_gen: SpeciesId,
}

impl SpeciesManager {
    pub(crate) fn total(&self) -> usize {
        todo!()
    }
}

impl SpeciesManager {
    fn new() -> Self {
        Self {
            total_fitness: 0.0,
            species: vec![],
            species_id_gen: 0,
        }
    }
}

pub(crate) struct Runner {
    pub(crate) history: Vec<GenerationHistory>,
    pub(crate) generations: Generation,
    pub(crate) limit: Fitness,
    pub(crate) min_fitness: Fitness,
    pub(crate) max_fitness: Fitness,
    pub(crate) fitness_range: Fitness
}
impl Runner {
    pub(crate) fn new(generations: Generation, limit: Fitness) -> Self {
        Self {
            history: vec![],
            generations,
            limit,
            min_fitness: 0.0,
            max_fitness: 0.0,
            fitness_range: 0.0,
        }
    }
}
pub(crate) struct GenerationHistory {
    pub(crate) generation: Generation,
    pub(crate) best: Genome,
    pub(crate) total_species: usize,
}

impl GenerationHistory {
    fn new(generation: Generation, best: Genome, total_species: usize) -> GenerationHistory {
        Self {
            generation,
            best,
            total_species,
        }
    }
}

pub(crate) struct Environment {
    pub(crate) c1: f32,
    pub(crate) c2: f32,
    pub(crate) c3: f32,
    pub(crate) threshold: f32,
    pub(crate) elitism: f32,
    pub(crate) only_mutate: f32,
    pub(crate) stagnation_threshold: Generation,
    pub(crate) add_node: f32,
    pub(crate) add_connection: f32,
    pub(crate) inherit_disable: f32,
}
impl Environment {
    pub(crate) fn new() -> Self {
        Self {
            c1: 0.0,
            c2: 0.0,
            c3: 0.0,
            threshold: 0.0,
            elitism: 0.0,
            only_mutate: 0.0,
            stagnation_threshold: 0,
            add_node: 0.0,
            add_connection: 0.0,
            inherit_disable: 0.0,
        }
    }
    pub(crate) fn c1(&mut self, c1: f32) {
        self.c1 = c1;
    }

    pub fn set_c1(&mut self, c1: f32) {
        self.c1 = c1;
    }

    pub fn set_c2(&mut self, c2: f32) {
        self.c2 = c2;
    }

    pub fn set_c3(&mut self, c3: f32) {
        self.c3 = c3;
    }

    pub fn set_threshold(&mut self, threshold: f32) {
        self.threshold = threshold;
    }

    pub fn set_elitism(&mut self, elitism: f32) {
        self.elitism = elitism;
    }

    pub fn set_only_mutate(&mut self, only_mutate: f32) {
        self.only_mutate = only_mutate;
    }

    pub fn set_stagnation_threshold(&mut self, stagnation_threshold: Generation) {
        self.stagnation_threshold = stagnation_threshold;
    }

    pub fn set_add_node(&mut self, add_node: f32) {
        self.add_node = add_node;
    }

    pub fn set_add_connection(&mut self, add_connection: f32) {
        self.add_connection = add_connection;
    }

    pub fn set_inherit_disable(&mut self, inherit_disable: f32) {
        self.inherit_disable = inherit_disable;
    }
}
pub(crate) fn creates_cycle(from: NodeId, to: NodeId, genome: &Genome) -> bool {
    if from == to {
        return true;
    }
    let mut visited = vec![to];
    while true {
        let mut num_added = 0;
        for c in genome.connections.iter() {
            if visited.iter().find(|v| **v == c.from).is_some()
                && visited.iter().find(|v| **v == c.to).is_none()
            {
                if c.to == from {
                    return true;
                } else {
                    visited.push(c.to);
                    num_added += 1;
                }
            }
        }
        if num_added == 0 {
            return false;
        }
    }
    false
}
