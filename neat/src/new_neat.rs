use std::collections::HashMap;
use rand::Rng;

pub(crate) const XOR_INPUT: [[f32; 2]; 4] = [[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]];
pub(crate) const XOR_OUTPUT: [f32; 4] = [0.0, 1.0, 1.0, 0.0];
pub(crate) const INPUT_DIM: usize = 2;
pub(crate) const OUTPUT_DIM: usize = 1;
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
    let mut population = Population::new(150, INPUT_DIM, OUTPUT_DIM);
    let mut species_manager = SpeciesManager::new(population.count, INPUT_DIM, OUTPUT_DIM);
    let mut existing_innovation = ExistingInnovation::new(INPUT_DIM, OUTPUT_DIM);
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
        for species in species_manager.species.iter_mut() {
            let max = species
                .members
                .iter()
                .map(|id| population.genomes.get(*id).unwrap().fitness)
                .max_by(|a, b| a.partial_cmp(b).unwrap())
                .unwrap_or_default();
            if max > species.max_fitness {
                species.max_fitness = max;
                species.last_improvement = gen;
            }
            if gen > species.last_improvement + environment.stagnation_threshold {
                // cull
                runner.to_cull.push(species.id);
            }
        }
        for culled in runner.to_cull.drain(..) {
            if species_manager.species.len() == 1 {
                // reset somehow
                return;
            }
            // remove all members from population so cannot be selected for reproduction
        }
        runner.min_fitness = population
            .genomes
            .iter()
            .min_by(|a, b| a.fitness.partial_cmp(&b.fitness).unwrap())
            .unwrap()
            .fitness;
        runner.best_genome = population
            .genomes
            .iter()
            .max_by(|a, b| a.fitness.partial_cmp(&b.fitness).unwrap()).cloned()
            .unwrap();
        runner.max_fitness = runner.best_genome.fitness;
        runner.fitness_range = (runner.max_fitness - runner.min_fitness).max(1.0);
        for species in species_manager.species.iter_mut() {
            species.explicit_fitness_sharing = 0.0;
            for id in species.members.iter() {
                let genome = population.genomes.get(*id).unwrap();
                species.explicit_fitness_sharing += genome.fitness;
            }
            species.explicit_fitness_sharing /= species.members.len() as f32;
            species.explicit_fitness_sharing -= runner.min_fitness;
            species.explicit_fitness_sharing /= runner.fitness_range;
        }
        runner.total_fitness = species_manager
            .species
            .iter()
            .map(|s| s.explicit_fitness_sharing)
            .sum();
        for species in species_manager.species.iter_mut() {
            species.percent_total = species.explicit_fitness_sharing / runner.total_fitness;
        }
        runner.next_gen_remaining = population.count;
        for species in species_manager.species.iter_mut() {
            let mut offspring_count = (species.percent_total * population.count as f32).floor();
            runner.next_gen_remaining -= offspring_count as usize;
            if runner.next_gen_remaining <= 0 {
                offspring_count += runner.next_gen_remaining as f32;
            }
            let only_mutate = (offspring_count * environment.only_mutate).floor();
            let to_crossover = offspring_count - only_mutate;
            let mut members = species
                .members
                .iter()
                .map(|m| population.genomes.get(*m).unwrap())
                .cloned()
                .collect::<Vec<_>>();
            members.sort_by(|a, b| a.fitness.partial_cmp(&b.fitness).unwrap());
            members.reverse();
            let elite_bound = ((environment.elitism * members.len() as f32) as usize)
                .min(members.len())
                .max(1);
            let elites = members.get(0..elite_bound).unwrap().to_vec();
            for _om in 0..only_mutate as usize {
                let selected = elites.get(rand::thread_rng().gen_range(0..elites.len())).cloned().unwrap();
                let mutated = environment.mutate(selected, &mut existing_innovation);
                population.next_gen.push(mutated);
            }
            for _tc in 0..to_crossover as usize {
                let parent1 = elites.get(rand::thread_rng().gen_range(0..elites.len())).cloned().unwrap();
                let mut parent2 = elites.get(rand::thread_rng().gen_range(0..elites.len())).cloned().unwrap();
                while parent1.id == parent2.id && elites.len() > 1 {
                    parent2 = elites.get(rand::thread_rng().gen_range(0..elites.len())).cloned().unwrap();
                }
                let (best, other) = if parent1.fitness > parent2.fitness {
                    (parent1, parent2)
                } else if parent2.fitness > parent1.fitness {
                    (parent2, parent1)
                } else {
                    if rand::thread_rng().gen_range(0.0..1.0) < 0.5 {
                        (parent2, parent1)
                    } else {
                        (parent1, parent2)
                    }
                };
                let crossover = environment.crossover(best, other);
                let mutated_crossover = environment.mutate(crossover, &mut existing_innovation);
                population.next_gen.push(mutated_crossover);
            }
        }
        population.genomes = population.next_gen.drain(..).collect();
        species_manager.speciate(&mut population.genomes);
        runner.history.push(GenerationHistory::new(gen, runner.best_genome.clone(), species_manager.total()));
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
    pub(crate) count: usize,
    pub(crate) genomes: Vec<Genome>,
    pub(crate) next_gen: Vec<Genome>,
}

impl Population {
    pub(crate) fn new(count: usize, input_dim: usize, output_dim: usize) -> Population {
        todo!()
    }
}

pub(crate) type GenomeId = usize;
#[derive(Clone)]
pub(crate) struct Genome {
    pub(crate) id: GenomeId,
    pub(crate) connections: Vec<Connection>,
    pub(crate) nodes: Vec<Node>,
    pub(crate) node_id_gen: NodeId,
    pub(crate) species_id: SpeciesId,
    pub(crate) fitness: Fitness,
}

impl Genome {
    pub(crate) fn new(inputs: usize, outputs: usize) -> Self {
        todo!()
    }
}

impl Genome {
    pub(crate) fn activate<I: Into<Input>>(&self, input: I) -> Output {
        todo!()
    }
}
pub(crate) type NodeId = usize;
#[derive(Copy, Clone)]
pub(crate) struct Node {
    pub(crate) id: NodeId,
}
#[derive(Clone, Copy)]
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

impl ExistingInnovation {
    fn new(inputs: usize, outputs: usize) -> Self {
        let mut innov = 0;
        Self {
            existing: {
                let mut set = HashMap::new();
                for i in 0..inputs {
                    for o in inputs..(inputs + outputs) {
                        set.insert((i, o), innov);
                        innov += 1;
                    }
                }
                for i in (inputs + outputs)..(inputs + outputs * 2) {
                    for o in inputs..(inputs + outputs) {
                        set.insert((i, o), innov);
                        innov += 1;
                    }
                }
                set
            },
            generator: innov,
        }
    }
    pub(crate) fn checked_innovation(&mut self, from: NodeId, to: NodeId) -> Innovation {
        let pair = (from, to);
        if let Some(k) = self.existing.get(&pair) {
            k.clone()
        } else {
            self.generator += 1;
            let idx = self.generator;
            self.existing.insert(pair, idx);
            idx
        }
    }
}

pub(crate) type Fitness = f32;
pub(crate) struct Evaluation<FFN: Fn(&Genome, Data, Output) -> Fitness> {
    pub(crate) func: FFN,
}
impl<FFN: Fn(&Genome, Data, Output) -> Fitness> Evaluation<FFN> {
    pub(crate) fn new(evaluation: FFN) -> Self {
        Self { func: evaluation }
    }
}
pub(crate) type SpeciesId = usize;
pub(crate) struct Species {
    pub(crate) id: SpeciesId,
    pub(crate) members: Vec<GenomeId>,
    pub(crate) representative: Genome,
    pub(crate) explicit_fitness_sharing: Fitness,
    pub(crate) max_fitness: Fitness,
    pub(crate) last_improvement: Generation,
    percent_total: f32,
}

impl Species {
    pub(crate) fn new(id: SpeciesId, representative: Genome) -> Species {
        todo!()
    }
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
    pub(crate) total_fitness: Fitness,
    pub(crate) species: Vec<Species>,
    pub(crate) species_id_gen: SpeciesId,
}

impl SpeciesManager {
    pub(crate) fn speciate(&self, genomes: &mut Vec<Genome>) {
        todo!()
    }
}

impl SpeciesManager {
    pub(crate) fn total(&self) -> usize {
        todo!()
    }
}

impl SpeciesManager {
    fn new(population_count: usize, inputs: usize, outputs: usize) -> Self {
        Self {
            total_fitness: 0.0,
            species: vec![Species::new(0, Genome::new(inputs, outputs))],
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
    pub(crate) fitness_range: Fitness,
    pub(crate) total_fitness: Fitness,
    pub(crate) next_gen_remaining: usize,
    pub(crate) to_cull: Vec<SpeciesId>,
    pub(crate) best_genome: Genome,
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
            total_fitness: 0.0,
            next_gen_remaining: 0,
            to_cull: vec![],
            best_genome: Genome::new(0, 0),
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
    pub(crate) fn crossover(&self, best: Genome, other: Genome) -> Genome {
        todo!()
    }
}

impl Environment {
    pub(crate) fn mutate(&self, genome: Genome, existing_innovation: &mut ExistingInnovation) -> Genome {
        todo!()
    }
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
