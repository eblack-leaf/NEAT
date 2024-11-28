use rand::Rng;
use std::collections::HashMap;
use std::fmt::{Display, Formatter, Write};

pub(crate) const XOR_INPUT: [[f32; 2]; 4] = [[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]];
pub(crate) const XOR_OUTPUT: [f32; 4] = [0.0, 1.0, 1.0, 0.0];
pub(crate) const INPUT_DIM: usize = 2;
pub(crate) const OUTPUT_DIM: usize = 1;
pub(crate) fn neat() -> Option<(Genome, Generation)> {
    let mut environment = Environment::new(INPUT_DIM, OUTPUT_DIM);
    environment.set_c1(1.0);
    environment.set_c2(1.0);
    environment.set_c3(0.4);
    environment.set_compatibility_threshold(3.0);
    environment.set_elitism(0.2);
    environment.set_stagnation_threshold(15);
    environment.set_add_connection(0.2);
    environment.set_add_node(0.07);
    environment.set_inherit_disable(0.75);
    environment.set_only_mutate(0.25);
    environment.set_connection_weight(0.8);
    environment.set_perturb(0.9);
    environment.set_crossover_only(0.2);
    let evaluation = Evaluation::new(|genome: &Genome, input: Data, actual: Output| -> Fitness {
        let mut fitness = 4.0;
        for (i, xi) in input.data.iter().enumerate() {
            let output = genome.activate(xi.to_vec());
            fitness -= (output.data[0] - actual.data[i]).powi(2);
        }
        fitness
    });
    let mut population = Population::new(150, INPUT_DIM, OUTPUT_DIM);
    let mut species_manager = SpeciesManager::new(INPUT_DIM, OUTPUT_DIM);
    species_manager.speciate(&mut population.genomes, &environment, 0);
    let mut existing_innovation = ExistingInnovation::new(INPUT_DIM, OUTPUT_DIM);
    let mut runner = Runner::new(150, 3.9);
    for gen in 0..runner.generations {
        // println!("generation {} ------------------------------------------------------------", gen);
        let mut max_found = None;
        for genome in population.genomes.iter_mut() {
            genome.fitness = (evaluation.func)(genome, XOR_INPUT.into(), XOR_OUTPUT.into());
            if genome.fitness >= runner.limit {
                max_found.replace(genome.clone());
                runner.limit = genome.fitness + Runner::NEW_LIMIT_DELTA;
            }
            // println!("id: {} fitness: {} in species: {}", genome.id, genome.fitness, genome.species_id);
        }
        if let Some(max) = max_found {
            runner.history.push(GenerationHistory::new(
                gen,
                max.clone(),
                max.clone(),
                species_manager.total(),
                runner.mutations.drain(..).collect::<Vec<_>>(),
                runner.lineage.drain(..).collect::<Vec<_>>(),
            ));
            // println!("max-found: id {} w/ fitness {} species {}", max.id, max.fitness, max.species_id);
            // for node in max.nodes.iter() {
            //     println!("node: {}", node);
            // }
            // for conn in max.connections.iter() {
            //     println!("conn: {}", conn);
            // }
            // TODO save history / print-out
            return Some((max, gen));
        }
        for species in species_manager.species.iter_mut() {
            let max = species
                .members
                .iter()
                .map(|id| population.genomes.get(*id).unwrap().fitness)
                .max_by(|a, b| a.partial_cmp(b).unwrap())
                .unwrap_or_default();
            if max > species.max_fitness {
                // println!("improved to {} @ {}", max, gen);
                species.max_fitness = max;
                species.last_improvement = gen;
            }
            if gen > species.last_improvement + environment.stagnation_threshold {
                runner.to_cull.push(species.id);
            }
        }
        for id in runner.to_cull.iter_mut() {
            *id = species_manager
                .species
                .iter()
                .cloned()
                .position(|s| s.id == *id)
                .unwrap();
        }
        runner.to_cull.sort();
        runner.to_cull.reverse();
        for idx in runner.to_cull.drain(..) {
            if species_manager.species.len() == 1 {
                // reset somehow?
                // println!(
                //     "skipping cull of {} for len() == 1",
                //     species_manager.species.get(idx).unwrap().id
                // );
                continue;
            }
            // println!("culling {}", species_manager.species.get(idx).unwrap().id);
            species_manager.species.remove(idx);
        }
        runner.min_fitness = population
            .genomes
            .iter()
            .min_by(|a, b| a.fitness.partial_cmp(&b.fitness).unwrap())
            .unwrap()
            .fitness;
        let current_best = population
            .genomes
            .iter()
            .max_by(|a, b| a.fitness.partial_cmp(&b.fitness).unwrap())
            .cloned()
            .unwrap();
        if current_best.fitness > runner.best_genome.fitness {
            // println!(
            //     "new-best: {} id: {} @ gen: {}",
            //     current_best.fitness, current_best.id, gen
            // );
            runner.best_genome = current_best.clone();
        }
        runner.max_fitness = current_best.fitness;
        runner.fitness_range = (runner.max_fitness - runner.min_fitness).max(1.0);
        // println!(
        //     "max {} min {} range {}",
        //     runner.max_fitness, runner.min_fitness, runner.fitness_range
        // );
        for species in species_manager.species.iter_mut() {
            species.explicit_fitness_sharing = 0.0;
            if species.members.is_empty() {
                println!(
                    "skipping empty species 888888888888888888888888888888888888888888888888888"
                );
            }
            for id in species.members.iter() {
                let genome = population.genomes.get(*id).unwrap();
                // println!("adding fitness: {} from {} in species {}", genome.fitness, genome.id, species.id);
                species.explicit_fitness_sharing += genome.fitness;
            }
            if species.explicit_fitness_sharing <= 0.0 {
                println!("negative or 0 888888888888888888888888888888888888888888888888888888888");
                continue;
            }
            species.explicit_fitness_sharing /= species.members.len() as f32;
            // species.explicit_fitness_sharing -= runner.min_fitness;
            // species.explicit_fitness_sharing /= runner.fitness_range;
            // println!(
            //     "species: {} explicit-fitness: {}",
            //     species.id, species.explicit_fitness_sharing
            // );
        }
        runner.total_fitness = species_manager
            .species
            .iter()
            .map(|s| s.explicit_fitness_sharing)
            .sum();
        // println!("total-fitness: {}", runner.total_fitness);
        for species in species_manager.species.iter_mut() {
            species.percent_total = species.explicit_fitness_sharing / runner.total_fitness;
            // println!(
            //     "species: {} percent-total: {}",
            //     species.id, species.percent_total
            // );
        }
        runner.next_gen_remaining = population.count;
        runner.next_gen_id = 0;
        for species in species_manager.species.iter_mut() {
            let mut offspring_count = (species.percent_total * population.count as f32).floor();
            runner.next_gen_remaining -= offspring_count as usize;
            if runner.next_gen_remaining <= 0 {
                offspring_count += runner.next_gen_remaining as f32;
            }
            // println!("offspring_count {} for {}", offspring_count, species.id);
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
            // println!("elite-bound: {} member-len: {}", elite_bound, members.len());
            let elites = members.get(0..elite_bound).unwrap().to_vec();
            for _om in 0..only_mutate as usize {
                let selected = elites
                    .get(rand::thread_rng().gen_range(0..elites.len()))
                    .cloned()
                    .unwrap();
                let mut mutated = environment.mutate(selected, &mut existing_innovation);
                mutated.id = runner.next_gen_id;
                runner.next_gen_id += 1;
                population.next_gen.push(mutated);
            }
            for _tc in 0..to_crossover as usize {
                let parent1 = elites
                    .get(rand::thread_rng().gen_range(0..elites.len()))
                    .cloned()
                    .unwrap();
                let mut parent2 = elites
                    .get(rand::thread_rng().gen_range(0..elites.len()))
                    .cloned()
                    .unwrap();
                while parent1.id == parent2.id && elites.len() > 1 {
                    parent2 = elites
                        .get(rand::thread_rng().gen_range(0..elites.len()))
                        .cloned()
                        .unwrap();
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
                let crossover = environment.crossover(runner.next_gen_id, best, other);
                runner.next_gen_id += 1;
                let crossover = if rand::thread_rng().gen_range(0.0..1.0) < environment.crossover_only {
                    crossover
                } else {
                    environment.mutate(crossover, &mut existing_innovation)
                };
                population.next_gen.push(crossover);
            }
        }
        for genome in population.next_gen.iter_mut() {
            genome.network_depth = genome.max_depth();
        }
        runner.history.push(GenerationHistory::new(
            gen,
            runner.best_genome.clone(),
            current_best,
            species_manager.total(),
            runner.mutations.drain(..).collect(),
            runner.lineage.drain(..).collect(),
        ));
        // println!(
        //     "best-genome: id: {} fitness: {}",
        //     runner.best_genome.id, runner.best_genome.fitness
        // );
        population.genomes = population.next_gen.drain(..).collect();
        species_manager.speciate(&mut population.genomes, &environment, gen);
    }
    None
}
pub(crate) struct Data {
    pub(crate) data: Vec<Vec<f32>>,
}
impl<const N: usize, const M: usize> From<[[f32; N]; M]> for Data {
    fn from(value: [[f32; N]; M]) -> Self {
        Self {
            data: {
                let mut data = vec![vec![0.0; N]; M];
                for (i, v) in value.iter().enumerate() {
                    for (j, b) in v.iter().enumerate() {
                        data[i][j] = *b;
                    }
                }
                data
            },
        }
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

impl Output {
    fn new(solved: Vec<f32>) -> Self {
        Self { data: solved }
    }
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
        Self {
            count,
            genomes: (0..count)
                .into_iter()
                .map(|i| Genome::new(i, input_dim, output_dim))
                .collect(),
            next_gen: vec![],
        }
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
    pub(crate) inputs: usize,
    pub(crate) outputs: usize,
    pub(crate) network_depth: usize,
}

impl Genome {
    pub(crate) const ACTIVATION_SCALE: f32 = 4.9;
    pub(crate) fn new(id: GenomeId, inputs: usize, outputs: usize) -> Self {
        let mut connections = vec![];
        let mut nodes = vec![];
        if inputs > 0 && outputs > 0 {
            for input in 0..inputs {
                // println!("input: {}", input);
                nodes.push(Node::new(input, NodeType::Input));
            }
            for output in inputs..inputs + outputs {
                // println!("output: {}", output);
                nodes.push(Node::new(output, NodeType::Output));
            }
            for bias in inputs + outputs..inputs + outputs * 2 {
                // println!("bias: {}", bias);
                nodes.push(Node::new(bias, NodeType::Bias));
            }
            let node_id_gen = nodes.len();
            let mut innovation = 0;
            for i in 0..inputs {
                for o in inputs..inputs + outputs {
                    let connection =
                        Connection::new(i, o, rand::thread_rng().gen_range(0.0..1.0), innovation);
                    connections.push(connection);
                    innovation += 1;
                }
            }
            for bias in inputs + outputs..inputs + outputs * 2 {
                for o in inputs..inputs + outputs {
                    let connection = Connection::new(
                        bias,
                        o,
                        rand::thread_rng().gen_range(0.0..1.0),
                        innovation,
                    );
                    connections.push(connection);
                    innovation += 1;
                }
            }
            return Self {
                id,
                connections,
                nodes,
                node_id_gen,
                species_id: 0,
                fitness: 0.0,
                inputs,
                outputs,
                network_depth: 1,
            };
        }
        Self {
            id,
            connections,
            node_id_gen: 0,
            nodes,
            species_id: 0,
            fitness: 0.0,
            inputs,
            outputs,
            network_depth: 1,
        }
    }
    pub(crate) fn max_depth(&self) -> usize {
        let mut max = 0;
        // println!("START MAX DEPTH ---------------------------------------------------------------");
        for o in self.inputs..(self.inputs + self.outputs) {
            // println!("checking output {}", o);
            let (current, aborted) = self.depth(0, o);
            if aborted {
                return 10;
            }
            if current > max {
                max = current;
            }
        }
        max
    }
    pub(crate) fn depth(&self, count: usize, to: NodeId) -> (usize, bool) {
        let mut max = count;
        if count > 100 {
            // println!("aborting depth @ {}", count);
            return (10, true);
        }
        for c in self.connections.iter() {
            if c.to == to {
                // println!("incoming-connection from: {} to: {}", c.from, c.to);
                let (current, aborted) = self.depth(count + 1, c.from);
                if aborted {
                    return (current, true);
                }
                if current > max {
                    max = current;
                }
            }
        }
        // println!("max {}", max);
        (max, false)
    }
    pub(crate) fn activate<I: Into<Input>>(&self, input: I) -> Output {
        let input = input.into();
        let mut solved_outputs = Output::new(vec![0.0; self.outputs]);
        let mut summations = vec![0f32; self.nodes.len()];
        let mut activations = vec![0f32; self.nodes.len()];
        // println!("max-depth: {} for {}", self.network_depth, self.id);
        for _relax in 0..self.network_depth.max(1) {
            let mut solved = vec![false; self.outputs];
            let mut valid = vec![false; self.nodes.len()];
            for i in 0..self.inputs {
                *activations.get_mut(i).unwrap() = input.data[i];
                *summations.get_mut(i).unwrap() = input.data[i];
                *valid.get_mut(i).unwrap() = true;
            }
            for bias in (self.inputs + self.outputs)..(self.inputs + self.outputs * 2) {
                *activations.get_mut(bias).unwrap() = 1.0;
                *summations.get_mut(bias).unwrap() = 1.0;
                *valid.get_mut(bias).unwrap() = true;
            }
            const ABORT: usize = 20;
            let mut abort = 0;
            let non_input = self
                .nodes
                .iter()
                .filter(|n| n.ty != NodeType::Input)
                .copied()
                .collect::<Vec<_>>();
            while solved.iter().any(|s| *s == false) && abort < ABORT {
                if abort == ABORT {
                    println!("aborting {}", self.id);
                    return solved_outputs;
                }
                for non in non_input.iter() {
                    *summations.get_mut(non.id).unwrap() = 0.0;
                    *valid.get_mut(non.id).unwrap() = false;
                    let incoming = self
                        .connections
                        .iter()
                        .filter(|c| c.to == non.id)
                        .cloned()
                        .collect::<Vec<_>>();
                    let current_values = incoming
                        .iter()
                        .map(|i| activations.get(i.from).copied().unwrap_or_default())
                        .collect::<Vec<_>>();
                    let sum = current_values
                        .iter()
                        .enumerate()
                        .map(|(i, a)| *a * incoming.get(i).unwrap().weight)
                        .sum::<f32>();
                    *summations.get_mut(non.id).unwrap() += sum;
                    if valid.iter().any(|a| *a == true) {
                        *valid.get_mut(non.id).unwrap() = true;
                    }
                }
                for non in non_input.iter() {
                    if *valid.get(non.id).unwrap() {
                        let out = summations.get(non.id).copied().unwrap();
                        *activations.get_mut(non.id).unwrap() =
                            sigmoid(Self::ACTIVATION_SCALE * out);
                        for output_test in self.inputs..self.inputs + self.outputs {
                            if output_test == non.id {
                                solved[output_test - self.inputs] = true;
                            }
                        }
                    }
                }
                abort += 1;
            }
            // println!("activations {} : {:?}", self.id, activations);
            for i in self.inputs..self.inputs + self.outputs {
                solved_outputs.data[i - self.inputs] = *activations.get(i).unwrap();
            }
        }
        solved_outputs
    }
}
pub(crate) fn sigmoid(z: f32) -> f32 {
    1.0 / (1.0 + (-z).exp())
}
pub(crate) type NodeId = usize;
#[derive(Clone, Copy, PartialEq, Debug)]
pub(crate) enum NodeType {
    Input,
    Output,
    Bias,
    Hidden,
}
#[derive(Copy, Clone, Debug)]
pub(crate) struct Node {
    pub(crate) id: NodeId,
    pub(crate) ty: NodeType,
}
impl Display for Node {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        f.write_fmt(format_args!("id: {} ty: {:?}", self.id, self.ty))
    }
}
impl Node {
    pub(crate) fn new(id: usize, ty: NodeType) -> Self {
        Self { id, ty }
    }
}

#[derive(Clone, Copy)]
pub(crate) struct Connection {
    pub(crate) from: NodeId,
    pub(crate) to: NodeId,
    pub(crate) weight: f32,
    pub(crate) enabled: bool,
    pub(crate) innovation: Innovation,
}
impl Display for Connection {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        f.write_fmt(format_args!("\nfrom: {}\n", self.from))?;
        f.write_fmt(format_args!("to: {}\n", self.to))?;
        f.write_fmt(format_args!("weight: {}\n", self.weight))?;
        f.write_fmt(format_args!("enabled: {}\n", self.enabled))?;
        f.write_fmt(format_args!("innovation: {}", self.innovation))
    }
}
impl Connection {
    pub(crate) fn new(from: NodeId, to: NodeId, weight: f32, innovation: Innovation) -> Self {
        Self {
            from,
            to,
            weight,
            enabled: true,
            innovation,
        }
    }
}
pub(crate) type Innovation = usize;
pub(crate) struct ExistingInnovation {
    pub(crate) existing: HashMap<(NodeId, NodeId), Innovation>,
    pub(crate) generator: Innovation,
}

impl ExistingInnovation {
    pub(crate) fn new(inputs: usize, outputs: usize) -> Self {
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
#[derive(Clone)]
pub(crate) struct Species {
    pub(crate) id: SpeciesId,
    pub(crate) members: Vec<GenomeId>,
    pub(crate) representative: Genome,
    pub(crate) explicit_fitness_sharing: Fitness,
    pub(crate) max_fitness: Fitness,
    pub(crate) last_improvement: Generation,
    pub(crate) percent_total: f32,
}

impl Species {
    pub(crate) fn new(id: SpeciesId, representative: Genome, gen: Generation) -> Species {
        Self {
            id,
            members: vec![representative.id],
            representative,
            explicit_fitness_sharing: 0.0,
            max_fitness: 0.0,
            last_improvement: gen,
            percent_total: 0.0,
        }
    }
}

pub(crate) struct Compatibility {
    pub(crate) excess: f32,
    pub(crate) disjoint: f32,
    pub(crate) weight_difference: f32,
    pub(crate) n: f32,
}
impl Compatibility {
    pub(crate) fn new(better: &Genome, other: &Genome) -> Self {
        let mut excess = 0.0;
        let mut disjoint = 0.0;
        let mut weight_difference = 0.0;
        let mut num_weights = 0.0;
        let lesser_innovation_max = other
            .connections
            .iter()
            .map(|c| c.innovation)
            .max()
            .unwrap_or_default();
        for conn in better.connections.iter() {
            let other_gene = other
                .connections
                .iter()
                .find(|c| c.innovation == conn.innovation);
            if let Some(matching) = other_gene {
                weight_difference += conn.weight - matching.weight;
                num_weights += 1.0;
            }
            if conn.innovation > lesser_innovation_max {
                excess += 1.0;
            } else if other_gene.is_none() {
                disjoint += 1.0;
            }
        }
        let lesser_node_id_max = other
            .nodes
            .iter()
            .max_by(|a, b| a.id.partial_cmp(&b.id).unwrap())
            .cloned()
            .unwrap_or(Node::new(0, NodeType::Hidden))
            .id;
        for node in better.nodes.iter() {
            if node.id > lesser_node_id_max {
                excess += 1.0;
            } else if other.nodes.iter().find(|n| n.id == node.id).is_none() {
                disjoint += 1.0;
            }
        }
        let n = better.connections.len().max(other.connections.len());
        let n = if n < 20 { 1.0 } else { n as f32 };
        Self {
            excess,
            disjoint,
            weight_difference: weight_difference / num_weights,
            n,
        }
    }
    pub(crate) fn distance(&self, environment: &Environment) -> f32 {
        environment.c1 * self.excess / self.n
            + environment.c2 * self.disjoint / self.n
            + environment.c3 * self.weight_difference
    }
}
pub(crate) struct SpeciesManager {
    pub(crate) total_fitness: Fitness,
    pub(crate) species: Vec<Species>,
    pub(crate) species_id_gen: SpeciesId,
}

impl SpeciesManager {
    pub(crate) fn total(&self) -> usize {
        self.species.len()
    }
    pub(crate) fn speciate(
        &mut self,
        genomes: &mut Vec<Genome>,
        environment: &Environment,
        gen: Generation,
    ) {
        for species in self.species.iter_mut() {
            species.members.clear();
        }
        for genome in genomes.iter_mut() {
            let mut found = None;
            for species in self.species.iter() {
                let distance =
                    Compatibility::new(&species.representative, &genome).distance(environment);
                if distance < environment.compatibility_threshold {
                    found = Some(species.id);
                    break;
                }
            }
            if let Some(f) = found {
                let idx = self.species.iter().position(|s| s.id == f).unwrap();
                self.species.get_mut(idx).unwrap().members.push(genome.id);
                genome.species_id = f;
            } else {
                let id = self.species_id_gen;
                self.species_id_gen += 1;
                self.species.push(Species::new(id, genome.clone(), gen));
            }
        }
        let mut empty = vec![];
        for species in self.species.iter_mut() {
            if !species.members.is_empty() {
                let rand_idx = rand::thread_rng().gen_range(0..species.members.len());
                let rand_rep = *species.members.get(rand_idx).unwrap();
                let representative = genomes.get(rand_rep).unwrap().clone();
                species.representative = representative;
            } else {
                // println!("empty @ {}", species.members.len());
                empty.push(species.id);
            }
        }
        for id in empty.iter_mut() {
            *id = self.species.iter().position(|s| s.id == *id).unwrap();
        }
        empty.sort();
        empty.reverse();
        for idx in empty {
            // println!(
            //     "removing species w/ member-count: {} and id {}",
            //     self.species.get(idx).unwrap().members.len(),
            //     idx
            // );
            self.species.remove(idx);
        }
    }
    pub(crate) fn new(inputs: usize, outputs: usize) -> Self {
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
    pub(crate) fitness_range: Fitness,
    pub(crate) total_fitness: Fitness,
    pub(crate) next_gen_remaining: usize,
    pub(crate) to_cull: Vec<usize>,
    pub(crate) best_genome: Genome,
    pub(crate) mutations: Vec<MutationHistory>,
    pub(crate) lineage: Vec<Lineage>,
    pub(crate) next_gen_id: usize,
}
impl Runner {
    pub(crate) const NEW_LIMIT_DELTA: f32 = 0.0000001;
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
            best_genome: Genome::new(0, 0, 0),
            mutations: vec![],
            lineage: vec![],
            next_gen_id: 0,
        }
    }
}
pub(crate) struct Lineage {
    pub(crate) best: Genome,
    pub(crate) other: Genome,
    pub(crate) crossover: Genome,
}
impl Lineage {
    pub(crate) fn new(best: Genome, other: Genome, crossover: Genome) -> Self {
        Self {
            best,
            other,
            crossover,
        }
    }
}
pub(crate) enum MutationHistory {
    Node(NodeMutation),
    Connection(ConnectionMutation),
}
pub(crate) struct NodeMutation {}
pub(crate) struct ConnectionMutation {}
pub(crate) struct GenerationHistory {
    pub(crate) generation: Generation,
    pub(crate) best: Genome,
    pub(crate) top_of_generation: Genome,
    pub(crate) total_species: usize,
    pub(crate) mutations: Vec<MutationHistory>,
    pub(crate) lineage: Vec<Lineage>,
}

impl GenerationHistory {
    pub(crate) fn new(
        generation: Generation,
        best: Genome,
        top_of_generation: Genome,
        total_species: usize,
        mutations: Vec<MutationHistory>,
        lineage: Vec<Lineage>,
    ) -> GenerationHistory {
        Self {
            generation,
            best,
            top_of_generation,
            total_species,
            mutations,
            lineage,
        }
    }
}

pub(crate) struct Environment {
    pub(crate) c1: f32,
    pub(crate) c2: f32,
    pub(crate) c3: f32,
    pub(crate) compatibility_threshold: f32,
    pub(crate) elitism: f32,
    pub(crate) only_mutate: f32,
    pub(crate) stagnation_threshold: Generation,
    pub(crate) add_node: f32,
    pub(crate) add_connection: f32,
    pub(crate) inherit_disable: f32,
    pub(crate) connection_weight: f32,
    pub(crate) perturb: f32,
    pub(crate) inputs: usize,
    pub(crate) outputs: usize,
    pub(crate) crossover_only: f32
}

impl Environment {
    pub(crate) fn crossover(&self, id: GenomeId, best: Genome, other: Genome) -> Genome {
        let mut child = Genome::new(id, self.inputs, self.outputs);
        // println!(
        //     "CROSSOVER ----------------------------------------------------------------------"
        // );
        for conn in best.connections.iter() {
            let mut gene = conn.clone();
            let mut from_type = best.nodes.iter().find(|n| n.id == gene.from).unwrap().ty;
            let mut to_type = best.nodes.iter().find(|n| n.id == gene.to).unwrap().ty;
            // println!(
            //     "best-connection: {}\n from: {} to: {}",
            //     conn,
            //     best.nodes.iter().find(|n| n.id == gene.from).unwrap(),
            //     best.nodes.iter().find(|n| n.id == gene.to).unwrap()
            // );
            if let Some(matching) = other
                .connections
                .iter()
                .find(|c| c.innovation == conn.innovation)
            {
                if rand::thread_rng().gen_range(0.0..1.0) < 0.5 {
                    gene = matching.clone();
                    from_type = other.nodes.iter().find(|n| n.id == gene.from).unwrap().ty;
                    to_type = other.nodes.iter().find(|n| n.id == gene.to).unwrap().ty;
                    // println!(
                    //     "matching-connection: {}\n from: {} to: {}",
                    //     conn,
                    //     other.nodes.iter().find(|n| n.id == gene.from).unwrap(),
                    //     other.nodes.iter().find(|n| n.id == gene.to).unwrap()
                    // );
                }
                if !conn.enabled || !matching.enabled {
                    if rand::thread_rng().gen_range(0.0..1.0) < self.inherit_disable {
                        // println!("disabling gene: {} {}", gene.from, gene.to);
                        gene.enabled = false;
                    }
                }
            }
            if child.nodes.iter().find(|n| n.id == gene.from).is_none() {
                let n = Node::new(gene.from, from_type);
                // println!("adding missing node: {}", n);
                child.nodes.push(n);
            }
            if child.nodes.iter().find(|n| n.id == gene.to).is_none() {
                let n = Node::new(gene.to, to_type);
                // println!("adding missing node: {}", n);
                child.nodes.push(n);
            }
            if child
                .connections
                .iter()
                .find(|c| c.from == gene.from && c.to == gene.to)
                .is_none()
            {
                child.connections.push(gene);
            }
        }
        // child.inputs = self.inputs;
        // child.outputs = self.outputs;
        child.node_id_gen = child.nodes.len();
        // for node in child.nodes.iter() {
        //     println!("crossover:nodes: {}", node);
        // }
        // for conn in child.connections.iter() {
        //     println!("crossover:connections: {}", conn);
        // }
        // println!(
        //     "END CROSSOVER ||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||"
        // );
        child
    }
    pub(crate) fn mutate(
        &self,
        mut genome: Genome,
        existing_innovation: &mut ExistingInnovation,
    ) -> Genome {
        for conn in genome.connections.iter_mut() {
            if rand::thread_rng().gen_range(0.0..1.0) < self.connection_weight {
                if rand::thread_rng().gen_range(0.0..1.0) < self.perturb {
                    let perturb = rand::thread_rng().gen_range(-1.0..1.0);
                    conn.weight += perturb;
                } else {
                    conn.weight = rand::thread_rng().gen_range(0.0..1.0);
                }
            }
        }
        if rand::thread_rng().gen_range(0.0..1.0) < self.add_node {
            if genome.connections.is_empty() {
                return genome;
            }
            let new = Node::new(genome.node_id_gen, NodeType::Hidden);
            // println!("adding node {}", new.id);
            genome.node_id_gen += 1;
            let idx = rand::thread_rng().gen_range(0..genome.connections.len());
            let existing_connection = genome.connections.get(idx).cloned().unwrap();
            genome.connections.get_mut(idx).unwrap().enabled = false;
            let a = Connection::new(
                existing_connection.from,
                new.id,
                1.0,
                existing_innovation.checked_innovation(existing_connection.from, new.id),
            );
            let b = Connection::new(
                new.id,
                existing_connection.to,
                existing_connection.weight,
                existing_innovation.checked_innovation(new.id, existing_connection.to),
            );
            genome.connections.push(a);
            genome.connections.push(b);
            genome.nodes.push(new);
        } else if rand::thread_rng().gen_range(0.0..1.0) < self.add_connection {
            if let Some((input, output)) = self.select_connection_nodes(&genome) {
                // println!(
                //     "adding connection: from: {}:{:?} to: {}:{:?}",
                //     input.id, input.ty, output.id, output.ty
                // );
                let connection = Connection::new(
                    input.id,
                    output.id,
                    rand::thread_rng().gen_range(0.0..1.0),
                    existing_innovation.checked_innovation(input.id, output.id),
                );
                genome.connections.push(connection);
            }
        }
        genome
    }
    pub(crate) fn select_connection_nodes(&self, genome: &Genome) -> Option<(Node, Node)> {
        let potential_inputs = genome
            .nodes
            .iter()
            .filter(|n| n.ty != NodeType::Output)
            .copied()
            .collect::<Vec<_>>();
        let potential_outputs = genome
            .nodes
            .iter()
            .filter(|n| n.ty != NodeType::Input && n.ty != NodeType::Bias)
            .copied()
            .collect::<Vec<_>>();
        if potential_inputs.is_empty() || potential_outputs.is_empty() {
            return None;
        }
        let idx = rand::thread_rng().gen_range(0..potential_inputs.len());
        let mut input = potential_inputs.get(idx).copied().unwrap();
        let idx = rand::thread_rng().gen_range(0..potential_outputs.len());
        let mut output = potential_outputs.get(idx).copied().unwrap();
        while input.id == output.id && potential_inputs.len() > 1 {
            let idx = rand::thread_rng().gen_range(0..potential_inputs.len());
            input = potential_inputs.get(idx).copied().unwrap();
        }
        while input.id == output.id && potential_outputs.len() > 1 {
            let idx = rand::thread_rng().gen_range(0..potential_outputs.len());
            output = potential_outputs.get(idx).copied().unwrap();
        }
        if input.id == output.id {
            return None;
        }
        if genome
            .connections
            .iter()
            .find(|c| c.from == input.id && c.to == output.id)
            .is_some()
        {
            // recursive create_connection if can
            // if potential_inputs.len() > 1 || potential_outputs.len() > 1 {
            //     return self.select_connection_nodes(&genome);
            // }
            return None;
        }
        Some((input, output))
    }
    pub(crate) fn new(inputs: usize, outputs: usize) -> Self {
        Self {
            c1: 0.0,
            c2: 0.0,
            c3: 0.0,
            compatibility_threshold: 0.0,
            elitism: 0.0,
            only_mutate: 0.0,
            stagnation_threshold: 0,
            add_node: 0.0,
            add_connection: 0.0,
            inherit_disable: 0.0,
            connection_weight: 0.0,
            perturb: 0.0,
            inputs,
            outputs,
            crossover_only: 0.0,
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
    pub fn set_compatibility_threshold(&mut self, compatibility_threshold: f32) {
        self.compatibility_threshold = compatibility_threshold;
    }
    pub fn set_connection_weight(&mut self, connection_weight: f32) {
        self.connection_weight = connection_weight;
    }
    pub fn set_perturb(&mut self, perturb: f32) {
        self.perturb = perturb;
    }
    pub fn set_crossover_only(&mut self, mate_only: f32) {
        self.crossover_only = mate_only;
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
