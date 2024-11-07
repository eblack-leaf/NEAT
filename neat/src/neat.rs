use rand::Rng;
use std::collections::HashMap;
pub(crate) const XOR_INPUT: [[f32; 2]; 4] = [[1.0, 1.0], [1.0, 1.0], [1.0, 1.0], [1.0, 1.0]];
pub(crate) const XOR_OUTPUT: [f32; 4] = [0.0, 1.0, 1.0, 0.0];
pub(crate) const INPUT_DIM: usize = 2;
pub(crate) const OUTPUT_DIM: usize = 1;
pub(crate) const POPULATION_COUNT: usize = 150;
pub(crate) const GENERATIONS: usize = 300;
pub(crate) const FITNESS_THRESHOLD: f32 = 3.9;
#[test]
fn neat() {
    let mut population = Population::new(INPUT_DIM, OUTPUT_DIM, POPULATION_COUNT);
    let compatibility = Compatibility::new(1.0, 1.0, 0.4, 3.0);
    let perfect_fitness = 1.0 * XOR_INPUT.len() as f32;
    let environment = Environment::new();
    let mut existing_innovation = ExistingInnovations::new(INPUT_DIM, OUTPUT_DIM);
    let mut evaluation = Evaluation::new(GENERATIONS, FITNESS_THRESHOLD);
    let mut species_tree = SpeciesTree::new();
    species_tree.speciate(&mut population.genomes, &compatibility);
    for g in 0..evaluation.generations {
        let mut next_gen = vec![];
        for (i, genome) in population.genomes.iter_mut().enumerate() {
            genome.fitness = perfect_fitness;
            for (i, xi) in XOR_INPUT.iter().enumerate() {
                let predicted = genome.activate(xi.to_vec());
                let xo = XOR_OUTPUT[i];
                genome.fitness -= (predicted[0] - xo).powi(2);
            }
            species_tree
                .order
                .get_mut(genome.species_id)
                .unwrap()
                .explicit_fitness_sharing += genome.fitness;
        }
        let best_genome = population
            .genomes
            .iter()
            .max_by(|g, o| g.fitness.partial_cmp(&o.fitness).unwrap())
            .cloned()
            .unwrap();
        if best_genome.fitness >= evaluation.fitness_threshold {
            evaluation.history.push(GenerationMetrics::new(
                best_genome,
                g,
                species_tree.clone(),
                population.genomes.clone(),
            ));
            break;
        }
        let min_fitness = population
            .genomes
            .iter()
            .min_by(|g, o| g.fitness.partial_cmp(&o.fitness).unwrap())
            .unwrap()
            .fitness;
        let max_fitness = population
            .genomes
            .iter()
            .max_by(|g, o| g.fitness.partial_cmp(&o.fitness).unwrap())
            .unwrap()
            .fitness;
        let fit_range = (max_fitness - min_fitness).max(1.0);
        let mut culled_organisms = vec![];
        for species in species_tree.order.iter_mut() {
            if species.count == 0 {
                continue;
            }
            let species_max = species
                .current_organisms
                .iter()
                .map(|gi| -> f32 { population.genomes.get(*gi).unwrap().fitness })
                .max_by(|lhs, rhs| lhs.partial_cmp(&rhs).unwrap())
                .unwrap();
            if species.max_fitness < species_max {
                species.max_fitness = species_max;
                species.last_improvement = g;
            }
            if g > species.last_improvement + environment.stagnation_threshold {
                species.count = 0;
                for s in species.current_organisms.drain(..) {
                    culled_organisms.push(s);
                }
                continue;
            }
            let average_of_species = species.explicit_fitness_sharing / species.count as f32;
            species.explicit_fitness_sharing = (average_of_species - min_fitness) / fit_range;
            species_tree.total_fitness += species.explicit_fitness_sharing;
        }
        for culled in culled_organisms {
            let mut new_designation = None;
            while new_designation.is_none() {
                let attempted_conversion =
                    rand::thread_rng().gen_range(0..species_tree.order.len());
                let c = species_tree.order.get(attempted_conversion).unwrap().count;
                if c > 0 {
                    new_designation = Some(attempted_conversion);
                }
            }
            *population.genomes.get_mut(culled).unwrap() = species_tree
                .order
                .get(new_designation.unwrap())
                .unwrap()
                .representative
                .clone();
        }
        population
            .genomes
            .sort_by(|a, b| a.fitness.partial_cmp(&b.fitness).unwrap());
        population.genomes.reverse();
        let mut selection = population
            .genomes
            .get(0..(environment.elitism_percent * population.count as f32) as usize)
            .unwrap()
            .iter()
            .cloned()
            .collect::<Vec<Genome>>();
        let mut total_remaining = population.count;
        for (i, species) in species_tree.order.iter().enumerate() {
            if species.count > 0 {
                let requested_offspring = if i + 1 == species_tree.num_active_species() {
                    total_remaining
                } else {
                    let species_percent =
                        species.explicit_fitness_sharing / species_tree.total_fitness;
                    let of_population = species_percent * population.count as f32;
                    let requested_offspring = of_population as usize;
                    total_remaining = total_remaining
                        .checked_sub(requested_offspring)
                        .unwrap_or_default();
                    requested_offspring
                };
                let requested_offspring = if species.count > environment.champion_network_count {
                    let champion_id = *species
                        .current_organisms
                        .iter()
                        .max_by(|l, r| {
                            population
                                .genomes
                                .get(**l)
                                .unwrap()
                                .fitness
                                .partial_cmp(&population.genomes.get(**r).unwrap().fitness)
                                .unwrap()
                        })
                        .unwrap();
                    let champion = population.genomes.get(champion_id).unwrap().clone();
                    next_gen.push(champion);
                    requested_offspring.checked_sub(1).unwrap_or_default()
                } else {
                    requested_offspring
                };
                let skip_crossover =
                    (requested_offspring as f32 * environment.skip_crossover) as usize;
                let normal = requested_offspring
                    .checked_sub(skip_crossover)
                    .unwrap_or_default();
                for offspring_request in 0..skip_crossover {
                    // mutate
                    // next_gen.push(mutated)
                }
                for offspring_request in 0..normal {
                    // parent1 = random (from species.nodes)
                    // parent2 = random (from species.nodes) or environment.interspecies [all]
                    // crossover
                    // mutate
                    // next_gen.push(mutated_crossover);
                }
            }
        }
        evaluation.history.push(GenerationMetrics::new(
            best_genome,
            g,
            species_tree.clone(),
            population.genomes.clone(),
        ));
        population.genomes = next_gen;
        species_tree.speciate(&mut population.genomes, &compatibility);
    }
    println!("evaluation: {:?}", evaluation.history);
}
pub(crate) struct Evaluation {
    pub(crate) history: Vec<GenerationMetrics>,
    pub(crate) generations: usize,
    pub(crate) fitness_threshold: f32,
}
impl Evaluation {
    pub(crate) fn new(generations: usize, fitness_threshold: f32) -> Self {
        Self {
            history: vec![],
            generations,
            fitness_threshold,
        }
    }
}
pub(crate) struct ExistingInnovations {
    pub(crate) set: HashMap<(NodeId, NodeId), Innovation>,
    pub(crate) current: Innovation,
}
impl ExistingInnovations {
    pub(crate) fn new(inputs: usize, outputs: usize) -> Self {
        Self {
            set: Default::default(),
            current: Innovation::new(inputs * outputs + 1),
        }
    }
    pub(crate) fn checked_innovation(&mut self, from: NodeId, to: NodeId) -> Innovation {
        let pair = (from, to);
        if let Some(k) = self.set.get(&pair) {
            k.clone()
        } else {
            let idx = self.current.increment();
            self.set.insert(pair, idx);
            idx
        }
    }
}
#[derive(Debug)]
pub(crate) struct GenerationMetrics {
    pub(crate) best_genome: Genome,
    pub(crate) generation: usize,
    pub(crate) species: SpeciesTree,
    pub(crate) population: Vec<Genome>,
}
impl GenerationMetrics {
    pub(crate) fn new(
        best_genome: Genome,
        generation: usize,
        species: SpeciesTree,
        population: Vec<Genome>,
    ) -> Self {
        Self {
            best_genome,
            generation,
            species,
            population,
        }
    }
}
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
#[derive(Clone, Debug)]
pub(crate) enum NodeType {
    Input,
    Hidden,
    Output,
    Bias,
}
pub(crate) fn sigmoid(z: f32) -> f32 {
    1.0 / (1.0 + (-z).exp())
}
#[derive(Clone, Debug)]
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
#[derive(Clone, Debug)]
pub(crate) struct Connection {
    pub(crate) from: NodeId,
    pub(crate) to: NodeId,
    pub(crate) weight: f32,
    pub(crate) enabled: bool,
    pub(crate) innovation: Innovation,
}
impl Connection {
    pub(crate) fn new(
        from: NodeId,
        to: NodeId,
        weight: f32,
        enabled: bool,
        innovation: Innovation,
    ) -> Self {
        Self {
            from,
            to,
            weight,
            enabled,
            innovation,
        }
    }
}
#[derive(PartialOrd, PartialEq, Hash, Copy, Clone, Debug)]
pub(crate) struct Innovation {
    pub(crate) idx: usize,
}
impl Innovation {
    pub(crate) fn new(idx: usize) -> Self {
        Self { idx }
    }
    pub(crate) fn increment(&mut self) -> Self {
        let s = self.idx;
        self.idx += 1;
        Self { idx: s }
    }
}
pub(crate) type SpeciesId = usize;
#[derive(Clone, Debug)]
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
        let mut local_innovation_for_setup = Innovation::new(0);
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
                    local_innovation_for_setup.increment(),
                ));
            }
        }
        nodes.push(Node::new(nodes.len(), NodeType::Bias).value(1.0));
        connections.push(Connection::new(
            nodes.len().checked_sub(1).unwrap_or_default(),
            inputs + 1,
            rand::thread_rng().gen_range(0.0..1.0),
            true,
            local_innovation_for_setup.increment(),
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
#[derive(Clone, Debug)]
pub(crate) struct Species {
    pub(crate) current_organisms: Vec<GenomeId>,
    pub(crate) representative: Genome,
    pub(crate) count: usize,
    pub(crate) explicit_fitness_sharing: f32,
    pub(crate) max_fitness: f32,
    pub(crate) last_improvement: usize,
}
impl Species {
    pub(crate) fn new(genome: Genome) -> Self {
        Self {
            current_organisms: vec![],
            representative: genome,
            count: 0,
            explicit_fitness_sharing: 0.0,
            max_fitness: 0.0,
            last_improvement: 0,
        }
    }
}
#[derive(Clone, Debug)]
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
        self.order
            .iter()
            .filter_map(|s| if s.count > 0 { Some(1) } else { None })
            .sum()
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
                    compatibility.distance(genome.compatibility_metrics(&species.representative));
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
                let idx = self.order.len();
                let mut g = genome.clone();
                g.species_id = idx;
                self.order.push(Species::new(g));
                idx
            };
            genome.species_id = s;
            self.order.get_mut(s).unwrap().count += 1;
        }
        for species in self.order.iter_mut() {
            if species.count > 0 {
                let rand_idx = rand::thread_rng().gen_range(0..species.current_organisms.len());
                let rand_rep = *species.current_organisms.get(rand_idx).unwrap();
                let representative = population.get(rand_rep).unwrap().clone();
                species.representative = representative;
            }
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
    pub(crate) elitism_percent: f32,
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
            elitism_percent: 0.3,
        }
    }
}
