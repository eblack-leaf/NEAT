use rand::Rng;
use std::cmp::PartialEq;
use std::collections::{HashMap, HashSet};
use std::fmt::{Display, Formatter, Write};

pub(crate) const XOR_INPUT: [[f32; 2]; 4] = [[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]];
pub(crate) const XOR_OUTPUT: [f32; 4] = [0.0, 1.0, 1.0, 0.0];
pub(crate) const INPUT_DIM: usize = 2;
pub(crate) const OUTPUT_DIM: usize = 1;
pub(crate) const POPULATION_COUNT: usize = 150;
pub(crate) const GENERATIONS: usize = 300;
pub(crate) const FITNESS_THRESHOLD: f32 = 3.9;
pub(crate) const ACTIVATION_SCALE: f32 = 4.9;

pub(crate) fn neat() {
    let mut population = Population::new(INPUT_DIM, OUTPUT_DIM, POPULATION_COUNT);
    let compatibility = Compatibility::new(1.0, 1.0, 0.5, 3.0);
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
            println!(
                "evaluation@{}: {}",
                g,
                evaluation.history.last().unwrap().best_genome
            );
            return;
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
        let mut current_active_species = species_tree.num_active_species();
        println!("current-active-species: {}", current_active_species);
        for (i, species) in species_tree.order.iter_mut().enumerate() {
            if species.count == 0 || species.culled {
                // println!(
                //     "skipping fitness-evaluation for species {} because {} or {}",
                //     i, species.count, species.culled
                // );
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
                current_active_species -= 1;
                if current_active_species != 0 {
                    println!("culling: {}", i);
                    species.count = 0;
                    species.culled = true;
                    for s in species.current_organisms.drain(..) {
                        culled_organisms.push(s);
                    }
                    continue;
                } else {
                    println!("skipping cull for {}", i);
                }
            }
            let average_of_species = species.explicit_fitness_sharing / species.count as f32;
            species.explicit_fitness_sharing = (average_of_species - min_fitness) / fit_range;
            // species.explicit_fitness_sharing = average_of_species;
            species_tree.total_fitness += species.explicit_fitness_sharing;
            // println!(
            //     "species[{}].fitness: {} total-fitness: {} @ count: {}",
            //     i, species.explicit_fitness_sharing, species_tree.total_fitness, species.count
            // );
        }
        // for culled in culled_organisms {
        //     let mut new_designation = None;
        //     while new_designation.is_none() {
        //         let attempted_conversion =
        //             rand::thread_rng().gen_range(0..species_tree.order.len());
        //         let c = species_tree.order.get(attempted_conversion).unwrap().count;
        //         if c > 0 {
        //             new_designation = Some(attempted_conversion);
        //         }
        //     }
        //     let mut to_copy_from = species_tree
        //         .order
        //         .get(new_designation.unwrap())
        //         .unwrap()
        //         .representative
        //         .clone();
        //     to_copy_from.id = culled;
        //     to_copy_from.species_id = new_designation.unwrap();
        //     *population.genomes.get_mut(culled).unwrap() = to_copy_from;
        //     species_tree
        //         .order
        //         .get_mut(new_designation.unwrap())
        //         .unwrap()
        //         .count += 1;
        // }
        let mut total_remaining = population.count;
        // println!("total-remaining: {}", total_remaining);
        let mut g_id = 0;
        let mut last_species_id = species_tree.order.len();
        for (i, s) in species_tree.order.iter().enumerate().rev() {
            if !s.culled {
                last_species_id = i;
                break;
            }
        }
        for (species_id, species) in species_tree.order.iter().enumerate() {
            if species.count > 0 && !species.culled {
                let requested_offspring = if species_id == last_species_id {
                    // println!("total-remaining: {} for species[{}].fitness: {} / total: {} = {} * population: {} = {}",
                    //          total_remaining,
                    //     species_id,
                    //     species.explicit_fitness_sharing,
                    //     species_tree.total_fitness,
                    //     species.explicit_fitness_sharing / species_tree.total_fitness,
                    //     population.count,
                    //          species.explicit_fitness_sharing / species_tree.total_fitness * population.count as f32
                    // );
                    // println!("total-remaining: {} for {}", total_remaining, species_id);
                    total_remaining
                } else {
                    let species_percent =
                        species.explicit_fitness_sharing / species_tree.total_fitness;
                    let of_population = species_percent * population.count as f32;
                    // println!(
                    //     "species[{}].fitness: {} / total: {} = {} * population: {} = {}",
                    //     species_id,
                    //     species.explicit_fitness_sharing,
                    //     species_tree.total_fitness,
                    //     species_percent,
                    //     population.count,
                    //     of_population
                    // );
                    let requested_offspring = of_population as usize;
                    total_remaining = total_remaining
                        .checked_sub(requested_offspring)
                        .unwrap_or_default();
                    requested_offspring
                };
                // println!("requested[{}]: {}", species_id, requested_offspring);
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
                    let mut champion = population.genomes.get(champion_id).unwrap().clone();
                    champion.id = g_id;
                    next_gen.push(champion);
                    g_id += 1;
                    requested_offspring.checked_sub(1).unwrap_or_default()
                } else {
                    requested_offspring
                };
                let skip_crossover =
                    (requested_offspring as f32 * environment.skip_crossover) as usize;
                let normal = requested_offspring
                    .checked_sub(skip_crossover)
                    .unwrap_or_default();
                // println!("skip: {} normal: {}", skip_crossover, normal);
                let mut species_selection = species
                    .current_organisms
                    .iter()
                    .filter_map(|gi| Some(population.genomes.get(*gi).unwrap().clone()))
                    .collect::<Vec<Genome>>();
                species_selection.sort_by(|a, b| a.fitness.partial_cmp(&b.fitness).unwrap());
                species_selection.reverse();
                let elitist_percent =
                    (environment.elitism_percent * species_selection.len() as f32).ceil() as usize;
                let elitist_percent = if elitist_percent > species_selection.len() {
                    species_selection.len()
                } else {
                    elitist_percent.max(1)
                };
                // println!("elitist-percent: {}", elitist_percent);
                species_selection = species_selection.get(0..elitist_percent).unwrap().to_vec();
                if species_selection.is_empty() {
                    total_remaining += requested_offspring;
                    // println!("skipping: {}", species_id);
                    continue;
                }
                for _offspring_request in 0..skip_crossover {
                    let rand_idx = rand::thread_rng().gen_range(0..species_selection.len());
                    let selected = species_selection.get(rand_idx).unwrap().clone();
                    let mut mutated = environment.mutate(selected, &mut existing_innovation);
                    mutated.id = g_id;
                    next_gen.push(mutated);
                    g_id += 1;
                }
                for _offspring_request in 0..normal {
                    let parent1_idx = rand::thread_rng().gen_range(0..species_selection.len());
                    let parent1 = species_selection.get(parent1_idx).unwrap().clone();
                    let mut parent2_idx = parent1_idx;
                    while parent2_idx == parent1_idx && species_selection.len() > 1 {
                        parent2_idx = rand::thread_rng().gen_range(0..species_selection.len());
                    }
                    let parent2 = species_selection.get(parent2_idx).unwrap().clone();
                    let crossover = crossover(g_id, parent1, parent2, &environment);
                    let mutated_crossover = environment.mutate(crossover, &mut existing_innovation);
                    next_gen.push(mutated_crossover);
                    g_id += 1;
                }
            }
        }
        let metrics = GenerationMetrics::new(
            best_genome,
            g,
            species_tree.clone(),
            population.genomes.clone(),
        );
        println!("metrics: {:?} @ {}", metrics.best_genome.fitness, g,);
        evaluation.history.push(metrics);
        population.genomes = next_gen;
        species_tree.speciate(&mut population.genomes, &compatibility);
    }
    println!(
        "evaluation: {}",
        evaluation.history.last().unwrap().best_genome
    );
}
pub(crate) fn crossover(
    id: GenomeId,
    parent1: Genome,
    parent2: Genome,
    environment: &Environment,
) -> Genome {
    let mut child = Genome::blank(id);
    let (best_parent, other) = if parent1.fitness == parent2.fitness {
        (parent1, parent2)
    } else if parent1.fitness > parent2.fitness {
        (parent1, parent2)
    } else {
        (parent2, parent1)
    };
    child.nodes.resize(
        best_parent.nodes.len(),
        Node::new(NodeId::MAX, NodeType::Hidden),
    );
    child.node_id_generator = best_parent.nodes.len();
    for c in best_parent.connections.iter() {
        let mut gene = c.clone();
        let mut nodes = (None, None);
        nodes.0 = Some(best_parent.nodes.get(gene.from).unwrap().clone());
        nodes.1 = Some(best_parent.nodes.get(gene.to).unwrap().clone());
        if let Some(matching) = other
            .connections
            .iter()
            .find(|b| b.innovation == c.innovation)
        {
            if !matching.enabled {
                gene.enabled = false;
            }
            if rand::thread_rng().gen_range(0.0..1.0) < 0.5 {
                gene = matching.clone();
                if !c.enabled {
                    gene.enabled = false;
                }
                nodes.0 = Some(other.nodes.get(gene.from).unwrap().clone());
                nodes.1 = Some(other.nodes.get(gene.to).unwrap().clone());
            }
        }
        if !gene.enabled {
            let should_enable = rand::thread_rng().gen_range(0.0..1.0) < environment.reenable_gene;
            // println!("reenable: {}", should_enable);
            gene.enabled = should_enable;
        }
        child.connections.push(gene.clone());
        *child.nodes.get_mut(gene.from).unwrap() = nodes.0.unwrap();
        *child.nodes.get_mut(gene.to).unwrap() = nodes.1.unwrap();
        // println!("adding gene: {:?}", gene);
    }
    for node in best_parent.nodes.iter() {
        let mut co = node.clone();
        if child.nodes.iter().find(|n| n.id == node.id).is_some() {
            continue;
        }
        if let Some(matching) = other.nodes.get(node.id) {
            if rand::thread_rng().gen_range(0.0..1.0) < 0.5 {
                co = matching.clone();
            }
        }
        *child.nodes.get_mut(node.id).unwrap() = co;
    }
    child
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
            set: {
                let mut set = HashMap::new();
                let mut innov = 0;
                for i in 0..inputs {
                    for o in inputs..(inputs + outputs) {
                        set.insert((i, o), Innovation::new(innov));
                        innov += 1;
                    }
                }
                for i in (inputs + outputs)..(inputs + outputs * 2) {
                    for o in inputs..(inputs + outputs) {
                        set.insert((i, o), Innovation::new(innov));
                        innov += 1;
                    }
                }
                set
            },
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
#[derive(Clone, Debug, PartialEq)]
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
impl Display for Genome {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        f.write_fmt(format_args!(
            "Genome: {} w/ fitness: {}\n",
            self.id, self.fitness
        ))?;
        f.write_str("nodes:\n")?;
        for n in self.nodes.iter() {
            f.write_fmt(format_args!("id: {}@{:?}\n", n.id, n.ty))?;
        }
        f.write_str("connections:\n")?;
        for c in self.connections.iter() {
            f.write_fmt(format_args!(
                "from: {} to: {} weight: {} enabled: {} innov: {}\n",
                c.from, c.to, c.weight, c.enabled, c.innovation.idx
            ))?;
        }
        f.write_fmt(format_args!("species: {}\n", self.species_id))?;
        Ok(())
    }
}
impl Genome {
    pub(crate) fn blank(id: GenomeId) -> Self {
        Self {
            nodes: vec![],
            connections: vec![],
            fitness: 0.0,
            node_id_generator: 0,
            species_id: 0,
            id,
        }
    }
    pub(crate) fn compatibility_metrics(&self, other: &Self) -> CompatibilityMetrics {
        let mut excess = 0;
        let mut disjoint = 0;
        let max_innovation = other
            .connections
            .iter()
            .map(|c| c.innovation.idx)
            .max()
            .unwrap_or_default();
        for primary in self.connections.iter() {
            if primary.innovation.idx > max_innovation {
                excess += 1;
            } else if other
                .connections
                .iter()
                .find(|secondary| secondary.innovation == primary.innovation)
                .is_none()
            {
                disjoint += 1;
            }
        }
        let max_node_id = other
            .nodes
            .iter()
            .max_by(|a, b| a.id.partial_cmp(&b.id).unwrap()).cloned()
            .unwrap_or(Node::new(0, NodeType::Hidden))
            .id;
        for node in self.nodes.iter() {
            if node.id > max_node_id {
                excess += 1;
            } else if other
                .nodes
                .iter()
                .find(|secondary| secondary.id == node.id)
                .is_none()
            {
                disjoint += 1;
            }
        }
        let mut num_weights = 0.0;
        let mut weight_difference = 0.0;
        for c in self.connections.iter() {
            if let Some(matching) = other
                .connections
                .iter()
                .find(|b| b.innovation == c.innovation)
            {
                weight_difference += c.weight - matching.weight;
                num_weights += 1.0;
            }
        }
        if num_weights == 0.0 {
            num_weights = 1.0;
        }
        weight_difference /= num_weights;
        let n = self.connections.len().max(other.connections.len());
        let n = if n < 20 { n } else { n } as f32;
        CompatibilityMetrics {
            excess: excess as f32,
            disjoint: disjoint as f32,
            weight_difference,
            n,
        }
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
                let output_id = inputs + o;
                let innovation = local_innovation_for_setup.increment();
                connections.push(Connection::new(
                    i,
                    output_id,
                    rand::thread_rng().gen_range(0.0..1.0),
                    true,
                    innovation,
                ));
            }
        }
        let mut last = 0;
        for i in (inputs + outputs)..(inputs + outputs * 2) {
            for o in inputs..(inputs + outputs) {
                nodes.push(Node::new(i, NodeType::Bias).value(1.0));
                connections.push(Connection::new(
                    i,
                    o,
                    rand::thread_rng().gen_range(0.0..1.0),
                    true,
                    local_innovation_for_setup.increment(),
                ));
                last = i;
            }
        }
        Self {
            nodes,
            connections,
            fitness: 0.0,
            node_id_generator: last + 1,
            species_id: 0,
            id,
        }
    }
    pub(crate) fn activate(&self, inputs: Vec<f32>) -> Vec<f32> {
        let mut outputs = vec![];
        let ordered = recursive_order(&self);
        let mut staged_output = HashMap::new();
        for (i, d) in inputs.iter().enumerate() {
            staged_output.insert(i, *d);
        }
        for i in (inputs.len() + OUTPUT_DIM)..(inputs.len() + 2 * OUTPUT_DIM) {
            staged_output.insert(i, 1.0);
        }
        // println!("-------------------------------------------------------------------------------");
        // println!("unordered: {:?}", self.nodes);
        // println!("ordered: {:?}", ordered);
        for o in ordered {
            // println!("getting ordered node: {}", o);
            let node = self.nodes.get(o).unwrap();
            // println!("ordered-node: {:?}", node);
            let mut W = vec![];
            if node.ty != NodeType::Input && node.ty != NodeType::Bias {
                let input_ids = self
                    .connections
                    .iter()
                    .filter_map(|c| {
                        if c.enabled && c.to == node.id {
                            W.push(c.weight);
                            Some(c.from)
                        } else {
                            None
                        }
                    })
                    .collect::<Vec<_>>();
                // println!("input-ids: {:?} for {:?}", input_ids, o);
                let stage_input = input_ids
                    .iter()
                    .map(|id| {
                        // println!("getting-staged: {}", id);
                        if staged_output.get(id).is_none() {
                            println!(
                                "invalid-nodes: {:?} w/ connections: {:?}",
                                self.nodes, self.connections
                            );
                        }
                        staged_output.get(id).unwrap().clone()
                    })
                    .collect::<Vec<f32>>();
                let mut out = 0.0;
                for (i, x) in stage_input.iter().enumerate() {
                    let w = W.get(i).unwrap();
                    out += *w * *x;
                }
                let adjusted_out = sigmoid(out * ACTIVATION_SCALE);
                staged_output.insert(o, adjusted_out);
            }
        }
        for i in inputs.len()..(inputs.len() + OUTPUT_DIM) {
            outputs.push(staged_output.get(&i).unwrap().clone());
        }
        // println!("outputs: {:?}", outputs);
        outputs
    }
}
pub(crate) fn recursive_order(genome: &Genome) -> Vec<NodeId> {
    let mut ordered = vec![];
    let mut visited = HashSet::new();
    for n in genome.nodes.iter() {
        if !visited.contains(&n.id) {
            inner_recursion(genome, n.id, &mut visited, &mut ordered);
        }
    }
    ordered
}
pub(crate) fn inner_recursion(
    genome: &Genome,
    node_id: NodeId,
    visited: &mut HashSet<NodeId>,
    ordered: &mut Vec<NodeId>,
) {
    visited.insert(node_id);
    let stage_outputs = genome
        .connections
        .iter()
        .filter_map(|c| {
            if c.enabled && c.to == node_id {
                Some(c.from)
            } else {
                None
            }
        })
        .collect::<Vec<_>>();
    for out_node in stage_outputs {
        if !visited.contains(&out_node) {
            // println!("processing out-node: {}", out_node);
            inner_recursion(genome, out_node, visited, ordered);
        }
    }
    // println!("adding original-node: {}", node_id);
    ordered.push(node_id);
}
pub(crate) struct Compatibility {
    pub(crate) c1: f32,
    pub(crate) c2: f32,
    pub(crate) c3: f32,
    pub(crate) threshold: f32,
}
pub(crate) struct CompatibilityMetrics {
    pub(crate) excess: f32,
    pub(crate) disjoint: f32,
    pub(crate) weight_difference: f32,
    pub(crate) n: f32,
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
    pub(crate) culled: bool,
}
impl Species {
    pub(crate) fn new(genome: Genome) -> Self {
        Self {
            current_organisms: vec![genome.id],
            representative: genome,
            count: 1,
            explicit_fitness_sharing: 0.0,
            max_fitness: 0.0,
            last_improvement: 0,
            culled: false,
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
                if species.culled {
                    continue;
                }
                let distance =
                    compatibility.distance(genome.compatibility_metrics(&species.representative));
                // println!("distance: {} threshold: {}", distance, compatibility.threshold);
                if distance < compatibility.threshold.abs() {
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
                self.order.get_mut(f).unwrap().count += 1;
                f
            } else {
                let idx = self.order.len();
                // println!("new species: {}", idx);
                let mut g = genome.clone();
                g.species_id = idx;
                self.order.push(Species::new(g));
                idx
            };
            // println!("setting genome[{}] to {}", genome.id, s);
            genome.species_id = s;
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
    pub(crate) reenable_gene: f32,
    pub(crate) delete_node: f32,
    pub(crate) delete_connection: f32,
}

impl Environment {
    pub(crate) fn new() -> Self {
        Self {
            connection_weight: (0.8, 0.9, 0.1),
            disable_gene: 0.75,
            skip_crossover: 0.25,
            interspecies: 0.001,
            add_node: 0.15,
            add_connection: 0.75,
            stagnation_threshold: 20,
            champion_network_count: 5,
            elitism_percent: 0.3,
            reenable_gene: 0.25,
            delete_node: 0.05,
            delete_connection: 0.35,
        }
    }
    pub(crate) fn mutate(
        &self,
        mut genome: Genome,
        existing_innovations: &mut ExistingInnovations,
    ) -> Genome {
        for c in genome.connections.iter_mut() {
            if rand::thread_rng().gen_range(0.0..1.0) < self.connection_weight.0 {
                if rand::thread_rng().gen_range(0.0..1.0) < self.connection_weight.1 {
                    let perturb = rand::thread_rng().gen_range(-1.0..1.0);
                    c.weight += perturb;
                } else {
                    c.weight = rand::thread_rng().gen_range(0.0..1.0);
                }
            }
        }
        if rand::thread_rng().gen_range(0.0..1.0) < self.delete_node {
            let available = genome.nodes.iter().filter(|n| n.ty == NodeType::Hidden).cloned().collect::<Vec<_>>();
            if !available.is_empty() {
                let choice = available.get(rand::thread_rng().gen_range(0..available.len())).unwrap();
                let mut to_remove = vec![];
                for c_idx in 0..genome.connections.len() {
                    let c = genome.connections.get(c_idx).unwrap().clone();
                    if c.from == choice.id || c.to == choice.id {
                        to_remove.push(c_idx);
                    }
                }
                to_remove.sort();
                to_remove.reverse();
                for idx in to_remove {
                    genome.connections.remove(idx);
                }
            }
        }
        if rand::thread_rng().gen_range(0.0..1.0) < self.delete_connection {
            if !genome.connections.is_empty() {
                let choice = rand::thread_rng().gen_range(0..genome.connections.len());
                genome.connections.get_mut(choice).unwrap().enabled = false;
                // println!("removing connection: {:?}", choice);
            }
        }
        if rand::thread_rng().gen_range(0.0..1.0) < self.add_node {
            if !genome.connections.is_empty() {
                let new_node = Node::new(genome.node_id_generator, NodeType::Hidden);
                genome.node_id_generator += 1;
                // println!("id-gen: {}", genome.node_id_generator);
                let idx = rand::thread_rng().gen_range(0..genome.connections.len());
                let existing_connection = genome.connections.get(idx).unwrap().clone();
                genome.connections.get_mut(idx).unwrap().enabled = false;
                let new_first = Connection::new(
                    existing_connection.from,
                    new_node.id,
                    1.0,
                    true,
                    existing_innovations.checked_innovation(existing_connection.from, new_node.id),
                );
                let new_second = Connection::new(
                    new_node.id,
                    existing_connection.to,
                    existing_connection.weight,
                    true,
                    existing_innovations.checked_innovation(new_node.id, existing_connection.to),
                );
                // println!("adding first: {:?}", new_first);
                genome.connections.push(new_first);
                // println!("adding second: {:?}", new_second);
                genome.connections.push(new_second);
                // println!("before-nodes: {:?}", genome.nodes);
                genome.nodes.push(new_node);
                // println!("after-nodes: {:?}", genome.nodes);
            }
        }
        if rand::thread_rng().gen_range(0.0..1.0) < self.add_connection {
            let potential_inputs = genome
                .nodes
                .iter()
                .filter(|n| n.ty != NodeType::Output)
                .cloned()
                .collect::<Vec<Node>>();
            let potential_outputs = genome
                .nodes
                .iter()
                .filter(|n| n.ty != NodeType::Input && n.ty != NodeType::Bias)
                .cloned()
                .collect::<Vec<Node>>();
            if !potential_inputs.is_empty() && ! potential_outputs.is_empty() {
                let selected_input = potential_inputs
                    .get(rand::thread_rng().gen_range(0..potential_inputs.len()))
                    .cloned()
                    .unwrap();
                let selected_output = potential_outputs
                    .get(rand::thread_rng().gen_range(0..potential_outputs.len()))
                    .cloned()
                    .unwrap();
                if creates_cycle(selected_input.id, selected_output.id, &genome) {
                    return genome;
                }
                let predicate_found = genome.connections.iter().find(|c| {
                    c.from == selected_input.id && c.to == selected_output.id
                        || c.from == selected_output.id && c.to == selected_input.id
                });
                if predicate_found.is_none() {
                    let connection = Connection::new(
                        selected_input.id,
                        selected_output.id,
                        rand::thread_rng().gen_range(0.0..1.0),
                        true,
                        existing_innovations.checked_innovation(selected_input.id, selected_output.id),
                    );
                    // println!("creating: {:?}", connection);
                    genome.connections.push(connection);
                    if genome
                        .nodes
                        .iter()
                        .find(|n| n.id == selected_input.id)
                        .is_none()
                    {
                        genome.nodes.push(selected_input);
                    }
                    if genome
                        .nodes
                        .iter()
                        .find(|n| n.id == selected_output.id)
                        .is_none()
                    {
                        genome.nodes.push(selected_output);
                    }
                }
            }
        }
        genome
    }
}
pub(crate) fn creates_cycle(
    selected_input_id: NodeId,
    selected_output_id: NodeId,
    genome: &Genome,
) -> bool {
    if selected_input_id == selected_output_id {
        return true;
    }
    let mut visited = vec![selected_output_id];
    while true {
        let mut num_added = 0;
        for c in genome.connections.iter() {
            if visited.iter().find(|v| **v == c.from).is_some()
                && visited.iter().find(|v| **v == c.to).is_none()
            {
                if c.to == selected_input_id {
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
