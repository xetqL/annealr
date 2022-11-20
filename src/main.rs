use core::panic;
use std::{vec, fs::File, io::Write, path::Path};
use rand::{Rng, rngs::ThreadRng};

#[derive(Debug)]
struct Graph {
    N: usize,
    adj: Vec<Vec<i32>>,
    edge_weight: Vec<Vec<f32>>,
    vertex_weight: Vec<f32>
}

fn rand_adjacency(N: usize, rng: &mut ThreadRng) -> Vec<Vec<i32>> { 
    let mut arr = vec![ vec![0; N]; N];
    for i in 0..N {
        for j in 0..N {
            arr[i][j] = rng.gen_range(0..=1)
        }
        arr[i][i] = 0;   
    }
    arr
}

impl Graph {
    fn new_null(N: usize) -> Self {
        Graph { N, adj:vec![ vec![0; N]; N], edge_weight:vec![ vec![0. as f32; N]; N], vertex_weight:vec![0. as f32; N], }
    }

    fn new_randomized(N: usize) -> Self {
        let mut rng = rand::thread_rng();
        let adj  = rand_adjacency(N, &mut rng);
        let mut edge : Vec<Vec<f32>> = vec![ vec![0.0; N]; N];

        for i in 0..N {
            for j in 0..N {
                if adj[i][j] > 0 {
                    edge[i][j] = rng.gen(); 
                }
            }
        }

        let mut vertex: Vec<f32> = vec![0.0; N];
        rng.fill(&mut vertex[..]); 
        Graph { 
            N, 
            adj, 
            edge_weight: edge, 
            vertex_weight: vertex, 
        }
    }

    fn new_grid<F>(grid_size: usize, ) -> Self {
        todo!()
    }

    fn import_from_file(path: &Path) -> Self{
        todo!()
    }

}

#[derive(Clone, Debug, PartialEq)]
enum Partition {
    None, Id(usize)
}

#[derive(Clone)]
struct PartitioningStatistics {
    //load_imbalance: f32,
    load_per_partition: Vec<f32>,
    ncut: usize,
    cut_cost: f32,
    // ... other metric can be implemented here ... 
}

struct Partitioning<'a> {
    n_part: usize,
    vertex_partition: Vec<Partition>,
    graph: &'a Graph,
    details: PartitioningStatistics
}

fn compute_cut_contribution(vertices: &[usize], partitioning: &Partitioning) -> f32 {
    let mut cut_cost = 0f32;
    for ivertex in vertices {
        let pvertex = &partitioning.vertex_partition[*ivertex];
        for (ineighbor, neighbor_edge_weight) in partitioning.graph.edge_weight[*ivertex].iter().enumerate() {
            if neighbor_edge_weight > &0f32 && &partitioning.vertex_partition[ineighbor] == pvertex {
                cut_cost += neighbor_edge_weight;
            }
        }
    }
    cut_cost
}

impl<'a> Partitioning<'a> {

    fn full_compute_statistics(mut self) -> Self {
        
        let mut load_per_partition = vec![0.; self.n_part];
        let mut ncut : usize = 0;
        let mut cut_cost: f32 = 0.;
        
        for (ivertex, wvertex) in self.graph.vertex_weight.iter().enumerate() {
            match self.vertex_partition[ivertex] {
                Partition::Id(id) => {
                    load_per_partition[id] += wvertex;
                    for (ineighbor, neighbor_edge_weight) in self.graph.edge_weight[ivertex].iter().skip(ivertex).enumerate() {
                        if neighbor_edge_weight > &0f32 && self.vertex_partition[ineighbor] == self.vertex_partition[ivertex] {
                            ncut += 1;
                            cut_cost += neighbor_edge_weight;
                        }
                    }
                },
                Partition::None => continue
            }
        }

        self.details.load_per_partition = load_per_partition;
        self.details.ncut = ncut;
        self.details.cut_cost = cut_cost;

        self
    }

    fn new_random(npart:usize, g:&'a Graph) -> Self {
        let n_per_partition = g.N / npart;
        let mut random_hat : Vec<usize> = (0..g.N).collect();
        let mut vertex_partition = vec![Partition::None; g.N];

        let mut rng = rand::thread_rng();

        // for each p part
        for p in 0..npart {
            // take out n_per_partition node randomly and assign them id(i)
            for _ in 0..n_per_partition {
                let candidate_index = rng.gen_range(0..random_hat.len());
                vertex_partition[random_hat[candidate_index]] = Partition::Id(p);
                random_hat.swap_remove(candidate_index);
            }
        }
        
        assert!(random_hat.is_empty());

        Partitioning {
            n_part: npart,
            vertex_partition,
            graph: g,
            details: PartitioningStatistics { load_per_partition: vec![0f32; npart], ncut: 0, cut_cost: 0. }
        }.full_compute_statistics()

    }

    fn new_simulated_annealing<Fitness: Fn(&Partitioning, &Graph) -> f32>(npart: usize, graph: &'a Graph, initial_temp: f32, fitness: Fitness) 
        -> Self {
        let current_solution = Partitioning::new_random(npart, graph);
        
        let current_fitness = fitness(&current_solution, graph);
        let mut rng = rand::thread_rng();
        
        let i = rng.gen_range(0..=graph.N);
        let j = rng.gen_range(0..=graph.N);

        let next_solution = current_solution.swap(i, j);
        let next_fitness = fitness(&current_solution, graph);

        let current_solution = if next_fitness < current_fitness {next_solution} else {current_solution}

        current_solution
    }

    fn swap(&'a self, i:usize, j:usize) -> Self {
        let mut new_part = Partitioning { 
            n_part: self.n_part, 
            vertex_partition: self.vertex_partition.clone(), 
            graph: self.graph, 
            details: self.details.clone() 
        };
        
        // update cut cost
        let contrib = compute_cut_contribution(&[i, j], &new_part);
        new_part.details.cut_cost -= contrib;
        new_part.vertex_partition.swap(i, j);
        let contrib = compute_cut_contribution(&[i, j], &new_part);
        new_part.details.cut_cost += contrib;

        new_part
    }

}

fn export_as_csv(folder: &Path, partitioning: &Partitioning) -> std::io::Result<()> {
    let mut fgraph = File::create(folder.join("vertices.data"))?;
    let mut fpart  = File::create(folder.join("part.data"))?;
    let g = partitioning.graph;
    fgraph.write(b"#size\n")?;
    fgraph.write(g.N.to_string().as_bytes())?;
    fgraph.write("\n#vertices\n".as_bytes())?;
    fgraph.write(g.vertex_weight.iter().map(|x| x.to_string()).collect::<Vec<String>>().join(",").as_bytes())?;
    fgraph.write("\n#adjacency\n".as_bytes())?;
    for row in &g.adj {
        fgraph.write(row.iter().map(|x| x.to_string()).collect::<Vec<String>>().join(" ").as_bytes())?;
        fgraph.write("\n".as_bytes())?;
    }  
    fgraph.write("#edges\n".as_bytes())?;
    for row in &g.edge_weight {
        fgraph.write(row.iter().map(|x| x.to_string()).collect::<Vec<String>>().join(" ").as_bytes())?;
        fgraph.write("\n".as_bytes())?;
    }  
    fpart.write(b"#size\n")?;
    fpart.write(g.N.to_string().as_bytes())?;
    fpart.write("\n#partitions\n".as_bytes())?;
    fpart.write(partitioning.vertex_partition.iter().map(|x| match x  {
        Partition::None => panic!("incomplete partition"),
        Partition::Id(i) => i.to_string()
    }).collect::<Vec<String>>().join(",").as_bytes())?;

    Ok(())
}

fn main() {
    let graph = Graph::new_randomized(100);
    let part = Partitioning::new_random(4, &graph);
    export_as_csv(Path::new("/tmp/"), &part).unwrap();
}
