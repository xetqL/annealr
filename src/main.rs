use std::{vec, fs::File, io::Write, path::Path};

use rand::{Rng, rngs::ThreadRng};

#[derive(Debug)]
struct Graph {
    N: usize,
    adj: Vec<Vec<i32>>,
    edge: Vec<Vec<f32>>,
    vertex: Vec<f32>
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
        Graph { N, adj:vec![ vec![0; N]; N], edge:vec![ vec![0. as f32; N]; N], vertex:vec![0. as f32; N], }
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
            edge, 
            vertex, 
        }
    }
}

#[derive(Clone, Debug)]
enum Partition {
    None, Id(usize)
}

struct PartitioningDetails {
    load_imbalance: f32,
    ncut: i32,
    cut_cost: f32,
    // ... other metric can be implemented here ... 
}

struct Partitioning<'a> {
    color: Vec<Partition>,
    graph: &'a Graph,
    details: PartitioningDetails
}

impl<'a> Partitioning<'a> {
    
    fn new_empty(g:&'a Graph) -> Self {
        Partitioning { 
            color: vec![Partition::None; g.N], 
            graph: g,
            details: PartitioningDetails { 
                load_imbalance: f32::MAX, 
                ncut: 0, 
                cut_cost: 0. 
            }
        }
    }
    
    fn set(&mut self, i: usize, id: Partition) {
        self.color[i] = id;
    }

    fn swap(&mut self, i: usize, j: usize) -> &PartitioningDetails{
        self.color.swap(i, j);
        
        &self.details
    }

}

trait PartitioningStrategy {
    fn run<'a>(&'a self, npart: usize, g: &'a Graph) -> Partitioning;
    
}

struct RandomAssignation;
impl PartitioningStrategy for RandomAssignation {
    fn run<'a>(&'a self, npart: usize, g: &'a Graph) -> Partitioning {
        let n_per_partition = g.N / npart;
        let mut random_hat : Vec<usize> = (0..g.N).collect();
        let mut partitions = Partitioning::new_empty(g);
        let mut rng = rand::thread_rng();

        // for each p part
        for p in 0..npart {
            // take out n_per_partition node randomly and assign them id(i)
            for j in 0..n_per_partition {
                let candidate_index = rng.gen_range(0..random_hat.len());
                partitions.set(random_hat[candidate_index], Partition::Id(p));
                random_hat.swap_remove(candidate_index);
            }
        }
        
        assert!(random_hat.is_empty());
        partitions
    }
}


struct SimulatedAnnealingStrategy {
    temperature: f32,
    //fitness: Fn(&Graph, &Partitioning) -> f32,
}

impl PartitioningStrategy for SimulatedAnnealingStrategy {
    fn run(&self, npart: usize, g: &Graph) -> Partitioning {
        todo!()
    }
}

fn export_as_csv(folder: &Path, partitioning: &Partitioning) -> std::io::Result<()> {
    let mut fgraph = File::create(folder.join("vertices.data"))?;
    let mut fpart  = File::create(folder.join("part.data"))?;
    let g = partitioning.graph;
    fgraph.write(b"#size\n")?;
    fgraph.write(g.N.to_string().as_bytes())?;
    fgraph.write("\n#vertices\n".as_bytes())?;
    fgraph.write(g.vertex.iter().map(|x| x.to_string()).collect::<Vec<String>>().join(",").as_bytes())?;
    fgraph.write("\n#adjacency\n".as_bytes())?;
    for row in &g.adj {
        fgraph.write(row.iter().map(|x| x.to_string()).collect::<Vec<String>>().join(" ").as_bytes())?;
        fgraph.write("\n".as_bytes())?;
    }  
    fgraph.write("#edges\n".as_bytes())?;
    for row in &g.edge {
        fgraph.write(row.iter().map(|x| x.to_string()).collect::<Vec<String>>().join(" ").as_bytes())?;
        fgraph.write("\n".as_bytes())?;
    }  
    fpart.write(b"#size\n")?;
    fpart.write(g.N.to_string().as_bytes())?;
    fpart.write("\n#partitions\n".as_bytes())?;
    fpart.write(partitioning.color.iter().map(|x| match x  {
        Partition::None => panic!("incomplete partition"),
        Partition::Id(i) => i.to_string()
    }).collect::<Vec<String>>().join(",").as_bytes())?;

    Ok(())
}


fn main() {
    let graph = Graph::new_randomized(100);
    let partitioner = RandomAssignation{};
    let part = partitioner.run(4, &graph);
    export_as_csv(Path::new("/tmp/"), &part).unwrap();
}
