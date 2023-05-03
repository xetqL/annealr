use std::collections::LinkedList;
use std::path::Path;

use rand::{Rng, rngs::ThreadRng};

pub fn load_imbalance(load_per_partition: &[f32]) ->f32 {
    let max = load_per_partition.iter().max_by(|a, b| a.partial_cmp(b).unwrap()).unwrap().clone();
    let mean : f32 = (load_per_partition.iter().sum::<f32>()) / load_per_partition.len() as f32;
    assert!(max >= mean);
    return max / mean;
}

pub fn var(data: &[f32]) -> f32 {
    let mean : f32 = (data.iter().sum::<f32>()) / data.len() as f32;
    data.iter().map(|value| {
        let diff = mean - *value;
        diff * diff
    }).sum::<f32>()
}

pub fn variance(load_per_partition: &[f32]) ->f32 {
    let mean_of_squares = load_per_partition.iter().fold(0f32, |acc, v| acc + v*v) / load_per_partition.len() as f32;
    let mean : f32 = (load_per_partition.iter().sum::<f32>()) / load_per_partition.len() as f32;
    return mean_of_squares - (mean * mean);
}

pub fn get_as_2d<T: Copy>(matrix_2d: &Vec<T>, row: usize, col:usize, n_cities: usize) -> T {
    matrix_2d[col + row * n_cities]
}

pub fn rand_adjacency(N: usize, rng: &mut ThreadRng) -> Vec<Vec<i32>> { 
    let mut arr = vec![ vec![0; N]; N];
    for i in 0..N {
        for j in 0..N {
            arr[i][j] = rng.gen_range(0..=1)
        }
        arr[i][i] = 0;   
    }
    arr
}

#[derive(Debug)]
pub struct Graph {
    pub n_nodes: usize,
    pub adj: Vec<Vec<i32>>,
    pub adj_list: Vec<LinkedList<i32>>,
    pub edge_weight: Vec<Vec<f32>>,
    pub vertex_weight: Vec<f32>
}

fn von_neumann_neighbors(x: usize, y: usize, xlim: usize, ylim: usize) -> [(usize, usize); 8] {
    let mut neighbors = [(0, 0); 8];

    let top_left_x = if x == 0 { xlim } else { x-1 };
    let top_left_y = if y == 0 { ylim } else { y-1 };
    neighbors[0] = (top_left_x, top_left_y);

    let top_center_x = x;
    let top_center_y = if y == 0 { ylim } else { y-1 };
    neighbors[1] = (top_center_x, top_center_y);

    let top_right_x = if x == (xlim-1) { 0 } else { x+1 };
    let top_right_y = if y == 0 { ylim } else { y-1 };
    neighbors[2] = (top_right_x, top_right_y);

    let middle_left_x = if x == 0 { xlim } else { x-1 };
    let middle_left_y = y;
    neighbors[3] = (middle_left_x, middle_left_y);

    let middle_right_x = if x == (xlim-1) { 0 } else { x+1 };
    let middle_right_y = y;
    neighbors[4] = (middle_right_x, middle_right_y);

    let bottom_left_x = if x == 0 { xlim } else { x-1 };
    let bottom_left_y = if y == (ylim-1) { 0 } else { y+1 };
    neighbors[5] = (bottom_left_x, bottom_left_y);

    let bottom_center_x = x;
    let bottom_center_y = if y == (ylim-1) { 0 } else { y+1 };
    neighbors[6] = (bottom_center_x, bottom_center_y);

    let bottom_right_x = if x == (xlim-1) { 0 } else { x+1 };
    let bottom_right_y = if y == (ylim-1) { 0 } else { y+1 };
    neighbors[7] = (bottom_right_x, bottom_right_y);

    neighbors
}

fn xy_to_i(x: usize, y: usize, xlim: usize, ylim: usize) -> usize { x + y * xlim }

impl Graph {

    pub fn new_null(n_nodes: usize) -> Self {
        Graph {
            n_nodes: n_nodes,
            adj: vec![ vec![0; n_nodes]; n_nodes],
            edge_weight:vec![ vec![0. as f32; n_nodes]; n_nodes],
            vertex_weight:vec![0. as f32; n_nodes],
            adj_list: vec![LinkedList::new(); n_nodes]
        }
    }

    pub fn new_randomized(N: usize) -> Self {
        let mut rng = rand::thread_rng();
        let adj  = rand_adjacency(N, &mut rng);
        let mut edge : Vec<Vec<f32>> = vec![ vec![0.0; N]; N];
        let mut vertex: Vec<f32> = vec![0.0; N];

        for i in 0..N {
            vertex[i] = rng.gen();
            for j in i..N {
                if adj[i][j] > 0 {
                    let v = rng.gen();
                    edge[i][j] = v; 
                    edge[j][i] = v;
                }
            }
        }

        let mut g = Graph::new_null(N);
        g.adj = adj;
        g.edge_weight = edge;
        g.vertex_weight = vertex;

        g
    }

    pub fn new_grid(grid_size_x: usize, grid_size_y: usize, edge_w: i32) -> Self {
        let mut g = Graph::new_null(grid_size_x * grid_size_y);

        for y in 0..grid_size_y {
            for x in 0..grid_size_x {
                let neighbors = von_neumann_neighbors(x, y, grid_size_x, grid_size_y);
                for (neighbor_x, neighbor_y) in neighbors {
                    let src = x + y*grid_size_x;
                    let dst = neighbor_x + neighbor_y*grid_size_x;
                    g.adj[src][dst] = edge_w;
                }
            }
        }

        g
    }

    pub fn set_vertex_weight(self, src: (usize, usize), dst: (usize, usize), weight: f32) {
        let src = xy_to_i(src.0, src.1, 0, 0);
        todo!()
    }

    fn import_from_file(path: &Path) -> Self{
        todo!()
    }

}

