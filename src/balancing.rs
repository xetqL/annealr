use std::cmp::{max, min};
use std::fmt::{Display, Formatter, Pointer};
use std::fs::File;
use std::io::BufRead;
use std::slice::{Iter, IterMut, SplitN};
use std::str::{from_boxed_utf8_unchecked, FromStr};
use std::rc::Rc;
use std::cell::RefCell;
use crate::point::{Point};
use crate::solution::{HasFitness, HasSize};


fn i_to_xy_colwise_nodiag(i: usize, n: usize) -> Result<(usize, usize), ()> {
    let triangle_size = n * (n - 1) / 2;
    if i < triangle_size {
        let kp = triangle_size - (i +1);
        let p = ((((1 + 8 * kp) as f32).sqrt() - 1f32) / 2f32).floor() as usize;
        let i = n - (kp - p * (p + 1) / 2);
        let j = n - 1 - p;
        Ok((i-1, j-1))
    } else {
        Err(())
    }
}

/**
sum c=1 to c=(y+1): (n-c) = (y+1) * n - (y+1)(y+2) / 2 = 0.5(y+1)(2n - y - 2) / (y+1)
E[w(y, n)] = 0.5(2n-y-3) if y != 0 else 0
x + E[w(y, n)] * y - (y+1)
**/
fn xy_to_i_colwise_nodiag(xy: (&usize, &usize), n: usize) -> Result<usize, ()> {
    if xy.0 != xy.1 {
        let x = *max(xy.0, xy.1) as f32;
        let y = *min(xy.0, xy.1) as f32;
        let e : f32 = if y == 0f32 {0f32} else {0.5 * (2f32*n as f32-(y-1f32)-2f32)};
        Ok(((x) + e * y - (y+1f32)) as usize)
    } else {
        Err(())
    }
}

#[derive(Debug, Copy, Clone)]
pub struct PartitionGridPoint {
    neighbors: [Option<Rc<RefCell<PartitionGridPoint>>>; 8],

    x: f32, y: f32,

    partition_id: u32
}

impl PartitionGridPoint {
    fn connect(&mut self, other: Rc<RefCell<PartitionGridPoint>>) {
        for group in self.neighbors.split_inclusive_mut(|maybe_point| maybe_point.is_none()) {
            let idx = group.len()-1;
            group[idx] = Some(other);
            break
        }
    }
}

impl Into<PartitionGridPoint> for Vec<f32> {
    fn into(self) -> PartitionGridPoint {
        PartitionGridPoint {
            neighbors: [None; 8],
            x: self[0], y: self[1],
            partition_id: 0
        }
    }
}

struct Graph<T> {
    elements: Vec<T>,
    weights: Vec<f32>,
    edges: Vec<f32> // lower diag matrix
}

impl From<File> for Graph<Point> {
    fn from(value: File) -> Self {
        let mut reader = std::io::BufReader::new(value);
        let mut buf = String::with_capacity(256);
        reader.read_line(&mut buf).expect("Number of elements is missing");
        let n_elements = usize::from_str(buf.as_str()).unwrap();
        let mut elements : Vec<Point> = Vec::with_capacity(n_elements);
        let mut weights : Vec<f32> = Vec::with_capacity(n_elements);
        // read points
        for i_element in 0..n_elements {
            reader.read_line(&mut buf).expect("Missing element");

            let mut point_and_weight = buf.splitn(2, ' ').map(|s| f32::from_str(s).unwrap());
            let weight = point_and_weight.by_ref().last().unwrap();

            let point: Vec<f32> = point_and_weight.by_ref().take(2).collect();

            elements.push_within_capacity(point.into()).unwrap();
            weights.push_within_capacity(weight).unwrap();

            buf.clear();
        }

        // read edges as semi-diag
        reader.read_line(&mut buf).expect("Missing edge");
        let semi_diag_length = (n_elements * (n_elements-1)) / 2; // sum_1^n-1
        let edges : Vec<f32> = buf.splitn(semi_diag_length-1, ' ')
            .map(|s| f32::from_str(s).unwrap()).collect();

        Self {
            elements,
            weights,
            edges,
        }
    }
}

impl From<File> for Graph<Rc<RefCell<PartitionGridPoint>>> {
    fn from(value: File) -> Self {
        let mut reader = std::io::BufReader::new(value);
        let mut buf = String::with_capacity(256);
        reader.read_line(&mut buf).expect("Number of elements is missing");
        let n_elements = usize::from_str(buf.as_str()).unwrap();
        let mut elements : Vec<Rc<RefCell<PartitionGridPoint>>> = Vec::with_capacity(n_elements);
        let mut weights : Vec<f32> = Vec::with_capacity(n_elements);

        // read points
        for i_element in 0..n_elements {
            reader.read_line(&mut buf).expect("Missing element");

            let mut point_and_weight = buf.splitn(2, ' ').map(|s| f32::from_str(s).unwrap());
            let weight = point_and_weight.by_ref().last().unwrap();

            let point: Vec<f32> = point_and_weight.by_ref().take(2).collect();

            elements.push_within_capacity(Rc::new(RefCell::new(point.into()))).unwrap();
            weights.push_within_capacity(weight).unwrap();

            buf.clear();
        }

        // read edges as semi-diag
        reader.read_line(&mut buf).expect("Missing edge");
        let semi_diag_length = (n_elements * (n_elements-1)) / 2; // sum_1^n-1
        let edges : Vec<f32> = buf.splitn(semi_diag_length-1, ' ')
            .map(|s| f32::from_str(s).unwrap()).collect();

        unsafe {
            for (i, w) in edges.iter().enumerate() {
                match i_to_xy_colwise_nodiag(i, n_elements) {

                    Ok((x, y)) => { // connect x and y
                        elements[x].connect(elements[y]);
                        elements.get_mut(y).unwrap().connect(elements.get(x).unwrap());
                    }

                    _ => panic!("algorithmic problem in i_to_xy")
                }
            }

            Self {
                elements,
                weights,
                edges,
            }
        }
    }
}

struct ContiguousPartition<'a> {
    frontier: Vec<usize>,
    body:     Vec<usize>,

    points: Vec<Rc<RefCell<PartitionGridPoint>>>,
    iter: Iter<'a, &'a Rc<RefCell<PartitionGridPoint>>>
}

impl<'a> ContiguousPartition<'a> {
    fn new() -> Self {
        let points = Vec::new();
        Self { frontier: Vec::new(), body: Vec::new(), iter: points.iter(), points }
    }
}

impl<'a> Iterator for ContiguousPartition<'a> {
    type Item = &'a Rc<RefCell<PartitionGridPoint>>;

    fn next(&mut self) -> Option<Self::Item> {
        self.iter.next().copied()
    }
}

impl ExactSizeIterator for ContiguousPartition<'_> {

}

struct LBSolution<'a, const NPartitions: usize> {
    partitions: [ContiguousPartition<'a>; NPartitions],
    problem_data: &'a Graph<Rc<RefCell<PartitionGridPoint>>>
}

struct MinMax(f32, f32);
struct Block {
    bounds_x: MinMax,
    bounds_y: MinMax,
}

fn within(block: Block, x: f32, y: f32) -> bool {
    block.bounds_x.0 <= x && x < block.bounds_x.1 &&
    block.bounds_y.0 <= y && y < block.bounds_y.1
}

impl<'a, const NPartitions: usize> LBSolution<'a, NPartitions> {
    fn new_block_partitioning(n_block_x: usize, n_block_y: usize, problem_data: &Graph<Rc<RefCell<PartitionGridPoint>>>) -> Self {
        assert_eq!(n_block_y*n_block_x, NPartitions);

        let minx = problem_data.elements.iter().map(|p| p.x).min_by(|a, b| a.total_cmp(b)).unwrap();
        let miny = problem_data.elements.iter().map(|p| p.y).min_by(|a, b| a.total_cmp(b)).unwrap();
        let maxx = problem_data.elements.iter().map(|p| p.x).max_by(|a, b| a.total_cmp(b)).unwrap();
        let maxy = problem_data.elements.iter().map(|p| p.y).max_by(|a, b| a.total_cmp(b)).unwrap();
        let width = maxx-minx;
        let width_per_block = width / n_block_x as f32;
        let height = maxy-miny;
        let height_per_block = height / n_block_y as f32;

        let mut blocks : Vec<Block> = Vec::with_capacity(NPartitions);
        for i_block_x in 0..n_block_x {
            for i_block_y in 0..n_block_y {
                let min_block_x = minx + width_per_block * i_block_x as f32;
                let min_block_y = miny + height_per_block * i_block_y  as f32;
                let block = Block {
                    bounds_x: MinMax(min_block_x, min_block_x+width_per_block),
                    bounds_y: MinMax(min_block_y, min_block_y+height_per_block),
                };

                ContiguousPartition::new(problem_data.elements.iter().filter(|p| within(block, p.borrow().x, p.borrow().y)).map(|p|p.clone()).collect());
            }
        }

        Self {
            problem_data,
            partitions: [ContiguousPartition<'a>; NPartitions]
        }
    }
}

trait Partitioning {
    fn load_imbalance(&self) -> f32;
    fn compute_edge_cut_cost(&self) -> f32;
}

impl<const NPartitions: usize> Partitioning for LBSolution<'_, NPartitions> {

    fn load_imbalance(&self) -> f32 {
        let part_sizes : Vec<usize>= self.partitions.iter().map(|l| l.len()).collect();
        let max = *part_sizes.iter().max().unwrap() as f32;
        let mu = part_sizes.iter().sum::<usize>() as f32 / NPartitions as f32;

        max - mu
    }

    fn compute_edge_cut_cost(&self) -> f32 {
       todo!()
    }
}

impl<const NPartitions: usize> Display for LBSolution<'_, NPartitions> {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        todo!()
    }
}

impl<const NPartitions: usize> HasSize for LBSolution<'_, NPartitions> {
    fn size(&self) -> usize {
        self.problem_data.elements.len()
    }
}

fn johnson_bipartition_fitness(cut: f32, alpha: f32, load_imbalance: f32) -> f32 {
    cut + alpha * f32::powi(load_imbalance, 2)
}

impl<const NPartitions: usize> HasFitness for LBSolution<'_, NPartitions> {
    // Complexity = E^2 :-(
    fn fitness(&self) -> f32 {
        johnson_bipartition_fitness(self.compute_edge_cut_cost(), 1.5f32, self.load_imbalance())
    }
}

struct MoveFrontier {}