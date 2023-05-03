
use std::path::Path;
use std::{fs::File, vec};
use std::fmt::{Display, Formatter};
use std::io::Write;
use std::marker::PhantomData;

use rand::{thread_rng};
use rand::{Rng, rngs::ThreadRng};
use rand::seq::SliceRandom;
use crate::solution::{*};
use crate::point::Point;

pub struct TSPProblem {
    dimension: usize,
    distances: Vec<f32>,
    positions: Vec<Point>
}

impl TSPProblem {
    pub fn distance(&self, i: usize, j: usize) -> f32 {
        self.distances.get(i + self.positions.len() * j).unwrap().clone()
    }
}

pub struct Swap<T> {
    prev: Option<(usize, usize)>,
    _phantom: PhantomData<T>
}

impl<T> Swap<T>{
    pub fn new() -> Self {
        Self {
            prev: None,
            _phantom: PhantomData::default()
        }
    }
}

pub struct TwoOptSwap;
pub struct SimpleSwap;

impl HasNext<TSPSolution<'_>> for Swap<TwoOptSwap> {
    fn next(&mut self, solution: &mut TSPSolution) -> f32{
        let len = solution.path.len();

        let i = rand::thread_rng().gen_range(0..len);
        let j = rand::thread_rng().gen_range(0..len);

        let (from, to) = if i < j {
            (i, j)
        } else {
            (i, len+j)
        };

        let mid = (to + from) / 2;

        self.prev = Some((from, to));

        for i in from+1..=mid {
            let with = to - (i - from); // to - (i - from)
            solution.path.swap(i%len, with%len);
        }

        solution.fitness()
    }

}
impl HasPrev<TSPSolution<'_>> for Swap<TwoOptSwap> {
    fn prev(&mut self, solution: &mut TSPSolution) -> Result<f32, ()> {
        if let Some((from, to)) = self.prev {
            let len = solution.path.len();
            let mid = (to + from) / 2;
            for i in from+1..=mid {
                let with = to - (i - from); // to - (i - from)
                solution.path.swap(i%len, with%len);
            }
            Ok(solution.fitness())
        } else {
            Err(())
        }
    }
}

impl HasNext<TSPSolution<'_>> for Swap<SimpleSwap> {
    fn next(&mut self, solution: &mut TSPSolution) -> f32 {
        let len = solution.path.len();
        let i = rand::thread_rng().gen_range(0..len);
        let j = rand::thread_rng().gen_range(0..len);
        self.prev = Some((i, j));
        solution.path.swap(i, j);
        solution.fitness()
    }

}
impl HasPrev<TSPSolution<'_>> for Swap<SimpleSwap> {
    fn prev(&mut self, solution: &mut TSPSolution) -> Result<f32, ()> {
        if let Some((previ, prevj)) = self.prev {
            solution.path.swap(previ, prevj);
            Ok(solution.fitness())
        } else {
            Err(())
        }
    }
}

pub struct TSPSolution<'a> {
    path: Vec<usize>,
    problem_data: &'a TSPProblem,
}

impl<'a> TSPSolution<'a> {
    pub fn new_random(problem_data: &'a TSPProblem) -> Self {
        let mut path :Vec<usize> = (0..problem_data.dimension).collect();
        path.shuffle(&mut thread_rng());
        Self {
            path,
            problem_data
        }
    }

    pub fn export(&self, file: &mut File) -> std::io::Result<()> {
        file.write_all(
            self.path.iter()
                .map(|v| v.to_string())
                .collect::<Vec<String>>().join(" ").as_bytes())?;
        file.write_all(b"\n")?;
        Ok(())
    }
}

impl Display for TSPSolution<'_> {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        f.write_fmt(format_args!("{}", self.path.iter()
                .map(|v| v.to_string())
                .collect::<Vec<String>>().join(" ")))?;
        f.write_str("\n")?;
        Ok(())
    }
}

impl HasSize for TSPSolution<'_> {
    fn size(&self) -> usize {
        self.path.len()
    }
}

impl HasFitness for TSPSolution<'_> {
    fn fitness(&self) -> f32 {
        let mut fitness = 0f32;
        let len = self.path.len();
        for i in 1..=len {
            fitness += self.problem_data.distance(self.path[(i-1) % len], self.path[(i) % len]);
        }
        fitness
    }

}

impl TSPProblem {
    pub fn new_random(n_cities: usize) -> Self {
        let mut cities_x = vec![0f32; n_cities];
        rand::thread_rng().fill(&mut cities_x[..]);
        let mut cities_y = vec![0f32; n_cities];
        rand::thread_rng().fill(&mut cities_y[..]);
        let positions : Vec<Point>  = cities_x.into_iter().zip(cities_y).map(|x| x.into()).collect();
        
        let mut distances = vec![0f32; n_cities * n_cities];
        for i in 0..n_cities {
            for j in 0..=i {
                let dist = f32::sqrt(f32::powf(positions[i].x-positions[j].x, 2f32) +
                    f32::powf(positions[i].y-positions[j].y, 2f32));
                distances[i + j*n_cities] = dist;
                distances[j + i*n_cities] = dist;
            }
        }

        TSPProblem { dimension: n_cities, distances, positions}
    }
}

pub fn save(instance: &TSPProblem, path: &Path) -> std::io::Result<()> {
    let mut tsp_problem_file = File::create(path)?;
    for (i, point) in instance.positions.iter().enumerate() {
        tsp_problem_file.write_all(format!("{} {} {}\n", i, point.x, point.y).as_bytes())?;
    }        
    Ok(())
}

pub struct TSPLib {
    instance: tsplib::Instance
}

impl From<tsplib::Instance> for TSPProblem {
    fn from(instance: tsplib::Instance) -> Self {
        TSPProblem {
            dimension: 0,
            distances: todo!(),
            positions: vec![],
        }
    }
}

fn two_opt_swap<T: Copy + PartialEq>(v: &[T], i: usize, j: usize) -> Vec<T> {
    let mut result = Vec::with_capacity(v.len());
    result.resize(v.len(), v[0]);
    result.clone_from_slice(v);

    let len = v.len();
        
    let (from, to) = if i <= j {
        (i, j)
    } else {
        (i, len+j)
    };

    let mid = (to + from) / 2;

    for i in from+1..=mid {
        let with = to - (i - from); // to - (i - from)
        result.swap(i%len, with%len);
    }

    result
}

#[test]
fn test_two_opt_middle() {
    assert_eq!(two_opt_swap(&[1,2,3],0, 2),      [1,2,3]);
    //assert_eq!(two_opt_swap(&[1,2,3,4,5], 1, 4), [1,2,4,3,5]);
}

#[test]
fn two_opt_swap_once() {
    assert_eq!(two_opt_swap(&[1,2,3,4], 0, 3),   [1,3,2,4]);
}

#[test]
fn two_opt_same_idx(){
    assert_eq!(two_opt_swap(&[1,2,3,4], 1, 1),   [1,2,3,4]);
}

#[test]
fn two_opt_same_multiple(){
    assert_eq!(two_opt_swap(&[1,2,3,4,5], 0, 4), [1,4,3,2,5]);
}

#[test]
fn two_opt_same_rev() {
    assert_eq!(two_opt_swap(&[1,2,3,4,5], 2, 0), [1,2,3,5,4]);
}
