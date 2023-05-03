#![feature(file_create_new)]
#![feature(vec_push_within_capacity)]

use std::{fs::File, path::Path};
use std::fmt::Display;
use std::io::Write;
use clap::ArgMatches;
use rand::{Rng};
use tsp::{TSPProblem, TSPSolution, save, Swap, TwoOptSwap};
use solution::{HasFitness, HasNext, HasPrev, HasSize};

pub mod solution;
pub mod utils;
pub mod tsp;
mod balancing;
pub mod point;


#[derive(PartialEq, Copy, Clone)]
#[repr(u8)]
enum ExportPace {None=0b00, AtCooling=0b01, AtIter=0b10, All=0b11}

fn accept_export(pace: u8, event: u8) -> bool {
    match event {
        0b00 => true,
        e => (e & pace)  != 0,
    }
}

impl TryFrom<u8> for ExportPace {
    type Error = ();

    fn try_from(value: u8) -> Result<Self, Self::Error> {
        match value {
            0b00 => Ok(ExportPace::None),
            0b01 => Ok(ExportPace::AtCooling),
            0b10 => Ok(ExportPace::AtIter),
            0b11 => Ok(ExportPace::All),
            _ => Err(())
        }
    }
}

#[derive(PartialEq)]
#[repr(u8)]
enum ExportEvent { Init=0, Cooling=ExportPace::AtCooling as u8, NewIter=ExportPace::AtIter as u8}

#[derive(Debug, Clone)]
struct ExportError;
trait Exporter {
    fn export<D: Display>(&mut self, display: &D, event: ExportEvent) -> Option<Result<(), ExportError>>;
}

struct StdioExporter {
    pace: ExportPace
}
impl Exporter for StdioExporter {
    fn export<D: Display>(&mut self, display: &D, event: ExportEvent) -> Option<Result<(), ExportError>> {
        if accept_export(self.pace.clone() as u8, event as u8) {
            Some(Ok(println!("Export: {}", display)))
        } else {
            None
        }
    }
}

struct FileExporter {
    pace: ExportPace,
    file: File
}
impl FileExporter {
    pub fn new(path: &Path, pace: ExportPace) -> std::io::Result<Self>{
        Ok(FileExporter {
            pace,
            file: File::create(path)?
        })
    }
}
impl Exporter for FileExporter {
    fn export<D: Display>(&mut self, display: &D, event: ExportEvent) -> Option<Result<(), ExportError>> {
        if accept_export(self.pace.clone() as u8, event as u8) {
            Some(write!(self.file, "{}", display).map_err(|_| ExportError))
        } else {
            None
        }
    }
}

#[derive(Clone, Copy)]
struct CoolingFactor(f32);
impl From<CoolingFactor> for f32 {
    fn from(value: CoolingFactor) -> Self {
        value.0
    }
}

#[derive(Debug, Clone)]
struct RatioBoundError;

impl CoolingFactor {
    pub fn new(value: f32) -> Result<Self, RatioBoundError>{
        if 0f32 <= value && value <= 1f32 {
            Ok(CoolingFactor(value))
        } else {
            Err(RatioBoundError)
        }
    }
}

fn simulated_annealing<S: HasSize + HasFitness + Display,
                       N: HasNext<S> + HasPrev<S>,
                       E: Exporter> (
        cooling: CoolingFactor,
        initial_solution: S,
        strategy: &mut N,
        exporters: &mut [E]) -> S {
    const INITIAL_ACCEPTANCE_PROBABILITY : f32 = 0.9;
    const N_ITER_BOOTSTRAP : usize = 100;

    // bootstrap

    let mut current_solution = initial_solution;

    let mut average_energy = 0f32;

    for _ in 0..N_ITER_BOOTSTRAP {
        let current_fitness = current_solution.fitness();
        average_energy += (strategy.next(&mut current_solution) - current_fitness) / N_ITER_BOOTSTRAP as f32;
    } 

    let decrease_threshold_on_accept = 12 * current_solution.size() as u32;
    let decrease_threshold_on_tried  = 100 * current_solution.size() as u32;
    let stop_after_step_not_improving = 10;

    let mut rng = rand::thread_rng();
    let mut temperature = -f32::abs(average_energy) / f32::ln(INITIAL_ACCEPTANCE_PROBABILITY);
    let mut n_accept = 0u32;
    let mut n_iter = 0u32;
    let mut n_not_improving_step = 0u32;
    let mut best_fitness_step = current_solution.fitness();
    let mut all_time_best = f32::MAX;
    let mut improve = false;

    while n_not_improving_step < stop_after_step_not_improving {
        n_iter += 1;

        let current_fitness : f32 = {
            let current_fitness = current_solution.fitness();
            let next_fitness = strategy.next(&mut current_solution);
            if rng.gen_range(0f32..=1f32) < f32::exp(-(next_fitness-current_fitness) / temperature) {
                n_accept += 1;
                next_fitness
            } else {
                strategy.prev(&mut current_solution).unwrap()
            }
        };

        if best_fitness_step - current_fitness > best_fitness_step*0.001 {
            best_fitness_step = current_fitness;
            improve = true;
            exporters.iter_mut().for_each(
                |exporter| { exporter.export(&current_solution, ExportEvent::Init); });
        }

        if n_accept >= decrease_threshold_on_accept || n_iter >= decrease_threshold_on_tried {
            temperature *= f32::from(cooling);

            n_not_improving_step = if !improve {n_not_improving_step + 1} else {0};
            n_accept = 0;
            n_iter = 0;
            improve = false;

            println!("Best: {}\tCurrent fitness: {}\tTemperature: {}\t{}", best_fitness_step, current_fitness, temperature,n_not_improving_step);

            all_time_best = all_time_best.min( current_fitness);
            best_fitness_step = current_fitness;
        }
    }

    current_solution
}

fn tsp_args() -> ArgMatches {
    clap::Command::new("annealr")
    .version("0.0.1")
    .author("Anthony B. <shortab@pm.me>")
    .about("Simulated annealing")
    .arg(clap::arg!(--tsplib <FILE>))
    .arg(clap::arg!(--ncities <N>).value_parser(clap::value_parser!(usize)))
    .arg(clap::Arg::new("export-at-cooling").long("export-at-cooling").action(clap::ArgAction::SetTrue))
    .arg(clap::Arg::new("export-at-iter").long("export-at-iter").action(clap::ArgAction::SetTrue))
    .get_matches()
}

fn lb_args() -> ArgMatches {
    clap::Command::new("annealr")
    .version("0.0.1")
    .author("Anthony B. <shortab@pm.me>")
    .about("Simulated annealing")
    .arg(clap::Arg::new("export-at-cooling").long("export-at-cooling").action(clap::ArgAction::SetTrue))
    .arg(clap::Arg::new("export-at-iter").action(clap::ArgAction::SetTrue))
    .get_matches()
}

fn tsp_run() {
    let matches = tsp_args();

    let export_pace = (if matches.get_flag("export-at-iter") {
        ExportPace::AtIter
    } else {
        ExportPace::None
    } as u8) | (if matches.get_flag("export-at-cooling") {
        ExportPace::AtCooling
    } else {
        ExportPace::None
    } as u8);

    let ncities : usize = *matches.get_one::<usize>("ncities").expect("msg");

    let tsp : TSPProblem = TSPProblem::new_random(ncities);
    let s = TSPSolution::new_random(&tsp);
    let mut strat : Swap<TwoOptSwap> = Swap::new();

    save(&tsp, Path::new("cities.tsp")).unwrap();


    let pace = ExportPace::try_from(export_pace).unwrap();

    simulated_annealing(CoolingFactor::new(0.9f32).unwrap(), s,
                        &mut strat,
                        [FileExporter::new(Path::new("./tour.tsp"), pace).unwrap()].as_mut_slice());
}

fn lb_run() {
    /*
    let matches = lb_args();

    let export_pace = if matches.get_flag("export-at-iter") {
        ExportPace::EachIter
    } else if matches.get_flag("export-at-cooling") {
        ExportPace::EachCooling
    } else {
        ExportPace::None
    };

    let g = Graph::new_grid(25, 25, 1);

    let s = Partitioning::new_random(4, &g);

    simulated_annealing(0.95f32, s, export_pace);
    */
}

fn main() {
    tsp_run();
}
