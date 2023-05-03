
pub trait HasNext<T> {
    fn next(&mut self, solution: &mut T) -> f32;
}

pub trait HasPrev<T> {
    fn prev(&mut self, solution: &mut T) -> Result<f32, ()>;
}

pub trait HasFitness {
    fn fitness(&self) -> f32;
}

pub trait HasSize {
    fn size(&self) -> usize;
}
